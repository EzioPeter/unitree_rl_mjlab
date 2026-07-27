from __future__ import annotations

import copy
import json
import math
from collections import deque
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from flash_rl.buffers.torch_buffer import TorchUniformBuffer

from scripts.reinforcement_learning.rwm_trace.feedback_manager import (
    CodexBatchLabelProvider,
    CumulativeFeedbackManager,
    pair_quota,
)
from scripts.reinforcement_learning.rwm_trace.feedback_pairs import (
    COMMAND_REGIONS,
    PairQuota,
    _sample_ranks_without_replacement,
    build_batch_global_random_pairs,
    build_feedback_pairs,
    build_trace_feedback_pairs,
    command_region,
)
from scripts.reinforcement_learning.rwm_trace.go2_feedback_prompt import (
    build_go2_feedback_prompt,
)
from scripts.reinforcement_learning.rwm_trace.label_feedback_with_codex import (
    build_batch_prompt,
    build_request_hash,
    salvage_label_response,
)
from scripts.reinforcement_learning.rwm_trace.materializer import materialize_selected
from scripts.reinforcement_learning.rwm_trace.online_trace_manager import (
    Go2OnlineTraceManager,
    OnlineTraceConfig,
)
from scripts.reinforcement_learning.rwm_trace.online_scorer_update import (
    decayed_feedback_budget,
)
from scripts.reinforcement_learning.rwm_trace.policy_context import (
    attach_policy_context,
)
from scripts.reinforcement_learning.rwm_trace.proposal import (
    FlashSACActorDistributionSampler,
    ProposalConfig,
    V13MJLabProposalCollector,
)
from scripts.reinforcement_learning.rwm_trace.replay import (
    REPLAY_KEYS,
    MutableTraceReplayBuffer,
    mix_trace_within_synthetic,
)
from scripts.reinforcement_learning.rwm_trace.rule_bootstrap import (
    RuleBootstrapConfig,
    score_rule_summaries,
)
from scripts.reinforcement_learning.rwm_trace.schemas import (
    COMMAND_REGION_FEATURE_NAMES,
    PROMPT_HASH,
    SCORER_EXPANDED_FEATURE_NAMES,
    SCORER_FEATURE_NAMES,
)
from scripts.reinforcement_learning.rwm_trace.scorer import (
    Go2TraceScorer,
    ScorerBinding,
    fit_feature_stats,
    load_scorer_checkpoint,
    raw_feature_matrix,
    save_scorer_checkpoint,
    transform_features,
)
from scripts.reinforcement_learning.rwm_trace.selection import global_top_alpha
from scripts.reinforcement_learning.rwm_trace.trajectory import (
    build_go2_llm_display,
    summarize_go2_trajectory,
)
from scripts.reinforcement_learning.rwm_trace.v13_replay_adapter import (
    CanonicalV13ReplaySemantics,
)
from scripts.reinforcement_learning.rwm_flashsac.train_flashsac_world_model_go2 import (
    _load_training_state,
    _save_checkpoint,
    _start_async_checkpoint,
    _wait_async_checkpoint,
)
from scripts.reinforcement_learning.rwm_flashsac.world_model_env import (
    Go2RWMFlashSACWorldModelEnv,
)


def trajectory(
    identity: str,
    *,
    start: str,
    command=(0.5, 0.0, 0.0),
    actual=(0.4, 0.02, 0.0),
) -> dict:
    length = 6
    states = torch.zeros(length, 45)
    next_states = torch.zeros(length, 45)
    states[:, 8] = -1.0
    next_states[:, 8] = -1.0
    next_states[:, 0] = actual[0]
    next_states[:, 1] = actual[1]
    next_states[:, 5] = actual[2]
    return {
        "states": states,
        "next_states": next_states,
        "actions": torch.zeros(length, 12),
        "commands": torch.tensor(command).repeat(length, 1),
        "contacts": torch.ones(length, 4),
        "rewards": torch.arange(1, length + 1, dtype=torch.float32),
        "terminations": torch.zeros(length, dtype=torch.bool),
        "prev_actions": torch.zeros(length, 12),
        "trajectory_id": identity,
        "start_state_id": start,
        "start_state_key": start,
        "comparison_group_key": start,
        "realized_snapshot_hash": f"snapshot-{start}",
        "simulator_config_hash": "sim-config",
        "source_kind": "same_start_candidate",
        "step_dt": 0.02,
        "expected_trajectory_length": length,
        "command_active_thresholds": (0.03, 0.02, 0.03),
        "command_normalization_floors": (0.05, 0.05, 0.05),
        "action_saturation_threshold": 0.9,
    }


def summary(identity: str, *, start: str, command=(0.5, 0.0, 0.0)) -> dict:
    return summarize_go2_trajectory(
        trajectory(identity, start=start, command=command)
    )


def test_policy_context_adds_region_one_hot_and_two_dynamic_features() -> None:
    rows = attach_policy_context(
        [
            summary("front", start="a"),
            summary("left", start="b", command=(0.0, 0.2, 0.0)),
        ],
        planar_command_scales=(0.5, 0.2),
        replay_cohort_counts={"front": 20},
    )
    assert SCORER_FEATURE_NAMES[-2:] == (
        "policy_gap_score",
        "replay_shortage_score",
    )
    assert all(name in SCORER_FEATURE_NAMES for name in COMMAND_REGION_FEATURE_NAMES)
    assert all(
        name in SCORER_FEATURE_NAMES
        for name in (
            "command_vx_abs_mean",
            "command_vy_abs_mean",
            "command_yaw_abs_mean",
        )
    )
    assert sum(rows[0][name] for name in COMMAND_REGION_FEATURE_NAMES) == 1.0
    assert sum(rows[1][name] for name in COMMAND_REGION_FEATURE_NAMES) == 1.0
    assert rows[0]["command_region_front"] == 1.0
    assert rows[1]["command_region_left"] == 1.0
    assert rows[1]["policy_gap_score"] > rows[0]["policy_gap_score"]
    assert rows[1]["replay_shortage_score"] == 1.0
    assert rows[0]["replay_shortage_score"] == 0.0
    assert np.isfinite(raw_feature_matrix(rows)[:, -2:]).all()
    display = build_go2_llm_display(rows[1])
    assert display["current_policy_context"]["cohort"] == "left"
    assert all(
        "command_abs_mean" in axis
        for axis in display["velocity_tracking"].values()
    )


def test_dataset_eight_way_command_modes_never_reach_prompt_or_scorer() -> None:
    left = summary("left", start="s0", command=(0.5, 0.0, 0.0))
    right = summary("right", start="s1", command=(0.0, 0.2, 0.0))
    for row, mode in ((left, "pure_x"), (right, "xy_yaw")):
        row["command_mode"] = mode
        row["command_mode_id"] = 7
        row["metadata"] = {
            "command_modes": [
                "stand",
                "pure_x",
                "pure_y",
                "pure_yaw",
                "xy",
                "x_yaw",
                "y_yaw",
                "xy_yaw",
            ],
            "command_mode_weights": {"pure_x": 1.0},
        }
    pair = build_batch_global_random_pairs(
        [left, right],
        pair_count=1,
        pair_prefix="no-mode-leak",
        seed=9,
        planar_command_scales=(0.5, 0.2),
    )[0]
    serialized_pair = json.dumps(pair, sort_keys=True)
    assert "command_mode" not in serialized_pair
    assert "pure_x" not in serialized_pair
    assert "xy_yaw" not in serialized_pair
    assert pair["trajectory_i"]["command_vx_abs_mean"] == pytest.approx(
        left["command_vx_abs_mean"]
    ) or pair["trajectory_j"]["command_vx_abs_mean"] == pytest.approx(
        left["command_vx_abs_mean"]
    )
    prompt = build_go2_feedback_prompt(pair)
    assert "command_mode" not in prompt
    assert not any(name.startswith("command_mode_") for name in SCORER_FEATURE_NAMES)


def valid_label(pair_id: str) -> dict:
    return {
        "pair_id": pair_id,
        "feedback": "i",
        "confidence": 0.9,
        "reason": "i has lower vx tracking error and equal tilt_max.",
        "used_criteria": ["velocity_tracking", "posture"],
        "return_used_as_primary": False,
        "motion_quality_overrode_return": False,
        "velocity_tracking_used_as_primary": True,
        "posture_veto_applied": False,
    }


def test_offline_label_request_hash_binds_prompt_schema_model_and_reasoning(
    tmp_path,
) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text('{"type":"object"}', encoding="utf-8")
    rows = [{"pair_id": "p0", "prompt": "prompt"}]
    baseline = build_request_hash(
        rows,
        schema_path=schema,
        model="gpt-5.5",
        reasoning_effort="medium",
    )
    assert build_request_hash(
        [{"pair_id": "p0", "prompt": "different"}],
        schema_path=schema,
        model="gpt-5.5",
        reasoning_effort="medium",
    ) != baseline
    assert build_request_hash(
        rows,
        schema_path=schema,
        model="different-model",
        reasoning_effort="medium",
    ) != baseline
    assert build_request_hash(
        rows,
        schema_path=schema,
        model="gpt-5.5",
        reasoning_effort="high",
    ) != baseline
    schema.write_text('{"type":"array"}', encoding="utf-8")
    assert build_request_hash(
        rows,
        schema_path=schema,
        model="gpt-5.5",
        reasoning_effort="medium",
    ) != baseline


def test_feedback_budget_matches_d4rl_decay_schedule() -> None:
    budgets = [
        decayed_feedback_budget(200, 20, 0.8, refresh_index)
        for refresh_index in range(97)
    ]
    assert budgets[:5] == [200, 160, 128, 102, 81]
    assert budgets[-1] == 20
    assert all(left >= right for left, right in zip(budgets, budgets[1:]))


def test_codex_provider_runs_batches_concurrently(
    tmp_path, monkeypatch
) -> None:
    import threading
    import time

    monkeypatch.setattr(
        "scripts.reinforcement_learning.rwm_trace.feedback_manager.shutil.which",
        lambda _name: "/usr/bin/codex",
    )
    provider = CodexBatchLabelProvider(
        repo_root=tmp_path,
        schema_path=(
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "scripts/reinforcement_learning/rwm_trace"
            / "feedback_label_batch.schema.json"
        ),
        control_dir=tmp_path / "control",
        batch_size=2,
        repair_rounds=0,
        workers=4,
        max_retries=0,
    )
    lock = threading.Lock()
    active = 0
    max_active = 0
    call_count = 0

    def fake_call(rows, *, call_id):
        nonlocal active, max_active, call_count
        assert call_id.startswith("r00_batch")
        with lock:
            active += 1
            call_count += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return {
            "labels": [valid_label(str(row["pair_id"])) for row in rows]
        }

    monkeypatch.setattr(provider, "_call_with_retries", fake_call)
    pairs = [
        {"pair_id": f"p{index}", "prompt": f"prompt {index}"}
        for index in range(8)
    ]
    labels = provider(pairs)
    assert len(labels) == 8
    assert call_count == 4
    assert max_active == 4


def test_codex_provider_reuses_completed_batch_response(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "scripts.reinforcement_learning.rwm_trace.feedback_manager.shutil.which",
        lambda _name: "/usr/bin/codex",
    )
    provider = CodexBatchLabelProvider(
        repo_root=tmp_path,
        schema_path=(
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "scripts/reinforcement_learning/rwm_trace"
            / "feedback_label_batch.schema.json"
        ),
        control_dir=tmp_path / "control",
        batch_size=20,
        repair_rounds=0,
        workers=1,
        max_retries=0,
    )
    provider.control_dir.mkdir()
    pair = {"pair_id": "p0", "prompt": "prompt"}
    request_hash = provider._request_hash(build_batch_prompt([pair]))
    response = provider.control_dir / (
        f"r00_batch00000_try00_{request_hash}.json"
    )
    response.write_text(
        __import__("json").dumps({"labels": [valid_label("p0")]}),
        encoding="utf-8",
    )

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("A completed LLM batch must not be called again.")

    monkeypatch.setattr(
        "scripts.reinforcement_learning.rwm_trace.feedback_manager.subprocess.run",
        unexpected_call,
    )
    assert provider([pair]) == [
        valid_label("p0")
    ]


def test_codex_provider_does_not_reuse_same_call_id_for_new_request(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "scripts.reinforcement_learning.rwm_trace.feedback_manager.shutil.which",
        lambda _name: "/usr/bin/codex",
    )
    provider = CodexBatchLabelProvider(
        repo_root=tmp_path,
        schema_path=(
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "scripts/reinforcement_learning/rwm_trace"
            / "feedback_label_batch.schema.json"
        ),
        control_dir=tmp_path / "control",
        batch_size=20,
        repair_rounds=0,
        workers=1,
        max_retries=0,
        model="gpt-5.5",
        reasoning_effort="medium",
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(kwargs["input"])
        response = __import__("pathlib").Path(
            command[command.index("-o") + 1]
        )
        pair_id = "p0" if len(calls) == 1 else "p1"
        response.write_text(
            __import__("json").dumps(
                {"labels": [valid_label(pair_id)]}
            ),
            encoding="utf-8",
        )
        return type(
            "Result", (), {"returncode": 0, "stdout": "", "stderr": ""}
        )()

    monkeypatch.setattr(
        "scripts.reinforcement_learning.rwm_trace.feedback_manager.subprocess.run",
        fake_run,
    )
    first = provider._call(
        [{"pair_id": "p0", "prompt": "first"}],
        call_id="r00_batch00000_try00",
    )
    second = provider._call(
        [{"pair_id": "p1", "prompt": "second"}],
        call_id="r00_batch00000_try00",
    )
    assert first == {"labels": [valid_label("p0")]}
    assert second == {"labels": [valid_label("p1")]}
    assert len(calls) == 2
    assert len(list(provider.control_dir.glob("*.json"))) == 2


def test_codex_provider_request_hash_binds_schema_model_and_reasoning(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "scripts.reinforcement_learning.rwm_trace.feedback_manager.shutil.which",
        lambda _name: "/usr/bin/codex",
    )
    schema = tmp_path / "schema.json"
    schema.write_text('{"type":"object"}', encoding="utf-8")
    provider = CodexBatchLabelProvider(
        repo_root=tmp_path,
        schema_path=schema,
        control_dir=tmp_path / "control",
        model="gpt-5.5",
        reasoning_effort="medium",
    )
    prompt = "complete batch prompt"
    baseline = provider._request_hash(prompt)
    provider.model = "different-model"
    assert provider._request_hash(prompt) != baseline
    provider.model = "gpt-5.5"
    provider.reasoning_effort = "high"
    assert provider._request_hash(prompt) != baseline
    provider.reasoning_effort = "medium"
    schema.write_text('{"type":"array"}', encoding="utf-8")
    assert provider._request_hash(prompt) != baseline


def test_codex_provider_passes_explicit_model_and_reasoning_effort(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "scripts.reinforcement_learning.rwm_trace.feedback_manager.shutil.which",
        lambda _name: "/usr/bin/codex",
    )
    schema = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts/reinforcement_learning/rwm_trace"
        / "feedback_label_batch.schema.json"
    )
    provider = CodexBatchLabelProvider(
        repo_root=tmp_path,
        schema_path=schema,
        control_dir=tmp_path / "control",
        model="gpt-5.5",
        reasoning_effort="medium",
        workers=1,
        max_retries=0,
    )
    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        response = __import__("pathlib").Path(
            command[command.index("-o") + 1]
        )
        response.write_text(
            __import__("json").dumps({"labels": [valid_label("p0")]}),
            encoding="utf-8",
        )
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(
        "scripts.reinforcement_learning.rwm_trace.feedback_manager.subprocess.run",
        fake_run,
    )
    response = provider._call(
        [{"pair_id": "p0", "prompt": "prompt"}],
        call_id="smoke",
    )
    assert response == {"labels": [valid_label("p0")]}
    assert captured["command"][2:6] == [
        "--config",
        'model_reasoning_effort="medium"',
        "--model",
        "gpt-5.5",
    ]


def replay_batch(count: int, value: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "observation": torch.full((count, 48), value),
        "action": torch.full((count, 12), value),
        "reward": torch.full((count,), value),
        "terminated": torch.zeros(count),
        "truncated": torch.zeros(count),
        "next_observation": torch.full((count, 48), value),
    }


def test_summary_velocity_primary_and_return_visible() -> None:
    row = summary("a", start="s0")
    assert "command_mode" not in row
    assert not any(name.startswith("command_mode_") for name in SCORER_FEATURE_NAMES)
    assert row["command_mean"] == pytest.approx(
        {"vx": 0.5, "vy": 0.0, "yaw": 0.0}
    )
    assert row["tracking_vx_mae_full"] == pytest.approx(0.1)
    assert row["realization_vx_full"] == pytest.approx(0.8)
    assert math.isnan(row["tracking_vy_mae_full"])
    assert row["drift_abs_mean_vy"] == pytest.approx(0.02)
    assert row["simulator_return"] == pytest.approx(21.0)
    assert row["behavior_return"] == pytest.approx(21.0)
    assert row["tilt_max"] == pytest.approx(0.0)


def test_prompt_contains_go2_order_and_exact_label_contract() -> None:
    left, right = summary("a", start="s0"), summary("b", start="s0")
    pair = {
        "pair_id": "p",
        "comparison_type": "within_command_region",
        "command_region_i": "front",
        "command_region_j": "front",
        "same_command_region": True,
        "planar_command_scales": [0.5, 0.2],
        "trajectory_i": left,
        "trajectory_j": right,
    }
    prompt = build_go2_feedback_prompt(pair)
    assert "marginal coverage value" in prompt
    assert "already-mastered" in prompt
    assert "informative imperfection" in prompt
    assert "Do not reward an easy or common stand/front/back" in prompt
    assert "posture and survival veto" in prompt
    assert "simulator_return" in prompt
    assert "velocity_tracking_used_as_primary" in prompt
    assert "posture_veto_applied" in prompt
    assert '"command_region_i": "front"' in prompt
    assert '"same_command_region": true' in prompt
    assert "same-state advantage" not in prompt


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ((0.5, 0.0, 0.0), "front"),
        ((-0.5, 0.0, 0.0), "back"),
        ((0.0, 0.2, 0.0), "left"),
        ((0.0, -0.2, 0.0), "right"),
        ((0.0, 0.0, 0.4), "pure_yaw"),
        ((0.0, 0.0, 0.0), "stand"),
    ],
)
def test_all_six_command_regions(command, expected) -> None:
    row = summary(expected, start=expected, command=command)
    assert command_region(row, (0.5, 0.2)) == expected


def test_command_region_diagonal_tie_is_front() -> None:
    row = summary("diagonal", start="s", command=(0.25, 0.1, 0.0))
    assert command_region(row, (0.5, 0.2)) == "front"


def test_yaw_translation_region_uses_only_planar_direction() -> None:
    row = summary("yaw-translation", start="s", command=(0.1, -0.2, 0.4))
    assert command_region(row, (0.5, 0.2)) == "right"


def test_fixed_region_pair_quota_is_reproducible_and_score_free() -> None:
    rows = [
        summary(
            f"{region}-{index}",
            start=f"{region}-{index}",
            command=command,
        )
        for region, command in (
            ("front", (0.5, 0.0, 0.0)),
            ("back", (-0.5, 0.0, 0.0)),
            ("left", (0.0, 0.2, 0.0)),
            ("right", (0.0, -0.2, 0.0)),
            ("pure_yaw", (0.0, 0.0, 0.4)),
            ("stand", (0.0, 0.0, 0.0)),
        )
        for index in range(4)
    ]
    first = build_feedback_pairs(
        rows,
        quota=PairQuota(within_region=8, cross_region=2),
        pair_prefix="test",
        seed=7,
        planar_command_scales=(0.5, 0.2),
    )
    second = build_feedback_pairs(
        rows,
        quota=PairQuota(within_region=8, cross_region=2),
        pair_prefix="test",
        seed=7,
        planar_command_scales=(0.5, 0.2),
    )
    assert first == second
    assert len(first) == 10
    assert sum(row["same_command_region"] for row in first) == 8
    assert all("pair_score_gap" not in row for row in first)
    assert all("score" not in row for row in first)


def test_pair_quota_uses_exact_round_80_20() -> None:
    assert pair_quota(200, 0.2) == PairQuota(160, 40)
    assert pair_quota(81, 0.2) == PairQuota(65, 16)


def test_global_pair_sampling_is_unique_reproducible_and_not_quota_forced() -> None:
    rows = [
        summary(f"row-{index}", start=f"s{index}", command=command)
        for index, command in enumerate(
            [(0.5, 0.0, 0.0)] * 8
            + [(-0.5, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.4), (0.0, 0.0, 0.0)]
        )
    ]
    first = build_trace_feedback_pairs(
        rows, pair_count=20, pair_prefix="global", seed=42,
        planar_command_scales=(0.5, 0.2),
    )
    second = build_trace_feedback_pairs(
        rows, pair_count=20, pair_prefix="global", seed=42,
        planar_command_scales=(0.5, 0.2),
    )
    assert first == second
    identities = {
        tuple(sorted((row["trajectory_i"]["trajectory_id"], row["trajectory_j"]["trajectory_id"])))
        for row in first
    }
    assert len(identities) == len(first) == 20
    assert all(row["pair_sampling_mode"] == "trace_original" for row in first)


def test_batch_global_random_is_uniform_over_complete_batch_and_score_free() -> None:
    rows = [
        summary(
            f"row-{index}",
            start=("shared" if index < 8 else f"s{index}"),
            command=command,
        )
        for index, command in enumerate(
            [(0.5, 0.0, 0.0)] * 8
            + [
                (-0.5, 0.0, 0.0),
                (0.0, 0.2, 0.0),
                (0.0, -0.2, 0.0),
                (0.0, 0.0, 0.4),
                (0.0, 0.0, 0.0),
            ]
        )
    ]
    first = build_batch_global_random_pairs(
        rows,
        pair_count=30,
        pair_prefix="batch",
        seed=42,
        planar_command_scales=(0.5, 0.2),
    )
    second = build_batch_global_random_pairs(
        rows,
        pair_count=30,
        pair_prefix="batch",
        seed=42,
        planar_command_scales=(0.5, 0.2),
    )
    assert first == second
    identities = {
        tuple(
            sorted(
                (
                    row["trajectory_i"]["trajectory_id"],
                    row["trajectory_j"]["trajectory_id"],
                )
            )
        )
        for row in first
    }
    assert len(identities) == len(first) == 30
    assert all(row["pair_sampling_mode"] == "batch_global_random" for row in first)
    assert any(
        row["trajectory_i"]["comparison_group_key"]
        != row["trajectory_j"]["comparison_group_key"]
        for row in first
    )
    assert any(not row["same_command_region"] for row in first)
    assert all("score" not in row and "pair_score_gap" not in row for row in first)


def test_region_sampling_does_not_balance_region_quotas() -> None:
    rows = [
        *[
            summary(f"front-{index}", start=f"f{index}", command=(0.5, 0.0, 0.0))
            for index in range(4)
        ],
        *[
            summary(f"back-{index}", start=f"b{index}", command=(-0.5, 0.0, 0.0))
            for index in range(2)
        ],
        *[
            summary(f"left-{index}", start=f"l{index}", command=(0.0, 0.2, 0.0))
            for index in range(2)
        ],
    ]
    pairs = build_feedback_pairs(
        rows,
        quota=PairQuota(within_region=8, cross_region=0),
        pair_prefix="unbalanced",
        seed=1,
        planar_command_scales=(0.5, 0.2),
    )
    counts = {
        region: sum(
            row["command_region_i"] == region
            and row["command_region_j"] == region
            for row in pairs
        )
        for region in COMMAND_REGIONS
    }
    assert counts["front"] == 6
    assert counts["back"] == 1
    assert counts["left"] == 1


def test_large_pair_population_sampling_does_not_materialize_population() -> None:
    rng = np.random.default_rng(9)
    ranks = _sample_ranks_without_replacement(
        4096 * 4095 // 2,
        200,
        rng=rng,
        kind="test",
    )
    assert len(ranks) == len(set(ranks)) == 200
    assert min(ranks) >= 0
    assert max(ranks) < 4096 * 4095 // 2


def test_partial_label_salvage_keeps_good_pair() -> None:
    good = valid_label("p0")
    bad = {**valid_label("p1"), "confidence": 2.0}
    accepted, failures = salvage_label_response(
        {"labels": [good, bad]}, ["p0", "p1", "p2"]
    )
    assert set(accepted) == {"p0"}
    assert set(failures) == {"p1", "p2"}


def test_cumulative_feedback_persists_partial_valid_labels(tmp_path) -> None:
    rows = [
        summary("a0", start="a"),
        summary("a1", start="a"),
        summary("b0", start="b", command=(-0.5, 0.0, 0.0)),
        summary("b1", start="b", command=(-0.5, 0.0, 0.0)),
    ]

    def partial_provider(pairs):
        return [valid_label(pairs[0]["pair_id"])]

    manager = CumulativeFeedbackManager(
        label_store_path=tmp_path / "labels.jsonl",
        confidence_threshold=0.7,
        label_provider=partial_provider,
        pair_seed=3,
        planar_command_scales=(0.5, 0.2),
    )
    with pytest.raises(RuntimeError, match="missing/invalid"):
        manager.request(rows, feedback_budget=2, cross_region_fraction=0.5)
    assert manager.label_count == 1
    assert manager.refresh_count == 1
    assert len(manager.cumulative_training_labels()) == 1
    assert (tmp_path / "labels.jsonl").is_file()


def test_cumulative_feedback_starts_from_exact_initial_labels(tmp_path) -> None:
    rows = [summary("a0", start="a"), summary("a1", start="a")]
    pair = build_batch_global_random_pairs(
        rows,
        pair_count=1,
        pair_prefix="initial",
        seed=7,
        planar_command_scales=(0.5, 0.2),
    )[0]
    initial = {
        **{key: value for key, value in pair.items() if key != "prompt"},
        **valid_label(pair["pair_id"]),
        "prompt_hash": PROMPT_HASH,
    }
    initial_path = tmp_path / "initial.jsonl"
    initial_path.write_text(
        json.dumps(initial, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manager = CumulativeFeedbackManager(
        label_store_path=tmp_path / "run" / "cumulative.jsonl",
        initial_labels_path=initial_path,
        confidence_threshold=0.7,
        label_provider=lambda _pairs: [],
        pair_seed=3,
        pair_sampling_mode="batch_global_random",
        planar_command_scales=(0.5, 0.2),
    )
    assert manager.initial_label_count == 1
    assert manager.label_count == 1
    assert manager.cumulative_training_labels()[0]["pair_id"] == pair["pair_id"]
    assert (tmp_path / "run" / "cumulative.jsonl").read_text(
        encoding="utf-8"
    ) == initial_path.read_text(encoding="utf-8")


def test_cumulative_feedback_resume_restores_labels_to_new_run_path(tmp_path) -> None:
    rows = [summary("a0", start="a"), summary("a1", start="a")]

    def complete_provider(pairs):
        return [valid_label(pair["pair_id"]) for pair in pairs]

    first = CumulativeFeedbackManager(
        label_store_path=tmp_path / "old" / "labels.jsonl",
        confidence_threshold=0.7,
        label_provider=complete_provider,
        pair_seed=3,
        planar_command_scales=(0.5, 0.2),
    )
    first.request(rows, feedback_budget=1, cross_region_fraction=0.0)
    state = copy.deepcopy(first.state_dict())
    resumed = CumulativeFeedbackManager(
        label_store_path=tmp_path / "new" / "labels.jsonl",
        confidence_threshold=0.7,
        label_provider=complete_provider,
        pair_seed=3,
        planar_command_scales=(0.5, 0.2),
    )
    resumed.load_state_dict(state)
    assert resumed.labels_sha256 == first.labels_sha256
    assert resumed.label_count == first.label_count
    assert resumed.refresh_count == first.refresh_count
    assert (tmp_path / "new" / "labels.jsonl").is_file()
    expected_pairs, _ = first.request(
        rows, feedback_budget=1, cross_region_fraction=0.0
    )
    actual_pairs, _ = resumed.request(
        rows, feedback_budget=1, cross_region_fraction=0.0
    )
    assert [
        (
            row["pair_id"],
            row["trajectory_i"]["trajectory_id"],
            row["trajectory_j"]["trajectory_id"],
        )
        for row in actual_pairs
    ] == [
        (
            row["pair_id"],
            row["trajectory_i"]["trajectory_id"],
            row["trajectory_j"]["trajectory_id"],
        )
        for row in expected_pairs
    ]
    assert resumed.labels_sha256 == first.labels_sha256


def test_pre_region_feedback_checkpoint_fails_closed(tmp_path) -> None:
    manager = CumulativeFeedbackManager(
        label_store_path=tmp_path / "labels.jsonl",
        confidence_threshold=0.7,
        label_provider=lambda _pairs: [],
        pair_seed=3,
        planar_command_scales=(0.5, 0.2),
    )
    with pytest.raises(ValueError, match="predates six-region"):
        manager.load_state_dict(
            {
                "refresh_count": 1,
                "pair_seed": 3,
                "labels": [],
                "labels_sha256": "",
                "label_count": 0,
            }
        )


def test_feature_missingness_and_checkpoint_binding(tmp_path) -> None:
    rows = [summary("a", start="a"), summary("b", start="b")]
    raw = raw_feature_matrix(rows)
    assert np.isnan(raw).any()
    stats = fit_feature_stats(raw)
    transformed = transform_features(raw, stats)
    assert transformed.shape == (2, len(SCORER_EXPANDED_FEATURE_NAMES))
    assert np.isfinite(transformed).all()
    model = Go2TraceScorer(transformed.shape[1], 256)
    binding = ScorerBinding(
        task_id="task",
        dataset_id="dataset",
        dataset_sha256="dataset-hash",
        condition_id="g0",
        labels_sha256="labels-hash",
        split_sha256="split-hash",
    )
    path = tmp_path / "scorer.pt"
    save_scorer_checkpoint(
        path, model, stats, binding=binding, training_metrics={"loss": 1.0}
    )
    loaded, _, actual, _ = load_scorer_checkpoint(
        path, expected_binding=binding
    )
    assert isinstance(loaded, Go2TraceScorer)
    assert actual == binding
    mismatch = ScorerBinding(
        task_id="other",
        dataset_id="dataset",
        dataset_sha256="dataset-hash",
        condition_id="g0",
        labels_sha256="labels-hash",
        split_sha256="split-hash",
    )
    with pytest.raises(ValueError, match="does not match"):
        load_scorer_checkpoint(path, expected_binding=mismatch)


def test_global_top_alpha_is_single_global_ranking() -> None:
    rows = [summary(str(i), start=str(i // 2)) for i in range(5)]
    result = global_top_alpha(rows, [0.1, 0.9, 0.3, 0.8, 0.2], alpha=0.4, seed=3)
    assert result.selected_count == 2
    assert result.selected_indices == (1, 3)
    assert result.rejected_indices == (0, 2, 4)


def test_rule_bootstrap_prefers_tracking_at_equal_return() -> None:
    good = summary("good", start="a")
    poor = summarize_go2_trajectory(
        trajectory("poor", start="b", actual=(0.05, 0.02, 0.0))
    )
    poor["simulator_return"] = good["simulator_return"]
    result = score_rule_summaries([good, poor], RuleBootstrapConfig())
    assert result.eligible == (True, True)
    assert result.scores[0] > result.scores[1]


def test_rule_bootstrap_uses_equal_thirds_with_minmax_return() -> None:
    base = summary("base", start="a")
    low = {**base, "trajectory_id": "low", "simulator_return": -3.0}
    high = {**base, "trajectory_id": "high", "simulator_return": 7.0}
    result = score_rule_summaries([low, high], RuleBootstrapConfig())
    assert result.eligible == (True, True)
    assert result.diagnostics[0]["reward_quality"] == 0.0
    assert result.diagnostics[1]["reward_quality"] == 1.0
    assert float(result.scores[1] - result.scores[0]) == pytest.approx(1.0 / 3.0)
    assert result.diagnostics[0]["weights"] == {
        "response_tracking": 1.0 / 3.0,
        "dynamic_stability": 1.0 / 3.0,
        "reward": 1.0 / 3.0,
    }
    assert all(row["return_used"] for row in result.diagnostics)


def test_rule_bootstrap_scores_saturation_without_hard_rejection() -> None:
    stable = trajectory("stable", start="a")
    saturated = trajectory("saturated", start="b")
    saturated["actions"] = torch.ones_like(saturated["actions"])
    result = score_rule_summaries(
        [
            summarize_go2_trajectory(stable),
            summarize_go2_trajectory(saturated),
        ],
        RuleBootstrapConfig(),
    )
    assert result.eligible == (True, True)
    assert result.scores[0] > result.scores[1]
    assert (
        result.diagnostics[1]["stability_components"]["action_saturation"]
        < result.diagnostics[0]["stability_components"]["action_saturation"]
    )


def test_v13_proposal_actor_observation_preserves_full_48d() -> None:
    class FakeEnv:
        unwrapped = None

    env = FakeEnv()
    env.unwrapped = env
    collector = V13MJLabProposalCollector(
        env=env,
        extractor=object(),
        actor_sampler=object(),
        make_policy_observation=lambda state, command, previous_action: torch.arange(
            96, dtype=torch.float32
        ).reshape(2, 48),
        simulator_config_sha256="sim-config",
        randomization_disabled=True,
    )
    observation = collector._actor_observation(
        torch.zeros(2, 45),
        torch.zeros(2, 3),
        torch.zeros(2, 12),
    )
    assert observation.shape == (2, 48)
    torch.testing.assert_close(
        observation,
        torch.arange(96, dtype=torch.float32).reshape(2, 48),
    )


def test_v13_proposal_actor_observation_masks_fullstate_coordinates() -> None:
    class FakeEnv:
        unwrapped = None

    env = FakeEnv()
    env.unwrapped = env
    collector = V13MJLabProposalCollector(
        env=env,
        extractor=object(),
        actor_sampler=object(),
        make_policy_observation=lambda state, command, previous_action: torch.arange(
            48, dtype=torch.float32
        ).reshape(1, 48),
        simulator_config_sha256="sim-config",
        randomization_disabled=True,
        policy_observation_mask_indices=(0, 3, 47),
    )
    observation = collector._actor_observation(
        torch.zeros(1, 45),
        torch.zeros(1, 3),
        torch.zeros(1, 12),
    )
    assert observation.shape == (1, 45)
    assert observation[0].tolist() == [
        float(index) for index in range(48) if index not in {0, 3, 47}
    ]


def test_proposal_temperature_scales_actor_standard_deviation() -> None:
    class FakeActor:
        def get_mean_and_std(self, *, observations, training):
            assert training is False
            return torch.zeros_like(observations), torch.ones_like(observations)

    sampler = FlashSACActorDistributionSampler(FakeActor())
    observations = torch.zeros(2, 3, requires_grad=True)
    generator = torch.Generator(device="cpu").manual_seed(17)
    expected_generator = torch.Generator(device="cpu").manual_seed(17)
    expected_noise = torch.randn(
        observations.shape, generator=expected_generator
    )
    actual = sampler.sample(
        observations,
        generator=generator,
        temperature=4.0,
    )
    assert actual.requires_grad is False
    assert actual.grad_fn is None
    torch.testing.assert_close(actual, torch.tanh(4.0 * expected_noise))
    ProposalConfig(
        rollout_horizon=20,
        trajectories_per_start=4,
        actor_sample_temperature=4.0,
    ).validate()


def test_mutable_buffer_and_within_synthetic_mix_roundtrip() -> None:
    buffer = MutableTraceReplayBuffer(10, seed=4)
    buffer.add_batch(
        replay_batch(3, 5.0),
        trajectory_id="candidate",
        learned_score=123.0,
        refresh_step=7,
    )
    mixed, counts = mix_trace_within_synthetic(
        replay_batch(5, 1.0),
        buffer,
        trace_ratio_within_synthetic=0.4,
        shuffle=False,
    )
    assert counts == {"synthetic_count": 5, "rwm_count": 3, "trace_count": 2}
    assert torch.equal(mixed["reward"], torch.tensor([1, 1, 1, 5, 5.0]))
    assert set(mixed) == set(REPLAY_KEYS)
    restored = MutableTraceReplayBuffer(10)
    restored.load_state_dict(buffer.state_dict())
    assert len(restored) == 3
    assert restored.audit_rows[0]["learned_score"] == 123.0


def test_mutable_buffer_batch_append_preserves_fifo_order_after_wrap() -> None:
    sequential = MutableTraceReplayBuffer(5, seed=4)
    batched = MutableTraceReplayBuffer(5, seed=4)
    first = replay_batch(2)
    first["reward"] = torch.tensor([0.0, 1.0])
    second = replay_batch(4)
    second["reward"] = torch.tensor([2.0, 3.0, 4.0, 5.0])
    sequential.add_batch(
        first,
        trajectory_id="first",
        learned_score=1.0,
        refresh_step=1,
    )
    sequential.add_batch(
        second,
        trajectory_id="second",
        learned_score=2.0,
        refresh_step=2,
    )
    batched.add_batches(
        [first, second],
        trajectory_ids=["first", "second"],
        scores=[1.0, 2.0],
        refresh_step=2,
    )
    assert torch.equal(
        sequential._data["reward"], torch.arange(1.0, 6.0)
    )
    for key in REPLAY_KEYS:
        assert torch.equal(sequential._data[key], batched._data[key])
    assert [row["trajectory_id"] for row in batched.audit_rows] == [
        "first",
        "second",
        "second",
        "second",
        "second",
    ]
    restored = MutableTraceReplayBuffer(5, seed=4)
    restored.load_state_dict(copy.deepcopy(batched.state_dict()))
    tail = replay_batch(1)
    tail["reward"] = torch.tensor([6.0])
    for buffer in (batched, restored):
        buffer.add_batch(
            tail,
            trajectory_id="tail",
            learned_score=3.0,
            refresh_step=3,
        )
    assert torch.equal(restored._data["reward"], torch.arange(2.0, 7.0))
    for key in REPLAY_KEYS:
        assert torch.equal(restored._data[key], batched._data[key])


class FakeSemantics:
    gamma = 0.99
    n_step = 1
    reward_config_sha256 = "reward-hash"

    def materialize(self, trajectory_row):
        return replay_batch(len(trajectory_row["states"]), value=2.0)


def test_score_is_metadata_only_during_materialization() -> None:
    candidates = [
        trajectory("a", start="a"),
        trajectory("b", start="b"),
    ]
    first = MutableTraceReplayBuffer(20)
    second = MutableTraceReplayBuffer(20)
    materialize_selected(
        candidates, [0], [1.0, 0.0], semantics=FakeSemantics(), buffer=first, refresh_step=0
    )
    materialize_selected(
        candidates, [0], [999.0, 0.0], semantics=FakeSemantics(), buffer=second, refresh_step=0
    )
    for key in REPLAY_KEYS:
        assert torch.equal(first._data[key], second._data[key])
    assert first.audit_rows[0]["learned_score"] != second.audit_rows[0]["learned_score"]


def test_materializer_uses_batch_semantics_once() -> None:
    class BatchSemantics(FakeSemantics):
        def __init__(self) -> None:
            self.batch_calls = 0
            self.single_calls = 0

        def materialize_many(self, rows):
            self.batch_calls += 1
            return [
                replay_batch(len(row["states"]), value=float(index))
                for index, row in enumerate(rows)
            ]

        def materialize(self, trajectory_row):
            self.single_calls += 1
            return super().materialize(trajectory_row)

    semantics = BatchSemantics()
    candidates = [
        trajectory("a", start="a"),
        trajectory("b", start="b"),
    ]
    buffer = MutableTraceReplayBuffer(20)
    report = materialize_selected(
        candidates,
        [0, 1],
        [1.0, 2.0],
        semantics=semantics,
        buffer=buffer,
        refresh_step=3,
    )
    assert semantics.batch_calls == 1
    assert semantics.single_calls == 0
    assert report.transition_count == 12


def test_canonical_materialize_batch_matches_one_trajectory_batches() -> None:
    semantics = CanonicalV13ReplaySemantics.__new__(
        CanonicalV13ReplaySemantics
    )
    semantics.device = torch.device("cpu")
    semantics._gamma = 0.99
    semantics._n_step = 3
    semantics._wm_cfg = SimpleNamespace(
        policy_action_mask_indices=[],
        reward_termination_penalty=0.0,
    )

    def configure(_cfg, *, num_envs, action_dim, device):
        return SimpleNamespace(
            last_joint_vel=torch.zeros(num_envs, 12, device=device),
            last_action=torch.zeros(num_envs, action_dim, device=device),
            base_lin_vel_xy_ema=torch.zeros(num_envs, 2, device=device),
            base_yaw_vel_ema=torch.zeros(num_envs, device=device),
        )

    def reward(*, state, action, command, foot_contact, episode_length, reward_state, epistemic_uncertainty):
        del foot_contact, episode_length, reward_state, epistemic_uncertainty
        return state[:, 0] + action[:, 0] + command[:, 0], {}

    from scripts.reinforcement_learning.rwm_flashsac.build_go2_real_replay import (
        _build_n_step,
    )
    from src.tasks.rwm_velocity.mdp.extractors import make_go2_policy_obs

    semantics._configure_reward_state = configure
    semantics._compute_reward = reward
    semantics._make_policy_obs = make_go2_policy_obs
    semantics._build_n_step = _build_n_step
    rows = [
        trajectory("a", start="a"),
        trajectory("b", start="b", actual=(0.3, 0.0, 0.0)),
    ]
    rows[0]["actions"][:, 0] = 0.1
    rows[1]["actions"][:, 0] = 0.2
    together = semantics.materialize_many(rows)
    separate = [
        semantics._materialize_same_length([row], length=len(row["states"]))[0]
        for row in rows
    ]
    for actual, expected in zip(together, separate, strict=True):
        for key in REPLAY_KEYS:
            torch.testing.assert_close(actual[key], expected[key])


class FakeSourceSampler:
    def __init__(self):
        self.counter = 0

    def sample(self, count):
        assert count == 1
        self.counter += 1
        return {"unused": self.counter}, [f"source-{self.counter}"]

    def state_dict(self):
        return {"counter": self.counter}

    def load_state_dict(self, state):
        self.counter = int(state["counter"])


class FakeProposalCollector:
    def collect(
        self,
        _snapshot,
        *,
        start_state_ids,
        config,
        actor_generator,
        proposal_event,
    ):
        rows = []
        for branch in range(config.trajectories_per_start):
            actual = 0.2 + 0.02 * float(
                torch.randn((), generator=actor_generator)
            )
            row = trajectory(
                f"event{proposal_event}-branch{branch}",
                start=start_state_ids[0],
                actual=(actual, 0.0, 0.0),
            )
            row["candidate_seed"] = proposal_event
            rows.append(row)
        return rows


def make_manager() -> Go2OnlineTraceManager:
    seed_summaries = [summary("seed-a", start="a"), summary("seed-b", start="b")]
    raw = raw_feature_matrix(seed_summaries)
    stats = fit_feature_stats(raw)
    model = Go2TraceScorer(len(SCORER_EXPANDED_FEATURE_NAMES), 256)
    binding = ScorerBinding(
        task_id="task",
        dataset_id="dataset",
        dataset_sha256="dataset-hash",
        condition_id="g0",
        labels_sha256="labels",
        split_sha256="split",
    )
    return Go2OnlineTraceManager(
        config=OnlineTraceConfig(
            condition_id="g0",
            dataset_sha256="dataset-hash",
            runtime_config_sha256="runtime-config-hash",
            selection_backend="learned",
            reset_certificate_sha256="reset-hash",
            proposal_interval=1,
            feedback_interval=1,
            num_start_states=1,
            rollout_horizon=6,
            trajectories_per_start=4,
            select_alpha=0.5,
            trace_ratio_within_synthetic=0.25,
            buffer_capacity=100,
            command_active_thresholds=(0.03, 0.02, 0.03),
            command_normalization_floors=(0.05, 0.05, 0.05),
            action_saturation_threshold=0.9,
            proposal_seed=9,
        ),
        proposal_collector=FakeProposalCollector(),
        source_sampler=FakeSourceSampler(),
        replay_semantics=FakeSemantics(),
        scorer_model=model,
        scorer_stats=stats,
        scorer_binding=binding,
    )


def make_rule_manager() -> Go2OnlineTraceManager:
    return Go2OnlineTraceManager(
        config=OnlineTraceConfig(
            condition_id="g0",
            dataset_sha256="dataset-hash",
            runtime_config_sha256="rule-runtime-config-hash",
            selection_backend="rule_bootstrap",
            reset_certificate_sha256="reset-hash",
            proposal_interval=1,
            feedback_interval=1,
            num_start_states=1,
            rollout_horizon=6,
            trajectories_per_start=4,
            select_alpha=0.5,
            trace_ratio_within_synthetic=0.25,
            buffer_capacity=100,
            command_active_thresholds=(0.03, 0.02, 0.03),
            command_normalization_floors=(0.05, 0.05, 0.05),
            action_saturation_threshold=0.9,
            proposal_seed=9,
        ),
        proposal_collector=FakeProposalCollector(),
        source_sampler=FakeSourceSampler(),
        replay_semantics=FakeSemantics(),
        rule_config=RuleBootstrapConfig(),
    )


def test_learned_manager_scores_before_feedback_then_selects_with_update() -> None:
    manager = make_manager()
    initial_binding_sha256 = manager.scorer_binding.sha256
    seen = {}

    def updater(summaries, _trajectories, proposal_event):
        assert proposal_event == 0
        seen["pre_scores"] = [
            row.get("trace_score_before_feedback") for row in summaries
        ]
        updated = Go2TraceScorer(len(SCORER_EXPANDED_FEATURE_NAMES), 256)
        binding = ScorerBinding(
            task_id="task",
            dataset_id="dataset",
            dataset_sha256="dataset-hash",
            condition_id="g0",
            labels_sha256="initial-plus-refresh0",
            split_sha256="refresh0-split",
        )
        return updated, manager.scorer_stats, binding, {
            "cumulative_trainable_pairs": 3,
            "initial_cumulative_pairs": 2,
        }

    manager.scorer_updater = updater
    report = manager.maybe_propose(1)
    assert all(value is not None for value in seen["pre_scores"])
    assert report["feedback_status"] == "updated"
    assert (
        report["pre_feedback_scorer_binding_sha256"]
        == initial_binding_sha256
    )
    assert report["selector_binding_sha256"] == manager.scorer_binding.sha256
    assert report["selector_binding_sha256"] != initial_binding_sha256
    assert (
        report["updater_provenance"]["pre_feedback_scorer"][
            "used_for_pair_sampling"
        ]
        is False
    )
    assert (
        report["updater_provenance"]["pre_feedback_scorer"][
            "pair_sampling_reason"
        ]
        == "batch_global_random_is_score_independent"
    )


def test_rule_bootstrap_manager_runs_without_scorer_or_llm() -> None:
    manager = make_rule_manager()
    report = manager.maybe_propose(1)
    assert report is not None
    assert report["selection_backend"] == "rule_bootstrap"
    assert report["feedback_status"] == "disabled_rule_bootstrap"
    rule = report["updater_provenance"]["rule_bootstrap"]
    assert rule["llm_calls"] == 0
    assert report["scorer_binding_sha256"] is None
    assert manager.buffer.audit_rows[0]["score_source"] == "rule_bootstrap"
    assert "rule_score" in manager.buffer.audit_rows[0]


def test_rule_bootstrap_empty_eligible_pool_uses_no_fallback() -> None:
    class NoResponseProposalCollector(FakeProposalCollector):
        def collect(self, *args, **kwargs):
            rows = super().collect(*args, **kwargs)
            for row in rows:
                row["next_states"][:, 0] = 0.0
            return rows

    manager = make_rule_manager()
    manager.proposal_collector = NoResponseProposalCollector()
    report = manager.maybe_propose(1)
    assert report is not None
    assert report["proposal_status"] == "no_eligible_no_fallback"
    assert report["valid_candidate_count"] == 0
    assert report["selected_count"] == 0
    assert report["selected_score_mean"] is None
    assert report["updater_provenance"]["rule_bootstrap"]["llm_calls"] == 0
    assert len(manager.buffer) == 0


def test_online_manager_resume_next_event_equivalence() -> None:
    uninterrupted = make_manager()
    assert uninterrupted.maybe_propose(1) is not None
    state = copy.deepcopy(uninterrupted.state_dict())
    expected = uninterrupted.maybe_propose(2)

    resumed = make_manager()
    resumed.load_state_dict(state)
    actual = resumed.maybe_propose(2)
    assert actual["selected_indices"] == expected["selected_indices"]
    assert actual["trace_buffer_size"] == expected["trace_buffer_size"]
    for key in REPLAY_KEYS:
        assert torch.equal(resumed.buffer._data[key], uninterrupted.buffer._data[key])


def test_online_manager_rejects_pre_six_region_policy_context_checkpoint() -> None:
    manager = make_manager()
    state = copy.deepcopy(manager.state_dict())
    state["format_version"] = "go2_online_trace_manager_v2"
    with pytest.raises(ValueError, match="six-region-only"):
        manager.load_state_dict(state)


def test_learned_manager_checkpoint_restores_cumulative_llm_feedback(
    tmp_path,
) -> None:
    rows = [summary("a0", start="a"), summary("a1", start="a")]

    def complete_provider(pairs):
        return [valid_label(pair["pair_id"]) for pair in pairs]

    uninterrupted = make_manager()
    uninterrupted.feedback_manager = CumulativeFeedbackManager(
        label_store_path=tmp_path / "old" / "labels.jsonl",
        confidence_threshold=0.7,
        label_provider=complete_provider,
        pair_seed=3,
        planar_command_scales=(0.5, 0.2),
    )
    uninterrupted.feedback_manager.request(
        rows, feedback_budget=1, cross_region_fraction=0.0
    )
    state = copy.deepcopy(uninterrupted.state_dict())

    resumed = make_manager()
    resumed.feedback_manager = CumulativeFeedbackManager(
        label_store_path=tmp_path / "new" / "labels.jsonl",
        confidence_threshold=0.7,
        label_provider=complete_provider,
        pair_seed=3,
        planar_command_scales=(0.5, 0.2),
    )
    resumed.load_state_dict(state)
    assert (
        resumed.feedback_manager.labels_sha256
        == uninterrupted.feedback_manager.labels_sha256
    )
    assert (
        resumed.feedback_manager.refresh_count
        == uninterrupted.feedback_manager.refresh_count
    )
    assert (tmp_path / "new" / "labels.jsonl").is_file()


def test_rule_bootstrap_manager_resume_next_event_equivalence() -> None:
    uninterrupted = make_rule_manager()
    assert uninterrupted.maybe_propose(1) is not None
    state = copy.deepcopy(uninterrupted.state_dict())
    expected = uninterrupted.maybe_propose(2)
    resumed = make_rule_manager()
    resumed.load_state_dict(state)
    actual = resumed.maybe_propose(2)
    assert actual["selected_indices"] == expected["selected_indices"]
    assert actual["trace_buffer_size"] == expected["trace_buffer_size"]


def test_torch_uniform_buffer_resume_preserves_inflight_n_step(tmp_path) -> None:
    observation_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(2,), dtype=np.float32
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def make_buffer():
        return TorchUniformBuffer(
            observation_space,
            action_space,
            n_step=3,
            gamma=0.99,
            max_length=16,
            min_length=1,
            sample_batch_size=2,
            device_type="cpu",
        )

    def transition_at(step):
        return {
            "observation": np.full((1, 2), step, dtype=np.float32),
            "action": np.full((1, 1), step, dtype=np.float32),
            "reward": np.full((1,), step + 1, dtype=np.float32),
            "terminated": np.zeros(1, dtype=bool),
            "truncated": np.zeros(1, dtype=bool),
            "next_observation": np.full((1, 2), step + 1, dtype=np.float32),
        }

    uninterrupted = make_buffer()
    uninterrupted.add(transition_at(0))
    uninterrupted.add(transition_at(1))
    path = tmp_path / "replay.pt"
    uninterrupted.save(str(path))
    resumed = make_buffer()
    resumed.load(str(path))
    for buffer in (uninterrupted, resumed):
        buffer.add(transition_at(2))
    assert len(resumed) == len(uninterrupted) == 1
    for name in (
        "_observations",
        "_actions",
        "_rewards",
        "_terminateds",
        "_truncateds",
        "_next_observations",
    ):
        torch.testing.assert_close(
            getattr(resumed, name)[:1],
            getattr(uninterrupted, name)[:1],
        )


def test_torch_uniform_buffer_rolling_raw_slot_is_self_contained(
    tmp_path,
) -> None:
    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
    )
    action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(1,), dtype=np.float32
    )

    def make_buffer():
        return TorchUniformBuffer(
            observation_space,
            action_space,
            n_step=1,
            gamma=0.99,
            max_length=5,
            min_length=1,
            sample_batch_size=1,
            device_type="cpu",
        )

    def transition_at(start: int, count: int):
        values = torch.arange(start, start + count, dtype=torch.float32)
        return {
            "observation": torch.stack((values, values + 0.25), dim=-1),
            "action": values[:, None] / 10.0,
            "reward": values,
            "terminated": torch.zeros(count),
            "truncated": torch.zeros(count),
            "next_observation": torch.stack(
                (values + 1.0, values + 1.25), dim=-1
            ),
        }

    uninterrupted = make_buffer()
    uninterrupted.add(transition_at(0, 3))
    first_dir = tmp_path / "step1"
    first_dir.mkdir()
    base = first_dir / "replay_buffer.pt"
    uninterrupted.save(str(base))
    uninterrupted.add(transition_at(3, 4))
    third_dir = tmp_path / "step3"
    third_dir.mkdir()
    __import__("shutil").move(
        str(base), str(third_dir / "replay_buffer.pt")
    )
    __import__("shutil").move(
        str(first_dir / "replay_buffer_data"),
        str(third_dir / "replay_buffer_data"),
    )
    current = third_dir / "replay_buffer.pt"
    uninterrupted.save(str(current))

    payload = torch.load(current, map_location="cpu", weights_only=False)
    assert (
        payload["format_version"]
        == "flashsac_torch_uniform_buffer_v4_rolling_raw"
    )
    assert payload["updated_from_total_added"] == 3
    assert payload["updated_count"] == 4
    assert payload["storage_directory"] == "replay_buffer_data"

    resumed = make_buffer()
    resumed.load(str(current))
    assert len(resumed) == len(uninterrupted) == 5
    assert resumed._current_idx == uninterrupted._current_idx
    for name in (
        "_observations",
        "_actions",
        "_rewards",
        "_terminateds",
        "_truncateds",
        "_next_observations",
    ):
        torch.testing.assert_close(
            getattr(resumed, name),
            getattr(uninterrupted, name),
        )


def test_training_checkpoint_retains_two_and_reuses_fifo_slot(
    tmp_path,
) -> None:
    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
    )
    action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(1,), dtype=np.float32
    )

    def make_buffer():
        return TorchUniformBuffer(
            observation_space,
            action_space,
            n_step=1,
            gamma=0.99,
            max_length=5,
            min_length=1,
            sample_batch_size=1,
            device_type="cpu",
        )

    def transition_at(start: int, count: int):
        values = torch.arange(start, start + count, dtype=torch.float32)
        return {
            "observation": torch.stack((values, values + 0.25), dim=-1),
            "action": values[:, None],
            "reward": values,
            "terminated": torch.zeros(count),
            "truncated": torch.zeros(count),
            "next_observation": torch.stack(
                (values + 1.0, values + 1.25), dim=-1
            ),
        }

    class Agent:
        def __init__(self):
            self._replay_buffer = make_buffer()

        def save(self, path):
            target = __import__("pathlib").Path(path)
            target.mkdir(parents=True, exist_ok=True)
            (target / "actor.pt").write_text("actor")

        def save_replay_buffer(self, path):
            self._replay_buffer.save(
                str(
                    __import__("pathlib").Path(path)
                    / "replay_buffer.pt"
                )
            )

    from omegaconf import OmegaConf

    agent = Agent()
    cfg = OmegaConf.create({"seed": 0})
    agent._replay_buffer.add(transition_at(0, 3))
    _save_checkpoint(agent, tmp_path / "step1", cfg, save_replay=True)
    agent._replay_buffer.add(transition_at(3, 2))
    _save_checkpoint(agent, tmp_path / "step2", cfg, save_replay=True)
    agent._replay_buffer.add(transition_at(5, 2))
    _save_checkpoint(agent, tmp_path / "step3", cfg, save_replay=True)

    assert not (tmp_path / "step1").exists()
    assert (tmp_path / "step2" / "CHECKPOINT_COMPLETE").is_file()
    assert (tmp_path / "step3" / "CHECKPOINT_COMPLETE").is_file()
    payload = torch.load(
        tmp_path / "step3" / "replay_buffer.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["updated_from_total_added"] == 3
    assert payload["updated_count"] == 4
    resumed = make_buffer()
    resumed.load(str(tmp_path / "step3" / "replay_buffer.pt"))
    assert resumed._total_added == 7
    torch.testing.assert_close(
        resumed._observations, agent._replay_buffer._observations
    )


def test_rwm_environment_state_roundtrip_restores_mutable_tensors() -> None:
    env = Go2RWMFlashSACWorldModelEnv.__new__(
        Go2RWMFlashSACWorldModelEnv
    )
    env.num_envs = 2
    env._device = torch.device("cpu")
    tensor_names = (
        "state_history",
        "action_history",
        "episode_length_buf",
        "model_ids",
        "command_intervals",
        "command",
        "_ep_returns",
        "_ep_lengths",
        "_interface_action_history",
        "_interface_action_scale",
        "_interface_action_bias",
        "_interface_obs_bias",
    )
    for index, name in enumerate(tensor_names):
        setattr(env, name, torch.full((2, 2), float(index)))
    env.reward_state = SimpleNamespace(
        last_action=torch.ones(2, 12),
        scalar_config=0.5,
    )
    env._reward_buffer = deque([1.0, 2.0], maxlen=100)
    env._length_buffer = deque([3.0], maxlen=100)
    env._latest_log = {"metric": 4.0}
    state = env.state_dict()
    for name in tensor_names:
        getattr(env, name).zero_()
    env.reward_state.last_action.zero_()
    env._reward_buffer.clear()
    env.load_state_dict(state)
    for index, name in enumerate(tensor_names):
        assert torch.equal(
            getattr(env, name), torch.full((2, 2), float(index))
        )
    assert torch.equal(env.reward_state.last_action, torch.ones(2, 12))
    assert list(env._reward_buffer) == [1.0, 2.0]


def test_atomic_training_checkpoint_requires_all_resume_components(tmp_path) -> None:
    class Agent:
        def save(self, path):
            target = __import__("pathlib").Path(path)
            target.mkdir(parents=True, exist_ok=True)
            (target / "actor.pt").write_text("actor")

        def save_replay_buffer(self, path):
            (__import__("pathlib").Path(path) / "replay_buffer.pt").write_text(
                "replay"
            )

    class Trace:
        def save_checkpoint(self, path):
            target = __import__("pathlib").Path(path)
            target.mkdir(parents=True)
            (target / "trace_manager.pt").write_text("trace")
            (target / "COMPLETE").write_text("complete")

    from omegaconf import OmegaConf

    state = {
        "format_version": "go2_flashsac_training_state_v1",
        "interaction_step": 1000,
    }
    destination = tmp_path / "step1000"
    _save_checkpoint(
        Agent(),
        destination,
        OmegaConf.create({"seed": 0}),
        save_replay=True,
        trace_manager=Trace(),
        training_state=state,
    )
    assert (destination / "CHECKPOINT_COMPLETE").is_file()
    assert (destination / "replay_buffer.pt").is_file()
    assert (destination / "trace_manager" / "COMPLETE").is_file()
    assert _load_training_state(destination)["interaction_step"] == 1000


def test_atomic_training_checkpoint_marks_committed_replay_parent(
    tmp_path,
) -> None:
    class Replay:
        committed = None

        def mark_saved_checkpoint(self, path):
            self.committed = path

    class Agent:
        def __init__(self):
            self._replay_buffer = Replay()

        def save(self, path):
            target = __import__("pathlib").Path(path)
            target.mkdir(parents=True, exist_ok=True)
            (target / "actor.pt").write_text("actor")

        def save_replay_buffer(self, path):
            (__import__("pathlib").Path(path) / "replay_buffer.pt").write_text(
                "replay"
            )

    from omegaconf import OmegaConf

    agent = Agent()
    destination = tmp_path / "step1000"
    _save_checkpoint(
        agent,
        destination,
        OmegaConf.create({"seed": 0}),
        save_replay=True,
    )
    assert agent._replay_buffer.committed == str(
        destination / "replay_buffer.pt"
    )


def test_async_checkpoint_returns_before_replay_write_and_commits(
    tmp_path,
) -> None:
    class Agent:
        def save(self, path):
            target = __import__("pathlib").Path(path)
            target.mkdir(parents=True, exist_ok=True)
            (target / "actor.pt").write_text("actor")

        def save_replay_buffer(self, path):
            __import__("time").sleep(0.25)
            (__import__("pathlib").Path(path) / "replay_buffer.pt").write_text(
                "replay"
            )

    class Trace:
        def save_checkpoint(self, path):
            target = __import__("pathlib").Path(path)
            target.mkdir(parents=True)
            (target / "trace_manager.pt").write_text("trace")
            (target / "COMPLETE").write_text("complete")

    from omegaconf import OmegaConf

    destination = tmp_path / "step5000"
    started = __import__("time").perf_counter()
    writer = _start_async_checkpoint(
        Agent(),
        destination,
        OmegaConf.create({"seed": 0}),
        Trace(),
        {
            "format_version": "go2_flashsac_training_state_v1",
            "interaction_step": 5000,
        },
    )
    assert __import__("time").perf_counter() - started < 0.20
    assert _wait_async_checkpoint(writer, block=False) is writer
    assert _wait_async_checkpoint(writer, block=True) is None
    assert (destination / "CHECKPOINT_COMPLETE").is_file()
    assert (destination / "replay_buffer.pt").is_file()
    assert _load_training_state(destination)["interaction_step"] == 5000


def test_async_checkpoint_propagates_writer_failure(tmp_path) -> None:
    class Agent:
        def save(self, path):
            target = __import__("pathlib").Path(path)
            target.mkdir(parents=True, exist_ok=True)
            (target / "actor.pt").write_text("actor")

        def save_replay_buffer(self, path):
            raise RuntimeError("disk failure")

    class Trace:
        def save_checkpoint(self, path):
            target = __import__("pathlib").Path(path)
            target.mkdir(parents=True)
            (target / "trace_manager.pt").write_text("trace")
            (target / "COMPLETE").write_text("complete")

    from omegaconf import OmegaConf

    writer = _start_async_checkpoint(
        Agent(),
        tmp_path / "step5000",
        OmegaConf.create({"seed": 0}),
        Trace(),
        {
            "format_version": "go2_flashsac_training_state_v1",
            "interaction_step": 5000,
        },
    )
    with pytest.raises(RuntimeError, match="writer failed"):
        _wait_async_checkpoint(writer, block=True)
    assert not (tmp_path / "step5000" / "CHECKPOINT_COMPLETE").exists()


def test_async_checkpoint_replay_is_fork_time_generation(tmp_path) -> None:
    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
    )
    action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(1,), dtype=np.float32
    )

    def make_buffer():
        return TorchUniformBuffer(
            observation_space,
            action_space,
            n_step=1,
            gamma=0.99,
            max_length=8,
            min_length=1,
            sample_batch_size=1,
            device_type="cpu",
        )

    def transition_at(start: int, count: int):
        values = torch.arange(start, start + count, dtype=torch.float32)
        return {
            "observation": torch.stack((values, values), dim=-1),
            "action": values[:, None],
            "reward": values,
            "terminated": torch.zeros(count),
            "truncated": torch.zeros(count),
            "next_observation": torch.stack(
                (values + 1.0, values + 1.0), dim=-1
            ),
        }

    class Agent:
        def __init__(self):
            self._replay_buffer = make_buffer()

        def save(self, path):
            target = __import__("pathlib").Path(path)
            target.mkdir(parents=True, exist_ok=True)
            (target / "actor.pt").write_text("actor")

        def save_replay_buffer(self, path):
            __import__("time").sleep(0.1)
            self._replay_buffer.save(
                str(
                    __import__("pathlib").Path(path)
                    / "replay_buffer.pt"
                )
            )

    class Trace:
        def save_checkpoint(self, path):
            target = __import__("pathlib").Path(path)
            target.mkdir(parents=True)
            (target / "trace_manager.pt").write_text("trace")
            (target / "COMPLETE").write_text("complete")

    from omegaconf import OmegaConf

    agent = Agent()
    agent._replay_buffer.add(transition_at(0, 3))
    destination = tmp_path / "step5000"
    writer = _start_async_checkpoint(
        agent,
        destination,
        OmegaConf.create({"seed": 0}),
        Trace(),
        {
            "format_version": "go2_flashsac_training_state_v1",
            "interaction_step": 5000,
        },
    )
    agent._replay_buffer.add(transition_at(3, 2))
    assert _wait_async_checkpoint(writer, block=True) is None
    restored = make_buffer()
    restored.load(str(destination / "replay_buffer.pt"))
    assert restored._total_added == 3
    assert agent._replay_buffer._total_added == 5


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for this test"
)
def test_async_checkpoint_supports_cuda_replay_snapshot(tmp_path) -> None:
    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
    )
    action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(1,), dtype=np.float32
    )

    def make_buffer(device_type: str):
        return TorchUniformBuffer(
            observation_space,
            action_space,
            n_step=1,
            gamma=0.99,
            max_length=8,
            min_length=1,
            sample_batch_size=1,
            device_type=device_type,
        )

    class Agent:
        def __init__(self):
            self._replay_buffer = make_buffer("cuda:0")

        def save(self, path):
            target = __import__("pathlib").Path(path)
            target.mkdir(parents=True, exist_ok=True)
            (target / "actor.pt").write_text("actor")

    class Trace:
        def save_checkpoint(self, path):
            target = __import__("pathlib").Path(path)
            target.mkdir(parents=True)
            (target / "trace_manager.pt").write_text("trace")
            (target / "COMPLETE").write_text("complete")

    values = torch.arange(3, dtype=torch.float32, device="cuda:0")
    agent = Agent()
    agent._replay_buffer.add(
        {
            "observation": torch.stack((values, values), dim=-1),
            "action": values[:, None],
            "reward": values,
            "terminated": torch.zeros(3, device="cuda:0"),
            "truncated": torch.zeros(3, device="cuda:0"),
            "next_observation": torch.stack(
                (values + 1.0, values + 1.0), dim=-1
            ),
        }
    )
    destination = tmp_path / "step5000"
    writer = _start_async_checkpoint(
        agent,
        destination,
        __import__("omegaconf").OmegaConf.create({"seed": 0}),
        Trace(),
        {
            "format_version": "go2_flashsac_training_state_v1",
            "interaction_step": 5000,
        },
    )
    assert _wait_async_checkpoint(writer, block=True) is None
    restored = make_buffer("cpu")
    restored.load(str(destination / "replay_buffer.pt"))
    assert restored._total_added == 3
    torch.testing.assert_close(
        restored._observations[:3],
        agent._replay_buffer._observations[:3].cpu(),
    )
