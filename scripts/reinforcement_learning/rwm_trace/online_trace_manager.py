"""Online Go2 TRACE lifecycle: propose, refresh, score, select, append, resume."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from .materializer import V13ReplaySemantics, materialize_selected
from .proposal import ProposalConfig, V13MJLabProposalCollector
from .replay import MutableTraceReplayBuffer, mix_trace_within_synthetic
from .rule_bootstrap import RuleBootstrapConfig, score_rule_summaries
from .schemas import schema_manifest
from .scorer import (
    FeatureStats,
    Go2TraceScorer,
    ScorerBinding,
    score_summaries,
)
from .selection import (
    COMMAND_REGIONS,
    SelectionResult,
    command_region,
    command_region_stratified_top_alpha,
    global_top_alpha,
)
from .source_sampler import V13SnapshotSourceSampler
from .trajectory import summarize_go2_trajectory


@dataclass(frozen=True)
class OnlineTraceConfig:
    condition_id: str
    dataset_sha256: str
    runtime_config_sha256: str
    selection_backend: str
    proposal_interval: int
    feedback_interval: int
    num_start_states: int
    rollout_horizon: int
    trajectories_per_start: int
    select_alpha: float
    trace_ratio_within_synthetic: float
    buffer_capacity: int
    command_active_thresholds: tuple[float, float, float]
    command_normalization_floors: tuple[float, float, float]
    action_saturation_threshold: float
    proposal_seed: int
    reset_certificate_sha256: str
    actor_sample_temperature: float = 1.0
    command_region_sim_ratios: Mapping[str, float] | None = None

    def validate(self) -> None:
        if (
            not self.condition_id
            or not self.dataset_sha256
            or not self.reset_certificate_sha256
            or not self.runtime_config_sha256
        ):
            raise ValueError(
                "TRACE config requires condition, dataset, reset-certificate, and "
                "runtime-config hashes."
            )
        if self.proposal_interval < 1 or self.feedback_interval < 1:
            raise ValueError("TRACE proposal/feedback intervals must be positive.")
        if self.selection_backend not in {"rule_bootstrap", "learned"}:
            raise ValueError(
                "TRACE selection_backend must be rule_bootstrap or learned."
            )
        if not 0.0 < self.select_alpha <= 1.0:
            raise ValueError("TRACE select_alpha must be in (0,1].")
        if not 0.0 <= self.trace_ratio_within_synthetic <= 1.0:
            raise ValueError("TRACE-within-synthetic ratio must be in [0,1].")
        if self.buffer_capacity < 1:
            raise ValueError("TRACE buffer capacity must be positive.")
        if self.num_start_states < 1:
            raise ValueError("TRACE num_start_states must be positive.")
        ProposalConfig(
            rollout_horizon=self.rollout_horizon,
            trajectories_per_start=self.trajectories_per_start,
            actor_sample_temperature=self.actor_sample_temperature,
        ).validate()
        if any(value <= 0.0 for value in self.command_active_thresholds):
            raise ValueError("Command active thresholds must be positive.")
        if any(value <= 0.0 for value in self.command_normalization_floors):
            raise ValueError("Command normalization floors must be positive.")
        if self.action_saturation_threshold <= 0.0:
            raise ValueError("Action saturation threshold must be positive.")
        if self.command_region_sim_ratios is not None:
            ratios = {
                str(region): float(ratio)
                for region, ratio in self.command_region_sim_ratios.items()
            }
            if set(ratios) != set(COMMAND_REGIONS):
                raise ValueError(
                    "Command-region sim ratios must specify exactly "
                    f"{list(COMMAND_REGIONS)}."
                )
            if any(not 0.0 <= ratio <= 1.0 for ratio in ratios.values()):
                raise ValueError("Command-region sim ratios must be in [0,1].")


ScorerUpdater = Callable[
    [
        Sequence[Mapping[str, Any]],
        Sequence[Mapping[str, Any]],
        int,
    ],
    tuple[Go2TraceScorer, FeatureStats, ScorerBinding, Mapping[str, Any]],
]


class Go2OnlineTraceManager:
    def __init__(
        self,
        *,
        config: OnlineTraceConfig,
        proposal_collector: V13MJLabProposalCollector,
        source_sampler: V13SnapshotSourceSampler,
        replay_semantics: V13ReplaySemantics,
        scorer_model: Go2TraceScorer | None = None,
        scorer_stats: FeatureStats | None = None,
        scorer_binding: ScorerBinding | None = None,
        scorer_updater: ScorerUpdater | None = None,
        rule_config: RuleBootstrapConfig | None = None,
    ) -> None:
        config.validate()
        if config.selection_backend == "learned":
            if scorer_model is None or scorer_stats is None or scorer_binding is None:
                raise ValueError("Learned TRACE requires an initial scorer checkpoint.")
            if rule_config is not None:
                raise ValueError("Learned TRACE must not receive a rule scorer.")
            if scorer_binding.dataset_sha256 != config.dataset_sha256:
                raise ValueError("Initial scorer is not bound to the configured dataset.")
            if scorer_binding.condition_id != config.condition_id:
                raise ValueError("Initial scorer is not bound to the configured condition.")
            scorer_model = scorer_model.eval()
        else:
            if rule_config is None:
                raise ValueError("Rule-bootstrap TRACE requires its frozen rule config.")
            rule_config.validate()
            if any(
                value is not None
                for value in (
                    scorer_model,
                    scorer_stats,
                    scorer_binding,
                    scorer_updater,
                )
            ):
                raise ValueError(
                    "Rule-bootstrap TRACE forbids learned scorer and LLM updater."
                )
        self.config = config
        self.proposal_collector = proposal_collector
        self.source_sampler = source_sampler
        self.replay_semantics = replay_semantics
        self.scorer_model = scorer_model
        self.scorer_stats = scorer_stats
        self.scorer_binding = scorer_binding
        self.scorer_updater = scorer_updater
        self.rule_config = rule_config
        self.buffer = MutableTraceReplayBuffer(
            config.buffer_capacity,
            seed=config.proposal_seed + 2,
            command_region_sim_ratios=config.command_region_sim_ratios,
            command_active_thresholds=config.command_active_thresholds,
            command_region_planar_scales=(0.5, 0.2),
        )
        self.actor_generator = torch.Generator(device="cpu").manual_seed(
            config.proposal_seed
        )
        self.mix_generator = torch.Generator(device="cpu").manual_seed(
            config.proposal_seed + 1
        )
        self.proposal_event = 0
        self.last_proposal_step = -1
        self.last_feedback_step = -1
        self.last_report: dict[str, Any] = {}
        self.feedback_manager: Any | None = None

    def _summaries(
        self, trajectories: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        summaries = []
        for raw in trajectories:
            trajectory = {
                **dict(raw),
                "command_active_thresholds": self.config.command_active_thresholds,
                "command_normalization_floors": self.config.command_normalization_floors,
                "action_saturation_threshold": self.config.action_saturation_threshold,
                "condition_id": self.config.condition_id,
            }
            summaries.append(summarize_go2_trajectory(trajectory))
        return summaries

    def maybe_propose(
        self,
        training_step: int,
    ) -> Mapping[str, Any] | None:
        if int(training_step) % self.config.proposal_interval != 0:
            return None
        snapshots, start_state_ids = self.source_sampler.sample(
            self.config.num_start_states
        )
        trajectories = self.proposal_collector.collect(
            snapshots,
            start_state_ids=start_state_ids,
            config=ProposalConfig(
                rollout_horizon=self.config.rollout_horizon,
                trajectories_per_start=self.config.trajectories_per_start,
                actor_sample_temperature=self.config.actor_sample_temperature,
            ),
            actor_generator=self.actor_generator,
            proposal_event=self.proposal_event,
        )
        summaries = self._summaries(trajectories)
        feedback_due = (
            self.config.selection_backend == "learned"
            and self.scorer_updater is not None
            and int(training_step) % self.config.feedback_interval == 0
        )
        feedback_status = (
            "disabled_rule_bootstrap"
            if self.config.selection_backend == "rule_bootstrap"
            else "not_due"
        )
        updater_provenance: Mapping[str, Any] = {}
        if feedback_due:
            try:
                model, stats, binding, updater_provenance = self.scorer_updater(
                    summaries, trajectories, self.proposal_event
                )
                if binding.dataset_sha256 != self.config.dataset_sha256:
                    raise ValueError("Updated scorer dataset binding mismatch.")
                if binding.condition_id != self.config.condition_id:
                    raise ValueError("Updated scorer condition binding mismatch.")
                self.scorer_model = model.eval()
                self.scorer_stats = stats
                self.scorer_binding = binding
                self.last_feedback_step = int(training_step)
                feedback_status = "updated"
            except Exception as exc:
                # Existing compatible scorer remains active; there is no fallback
                # selector or fallback score.
                feedback_status = f"kept_previous:{type(exc).__name__}:{exc}"

        base_valid_mask = [
            bool(summary["survival_length"] >= 1)
            and not summary["missing_features"].count("tilt_max")
            for summary in summaries
        ]
        if self.config.selection_backend == "rule_bootstrap":
            assert self.rule_config is not None
            rule_result = score_rule_summaries(summaries, self.rule_config)
            scores = rule_result.scores
            valid_mask = [
                base and eligible
                for base, eligible in zip(
                    base_valid_mask, rule_result.eligible, strict=True
                )
            ]
            rejection_counts: dict[str, int] = {}
            for row in rule_result.diagnostics:
                reason = row.get("rejection_reason")
                if reason is not None:
                    rejection_counts[str(reason)] = (
                        rejection_counts.get(str(reason), 0) + 1
                    )
            updater_provenance = {
                "rule_bootstrap": {
                    "config_sha256": rule_result.config_sha256,
                    "eligible_count": sum(rule_result.eligible),
                    "rejection_counts": rejection_counts,
                    "llm_calls": 0,
                }
            }
            selector_binding_sha256 = rule_result.config_sha256
        else:
            assert self.scorer_model is not None
            assert self.scorer_stats is not None
            assert self.scorer_binding is not None
            scores = score_summaries(
                self.scorer_model, summaries, self.scorer_stats, device="cpu"
            )
            valid_mask = base_valid_mask
            selector_binding_sha256 = self.scorer_binding.sha256
        if self.config.selection_backend == "rule_bootstrap" and not any(
            valid_mask
        ):
            # The legacy rule gate explicitly allows no selected trajectories
            # and forbids fallback filling. Keep training on RWM replay and
            # audit the empty event so a later refresh can try again.
            selection = SelectionResult(
                selected_indices=(),
                valid_indices=(),
                rejected_indices=tuple(range(len(summaries))),
                alpha=self.config.select_alpha,
                selected_count=0,
                ranking=(),
            )
            proposal_status = "no_eligible_no_fallback"
        elif self.config.selection_backend == "rule_bootstrap":
            selection = command_region_stratified_top_alpha(
                summaries,
                scores,
                alpha=self.config.select_alpha,
                seed=self.config.proposal_seed + self.proposal_event,
                valid_mask=valid_mask,
            )
            proposal_status = "selected"
        else:
            selection = global_top_alpha(
                summaries,
                scores,
                alpha=self.config.select_alpha,
                seed=self.config.proposal_seed + self.proposal_event,
                valid_mask=valid_mask,
            )
            proposal_status = "selected"
        valid_command_region_counts = {
            region: sum(
                command_region(
                    summaries[index], planar_command_scales=(0.5, 0.2)
                )
                == region
                for index in selection.valid_indices
            )
            for region in COMMAND_REGIONS
        }
        selected_command_region_counts = {
            region: sum(
                command_region(
                    summaries[index], planar_command_scales=(0.5, 0.2)
                )
                == region
                for index in selection.selected_indices
            )
            for region in COMMAND_REGIONS
        }
        materialization = materialize_selected(
            trajectories,
            selection.selected_indices,
            scores,
            semantics=self.replay_semantics,
            buffer=self.buffer,
            refresh_step=int(training_step),
            score_source=self.config.selection_backend,
        )
        score_values = [float(value) for value in scores]
        selected_score_values = [
            score_values[index] for index in selection.selected_indices
        ]
        self.last_proposal_step = int(training_step)
        self.proposal_event += 1
        self.last_report = {
            "training_step": int(training_step),
            "proposal_event": self.proposal_event - 1,
            "candidate_count": len(trajectories),
            "valid_candidate_count": len(selection.valid_indices),
            "selected_count": selection.selected_count,
            "selected_indices": list(selection.selected_indices),
            "proposal_status": proposal_status,
            "selection_backend": self.config.selection_backend,
            "selection_policy": (
                "command_region_stratified_top_alpha"
                if self.config.selection_backend == "rule_bootstrap"
                else "global_top_alpha"
            ),
            "valid_command_region_counts": valid_command_region_counts,
            "selected_command_region_counts": selected_command_region_counts,
            "candidate_score_mean": sum(score_values) / len(score_values),
            "candidate_score_min": min(score_values),
            "candidate_score_max": max(score_values),
            "selected_score_mean": (
                sum(selected_score_values) / len(selected_score_values)
                if selected_score_values
                else None
            ),
            "feedback_status": feedback_status,
            "selector_binding_sha256": selector_binding_sha256,
            "scorer_binding_sha256": (
                self.scorer_binding.sha256
                if self.scorer_binding is not None
                else None
            ),
            "trace_buffer_size": len(self.buffer),
            "trace_buffer_total_inserted": self.buffer.total_inserted,
            "materialization": asdict(materialization),
            "updater_provenance": dict(updater_provenance),
        }
        return dict(self.last_report)

    def mix_synthetic_batch(
        self,
        rwm_synthetic_batch: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if len(self.buffer) == 0:
            return (
                {key: value for key, value in rwm_synthetic_batch.items()},
                {
                    "trace_warmup": True,
                    "trace_count": 0,
                    "rwm_count": int(rwm_synthetic_batch["reward"].shape[0]),
                    "synthetic_count": int(rwm_synthetic_batch["reward"].shape[0]),
                },
            )
        mixed, counts = mix_trace_within_synthetic(
            rwm_synthetic_batch,
            self.buffer,
            trace_ratio_within_synthetic=self.config.trace_ratio_within_synthetic,
            shuffle=True,
            generator=self.mix_generator,
        )
        return mixed, {"trace_warmup": False, **counts}

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": "go2_online_trace_manager_v2",
            "config": asdict(self.config),
            "schemas": schema_manifest(),
            "proposal_event": self.proposal_event,
            "last_proposal_step": self.last_proposal_step,
            "last_feedback_step": self.last_feedback_step,
            "last_report": self.last_report,
            "actor_generator_state": self.actor_generator.get_state(),
            "mix_generator_state": self.mix_generator.get_state(),
            "trace_buffer": self.buffer.state_dict(),
            "source_sampler": self.source_sampler.state_dict(),
            "rule_config": (
                asdict(self.rule_config) if self.rule_config is not None else None
            ),
            "scorer_binding": (
                self.scorer_binding.to_dict()
                if self.scorer_binding is not None
                else None
            ),
            "scorer_binding_sha256": (
                self.scorer_binding.sha256
                if self.scorer_binding is not None
                else None
            ),
            "scorer_stats": (
                self.scorer_stats.to_dict()
                if self.scorer_stats is not None
                else None
            ),
            "scorer_model_state_dict": (
                self.scorer_model.state_dict()
                if self.scorer_model is not None
                else None
            ),
            "torch_cpu_rng_state": torch.random.get_rng_state(),
            "feedback_manager": (
                self.feedback_manager.state_dict()
                if self.feedback_manager is not None
                else None
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("format_version") != "go2_online_trace_manager_v2":
            raise ValueError("Online TRACE manager checkpoint format mismatch.")
        if dict(state["config"]) != asdict(self.config):
            raise ValueError("Online TRACE manager config changed across resume.")
        if self.config.selection_backend == "learned":
            if self.scorer_model is None:
                raise ValueError("Learned TRACE manager lacks a scorer on resume.")
            binding = ScorerBinding.from_dict(state["scorer_binding"])
            if binding.sha256 != state["scorer_binding_sha256"]:
                raise ValueError("Online TRACE scorer binding hash is corrupt.")
            self.scorer_model.load_state_dict(
                state["scorer_model_state_dict"], strict=True
            )
            self.scorer_model.eval()
            self.scorer_stats = FeatureStats.from_dict(state["scorer_stats"])
            self.scorer_stats.validate()
            self.scorer_binding = binding
        else:
            if self.rule_config is None or dict(state["rule_config"]) != asdict(
                self.rule_config
            ):
                raise ValueError("Rule-bootstrap config changed across resume.")
            if any(
                state.get(key) is not None
                for key in (
                    "scorer_binding",
                    "scorer_binding_sha256",
                    "scorer_stats",
                    "scorer_model_state_dict",
                )
            ):
                raise ValueError("Rule-bootstrap checkpoint contains a learned scorer.")
        self.buffer.load_state_dict(state["trace_buffer"])
        self.source_sampler.load_state_dict(state["source_sampler"])
        self.actor_generator.set_state(state["actor_generator_state"])
        self.mix_generator.set_state(state["mix_generator_state"])
        torch.random.set_rng_state(state["torch_cpu_rng_state"])
        self.proposal_event = int(state["proposal_event"])
        self.last_proposal_step = int(state["last_proposal_step"])
        self.last_feedback_step = int(state["last_feedback_step"])
        self.last_report = dict(state["last_report"])
        feedback_state = state.get("feedback_manager")
        if (feedback_state is None) != (self.feedback_manager is None):
            raise ValueError("TRACE feedback-manager presence changed across resume.")
        if self.feedback_manager is not None:
            self.feedback_manager.load_state_dict(feedback_state)

    def save_checkpoint(self, directory: str | Path) -> None:
        destination = Path(directory).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing TRACE checkpoint: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        try:
            torch.save(self.state_dict(), temporary / "trace_manager.pt")
            (temporary / "COMPLETE").write_text("complete\n", encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def load_checkpoint(self, directory: str | Path) -> None:
        source = Path(directory).expanduser().resolve()
        if not (source / "COMPLETE").is_file():
            raise ValueError("TRACE checkpoint lacks atomic COMPLETE marker.")
        state = torch.load(
            source / "trace_manager.pt", map_location="cpu", weights_only=False
        )
        self.load_state_dict(state)
