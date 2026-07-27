"""Build the online TRACE manager against the canonical V13 training runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .artifact_manifest import sha256_path
from .feedback_manager import CodexBatchLabelProvider, CumulativeFeedbackManager
from .online_scorer_update import OnlineScorerUpdater
from .online_trace_manager import Go2OnlineTraceManager, OnlineTraceConfig
from .proposal import FlashSACActorDistributionSampler, V13MJLabProposalCollector
from .rule_bootstrap import RuleBootstrapConfig
from .schemas import canonical_sha256
from .scorer import load_scorer_checkpoint
from .source_sampler import V13SnapshotSourceSampler
from .v13_replay_adapter import CanonicalV13ReplaySemantics, V13_EXPECTED_COMMIT


def _configure_mjwarp_driver_fallback() -> bool:
    """Use MJWarp's non-conditional solver path on pre-CUDA-12.4 drivers.

    MJLab already disables CUDA graph capture on these drivers, and MJWarp's
    solver has an explicit ordinary-loop implementation for the same case.
    MJWarp 3.5 nevertheless checks conditional-graph support unconditionally
    in ``put_model``.  Keep the compatibility adjustment process-local and
    leave current-driver behavior untouched.
    """

    import mujoco_warp as mjwarp
    import warp as wp
    from mujoco_warp._src import warp_util

    wp.init()
    if wp.is_conditional_graph_supported():
        return False
    if bool(getattr(mjwarp.put_model, "_trace_driver_fallback", False)):
        return True

    original_put_model = mjwarp.put_model

    def put_model_without_conditional_graph(*args: Any, **kwargs: Any) -> Any:
        model = original_put_model(*args, **kwargs)
        model.opt.graph_conditional = False
        return model

    put_model_without_conditional_graph._trace_driver_fallback = True  # type: ignore[attr-defined]
    # The guard only rejects the driver before the model exists.  The wrapper
    # immediately selects the supported non-conditional solver path afterward.
    warp_util.check_toolkit_driver = wp.init
    mjwarp.put_model = put_model_without_conditional_graph
    return True


def _required(cfg: Any, key: str) -> Any:
    value = OmegaConf.select(cfg, key, default=None)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Formal online TRACE config requires {key}.")
    return value


def create_v13_online_trace_manager(
    *,
    cfg: Any,
    agent: Any,
    training_config_path: str | Path,
    save_root: str | Path,
    device: str,
    v13_repo_root: str | Path,
) -> tuple[Go2OnlineTraceManager, Any]:
    """Instantiate proposal simulator, scorer lifecycle, and mutable replay."""

    if not bool(OmegaConf.select(cfg, "trace.enabled", default=False)):
        raise ValueError("create_v13_online_trace_manager requires trace.enabled=true.")
    if OmegaConf.select(cfg, "trace.replay_path", default=None) is not None:
        raise ValueError("Legacy trace.replay_path is forbidden for online TRACE.")
    if float(OmegaConf.select(cfg, "trace.replay_ratio", default=0.0)) != 0.0:
        raise ValueError("Legacy trace.replay_ratio is forbidden for online TRACE.")
    if not bool(OmegaConf.select(cfg, "replay_mix.enabled", default=False)):
        raise ValueError("Online TRACE requires the formal V13 replay mixer.")
    if str(OmegaConf.select(cfg, "replay_mix.synthetic_mode")) != "mixed":
        raise ValueError("Online TRACE requires replay_mix.synthetic_mode=mixed.")
    if float(OmegaConf.select(cfg, "replay_mix.trace_ratio_within_synthetic")) != float(
        _required(cfg, "trace.trace_ratio_within_synthetic")
    ):
        raise ValueError("TRACE ratio differs between manager and replay mixer.")
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    import src.tasks.rwm_velocity  # noqa: F401
    from flash_rl.envs.mjlab import configure_mjlab_randomization
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends
    from src.tasks.rwm_velocity.mdp.extractors import (
        Go2RWMExtractor,
        make_go2_policy_obs,
    )

    legacy_driver_fallback = _configure_mjwarp_driver_fallback()
    if legacy_driver_fallback:
        print(
            "[TRACE] CUDA conditional graphs unavailable; "
            "using MJWarp ordinary-loop fallback."
        )

    v13_root = Path(v13_repo_root).expanduser().resolve()
    dataset_path = Path(_required(cfg, "trace.dataset_path")).expanduser().resolve()
    dataset_sha = str(_required(cfg, "trace.dataset_sha256"))
    if sha256_path(dataset_path) != dataset_sha:
        raise ValueError("TRACE source dataset hash mismatch.")
    reset_certificate = Path(
        _required(cfg, "trace.reset_certificate_path")
    ).expanduser().resolve()
    reset_certificate_sha = sha256_path(reset_certificate)
    expected_reset_sha = str(_required(cfg, "trace.reset_certificate_sha256"))
    if reset_certificate_sha != expected_reset_sha:
        raise ValueError("TRACE reset certificate hash mismatch.")
    condition_id = str(_required(cfg, "trace.condition_id"))
    task_id = str(_required(cfg, "trace.task_id"))
    dataset_id = str(_required(cfg, "trace.dataset_id"))
    selection_backend = str(_required(cfg, "trace.selection_backend"))
    if selection_backend not in {"rule_bootstrap", "learned"}:
        raise ValueError("trace.selection_backend must be rule_bootstrap or learned.")
    num_start_states = int(_required(cfg, "trace.num_start_states"))
    branches = int(_required(cfg, "trace.trajectories_per_start"))
    num_proposal_envs = num_start_states * branches

    imperfect_task = str(_required(cfg, "trace.imperfect_simulator_task_id"))
    configure_torch_backends()
    env_cfg = load_env_cfg(imperfect_task)
    env_cfg.scene.num_envs = num_proposal_envs
    env_cfg.seed = int(_required(cfg, "trace.proposal_seed"))
    configure_mjlab_randomization(
        env_cfg,
        use_domain_randomization=False,
        use_push_randomization=False,
        use_observation_noise=False,
    )
    command = env_cfg.commands.get("twist")
    if command is None or not hasattr(command, "resampling_time_range"):
        raise ValueError("Imperfect simulator task lacks the twist command.")
    command.resampling_time_range = (1.0e9, 1.0e9)
    bad_events = {
        str(name): str(getattr(term, "mode", ""))
        for name, term in env_cfg.events.items()
        if str(getattr(term, "mode", "")) in {"startup", "step", "interval"}
    }
    if bad_events:
        raise ValueError(f"Imperfect simulator retains rollout-changing events: {bad_events}")
    noisy = [
        str(name)
        for name, group in env_cfg.observations.items()
        if bool(getattr(group, "enable_corruption", False))
    ]
    if noisy:
        raise ValueError(f"Imperfect simulator retains observation noise: {noisy}")
    proposal_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    actor = getattr(agent, "_actor", None) or getattr(agent, "actor", None)
    if actor is None:
        proposal_env.close()
        raise ValueError("V13 FlashSAC agent does not expose its current actor.")
    simulator_hash = canonical_sha256(
        {
            "v13_commit": V13_EXPECTED_COMMIT,
            "task": imperfect_task,
            "domain_randomization": False,
            "push_randomization": False,
            "observation_noise": False,
            "payload_mass_kg": 0.0,
            "rr_calf_strength_scale": 1.0,
        }
    )
    collector = V13MJLabProposalCollector(
        env=proposal_env,
        extractor=Go2RWMExtractor(proposal_env.unwrapped),
        actor_sampler=FlashSACActorDistributionSampler(actor),
        make_policy_observation=make_go2_policy_obs,
        simulator_config_sha256=simulator_hash,
        randomization_disabled=True,
        policy_observation_mask_indices=tuple(
            OmegaConf.select(
                cfg, "world_model.policy_observation_mask_indices", default=[]
            )
        ),
    )
    source_sampler = V13SnapshotSourceSampler(
        dataset_path,
        expected_dataset_sha256=dataset_sha,
        condition_id=condition_id,
        seed=int(_required(cfg, "trace.proposal_seed")) + 31,
    )
    semantics = CanonicalV13ReplaySemantics(
        v13_repo_root=v13_root,
        training_config_path=training_config_path,
        resolved_config=cfg,
        expected_commit=V13_EXPECTED_COMMIT,
        # Batched canonical reward/observation materialization stays on the
        # training GPU.  Only the finalized replay rows cross to the CPU ring.
        device=device,
    )
    model = None
    stats = None
    binding = None
    rule_config = None
    feedback_enabled = bool(
        _required(cfg, "trace.online_feedback_enabled")
    )
    if selection_backend == "learned":
        initial_scorer = Path(
            _required(cfg, "trace.initial_scorer_checkpoint")
        ).expanduser().resolve()
        initial_labels = Path(
            _required(cfg, "trace.initial_labels_path")
        ).expanduser().resolve()
        model, stats, binding, _metrics = load_scorer_checkpoint(initial_scorer)
        if (
            binding.task_id != task_id
            or binding.dataset_id != dataset_id
            or binding.dataset_sha256 != dataset_sha
            or binding.condition_id != condition_id
        ):
            proposal_env.close()
            raise ValueError(
                "Initial TRACE scorer binding differs from the active "
                "task/dataset/condition."
            )
        if feedback_enabled:
            initial_labels_sha256 = sha256_path(initial_labels)
            if binding.labels_sha256 != initial_labels_sha256:
                proposal_env.close()
                raise ValueError(
                    "Initial TRACE scorer binding differs from the active "
                    "initial-label artifact."
                )
        else:
            expected_frozen_binding = OmegaConf.select(
                cfg, "trace.frozen_scorer_binding_sha256", default=None
            )
            if (
                not expected_frozen_binding
                or str(expected_frozen_binding) != binding.sha256
            ):
                proposal_env.close()
                raise ValueError(
                    "Frozen TRACE scorer must be pinned by its exact binding SHA256."
                )
    else:
        if OmegaConf.select(
            cfg, "trace.initial_scorer_checkpoint", default=None
        ) is not None:
            proposal_env.close()
            raise ValueError(
                "Rule-bootstrap mode must not load an initial learned scorer."
            )
        if feedback_enabled:
            proposal_env.close()
            raise ValueError(
                "Rule-bootstrap mode forbids online feedback and LLM calls."
            )
        raw_rule = OmegaConf.to_container(
            _required(cfg, "trace.rule_bootstrap"),
            resolve=True,
            throw_on_missing=True,
        )
        if not isinstance(raw_rule, dict):
            proposal_env.close()
            raise ValueError("trace.rule_bootstrap must resolve to a mapping.")
        rule_config = RuleBootstrapConfig(**raw_rule)
        rule_config.validate()

    world = cfg.world_model
    trace_runtime_config = OmegaConf.to_container(
        cfg.trace, resolve=True, throw_on_missing=True
    )
    if not isinstance(trace_runtime_config, dict):
        raise ValueError("cfg.trace must resolve to a mapping.")
    trace_runtime_config.pop("resume_checkpoint", None)
    trace_runtime_config.pop("resume_actor_sample_temperature_from", None)
    trace_runtime_config.pop("resume_runtime_config_sha256", None)
    runtime_config_sha = canonical_sha256(
        {
            "trace": trace_runtime_config,
            "replay_mix": {
                "enabled": bool(cfg.replay_mix.enabled),
                "synthetic_mode": str(cfg.replay_mix.synthetic_mode),
                "trace_ratio_within_synthetic": float(
                    cfg.replay_mix.trace_ratio_within_synthetic
                ),
            },
        }
    )
    feedback_manager = None
    updater = None
    if selection_backend == "learned" and feedback_enabled:
        output_root = Path(save_root).expanduser().resolve() / "trace"
        provider = CodexBatchLabelProvider(
            repo_root=v13_root,
            schema_path=Path(__file__).with_name("feedback_label_batch.schema.json"),
            control_dir=output_root / "codex_control",
            batch_size=int(_required(cfg, "trace.llm.batch_size")),
            repair_rounds=int(_required(cfg, "trace.llm.repair_rounds")),
            timeout_seconds=float(_required(cfg, "trace.llm.timeout_seconds")),
            model=OmegaConf.select(cfg, "trace.llm.model", default=None),
            reasoning_effort=OmegaConf.select(
                cfg, "trace.llm.reasoning_effort", default=None
            ),
            workers=int(
                OmegaConf.select(cfg, "trace.llm.workers", default=16)
            ),
            max_retries=int(
                OmegaConf.select(cfg, "trace.llm.max_retries", default=2)
            ),
            retry_backoff_seconds=float(
                OmegaConf.select(
                    cfg,
                    "trace.llm.retry_backoff_seconds",
                    default=2.0,
                )
            ),
        )
        feedback_manager = CumulativeFeedbackManager(
            label_store_path=output_root / "cumulative_labels.jsonl",
            initial_labels_path=initial_labels,
            confidence_threshold=float(_required(cfg, "trace.confidence_threshold")),
            label_provider=provider,
            pair_seed=int(_required(cfg, "trace.pair_seed")),
            pair_sampling_mode=str(_required(cfg, "trace.pair_sampling_mode")),
            planar_command_scales=tuple(
                _required(cfg, "trace.planar_command_scales")
            ),
        )
        updater = OnlineScorerUpdater(
            feedback=feedback_manager,
            feedback_budget_initial=int(
                _required(cfg, "trace.feedback_budget_initial")
            ),
            feedback_budget_min=int(
                _required(cfg, "trace.feedback_budget_min")
            ),
            feedback_budget_decay=float(
                _required(cfg, "trace.feedback_budget_decay")
            ),
            cross_region_fraction=float(
                _required(cfg, "trace.cross_region_pair_fraction")
            ),
            task_id=task_id,
            dataset_id=dataset_id,
            dataset_sha256=dataset_sha,
            condition_id=condition_id,
            checkpoint_directory=output_root / "scorers",
            epochs=int(_required(cfg, "trace.scorer_update_epochs")),
            learning_rate=float(_required(cfg, "trace.scorer_learning_rate")),
            validation_ratio=float(_required(cfg, "trace.scorer_validation_ratio")),
            seed=int(_required(cfg, "trace.scorer_seed")),
            cpu_threads=int(_required(cfg, "trace.scorer_cpu_threads")),
        )
    manager = Go2OnlineTraceManager(
        config=OnlineTraceConfig(
            condition_id=condition_id,
            dataset_sha256=dataset_sha,
            runtime_config_sha256=runtime_config_sha,
            selection_backend=selection_backend,
            reset_certificate_sha256=reset_certificate_sha,
            proposal_interval=int(_required(cfg, "trace.proposal_interval")),
            feedback_interval=int(_required(cfg, "trace.feedback_interval")),
            num_start_states=num_start_states,
            rollout_horizon=int(_required(cfg, "trace.rollout_horizon")),
            trajectories_per_start=branches,
            select_alpha=float(_required(cfg, "trace.select_alpha")),
            trace_ratio_within_synthetic=float(
                _required(cfg, "trace.trace_ratio_within_synthetic")
            ),
            buffer_capacity=int(_required(cfg, "trace.buffer_capacity")),
            command_active_thresholds=(
                float(world.reward_command_active_threshold_x),
                float(world.reward_command_active_threshold_y),
                float(world.reward_command_active_threshold_yaw),
            ),
            command_normalization_floors=(
                float(world.reward_response_command_scale_floor),
            )
            * 3,
            action_saturation_threshold=float(
                world.reward_action_saturation_threshold
            ),
            proposal_seed=int(_required(cfg, "trace.proposal_seed")),
            actor_sample_temperature=float(
                _required(cfg, "trace.actor_sample_temperature")
            ),
            resume_actor_sample_temperature_from=(
                float(value)
                if (
                    value := OmegaConf.select(
                        cfg,
                        "trace.resume_actor_sample_temperature_from",
                        default=None,
                    )
                )
                is not None
                else None
            ),
            resume_runtime_config_sha256=OmegaConf.select(
                cfg,
                "trace.resume_runtime_config_sha256",
                default=None,
            ),
        ),
        proposal_collector=collector,
        source_sampler=source_sampler,
        replay_semantics=semantics,
        scorer_model=model,
        scorer_stats=stats,
        scorer_binding=binding,
        scorer_updater=updater,
        rule_config=rule_config,
    )
    manager.buffer.bind_provenance(
        {
            "gamma": semantics.gamma,
            "n_step": semantics.n_step,
            "reward_config_sha256": semantics.reward_config_sha256,
            "condition_id": condition_id,
            "dataset_sha256": dataset_sha,
            "reset_certificate_sha256": reset_certificate_sha,
            "trace_protocol_version": "go2_trace_online_v1",
            "selection_backend": selection_backend,
        }
    )
    # Keep feedback state attached so the trainer checkpoint bridge can include
    # and validate its cumulative-label cursor.
    manager.feedback_manager = feedback_manager
    return manager, proposal_env
