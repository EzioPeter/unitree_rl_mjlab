"""Default config helpers for Go2 model-based RWM policy training."""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from scripts.reinforcement_learning.rwm.envs.go2_flat import Go2ImaginationCfg


@dataclass
class Go2FlatRWMConfig:
    experiment_name: str = "go2_flat_rwm_model_based"
    imagination: Go2ImaginationCfg = field(default_factory=Go2ImaginationCfg)


def unitree_go2_rwm_model_based_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.0001,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="go2_flat_rwm_model_based",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=500,
        logger="wandb",
        upload_model=False,
    )
