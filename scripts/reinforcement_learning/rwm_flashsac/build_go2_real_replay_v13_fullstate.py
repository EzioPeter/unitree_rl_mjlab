"""V13 full-state wrapper for the audited real-replay builder."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_flashsac import (  # noqa: E402
    build_go2_real_replay as baseline,
)


_build_real_replay = baseline.build_real_replay


def _build_v13_fullstate_replay(*args: Any, **kwargs: Any) -> dict[str, Any]:
    artifact = _build_real_replay(
        *args,
        allowed_collector_types=(0, 1, 2, 3, 4),
        **kwargs,
    )
    if artifact["observation"].shape[-1] != 48:
        raise ValueError("V13 real replay observation is not 48D.")
    if artifact["next_observation"].shape[-1] != 48:
        raise ValueError("V13 real replay next_observation is not 48D.")
    metadata = artifact["metadata"]
    if metadata.get("source_collector_types") != [0, 1, 2, 3, 4]:
        raise ValueError(
            "V13 real replay source must contain every collector type 0..4."
        )
    metadata["actor_observation_dim"] = 48
    metadata["critic_observation_dim"] = 48
    metadata["actor_contains_base_lin_vel"] = True
    metadata["critic_contains_base_lin_vel"] = True
    metadata["v13_policy_interface"] = "fullstate_48d_actor_48d_critic"
    metadata["source_policy"] = "v13_frozen_five_collector_mixed_dataset"
    return artifact


baseline.build_real_replay = _build_v13_fullstate_replay


if __name__ == "__main__":
    baseline.main()
