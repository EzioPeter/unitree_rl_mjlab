"""V13 full-state policy entry using the audited 5%/95% replay mixer."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_flashsac import (  # noqa: E402
    train_flashsac_world_model_go2 as baseline,
)
from scripts.reinforcement_learning.rwm_flashsac.agent_fullstate_replay import (  # noqa: E402
    create_go2_flashsac_fullstate_replay_agent,
)

_write_replay_manifest = baseline._write_formal_replay_manifest


def _write_v13_replay_manifest(**kwargs: object) -> None:
    _write_replay_manifest(**kwargs)
    path = kwargs["path"]
    if not isinstance(path, Path):
        raise TypeError("Replay manifest path must be a pathlib.Path.")
    import json

    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["format_version"] = "go2_v13_fullstate_replay_mix_manifest_v1"
    manifest["policy_interface"] = {
        "actor_observation_dim": 48,
        "critic_observation_dim": 48,
        "actor_contains_base_lin_vel": True,
        "critic_contains_base_lin_vel": True,
    }
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


baseline.create_go2_flashsac_agent = create_go2_flashsac_fullstate_replay_agent
baseline._write_formal_replay_manifest = _write_v13_replay_manifest


if __name__ == "__main__":
    baseline.main()
