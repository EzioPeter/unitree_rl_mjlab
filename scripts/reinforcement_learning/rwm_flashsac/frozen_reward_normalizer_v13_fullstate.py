"""V13 full-state frozen reward normalizer for the frozen mixed dataset."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_flashsac import (  # noqa: E402
    frozen_reward_normalizer as baseline,
)


_build_frozen_reward_normalizer = baseline.build_frozen_reward_normalizer


def _build_v13_frozen_reward_normalizer(
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    artifact = _build_frozen_reward_normalizer(
        *args,
        allowed_collector_types=(0, 1, 2, 3, 4),
        **kwargs,
    )
    metadata = artifact["metadata"]
    if metadata.get("source_collector_types") != [0, 1, 2, 3, 4]:
        raise ValueError(
            "V13 reward normalizer source must contain every collector type 0..4."
        )
    metadata["source_policy"] = "v13_frozen_five_collector_mixed_dataset"
    return artifact


baseline.build_frozen_reward_normalizer = _build_v13_frozen_reward_normalizer


if __name__ == "__main__":
    baseline.main()
