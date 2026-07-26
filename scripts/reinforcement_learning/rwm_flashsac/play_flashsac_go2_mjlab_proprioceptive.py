"""Visualize an proprioceptive-lin-vel Go2 FlashSAC-RWM policy in mjlab.

This reuses the regular Go2 FlashSAC-RWM play script and swaps only the agent
factory. The environment observation remains the normal 48-dim RWM observation;
the proprioceptive agent slices off base_lin_vel internally before actor inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_flashsac import play_flashsac_go2_mjlab as base_play
from scripts.reinforcement_learning.rwm_flashsac.agent_proprioceptive import (
    create_go2_flashsac_proprioceptive_agent,
)


def main() -> None:
    base_play.create_go2_flashsac_agent = create_go2_flashsac_proprioceptive_agent
    base_play.main()


if __name__ == "__main__":
    main()
