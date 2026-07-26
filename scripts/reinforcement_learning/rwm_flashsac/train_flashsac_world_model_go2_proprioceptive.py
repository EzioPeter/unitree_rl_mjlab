"""Train Go2 FlashSAC-RWM with proprioceptive actor input.

The critic still receives the original full 48-dim RWM policy observation. The
actor receives only the remaining 45 dims. Dynamics, reward, dataset sampling,
and action space are otherwise identical to the standard RWM-FlashSAC trainer.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_flashsac import train_flashsac_world_model_go2 as base_trainer
from scripts.reinforcement_learning.rwm_flashsac.agent_proprioceptive import (
    create_go2_flashsac_proprioceptive_agent,
)
from scripts.reinforcement_learning.rwm_flashsac.dynamics_loader import (
    load_any_go2_dynamics_checkpoint,
)

base_trainer.load_dynamics_checkpoint = load_any_go2_dynamics_checkpoint
base_trainer.create_go2_flashsac_agent = create_go2_flashsac_proprioceptive_agent


if __name__ == "__main__":
    base_trainer.main()
