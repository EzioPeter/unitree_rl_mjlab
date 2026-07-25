# V13 full-state wide-command training freeze

This branch freezes the exact code and settings used by the formal V13 run on
2026-07-25. It is a peer branch created directly from:

```text
origin/model_based
daa2eae7324255d4011964d2e74199547f347f26
```

Branch:

```text
v13/fullstate-widecmd-25k
```

The old `model_based_broken` branch is not this branch's base.

## Formal route

- Five condition-specific mixed simulator datasets: `g0`, `rr03`, `rr05`,
  `p5`, and `p75`.
- Each selected dataset is 25 complete 1,000-step trajectories: 25K
  transitions.
- Requested source collector mix: expert 45%, noisy expert 25%, medium 10%,
  failure-border 15%, random 5%.
- Dataset snapshots are resettable in the simulator.
- One condition-specific RWM is trained for each dataset.
- RWM state interface is 45D input to 45D output. `base_lin_vel[3]` is present
  in the input and is supervised in the target.
- Policy Actor and Critic both receive the same 48D full-state observation,
  including `base_lin_vel[3]` and the 3D command.
- Formal policy update is 5% real replay and 95% RWM replay, with no BC and no
  sim/TRACE replay in route one.

## Frozen dataset artifacts

Manifest:

```text
V13_SIM_DATASET_MANIFEST.json
SHA256 f507931ef4106d6c64952ddc25f4280d00c17330d8396bf30a788b6195bb28ff
```

Selected dataset SHA256:

```text
g0    39699589c47cf056df26fa9ea651ec78e3692b2099b4333f9678058051af96f2
rr03  3a2cb6487a5cf79b94cd02bba9037c95124ada85d2b629974c4b15facf5938f2
rr05  928b70fd6133201824fb5c7283c1f43fc5dc2c238735004a27c7f601704bc156
p5    e102ee455e7d512289f3f00359fc2c703f181ec402a8e169dbb3d5909641622d
p75   d152a35d4917b88f01008de5d532d918346bffad1fab5b4a3b8f25d8ffd88a59
```

Datasets and checkpoints are intentionally not committed to Git.

## Frozen RWM

Configuration:

```text
scripts/reinforcement_learning/rwm_dataset/configs/
  go2_offline_world_model_v13_full45.yaml
SHA256 f66f1f12500146b08224722094b0f1822391134ce1ff92d328420c9f615c73bc
```

Final RWM checkpoint SHA256:

```text
g0    aaa1516158a36aa9792b0df7ae6d0380a3d13a4079102566cbb31f2e3e93157d
rr03  b63eff7524e5c30752c7c78ec76889e8753361ce5d0229b7e4147bc9f58bb23d
rr05  eaa049d863b4d353118a6e89e8e38bdf8d08944e0fd8057c3993d6d929603e87
p5    93a894aef349c4da889b165b3045d4f28fce8fe821e31be5d8d84f7fbb33874e
p75   f913dfad8affcb9cd257d90255e6685f4fe0914aecb4ac9db170ddb3ec56e634
```

Launch:

```bash
scripts/reinforcement_learning/rwm_dataset/launch_v13_rwm_full45_fresh_queue.sh
```

## Frozen policy settings

```text
seed:                         0
nominal env-step budget:      50,000,000
actual env steps:             49,999,872
interaction steps:            48,828
imagination envs:             1,024
updates per interaction:      2
batch size:                   2,048
real / RWM / sim rows:        102 / 1,946 / 0
replay capacity/device:       10,000,000 / CPU
learning start:               100,000 transitions
gamma:                        0.99
n_step:                       3
reward version:               v1
reward normalization:         enabled, frozen, condition-specific
normalize_reward:             true
normalized_G_max:             5.0
uncertainty penalty:          -2.0
critic support:               [-5.0, 5.0]
actor_bc_alpha:               0.0
Actor / Critic observation:   48D / 48D
command resample interval:    120 to 300 steps
standing probability:         0.08
vx range:                     [-0.5, 0.5]
vy range:                     [-0.2, 0.2]
yaw range:                    [-0.4, 0.4]
save replay buffer:           true
```

Formal policy configuration:

```text
scripts/reinforcement_learning/rwm_flashsac/configs/
  go2_v13_fullstate_rwm_mixed1m_widecmd_policycfg_50m.yaml
SHA256 6884a9dd6b3f84ca61f0822062b42fdceb37f00085c4b179da89551f667685e0
```

Core policy code SHA256 captured for this run:

```text
train_flashsac_world_model_go2.py
  d3bfae3d140178e2371fa48d46db3b32a811c12d2c47c652496425d5cc9a1d65
train_flashsac_world_model_go2_v13_fullstate.py
  66774b226fb052d7f81805b2d3ee6b82ae0893a8afa7b4a1ca0a72cd66174859
replay_mixer.py
  1030b165de58a0d09225c24703b70e0af5c83abe81ec31b928c363449cdde980
agent_fullstate_replay.py
  3e8b5859a5d460d56d282e3307befbe4828aa47402dc661472f04610af4fc5d6
world_model_env.py
  7abcb144c591632996787a6ae4ef24e204f97961b9e32c6b7e81f0bb8f378ada
```

Launch:

```bash
scripts/reinforcement_learning/rwm_flashsac/
  launch_v13_fullstate_mixed1m_widecmd_policycfg_queue.sh
```

The earlier `vx +/-0.3`, `vy +/-0.15`, `yaw +/-0.3` run is a small-command
ablation and is not the formal run frozen here.

## Runtime and validation

Runtime used on the RTX 3090 host:

```text
/data1/xjy/venvs/unitree_rl_mjlab_model_based_daa2eae_20260725
Python 3.11.15
Torch 2.9.1+cu128
OmegaConf 2.3.0
```

The branch preserves the byte-identical training files used by the running
experiment. Ruff checks therefore explicitly ignore only `E402` path-bootstrap
imports and pre-existing `F401` imports; these are non-semantic style findings.
All other Ruff checks and the V13 replay, reward-normalizer, full-state,
selection, and BLV-supervision tests must pass.

At 2026-07-25T18:28+08:00, after all five formal policy processes had already
started, the mutable training directory received later TRACE-contract edits to
`train_flashsac_world_model_go2.py`, `replay_mixer.py`, and its replay-mixer
test. Those post-launch edits are deliberately excluded from this freeze. This
branch contains the pre-edit files copied before that timestamp; their hashes
above bind the route-one run that has `trace_ratio_within_synthetic=0.0`.

The authoritative route, physical-condition mapping, audit history, and
artifact binding are documented in `V13_TWO_ROUTE_EXPERIMENT_DESIGN_ZH.md`.
