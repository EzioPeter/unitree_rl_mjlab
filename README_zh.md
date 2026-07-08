# Unitree RL Mjlab


## ✳️ 概述

Unitree RL Mjlab 是一个基于 [mjlab](https://github.com/mujocolab/mjlab.git) 构建的强化学习项目，
使用 MuJoCo 作为物理仿真后端，当前支持 Unitree Go2, A2, As2, G1, R1, H1_2 和 H2 机器人。

Mjlab 结合了 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 的成熟高层 API 与 
[MuJoCo](https://github.com/google-deepmind/mujoco_warp) 的高精度物理引擎，
为强化学习机器人研究与 Sim-to-Real（仿真到实机） 部署提供了一个轻量化、模块化的框架。

<div align="center">

| <div align="center">  MuJoCo </div>                                                                                                                                           | <div align="center"> Physical </div>                                                                                                                                               |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/g1-velocity.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> | <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/g1-velocity-real.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> |

</div>


## 📦 安装配置

安装和配置步骤请参考 [setup.md](doc/setup_zh.md)


## 🔁 流程概览

使用强化学习实现机器人运动控制的基本流程如下：

`训练` → `仿真验证` → `仿真到实机`

- **训练**: 在 MuJoCo 模拟环境中让机器人与环境交互，并通过奖励函数最大化学习策略。
- **仿真验证**: 加载训练好的策略进行回放，验证策略行为是否符合预期。
- **仿真到实机**: 将策略部署到物理机器人上，实现真实环境中的运动控制。


## 🛠️ 使用指南

### 1. 速度跟踪训练

运行以下命令进行速度跟踪训练：

```bash
uv run python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

多 GPU 训练：使用 --gpu-ids 扩展到多块 GPU：

```bash
uv run python scripts/train.py Unitree-G1-Flat \
  --gpu-ids 0 1 \
  --env.scene.num-envs=4096
```

- 第一个参数(如 Mjlab-Velocity-Flat-Unitree-G1)为必选参数，确定要启用的训练环境。可选：
  - Unitree-Go2-Flat
  - Unitree-G1-Flat
  - Unitree-G1-23Dof-Flat
  - Unitree-H1_2-Flat
  - Unitree-A2-Flat
  - Unitree-R1-Flat

> [!NOTE]
> 更多有关详细说明，请参阅 mjlab 文档
> [mjlab documentation](https://mujocolab.github.io/mjlab/index.html).

### 2. 动作模仿训练

训练 Unitree G1 模仿参考动作序列。

<div style="margin-left: 20px;">

#### 2.1 准备动作文件

将准备好的 csv 格式的动作文件保存在 mjlab/motions/g1/ 目录下，执行下面的指令将其转为训练可用的 npz 文件：

```bash
uv run python scripts/csv_to_npz.py \
--input-file src/assets/motions/g1/dance1_subject2.csv \
--output-name dance1_subject2.npz \
--input-fps 30 \
--output-fps 50 \
--robot g1 # g1 or g1_23dof
```

**npz文件默认保存路径为**：`src/motions/g1/...`

#### 2.2 训练

确保有可用的npz文件之后，执行以下指令进行训练：

```bash
uv run python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --env.scene.num-envs=4096
```

可用任务:
  - Unitree-G1-Tracking-No-State-Estimation
  - Unitree-G1-23Dof-Tracking-No-State-Estimation

</div>

> [!NOTE]
> 有关动作模仿训练的详细说明，请参阅BeyondMimic 文档
> [BeyondMimic documentation](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup).

#### ⚙️  参数说明
- `--env.scene`: 仿真场景配置，包括环境数量（num_envs）、物理仿真步长、地面类型、重力、随机扰动等参数。
- `--env.observations`: 观测空间配置，控制训练时输入到策略网络的状态信息，如关节位置、速度、IMU等内容。
- `--env.rewards`: 奖励函数配置，定义每步训练时的优化目标。
- `--env.commands`: 控制命令配置，用于生成训练时随机或指定的速度 / 姿态 / 动作指令。
- `--env.terminations`: 终止条件配置，定义训练 episode 的结束条件。
- `--agent.seed`: 训练随机种子，用于结果复现，不同 seed 会导致策略略有差异。
- `--agent.resume`: 是否从上次中断的 checkpoint 继续训练。 设置为 True 时，会自动加载最近一次保存的 .pt 模型文件。
- `--agent.policy`: 策略网络结构配置，例如 MLP 层数、隐藏维度、激活函数等。
- `--agent.algorithm`: 强化学习算法配置。可设置优化超参数，如学习率、批量大小、GAE λ 等。

**默认保存训练结果**：`logs/rsl_rl/<robot>_(velocity | tracking)/<date_time>/model_<iteration>.pt`

### 3. FlashSAC G1 速度训练与导出

FlashSAC 使用当前仓库的 G1 mjlab 环境和同一套实物部署观测/动作接口。现有部署参考策略
`deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx` 的 ONNX 签名为：
`obs [1, 98] -> actions [1, 29]`。FlashSAC 导出脚本会以该文件为基准校验签名，不会修改部署端输入输出维度。

```bash
uv run python scripts/train_flashsac.py
```

当前推荐的 G1 DR 从零训练命令：

```bash
uv run python scripts/train_flashsac.py \
  --overrides exp_name=flat \
  --overrides agent_load_path=null \
  --overrides buffer_load_path=null \
  --overrides num_train_envs=1024 \
  --overrides updates_per_interaction_step=2 \
  --overrides agent.buffer_max_length=10000000 \
  --overrides agent.buffer_min_length=100000 \
  --overrides agent.buffer_device_type=cpu \
  --overrides agent.sample_batch_size=2048 \
  --overrides n_step=3 \
  --overrides env.use_domain_randomization=true \
  --overrides env.use_push_randomization=true \
  --overrides env.use_observation_noise=true
```

Go2 DR 从零训练命令如下。Go2 当前不使用 G1 的部署 ONNX 导出，因此需要加 `--no-export_deploy_policy`：

```bash
uv run python scripts/train_flashsac.py \
  --no-export_deploy_policy \
  --overrides group_name=go2_velocity_flashsac \
  --overrides exp_name=flat \
  --overrides env.env_name=Unitree-Go2-Flat \
  --overrides num_train_envs=1024 \
  --overrides updates_per_interaction_step=2 \
  --overrides agent.buffer_max_length=10000000 \
  --overrides agent.buffer_min_length=100000 \
  --overrides agent.buffer_device_type=cpu \
  --overrides agent.sample_batch_size=2048 \
  --overrides n_step=3 \
  --overrides env.use_domain_randomization=true \
  --overrides env.use_push_randomization=true \
  --overrides env.use_observation_noise=true
```

训练保存 checkpoint 时会同步导出最新策略到：
`deploy/robots/g1/config/policy/velocity/v1_flashsac/exported/policy.onnx`。

也可以手动从某个 FlashSAC checkpoint 导出：

```bash
uv run python scripts/export_flashsac_g1.py \
  --checkpoint_path models/g1_velocity_flashsac/flat/Unitree-G1-Flat/seed0-xxxx/stepxxxxx
```

在 mjlab 中回放：

```bash
uv run python scripts/play_flashsac_mjlab.py \
  --checkpoint_path models/g1_velocity_flashsac/flat/Unitree-G1-Flat/seed0-xxxx/stepxxxxx
```

### 4. Go2 RWM / MOPO 两阶段训练

当前仓库提供两套 Go2 flat RWM-friendly velocity 的两阶段训练流程：

- **MOPO-PPO**: Stage 1 用 PPO 在真实 mjlab 中采样并训练 world model；Stage 2 用 PPO 在 learned world model 中训练 policy。
- **MOPO-SAC**: Stage 1 用 FlashSAC 在真实 mjlab 中采样并训练 world model；Stage 2 用 FlashSAC 在 learned world model 中训练 policy。

#### 4.1 MOPO-PPO

Stage 1：PPO 采真实 mjlab 数据并训练 Go2 world model。

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run python scripts/reinforcement_learning/rwm/train_pretrain_go2.py \
  --task Unitree-Go2-Flat-RWM-Pretrain-Ens \
  --headless
```

输出目录：

```text
logs/rsl_rl/go2_flat_rwm_pretrain/<run>/
  model_<iter>.pt
  dataset.pt
  policy_<iter>.pt
```

Stage 2：加载 Stage 1 的 world model，在 imagination env 中训练 PPO policy。

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run python scripts/reinforcement_learning/rwm/train_model_based_go2.py \
  --task go2_flat \
  --model_resume_path logs/rsl_rl/go2_flat_rwm_pretrain/<run>/model_<iter>.pt \
  --dataset_path logs/rsl_rl/go2_flat_rwm_pretrain/<run>/dataset.pt \
  --run_num 0
```

输出目录：

```text
logs/model_based/go2_flat/<run>/
  policy_<iter>.pt
```

当前可复用的已训练 PPO world model 示例：

```text
logs/rsl_rl/go2_flat_rwm_pretrain/2026-06-04_20-35-46/model_10000.pt
logs/rsl_rl/go2_flat_rwm_pretrain/2026-06-04_20-35-46/dataset.pt
```

#### 4.2 MOPO-SAC

Stage 1：FlashSAC 采真实 mjlab 数据并训练 Go2 world model。该脚本同时保存 FlashSAC actor 和 RWM dynamics。

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run python scripts/reinforcement_learning/rwm_flashsac/train_pretrain_flashsac_go2.py \
  --task Unitree-Go2-Flat-RWM-Pretrain-Ens \
  --num_envs 512 \
  --num_env_steps 100000000 \
  --overrides updates_per_interaction_step=1 \
  --overrides agent.buffer_max_length=10000000 \
  --overrides agent.buffer_min_length=50000 \
  --overrides agent.buffer_device_type=cpu \
  --overrides agent.sample_batch_size=2048 \
  --overrides agent.n_step=3 \
  --overrides system_dynamics.batch_size=256 \
  --overrides system_dynamics.min_transitions=50000 \
  --overrides system_dynamics.updates_per_interaction_step=1 \
  --overrides env.use_domain_randomization=true \
  --overrides env.use_push_randomization=true \
  --overrides env.use_observation_noise=true
```

输出目录：

```text
logs/rsl_rl/go2_flat_rwm_flashsac_pretrain/<run>/
  model_<interaction_step>.pt
  dataset.pt
  step<interaction_step>/actor.pt
  step<interaction_step>/critic.pt
  step<interaction_step>/target_critic.pt
  step<interaction_step>/temperature.pt
```

Stage 2：加载 SAC Stage 1 训练出的 world model，在 imagination env 中训练 FlashSAC policy。

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run python scripts/reinforcement_learning/rwm_flashsac/train_flashsac_world_model_go2.py \
  --model_resume_path logs/rsl_rl/go2_flat_rwm_flashsac_pretrain/<run>/model_<interaction_step>.pt \
  --dataset_path logs/rsl_rl/go2_flat_rwm_flashsac_pretrain/<run>/dataset.pt \
  --num_imagination_envs 1024 \
  --num_env_steps 50000000 \
  --overrides save_path=logs/model_based/go2_flat_flashsac_rwm_sacwm/TIMESTAMP \
  --overrides updates_per_interaction_step=2 \
  --overrides agent.buffer_max_length=10000000 \
  --overrides agent.buffer_min_length=100000 \
  --overrides agent.buffer_device_type=cpu \
  --overrides agent.sample_batch_size=2048 \
  --overrides agent.n_step=3 \
  --overrides agent.normalize_reward=true \
  --overrides agent.normalized_G_max=5.0 \
  --overrides agent.critic_min_v=-5.0 \
  --overrides agent.critic_max_v=5.0
```

输出目录：

```text
logs/model_based/go2_flat_flashsac_rwm_sacwm/<run>/step<interaction_step>/
  actor.pt
  critic.pt
  target_critic.pt
  temperature.pt
  agent_state.pt
```

在真实 mjlab Go2 RWM task 中评估 FlashSAC-RWM policy：

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run python scripts/reinforcement_learning/rwm_flashsac/eval_flashsac_go2_mjlab.py \
  --checkpoint_path logs/model_based/go2_flat_flashsac_rwm_sacwm/<run>/step<interaction_step> \
  --num_envs 1024 \
  --steps 1001
```

当前可复用的已训练 SAC world model 和 policy 示例：

```text
logs/rsl_rl/go2_flat_rwm_flashsac_pretrain/2026-06-05_01-39-13/model_195312.pt
logs/rsl_rl/go2_flat_rwm_flashsac_pretrain/2026-06-05_01-39-13/dataset.pt
logs/model_based/go2_flat_flashsac_rwm_sacwm/2026-06-05_17-27-59/step48828
```

#### 4.3 200k mixed dataset 离线 world model + MOPO-SAC

这条流程先用已经采集好的 Go2 mixed sim dataset 离线训练 RWM-U dynamics，然后用该 world model 给 FlashSAC/MOPO-SAC 提供 imagination transition。

当前 200k dataset 直接复用 1M mixed dataset 的第一个 part：

```text
logs/rwm_datasets/go2_flat_mixed_1m/parts/dataset_part_000.pt
```

该文件约包含 200k transition。实际已跑通的离线 world model 训练命令如下：

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run python scripts/reinforcement_learning/rwm_dataset/train_world_model_offline_go2.py \
  --dataset_path logs/rwm_datasets/go2_flat_mixed_1m/parts/dataset_part_000.pt \
  --save_dir logs/rsl_rl/go2_flat_rwm_offline_mixed_200k \
  --max_iterations 5000 \
  --batch_size 1024 \
  --micro_batch_size 256 \
  --ensemble_size 5 \
  --history_horizon 32 \
  --forecast_horizon 8 \
  --device cuda:0
```

输出示例：

```text
logs/rsl_rl/go2_flat_rwm_offline_mixed_200k/2026-06-06_21-39-21/model_5000.pt
logs/rsl_rl/go2_flat_rwm_offline_mixed_200k/2026-06-06_21-39-21/final_metrics.json
```

然后用这个 200k offline world model 训练 MOPO-SAC policy：

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run python scripts/reinforcement_learning/rwm_flashsac/train_flashsac_world_model_go2.py \
  --model_resume_path logs/rsl_rl/go2_flat_rwm_offline_mixed_200k/2026-06-06_21-39-21/model_5000.pt \
  --dataset_path logs/rwm_datasets/go2_flat_mixed_1m/parts/dataset_part_000.pt \
  --num_imagination_envs 1024 \
  --num_env_steps 50000000 \
  --overrides save_path=logs/model_based/go2_flat_flashsac_rwm_mixed200k/TIMESTAMP \
  --overrides updates_per_interaction_step=2 \
  --overrides agent.buffer_max_length=10000000 \
  --overrides agent.buffer_min_length=100000 \
  --overrides agent.buffer_device_type=cpu \
  --overrides agent.sample_batch_size=2048 \
  --overrides agent.n_step=3 \
  --overrides uncertainty_penalty_weight=-1.0
```

输出示例：

```text
logs/model_based/go2_flat_flashsac_rwm_mixed200k/2026-06-06_22-19-16/step<interaction_step>/
```

真实 mjlab eval 命令：

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run python scripts/reinforcement_learning/rwm_flashsac/eval_flashsac_go2_mjlab.py \
  --checkpoint_path logs/model_based/go2_flat_flashsac_rwm_mixed200k/2026-06-06_22-19-16/step22000 \
  --num_envs 1024 \
  --steps 1001 \
  --device cuda:0
```

当前 200k run 的真实 mjlab eval 最佳 checkpoint 是：

```text
logs/model_based/go2_flat_flashsac_rwm_mixed200k/2026-06-06_22-19-16/step22000
```

对应结果约为 `mean_return=46.43`、`mean_episode_length=1001`、`terminated_count=0`。后续 checkpoint 会出现明显过训练，因此不建议直接使用 final `step48828`。

#### 4.4 1M mixed safety command coverage 队列

当前推荐的离线 RWM-U + MOPO-SAC 路线是使用 mixed safety command coverage 数据集。该队列会自动串联：

```text
采集 1M mixed Go2 sim transitions
  -> 离线训练 SystemDynamicsEnsemble / world model
  -> 使用该 world model 训练 FlashSAC / MOPO-SAC policy
```

一键从头运行：

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 bash scripts/reinforcement_learning/rwm_dataset/run_mixed_safety_command_coverage_1m_offline_wm_sac_queue.sh
```

如果数据集已经存在，只想重新训练 world model 和 SAC policy：

```bash
SKIP_COLLECT=true WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 bash scripts/reinforcement_learning/rwm_dataset/run_mixed_safety_command_coverage_1m_offline_wm_sac_queue.sh
```

默认输出路径：

```text
dataset:
logs/rwm_datasets/go2_flat_mixed_safety_command_coverage_1m/dataset.pt

world model:
logs/rsl_rl/go2_flat_rwm_offline_mixed_safety_command_coverage_1m_aligned_hidden/<run>/model_5000.pt

SAC policy:
logs/model_based/go2_flat_flashsac_rwm_mixed_safety_command_coverage_1m/<run>/step<interaction_step>/

queue logs:
logs/queues/offline_wm_sac_mixed_safety_cmdcov1m/<run>/
```

该队列默认采集分布：

```text
collector mix:
  expert          45%
  noisy_expert    25%
  medium          10%
  failure_border  15%
  random           5%

command mode weights:
  stand      8%
  pure_x    25%
  pure_y    10%
  pure_yaw   8%
  xy        14%
  x_yaw     17%
  y_yaw      5%
  xy_yaw    13%
```

采集 command 范围：

```text
vx:  signed, |vx| in [0.05, 0.5], plus stand mode at 0
vy:  signed, |vy| in [0.03, 0.2], plus modes where vy=0
yaw: signed, |yaw| in [0.05, 0.4], plus modes where yaw=0
```

SAC imagination 训练 command 范围更保守：

```text
vx:  [-0.3, 0.3]
vy:  [-0.15, 0.15]
yaw: [-0.3, 0.3]
```

可以用环境变量覆盖默认配置，例如缩小采集量或修改速度范围：

```bash
COLLECT_NUM_TRANSITIONS=200000 \
X_ABS_RANGE_MAX=0.4 \
Y_ABS_RANGE_MAX=0.15 \
YAW_ABS_RANGE_MAX=0.3 \
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 \
bash scripts/reinforcement_learning/rwm_dataset/run_mixed_safety_command_coverage_1m_offline_wm_sac_queue.sh
```

查看队列进度：

```bash
QUEUE_ROOT=logs/queues/offline_wm_sac_mixed_safety_cmdcov1m/<run>
cat ${QUEUE_ROOT}/state.txt
tail -f ${QUEUE_ROOT}/summary.txt
tail -f ${QUEUE_ROOT}/03_train_sac.log
```

本仓库当前跑通的一组结果：

```text
dataset:
logs/rwm_datasets/go2_flat_mixed_safety_command_coverage_1m/dataset.pt

world model:
logs/rsl_rl/go2_flat_rwm_offline_mixed_safety_command_coverage_1m_aligned_hidden/2026-06-09_12-14-43/model_5000.pt

best SAC policy:
logs/model_based/go2_flat_flashsac_rwm_mixed_safety_command_coverage_1m/2026-06-09_12-36-19/step32000
```

该 policy 在真实 mjlab fixed command `vx=0.3, vy=0, yaw=0` 下的 512-env eval 结果约为：

```text
mean_return = 59.85
mean_episode_length = 1001
terminated_count = 0
base_lin_vel_x = 0.258
error_vel_xy = 0.046
base_yaw_vel = 0.005
```

可视化最佳 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/reinforcement_learning/rwm_flashsac/play_flashsac_go2_mjlab.py \
  --checkpoint_path logs/model_based/go2_flat_flashsac_rwm_mixed_safety_command_coverage_1m/2026-06-09_12-36-19/step32000 \
  --num_envs 1 \
  --device cuda:0 \
  --viewer native \
  --fixed_command 0.3 0.0 0.0
```

随机小命令切换可视化：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/reinforcement_learning/rwm_flashsac/play_flashsac_go2_mjlab.py \
  --checkpoint_path logs/model_based/go2_flat_flashsac_rwm_mixed_safety_command_coverage_1m/2026-06-09_12-36-19/step32000 \
  --num_envs 1 \
  --device cuda:0 \
  --viewer native \
  --random_command \
  --command_switch_steps 500 \
  --random_lin_vel_x -0.3 0.3 \
  --random_lin_vel_y -0.15 0.15 \
  --random_ang_vel_z -0.3 0.3
```

#### 4.5 Go2 RR calf 0.5 电机能力 RWM + policy 训练 pipeline

当前坏/弱关节实验只保留 0.5 电机能力设定，不再使用完全置零关节作为默认流程。最小链路如下：

```text
curriculum 训练 0.5-strength expert policy
  -> 用该 expert policy 采集 0.5-strength proprioceptive dataset
  -> 用 dataset 离线训练 proprioceptive RWM
  -> 用 RWM 作为 imagination env 训练 FlashSAC policy
  -> 对比 unmasked 与 full-joint-masked 两组
```

这里的 `full-joint-masked` 指 `RR_calf_joint` 不进入任何网络：

```text
RWM state input: 45 -> 39, 删除 base_lin_vel[0:3] 和 RR_calf 的 pos/vel/force
RWM state output: 45 -> 42, 删除 RR_calf 的 pos/vel/force
RWM/action policy: 12 -> 11, 删除 RR_calf action index 8
policy observation: 删除 full RWM obs 中 RR_calf 的 pos/vel/last_action 维度
RWM imagination env 使用和向外暴露的是补零后的 full 45 维 state 和 12 维 action
```

Stage 0：用 actuator curriculum 训练能在 `RR_calf_joint=0.5` 下跟踪速度的 expert policy：

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 \
bash scripts/reinforcement_learning/go2_0p5_proprioceptive/01_train_curriculum_expert.sh
```

Stage 1：使用训练好的 0.5-strength expert policy 采集 1M mixed command coverage dataset：

```bash
EXPERT_POLICY_PATH=logs/model_based/go2_rr_calf_strength_0p5_flashsac_expert_proprioceptive/<run>/step<step> \
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 \
bash scripts/reinforcement_learning/go2_0p5_proprioceptive/02_collect_expert_dataset.sh
```

默认数据路径：

```text
logs/rwm_datasets/go2_rr_calf_strength_0p5_proprioceptive_mixed_1m/dataset.pt
```

Stage 2：已有 dataset 后，分别训练未 mask 和 full-joint-masked 两个 proprioceptive RWM：

```bash
DATASET_PATH=logs/rwm_datasets/go2_rr_calf_strength_0p5_proprioceptive_mixed_1m/dataset.pt \
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 \
bash scripts/reinforcement_learning/go2_0p5_proprioceptive/03_train_rwm_unmasked.sh
```

```bash
DATASET_PATH=logs/rwm_datasets/go2_rr_calf_strength_0p5_proprioceptive_mixed_1m/dataset.pt \
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 \
bash scripts/reinforcement_learning/go2_0p5_proprioceptive/04_train_rwm_full_joint_masked.sh
```

Stage 3：分别使用对应 RWM 作为 imagination env 训练 policy：

```bash
DATASET_PATH=logs/rwm_datasets/go2_rr_calf_strength_0p5_proprioceptive_mixed_1m/dataset.pt \
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 \
bash scripts/reinforcement_learning/go2_0p5_proprioceptive/05_train_policy_unmasked.sh
```

```bash
DATASET_PATH=logs/rwm_datasets/go2_rr_calf_strength_0p5_proprioceptive_mixed_1m/dataset.pt \
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 \
bash scripts/reinforcement_learning/go2_0p5_proprioceptive/06_train_policy_full_joint_masked.sh
```

也可以直接运行离线 RWM 训练命令。`full_joint_masked` 版本只需要额外添加：

```bash
--overrides "masked_joint_names=[RR_calf_joint]"
```

训练出的 masked RWM 日志应包含：

```text
model input: state_dim=39, action_dim=11
model output: state_dim=42
dropped_state_indices=[0, 1, 2, 17, 29, 41]
dropped_action_indices=[8]
policy_observation_mask_indices=[20, 32, 44]
```

真实 mjlab play 时仍然显式设置 0.5 电机能力：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/reinforcement_learning/rwm_flashsac/play_flashsac_go2_mjlab_proprioceptive.py \
  --checkpoint_path logs/model_based/go2_rr_calf_strength_0p5_proprioceptive_unmasked/<run>/step<step> \
  --config_path logs/model_based/go2_rr_calf_strength_0p5_proprioceptive_unmasked/<run>/step<step>/rwm_flashsac_config.yaml \
  --task Unitree-Go2-Flat-Broken-RR-Calf-Proprioceptive-Expert \  
  --num_envs 1 \
  --device cuda:0 \
  --viewer viser \
  --random_command \
  --random_lin_vel_x -0.3 0.3 \
  --random_lin_vel_y -0.15 0.15 \
  --random_ang_vel_z -0.3 0.3 \
  --joint_strength_scales RR_calf_joint=0.5
```

### 5. 仿真验证

如果想要在 MuJoCo 中查看训练效果，可以运行以下命令：

查看速度跟踪训练效果：
```bash
uv run python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/g1_velocity/2026-xx-xx_xx-xx-xx/model_xx.pt
```

查看动作模仿训练效果：
```bash
uv run python scripts/play.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --checkpoint_file=logs/rsl_rl/g1_tracking/2026-xx-xx_xx-xx-xx/model_xx.pt
```

**说明**：

- 训练时在每次保存模型时会同步导出 policy.onnx 文件在同层目录下，可用于实物部署。

**效果**：

| Go2                              | G1                             | H1_2                               | G1_mimic                          |
|----------------------------------|--------------------------------|------------------------------------|-----------------------------------|
| ![go2](doc/gif/go2-velocity.gif) | ![g1](doc/gif/g1-velocity.gif) | ![h1_2](doc/gif/h1_2-velocity.gif) | ![g1_mimic](doc/gif/g1-mimic.gif) |

### 6. 实物部署

实物部署前先确保主机安装了下列通信工具：
- [cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds.git)
- [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2.git)

<div style="margin-left: 20px;">

#### 6.1 启动机器人
将机器人在吊装状态下启动，并等待机器人进入 `零力矩模式`

#### 6.2 进入调试模式
确保机器人处于 `零力矩模式` 的情况下，按下遥控器的 `L2+R2`组合键；此时机器人会进入`调试模式`, `调试模式`下机器人关节处于阻尼状态。

#### 6.3 连接机器人
使用网线连接电脑与机器人网口，并修改网络配置如下：
- 地址：`192.168.123.222`
- 子网掩码：`255.255.255.0`

然后使用 `ifconfig` 命令查看与机器人连接的网卡名称，记录后用于启动参数。

#### 6.4 编译
以 Unitree G1 速度控制为例（其他机器人同理）。
将策略文件（`policy.onnx`）放入 `deploy/robots/g1/config/policy/velocity/<version>/exported` 下，并保证同级存在 `params/deploy.yaml`。FlashSAC 默认会导出到 `deploy/robots/g1/config/policy/velocity/v1_flashsac`，然后执行：

```bash
cd deploy/robots/g1
mkdir build && cd build
cmake .. && make
```

#### 6.5 部署

#### 6.5.1 仿真部署

在实物部署前，建议使用[unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)进行仿真部署，防止实物机器人出现异常动作。本框架已将其集成。

编译unitree_mujoco：

```bash
cd simulate
mkdir build && cd build
cmake .. && make -j8
```

启动仿真器(注意此处需连接上手柄才能启动)：

```bash
./simulate/build/unitree_mujoco
```

可在 `simulate/config` 中选择对应机器人

启动仿真控制程序：

```bash
cd deploy/robots/g1/build
./g1_ctrl --network=lo
```

#### 6.5.2 实物部署

启动实物控制程序：

```bash
cd deploy/robots/g1/build
./g1_ctrl --network=enp5s0
```

**参数说明**：
- `network`: 连接机器人网卡名称，仿真部署使用 `lo`，实物机器人如 `enp5s0`(可使用 `ifconfig` 指令查看)

</div>

**实物效果**：

| Go2                                                    | G1                                                    | H1_2                                                    | G1_mimic                                           |
|--------------------------------------------------------|-------------------------------------------------------|---------------------------------------------------------|----------------------------------------------------|
| <img src="doc/gif/go2-velocity-real.gif" width="300"/> | <img src="doc/gif/g1-velocity-real.gif" width="300"/> | <img src="doc/gif/h1_2-velocity-real.gif" width="300"/> | <img src="doc/gif/g1-mimic-real.gif" width="300"/> |


## 🎉  致谢

本仓库开发离不开以下开源项目的支持与贡献，特此感谢：

- [mjlab](https://github.com/mujocolab/mjlab.git): 构建训练与运行代码的基础。
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking.git): 用于动作跟踪的通用人形机器人控制框架。
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): 强化学习算法实现。
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git): 提供 GPU 加速渲染与仿真接口。
- [mujoco](https://github.com/google-deepmind/mujoco.git): 提供强大仿真功能。
