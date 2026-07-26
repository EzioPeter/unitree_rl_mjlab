# V13 两路线实验总体规范（主文档）

更新时间：2026-07-25

状态：V13 总体路线权威文档，同时是 Sim 侧路线与数据协议的主文档。本文冻结
Sim/Real 共同遵守的实验结构、维度、buffer 语义、公平性和验收原则，并冻结
Sim 侧从 pure-expert 切换回 mixed-data 的方向与 reference protocol。Real 侧的
数据来源、构造和专用验收流程仍以独立子文档为准，不能覆盖或替代本文。

配套文档：

```text
V13_REAL_SIDE_EXPERIMENT_DESIGN_ZH.md
V13_TRACE_HANDOFF_ZH.md
```

## 1. 总体目标

V13 从 V12 的六路线收敛为两条正式路线：

```text
路线一：RWM baseline
路线二：RWM + TRACE
```

两条路线共同遵守：

- 不使用行为克隆（BC）；
- 不加载 expert actor 初始化；
- Actor、Critic、target critic 和 temperature 按相同协议随机初始化；
- 每次 policy update 固定包含 5% realbuffer；
- 使用 `model_based` 默认 full-state、非 proprioceptive 路线；
- RWM 输入和输出都包含 `base_lin_vel`；
- RWM 必须预测并监督 `base_lin_vel`；
- Actor 和 Critic 都使用包含 `base_lin_vel[3]` 的48D full-state observation；
- 不启用 asymmetric/proprioceptive observation。

V13 当前分为两个数据侧：

```text
Sim 侧：
  负责 simulator 数据、Sim-side RWM/验证和 TRACE-selected simbuffer

Real 侧：
  负责 expert 实机数据、BLV 估算、Real-side RWM 和最终 policy 流程
```

两个数据侧共享本文定义的接口，但数据、RWM、replay、manifest 和结果必须明确命名，
不得静默混用。

### 1.1 Sim 侧 mixed-data 决定

V13 Sim 侧正式 RWM 数据不再使用 pure-expert 25K。Sim 侧回到
[`EzioPeter/unitree_rl_mjlab`](https://github.com/EzioPeter/unitree_rl_mjlab)
的 `model_based` 分支所定义的 mixed safety-command-coverage 方案，reference
commit 固定为：

```text
branch: model_based
commit: daa2eae7324255d4011964d2e74199547f347f26
```

对应的 reference 文件为：

```text
scripts/reinforcement_learning/rwm_dataset/
  run_mixed_safety_command_coverage_1m_offline_wm_sac_queue.sh
  collect_go2_expert_command_coverage_dataset.py
  slice_go2_dataset_by_env_trajectories.py
  train_world_model_offline_go2.py
scripts/reinforcement_learning/rwm_flashsac/
  train_flashsac_world_model_go2.py
  dataset.py
```

reference 流程不是“直接采25条纯 expert 轨迹”，而是：

```text
reference queue 名义上采集 1M mixed source pool
V13 为保证矩形1000步数据，每个 condition 实际采集 1024 × 1000 = 1,024,000 transitions
-> 保留 collector_types、episode_ids、timesteps、done/timeout 和 reset snapshot
-> selection=stratified, subset_seed=0
-> 完整保留25个 time-major environment columns
-> 得到25 × 1000 = 25K 的 condition-specific mixed dataset
```

这里的“25 trajectories”沿用 reference 脚本命名，准确含义是25个完整 env columns，
不保证每列只含一个 episode。发生 termination、timeout 或 autoreset 时，一列内可包含
多个 episode/fragments；RWM sequence、TRACE candidate 和 replay window 必须依据
`episode_ids/timesteps/done` 拒绝跨边界窗口。

V13 Sim 侧 mixed collector 比例冻结为 reference safety-command-coverage queue
中的比例：

```text
expert:         0.45
noisy_expert:   0.25
medium:         0.10
failure_border: 0.15
random:         0.05
```

动作噪声参数冻结为：

```text
noisy_expert action_noise_std:       0.12
medium medium_action_noise_std:      0.25
failure_border failure_action_noise_std: 0.45
random:                              Uniform(-1, 1)
最终 action:                         clamp(-1, 1)
```

reference queue 的 `MEDIUM_POLICY_PATH` 默认为空，因此 medium collector 实际回退为
“expert action + 0.25 Gaussian noise”。若 V13 要使用独立 medium checkpoint，必须
另行冻结 checkpoint/hash 并把它声明为相对 reference 的实验改动，不能静默替换。
GitHub reference 中记录的旧 expert 路径也不直接继承；V13 每个 mixed source pool
必须绑定当前实际加载的 expert checkpoint、config 和 SHA256。

reference command coverage 保持：

```text
command modes:
  stand,pure_x,pure_y,pure_yaw,xy,x_yaw,y_yaw,xy_yaw

command mode weights:
  stand:0.08,pure_x:0.25,pure_y:0.10,pure_yaw:0.08,
  xy:0.14,x_yaw:0.17,y_yaw:0.05,xy_yaw:0.13

vx abs range:  [0.05, 0.50], signed
vy abs range:  [0.03, 0.20], signed
yaw abs range: [0.05, 0.40], signed
command resample interval: [120, 300]
```

上述 weights 是每次 command resample 时使用的随机采样概率，不是 selected 25K
最终 transition occupancy 的硬配额。command 在每个环境初始化时采样，并在保持
120～300步到期或 episode reset 时重新采样。raw source pool 和 selected 25K 的
八类 command 实际计数、比例、正负方向和幅值范围必须审计并落盘，但不得用 V12
pure-expert 路线的 MILP 精确配比或“每类误差不超过3%”门槛拒绝数据。若未来要求
最终25K严格配比，属于新的实验变量，必须另立路线。

为保持与 reference queue 一致，collection 阶段：

```text
use_domain_randomization=false
use_push_randomization=false
use_observation_noise=false
```

V13 以 reference 的标准 full-state RWM/policy 为基线，并明确加入当前实验需要的
`25×1000`、五 condition 和 exact-reset snapshot 能力；不得静默切回
proprioceptive `42→45` RWM，也不得改变 Actor 48D / Critic 48D 接口。

五个 condition 仍为：

```text
g0, rr05, rr03, p5, p75
```

mixed collector 必须在每个 condition 内独立采集、独立 stratified selection、
独立冻结 SHA256，禁止从一个 condition 复制数据到另一个 condition。选择后的
实际 `collector_types` 计数和比例必须写入 manifest；不能只记录 requested mix。
`collector_type` 必须在 action 生成前保存，确保标签对应本 transition 的真实
collector；不得把 done/command resample 后的新 collector ID 回写给上一条 transition。

必须区分 requested collector probability 与最终 transition occupancy。reference
collector 只在初始化、command resample 或 episode reset 时重新抽取 collector，
不是每个 transition 独立重抽。不同 collector 在弱腿、payload 等 condition 下的
episode 存活时间不同，因此最终 1,024,000 source pool 和 selected 25K 的 actual
transition 比例允许随 condition 变化，不能用“必须接近45/25/10/15/5”的未冻结阈值
拒绝数据，也不能为了凑比例改成逐 transition 重抽。验收必须确认：

```text
requested mix 与 reference 完全一致
五类 collector 在 selected 25K 中都实际出现
collector_type 标签对应本 transition 的真实 action generator
requested/actual counts 和 ratios 全部落盘
```

如果未来要严格控制最终 transition occupancy，属于相对 reference 的新实验变量，
必须另立路线并重新冻结，不能在 V13 当前 mixed 路线中静默修改。

此前冻结的 pure-expert snapshot 25K 只保留为历史审计与 reset-pipeline 证据，
不得再用于正式 V13 Sim RWM、TRACE candidate source 或 simbuffer。正式执行前必须
重采 snapshot-enabled mixed source pool，重新生成 mixed 25K，并发布新的
Sim dataset manifest；旧 pure-expert manifest 不得继续作为正式输入。

明确撤销以下旧 manifest 的“正式 V13 Sim 输入”资格：

```text
file:   V13_SIM_DATASET_MANIFEST_PURE_EXPERT_SUPERSEDED.json
sha256: f809a40a61d488b148c75ae5fe96793ef6dc0d27ee7bd065da32e6eb23e1e60f
status: SUPERSEDED_DO_NOT_TRAIN
```

该文件及其中五组 pure-expert dataset hash 可以保留用于来源追溯，但任何 launcher、
RWM trainer 或 TRACE pipeline 若仍将该 hash 当成 active manifest，必须 fail closed。

当前 full-state active manifest 冻结为：

```text
file:   V13_SIM_DATASET_MANIFEST.json
schema: go2_v13_frozen_sim_mixed_fullstate_dataset_manifest_v3
sha256: f507931ef4106d6c64952ddc25f4280d00c17330d8396bf30a788b6195bb28ff
status: active_sim_dataset_input
```

上一版 mixed/proprioceptive 接口 manifest 已保存在
`V13_SIM_DATASET_MANIFEST_PROPRIOCEPTIVE_SUPERSEDED.json`，SHA256 为
`34a01e6c70ee43b5b3d658b76d580c9255d89d559e1d2d60c5dae9c2d8c99f2c`，
不得作为新训练入口。

## 2. 冻结维度

必须严格区分 RWM dynamics state 和 policy observation。

### 2.1 RWM dynamics

```text
RWM full state:       45D
RWM input:            45D
RWM output:           45D
Action:               12D
```

RWM full state 顺序：

```text
base_lin_vel[3]
base_ang_vel[3]
projected_gravity[3]
joint_pos_rel[12]
joint_vel[12]
actuator_force[12]
```

RWM 输入和输出都保留 BLV：

```text
RWM input  = state[0:45]   # 45D
RWM output = next_state    # 45D
```

### 2.2 Policy observation

```text
Actor observation:   48D
Critic observation:  48D
Command:              3D
Action:              12D
```

Actor/Critic observation 共同顺序：

```text
base_lin_vel[3]
base_ang_vel[3]
projected_gravity[3]
command[3]
joint_pos_rel[12]
joint_vel[12]
last_action[12]
```

Actor和Critic逐字段、逐顺序完全一致，不存在Critic独有的privileged BLV。

统一配置断言：

```text
input_state_dim=45
output_state_dim=45
actor_observation_dim=48
critic_observation_dim=48
command_dim=3
action_dim=12
```

V13 的正式入口固定为：

```text
train_world_model_offline_go2_v13_fullstate.py
run_v13_rwm45to45_fullstate.sh
configs/go2_offline_world_model_v13_full45.yaml
```

该薄入口复用 reference 标准 `train_world_model_offline_go2.py` 的45→45 dynamics，
只负责从 `state + command + prev_action` 重建共同的48D Actor/Critic observation
统计量；不得直接用数据文件中为旧 expert collector 保存的统计量充当新policy
normalizer。

禁止调用 `train_world_model_offline_go2_proprioceptive.py` 或在 agent 内执行
旧的42D proprioceptive路线。Actor不得再使用`observation[..., 3:48]`投影，
必须消费完整48D observation。

## 3. BLV 的共同语义

`base_lin_vel` 必须满足：

```text
进入 RWM current-state input
进入 RWM 45D output target
进入 Actor observation
进入 Critic observation
参与需要 BLV 的 reward 计算
```

正式 RWM 配置必须满足：

```text
input_dropped_state_indices:  []
output_dropped_state_indices: []
state_loss_ignored_indices:   []
```

RWM state loss：

\[
L_{\mathrm{state}}
=
\sum_{i=0}^{44}w_i
\ell(\hat{s}_{t+1,i},s_{t+1,i}),
\qquad
w_0,w_1,w_2>0
\]

必须单独记录 BLV loss、非 BLV state loss 和 total state loss。

不同数据侧的 BLV 来源不同：

```text
Sim dataset / simbuffer:
  simulator 真实 BLV

Real dataset / realbuffer:
  经冻结 estimator 产生并通过 Real 侧验收的 estimated BLV

RWMbuffer:
  frozen RWM 预测的 BLV
```

Real 侧 estimated BLV 必须明确标记为估算标签或伪监督，禁止声称为传感器真实值。

## 4. 三种 buffer

### 4.1 realbuffer

realbuffer 来自当前实验绑定的 dataset，参与 SAC update，是动力学锚点。Sim 侧
正式 baseline 的 realbuffer 必须来自对应 condition 的 frozen mixed 25K，不能
把其中非 expert collector 丢弃，也不能换回 superseded pure-expert 25K。

realbuffer 不是 BC：

- 不加载 expert actor；
- 不执行 expert-action supervised loss；
- 不恢复 BC optimizer；
- `actor_bc_alpha=0`。

realbuffer 必须与当前 dataset、BLV 来源、reward、reward normalization设置、
`gamma` 和 `n_step`
通过 manifest/hash 绑定。

### 4.2 RWMbuffer

RWMbuffer 由当前 policy 与对应数据侧的 frozen RWM 交互产生。RWM 必须由当前
实验绑定的 dataset 训练，不能跨数据侧或跨 dataset 静默复用。

### 4.3 simbuffer

simbuffer 由 Sim 侧生成，并经 TRACE rule 或 learned scorer 筛选。TRACE score
只用于选择 trajectory，不得写入 SAC transition reward。

Real 侧消费 simbuffer 前必须验证其 source commit、配置、reward、observation
顺序、action 顺序、`gamma/n_step`、selection manifest 和 SHA256。

## 5. 两条正式路线

### 5.1 路线一：RWM baseline

```text
绑定的 dataset（Sim 侧为 mixed 25K；Real 侧按 Real 子文档）
-> 训练 RWM（45D 输入、45D 输出、BLV 参与输入和监督）
-> 冻结 RWM
-> 随机初始化 Actor(48D)、Critic(48D)、temperature
-> 生成 RWMbuffer
-> update = 5% realbuffer + 95% RWMbuffer
-> final policy
```

正式比例：

```text
realbuffer = 5%
RWMbuffer  = 95%
simbuffer  = 0%
```

### 5.2 路线二：RWM + TRACE

路线二与路线一共享相同 dataset、BLV labels、RWM、初始化、reward 和训练预算，
只在 95% synthetic 部分加入 TRACE-selected simbuffer：

```text
update = 5% realbuffer
       + 95% × (RWMbuffer + TRACE-selected simbuffer)
```

定义 \(\rho_{\mathrm{sim}}\) 为 synthetic 内部的 sim 比例：

\[
N_{\mathrm{real}}=\lfloor0.05N\rfloor
\]

\[
N_{\mathrm{synthetic}}=N-N_{\mathrm{real}}
\]

\[
N_{\mathrm{sim}}
=
\lfloor\rho_{\mathrm{sim}}N_{\mathrm{synthetic}}\rfloor
\]

\[
N_{\mathrm{RWM}}
=
N_{\mathrm{synthetic}}-N_{\mathrm{sim}}
\]

必须始终满足：

```text
realbuffer = 5%
RWMbuffer + simbuffer = 95%
```

`\rho_sim` 未冻结前不得启动正式实验。若进行 r10/r25 消融，名称必须明确表示它是
synthetic 内部比例：

```text
r10 = 5% real + 85.5% RWM + 9.5% sim
r25 = 5% real + 71.25% RWM + 23.75% sim
```

## 6. Batch mixer

每次 update 必须分别采样、校验、拼接并全局打乱：

```text
路线一：
  real + RWM -> concatenate -> shuffle -> SAC update

路线二：
  real + RWM + sim -> concatenate -> shuffle -> SAC update
```

每个 batch 必须记录：

```text
requested_real/rwm/sim_count
actual_real/rwm/sim_count
actual_real/rwm/sim_ratio
trace_ratio_within_synthetic
```

禁止先采完整 RWM batch，再静默替换固定前缀。

## 7. Observation 构造

三种 buffer 都保存同一顺序的48D full observation：

```text
realbuffer:
  对应 expert dataset 的 BLV + 其余 full observation 字段

RWMbuffer:
  RWM 预测 BLV + 由 RWM state、command、last action 构造的 full observation

simbuffer:
  simulator BLV + simulator full observation
```

必须断言：

```text
critic_observation.shape[-1] == 48
actor_observation.shape[-1] == 48
actor_observation 与 critic_observation 字段顺序一致
```

禁止丢弃前三维 BLV；同时禁止直接把45D RWM state 当作48D policy observation，
因为 policy observation 还包含 command 和 last action，且不包含 actuator force。

## 8. Reward 与 n-step

三种 buffer 必须共享同一公共 reward 定义。线速度 reward 的 BLV 必须来自对应
transition source：

```text
realbuffer: 当前绑定 dataset 的 BLV label（Sim 为 simulator BLV）
RWMbuffer:  RWM predicted BLV
simbuffer:  simulator BLV
```

Sim侧路线一正式policy目标配置冻结为：

```text
seed:                         0
nominal policy env-step budget: 50,000,000
actual env steps:             49,999,872（48,828 × 1,024）
num imagination envs:         1,024
updates per interaction:      2
batch size:                   2048
gamma:                        0.99
n_step:                       3
replay capacity/device:       10,000,000 / CPU
learning starts:              100,000 transitions
reward version:               model_based reference v1
reward normalization:         enabled + frozen condition-specific normalizer
normalize_reward:             true
normalized_G_max:             5.0
uncertainty penalty weight:   -2.0
critic support:               [-5, 5]
actor_bc_alpha:               0
Actor/Critic observation:     48D / 48D
real rows per batch:          102
RWM rows per batch:           1946
sim rows per batch:           0
command duration:             120～300 steps
policy imagination ranges:    vx ±0.5, vy ±0.2, yaw ±0.4
save replay buffer:           true
```

此前生成的
`go2_v13_fullstate_rwm_mixed1m_policycfg_50m.yaml`使用
`vx±0.3、vy±0.15、yaw±0.3`，只能作为small-command-range消融，不能作为本节
正式0.5路线配置。正式重训必须新建wide-command配置、runner、queue和输出目录，
完成SHA256及完整diff冻结后再启动；禁止覆盖或修改已落盘的小范围实验配置。

上述 `102 + 1946 = 2048` 是当前 mixer 对 5%/95% 的确定性整数分配。路线二必须
保持同一 `gamma`、`n_step`、reward config、batch size 和总预算，
只在1946条 synthetic rows 内按冻结的 `rho_sim` 拆分 RWM/sim。
任何 buffer 都不得跨 episode、segment、reset、candidate 或无效数据边界拼接。

两条路线都必须开启相同的reward normalization，并使用从对应condition的冻结
mixed 25K、公共v1 reward、`gamma=0.99`、`n_step=3`构建的condition-specific
frozen normalizer；训练期间禁止继续更新normalizer统计量。不得因TRACE注入改变
reward定义、normalizer或reward scale。

## 9. Sim/Real 责任边界

### 9.1 Sim 侧

Sim 侧独立负责：

```text
每个 condition 的 mixed 1,024,000-transition source pool
stratified 25-trajectory / 25K selection
collector mix requested/actual 计数
snapshot-enabled simulator dataset
真实 simulator BLV
TRACE candidate generation
TRACE selection
simbuffer materialization
simbuffer artifact/manifest/hash
```

### 9.2 Real 侧

Real 侧独立负责：

```text
expert 实机原始日志
连续片段和安全边界
Real 25K 筛选
estimated BLV
Real RWM dataset
Real-side RWM
realbuffer
最终两路线 policy
```

Real 侧的具体实现与验收以
`V13_REAL_SIDE_EXPERIMENT_DESIGN_ZH.md` 为准。

## 10. 公平性约束

同一配对实验的两条路线必须共享：

```text
dataset/hash
BLV labels 和 estimator/hash
RWM checkpoint/hash
realbuffer
reward config
condition-specific frozen reward normalizer/hash
Actor/Critic/temperature 初始化 seed
optimizer 与学习率
batch size
gamma/n_step
gradient-update budget
评测协议
```

唯一主变量是 95% synthetic 中是否加入冻结比例的 TRACE-selected simbuffer。

## 11. 禁用路线

V13 正式实验禁止：

```text
BC-only
TRACE-only
BC+RWM
BC+TRACE
BC+RWM+TRACE
0% real + 100% RWM
0% real + 100% sim
```

launcher 必须 fail closed 拒绝 BC loss、expert actor initialization 和
0% realbuffer。

## 12. 强制日志

每个 RWM run 必须记录：

```text
version=V13
data_side=sim|real
dataset path/hash
Sim collector mix requested/actual（仅 Sim）
Sim source-pool/selection seed/hash（仅 Sim）
BLV source/estimator/hash
input_state_dim=45
output_state_dim=45
state_loss_ignored_indices=[]
BLV loss
non-BLV state loss
total state loss
```

每个 policy run 必须记录：

```text
route_name
data_side
actor_observation_dim=48
critic_observation_dim=48
BC enabled=false
expert actor loaded=false
real/RWM/sim counts and ratios
dataset/RWM/replay hashes
reward config hash / frozen reward normalizer hash
gradient update count
```

## 13. 强制验收

### 13.1 RWM

- 输入最后一维为 45；
- 输出和 target 最后一维为 45；
- BLV 三维参与 loss 且权重非零；
- BLV loss 能反向传播；
- checkpoint 不含 input/output drop 或 `[0,1,2]` loss ignore。
- Sim RWM dataset 是 condition-specific mixed 25K，不是 pure-expert 25K；
- Sim manifest 绑定 1M source pool、stratified seed、选中轨迹和实际 collector mix。

Sim mixed dataset 的 snapshot/reset 还必须满足：

```text
exact reset reconstruction: 比较 state[0:33]
state[33:45] actuator_force: dynamics-derived，不作为可恢复坐标
reset reconstruction max error <= 0.005
同一 snapshot + 同一 action 的重复一步 max error <= 0.005
当 source 与 rollout dynamics/action interface 相同时：
  sim(snapshot, source_action) 与 source next_state 的 max error <= 0.005
```

`0.005` 沿用已验证的 V12 exact-snapshot 验收门槛，不得在正式运行时静默修改。
V13 在 RTX 3090 上完成的 mixed g0 小规模验证结果为：

```text
reset reconstruction max error: 2.13368935e-07
repeated one-step max error:     2.67028809e-04
source transition max error:     3.73840332e-04
历史45D proprioceptive actor observation rebuild error: 0（仅历史诊断）
```

该smoke只证明mixed snapshot/reset数据链可工作，不是正式训练输入，也不能替代
五个condition的正式数据审计。该历史smoke的45D Actor接口已被废弃，不能代表
V13当前正式policy接口；正式运行必须由每个condition的preflight独立检查
Actor48/Critic48接口。

### 13.2 Policy observation

- Actor和Critic对所有来源均收到48D observation；
- 前三维BLV来自对应transition source；
- Actor/Critic字段与顺序必须完全一致。

### 13.3 Replay

- 路线一为 5% real、95% RWM；
- 路线二为 5% real，RWM+sim 合计 95%；
- sim 只来自冻结 TRACE selection artifact；
- 三种来源 reward、n-step、shape、dtype 和 device 对齐；
- concatenate 后统一 shuffle。

### 13.4 No-BC

- 不加载 expert/BC actor；
- `actor_bc_alpha=0`；
- 无 supervised action loss；
- 无 BC optimizer；
- route matrix 只有两路。

## 14. 执行顺序

```text
1. 按第1.1节重采并冻结 snapshot-enabled Sim mixed dataset；Real 侧按子文档冻结数据
2. 审计并训练标准 full-state 45→45 RWM
3. 冻结 RWM
4. 构建并验收 realbuffer
5. 验收 RWMbuffer
6. 接收并验收 TRACE simbuffer
7. 冻结 synthetic 内 sim 比例
8. 两路线 smoke
9. 配对 pilot
10. 配置 diff 无非预期变化后启动正式实验
11. 统一评测
```

任何关键配置缺少直接证据时必须停止，禁止根据历史目录名或旧实验默认值继续。

## 15. 2026-07-25 small-command-range批次审计记录

本批次复用了已经完成的五个RWM，没有重训RWM，但policy imagination范围误用了
`vx±0.3、vy±0.15、yaw±0.3`，不符合第8节冻结的正式0.5路线。整批policy已标记：

```text
INVALID_FOR_V13_FORMAL_WIDE_COMMAND_ROUTE
```

已完成的g0、rr05、p75只能作为small-command-range消融；尚未完成的rr03、p5在
确认范围错误后停止。任何该批policy都不得进入正式V13路线汇总。五个RWM checkpoint
仍然有效，不受此标记影响。

该批次的追溯信息为：

```text
RWM experiment:
  go2_rwm_v13_full45_25k_fresh_seed0_20260725
Policy experiment:
  go2_v13_fullstate_rwm_mixed1m_policycfg_50m_fresh_seed0_20260725
RWM queue:
  logs/queues/v13_rwm_full45_fresh_seed0_20260725
Policy queue:
  logs/queues/v13_policy_fullstate_mixed1m_policycfg_fresh_seed0_20260725
Policy config:
  scripts/reinforcement_learning/rwm_flashsac/configs/
    go2_v13_fullstate_rwm_mixed1m_policycfg_50m.yaml
Policy config SHA256:
  1d07d3845408823600d0a5e6c2d852f2839d11f8f601457dc72aa9f85a70aaf6
Policy runner:
  scripts/reinforcement_learning/rwm_flashsac/
    run_v13_fullstate_rwm_mixed1m_policycfg.sh
Policy runner SHA256:
  991e3fc22b73d4a894de1ad8eedd8c9ef276967a33c04d3b97bba16050348457
Invalid marker:
  logs/experiments/
    go2_v13_fullstate_rwm_mixed1m_policycfg_50m_fresh_seed0_20260725/
      INVALID_DO_NOT_USE_FOR_V13_FORMAL_ROUTE.txt
```

该批次调度为：

```text
GPU0: g0 -> rr03
GPU5: rr05 -> p5
GPU6: p75
GPU2/GPU7: 禁用
```

该批次启动前的120-interaction smoke使用1024 imagination env，并保留10M CPU
replay和100K learning start。`step120` checkpoint记录`update_step=42`，恰好等于
越过100K后的21次interaction乘以每次2次update，证明learning start和update频率
在该小范围消融中实际生效。`step1000` checkpoint记录`update_step=1802`。
batch实际记录为：

```text
real: 102
RWM:  1946
sim:  0
frozen reward normalizer: true
```

该批次每个condition独立构建并审计了：

```text
n_step=3 real replay
condition-specific frozen reward normalizer
policy preflight（dataset/RWM/config/interface/hash）
```

训练中若共享`.venv`软链接失效，已运行进程不得据此宣称后续队列可启动；必须重新
绑定到存在且通过核心导入和三组回归测试的固定环境。本批次恢复使用：

```text
/data1/xjy/venvs/unitree_rl_mjlab_model_based_daa2eae_20260725
Python 3.11.15
Torch 2.9.1+cu128
OmegaConf 2.3.0
```

环境故障发生前完成的checkpoint保持有效；尚未创建输出的condition可在修复环境后
从头启动。故障原因、旧status和PID必须归档，不得伪装成无故障连续队列。

### 15.1 正式wide-command policy重训记录

2026-07-25已在不重训RWM的前提下，按第8节正式范围新建独立配置、runner、queue
和输出目录。相对第15节small-command-range配置的完整递归diff只有实验命名和
以下六个上下界，未出现其他配置变化：

```text
vx:  [-0.3, 0.3]   -> [-0.5, 0.5]
vy:  [-0.15, 0.15] -> [-0.2, 0.2]
yaw: [-0.3, 0.3]   -> [-0.4, 0.4]
```

正式artifact为：

```text
Policy experiment:
  go2_v13_fullstate_rwm_mixed1m_widecmd_policycfg_50m_fresh_seed0_20260725
Policy queue:
  logs/queues/
    v13_policy_fullstate_mixed1m_widecmd_policycfg_fresh_seed0_20260725
Policy config:
  scripts/reinforcement_learning/rwm_flashsac/configs/
    go2_v13_fullstate_rwm_mixed1m_widecmd_policycfg_50m.yaml
Policy config SHA256:
  6884a9dd6b3f84ca61f0822062b42fdceb37f00085c4b179da89551f667685e0
Policy runner:
  scripts/reinforcement_learning/rwm_flashsac/
    run_v13_fullstate_rwm_mixed1m_widecmd_policycfg.sh
Policy runner SHA256:
  8359468bcb6385946ca1b09c185d971750a1e46b81c2e5a2f38046d6d446f11b
Queue launcher:
  scripts/reinforcement_learning/rwm_flashsac/
    launch_v13_fullstate_mixed1m_widecmd_policycfg_queue.sh
Queue launcher SHA256:
  41920e803c0c882f84bd93f9635ec3d8c08241d1a7086419f6d5d9de94466a11
Launch time:
  2026-07-25T16:51:40+08:00
```

启动前重新核验了五个mixed 25K dataset的manifest SHA256、五个唯一RWM
`model_5000.pt`及其condition-specific dataset绑定、45D state、12D action和
Actor48/Critic48接口。代码lint和三组独立回归测试通过。wide-command smoke使用
1024 imagination env运行120 interactions，实际得到：

```text
env steps:     122,880
update_step:   42
real/RWM/sim:  102 / 1946 / 0 per batch
```

smoke实际落盘配置确认指令范围为`vx±0.5、vy±0.2、yaw±0.4`，并确认
`2 updates/interaction、n_step=3、10M CPU replay、100K learning start、
frozen reward normalization、normalized_G_max=5、uncertainty=-2、
critic=[-5,5]`。正式队列调度为：

```text
GPU0: g0 -> rr03
GPU5: rr05 -> p5
GPU6: p75
GPU2/GPU7: 禁用
```

启动后的首批g0、rr05、p75进程参数、实际解析配置和replay manifest再次核验，
三者均使用正式wide-command config、对应condition的RWM与mixed 25K dataset，
batch均为`102 real + 1946 RWM + 0 sim`。三者的`step1000` checkpoint均记录
`update_step=1802`，与100K learning start后每次interaction更新2次完全一致；
同时未发现failure marker、Traceback、ERROR或NaN/Inf。

2026-07-25T17:03+08:00再次检查允许使用的GPU后，GPU0和GPU6分别只有约5.6GiB
显存占用，明显低于GPU1/3/4/5；GPU2和GPU7仍保持禁用。为提前启动等待中的两个
condition，原GPU0和GPU5串行launcher父进程在确认g0/rr05的runner与trainer子进程
仍存活后解除，防止前序完成后重复启动；现有g0和rr05训练未停止。随后独立启动：

```text
GPU0: rr03（与g0并行）
GPU6: p5（与p75并行）
```

重排记录及SHA256为：

```text
logs/queues/v13_policy_fullstate_mixed1m_widecmd_policycfg_fresh_seed0_20260725/
  reassignment_20260725_1703.json
aa0e5e5435232c0739d4a35c1de7cc3f6470b73259f18a67cfd28097dd824a30
```

rr03和p5均已通过各自preflight并真正进入GPU更新；两者的`step20000`均记录
`update_step=39802`，实际解析配置、dataset/RWM绑定、wide-command范围和
`102 real + 1946 RWM + 0 sim`再次核验通过，日志未发现Traceback、ERROR或
NaN/Inf。

## 16. MJLab可视化与评测物理条件冻结

condition到MJLab物理条件的唯一映射为：

```text
g0:   payload_mass_kg=0.0, rr_calf_strength=1.0
rr05: payload_mass_kg=0.0, rr_calf_strength=0.5
rr03: payload_mass_kg=0.0, rr_calf_strength=0.3
p5:   payload_mass_kg=5.0, rr_calf_strength=1.0
p75:  payload_mass_kg=7.5, rr_calf_strength=1.0
```

MJLab play必须同时满足：

```text
task: Unitree-Go2-Flat-Normal-FixStand-RWM-Pretrain-Ens
clean: true
num_envs: 1
manual command initial value: [0, 0, 0]
manual vx range:  [-0.5, 0.5]
manual vy range:  [-0.2, 0.2]
manual yaw range: [-0.4, 0.4]
```

不得只解析或打印`payload_mass_kg`/`rr_calf_strength`参数而不修改`env_cfg`。
payload condition的play必须创建可见的`fixed_payload_brick`刚体，并在env创建后
读取运行时body mass，确认与condition完全一致；p75日志必须包含：

```text
payload_mass_kg=7.5
realized_visible_payload_mass_kg=7.5
```

rr condition必须把`RR_calf_joint`的stiffness、damping和effort limit按condition
比例缩放；rr05日志必须包含：

```text
joint_strength_scales={'RR_calf_joint': 0.5}
rr_calf_strength=0.5
```

修正后的fullstate play脚本为：

```text
scripts/reinforcement_learning/rwm_flashsac/play_flashsac_go2_mjlab_fullstate.py
SHA256: f611cde50104e16da3935d2cdca3a71ddf82c1239af87af979a297486162d944
```

任何未真实应用condition物理参数的viewer必须停止并写入
`INVALID_DO_NOT_USE.txt`；可视化行为不能代替统一MJLab评测指标。
