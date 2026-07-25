# V13 TRACE 模块交接与移植规范

更新时间：2026-07-24

状态：交接规范；实施与实验语义必须服从同目录下
`V13_TWO_ROUTE_EXPERIMENT_DESIGN_ZH.md`。

V13 已拆分为 Sim 侧与 Real 侧。TRACE candidate generation、selection 和
simbuffer 生产由 Sim 侧负责，最终训练侧按冻结 artifact contract 验收和消费
TRACE-selected simbuffer。本文是总体 TRACE 交接规范，不属于任一数据侧的
专用路线文档；两侧责任边界以主文档为准。

同目录下的旧 manifest：

```text
V13_SIM_DATASET_MANIFEST.json
SHA256: f809a40a61d488b148c75ae5fe96793ef6dc0d27ee7bd065da32e6eb23e1e60f
status: SUPERSEDED_DO_NOT_TRAIN
```

只记录历史 pure-expert snapshot 25K，不再是正式 V13 Sim RWM、TRACE candidate
或 simbuffer 输入。V13 当前必须按主文档重采五个 condition 的 snapshot-enabled
mixed source pool，生成 condition-specific mixed 25K，并发布新的 active Sim
dataset manifest。在新 manifest 的绝对路径、condition、实际 collector counts、
snapshot contract 和 SHA256 冻结前，TRACE 正式 candidate generation 必须停止。

## 1. 文档目的与优先级

本文用于把已经解耦、在历史实验中做过验证的 TRACE core 与 TRACE scorer
交给 V13 实施方，并说明如何接到当前 V13 两路线 baseline。历史验证
不等于在当前 V13 baseline 上已验证有效；V13 有效性必须按本文第 6、7、14 节
重新建立。

规范优先级如下：

1. `V13_TWO_ROUTE_EXPERIMENT_DESIGN_ZH.md`：当前实验路线、比例、初始化、
   reward、评测和暂停点的唯一权威来源；
2. 本文：TRACE 移植边界、实施顺序和验收清单；
3. 两个 clean 分支中的 README/adapter 文档：模块 API 和软件契约；
4. 历史 V10/V11/V11.1/V12/V5 脚本：只能作为实现参考，不能覆盖当前 V13 语义。

特别注意：clean 分支中的旧 `V12_ADAPTATION.md` 仍保留早期
from-zero/no-BC baseline 描述。可以复用其模块代码、summary contract、
source-leakage audit 和验证方法，但不能把旧实验初始化、旧 replay 比例或旧
baseline 名称带入当前两路线。

## 2. 已解耦模块的准确来源

GitHub 仓库：

```text
https://github.com/jaoyee/unitree_rl_mjlab
```

### 2.1 TRACE core

```text
branch: zkq/trace-core-clean
commit: de722be372a610f08d004e65b3b4f05db2052b6a
path:   standalone/trace_core
```

核心内容：

```text
source row selection
-> multi-branch candidate collection
-> trajectory summary
-> eligibility gates
-> rule/scorer backend
-> command-distribution selection
-> immutable selection manifest
-> replay materialization adapter
-> downstream trainer adapter
```

优先阅读：

```text
standalone/trace_core/README.md
standalone/trace_core/ADAPTER_CONTRACT.md
standalone/trace_core/V12_ADAPTATION.md
standalone/trace_core/VALIDATION.md
standalone/trace_core/v12_current_baseline_adapter.template.json
```

### 2.2 TRACE scorer

```text
branch: zkq/trace-scorer-clean
commit: e9235f5143a12f7b31c8b447afee4077007da360
path:   standalone/trace_scorer
```

scorer 模块只接受 trajectory-level summary 并输出 scalar score。它不导入或
修改：

```text
RWM
simulator
actor/critic
transition reward
replay buffer
trainer
```

优先阅读：

```text
standalone/trace_scorer/README.md
standalone/trace_scorer/INTEGRATION.md
standalone/trace_scorer/V12_SUMMARY_ADAPTER.md
standalone/trace_scorer/checkpoints/scorer.pt
standalone/trace_scorer/checkpoints/portability_audit.json
```

可选离线交接包：

```text
exports/trace_clean_modules_20260723.zip
exports/trace_scorer_clean_20260723.zip
```

### 2.3 获取方式

```bash
git fetch origin
git worktree add ../trace-core-clean origin/zkq/trace-core-clean
git worktree add ../trace-scorer-clean origin/zkq/trace-scorer-clean
```

禁止把两个 clean 分支整体 merge 到 V13 baseline 后再删除冲突。正确方式是保留
模块边界，用 adapter 对接 V13 的 collector、reward、replay 和 trainer。

## 3. V13 TRACE 生产与交付协议

### 3.1 Candidate generation

主实验固定为：

```text
source dataset:                 新 active Sim manifest 中对应 condition 的 mixed 25K
required collector mix:         expert/noisy_expert/medium/failure_border/random
simulator:                      clean nominal simulator
action temperature T:          1
dataset start states:          1024
branches per start state:      4
rollout horizon H:             20
candidate trajectories:        4096
candidate one-step maximum:    81,920
trajectory select ratio:       0.25
simbuffer artifact/hash:        必须冻结
nominal mixed replay ratio:     最终训练正式运行前冻结
```

每个 source snapshot 的四个 branch 必须：

- 从同一个完整、已实现的 simulator snapshot 开始；
- 具有相同 state、command、previous action 和 simulator parameters；
- 只因 policy 在 `T=1` 下的随机采样而产生不同 action；
- automatic reset 关闭；
- terminal 后停止写入有效 transition，禁止把 reset 后状态拼回同一轨迹。

旧 pure-expert 五份25K保存了 `go2_trace_snapshot_v1` 字段，只能用于历史追溯和
reset-pipeline审计，禁止作为正式 V13 TRACE source。新 mixed dataset 必须保留
等价的完整 snapshot 字段，并执行真实 simulator restore smoke；普通采集产生的
`trace_reset_reconstruction_errors=NaN` 不能作为通过证据。生产侧必须在新 active
manifest 中提供 restore 证据，消费侧不得根据目录名推定已经通过。

历史 V12 nominal simulator 协议为：

```text
payload mass:             0.0 kg
RR calf strength:         1.0
domain randomization:     off
push randomization:       off
observation noise:        off
```

这段只用于解释历史 artifact。V13 的 source condition、snapshot 和 simulator
gap 必须由 Sim 侧权威配置明确记录；任何消费方都不得根据历史 V12 condition
名称自行推断或重新施加 gap。

### 3.2 五组独立候选

首次正式验证必须重采五组独立 candidate pools：

```text
candidate_group_index = 0, 1, 2, 3, 4
```

五组使用相同协议和不同 source-selection/rollout seed。每组 candidate pool
生成后必须不可变，并由 random、rule、scorer 三个 selection arm 共同复用。
禁止为不同 selector 重新采样候选，否则 selector 效果与候选随机性无法分离。

### 3.3 Candidate policy

候选动作来自对应路线的当前 policy actor：

- 只加载 `actor.pt`；
- 推理加载不得恢复 critic、temperature、optimizer、RWMbuffer 或 replay-mixer
  状态；
- 不得用 expert/E0 actor 替代当前 policy；
- 参数名即使暂时保留 `expert_policy_path`，传入内容也必须是当前 policy
  checkpoint，并应尽快改成中性的 `rollout_actor_checkpoint`。

## 4. TRACE core 与 scorer 的严格边界

### 4.1 TRACE core 负责

```text
snapshot/source selection
candidate collection orchestration
behavioral summary schema
hard eligibility
rule selection
command-distribution matching
selection manifest
replay/trainer adapter orchestration
```

### 4.2 TRACE scorer 只负责

```text
behavioral summary JSONL
-> schema/missingness validation
-> feature normalization
-> learned scalar score
-> immutable score manifest
```

scorer score：

- 不是 transition reward；
- 不得写入 replay reward；
- 不决定 actor/critic 初始化；
- 不决定三缓冲区比例；
- 不读取 RWM uncertainty、critic Q、world-model loss 或 expert label；
- 默认拒绝 reward/return shortcut features。

### 4.3 V13 adapter 负责

V13 项目侧保留以下具体实现：

```text
simulator snapshot restore
actor checkpoint inference
candidate tensor collection
Go2 trajectory summary extraction
public reward recomputation
n-step replay materialization
three-buffer mixer
checkpoint/resume
MJLab evaluation
```

## 5. Summary contract：先响应，再评价运动中稳定性

### 5.1 Stand command

当三维 command 均为零时，目标是静止稳定：

- 小 base linear/yaw velocity；
- 小 tilt 和 roll/pitch body-rate；
- 小 action delta；
- 无 saturation、illegal contact、terminal 或 nonfinite。

### 5.2 Nonzero command

当任一 command 分量非零时，稳定性不是“越不动越稳定”。必须按以下顺序：

1. 判断每个激活 command 分量是否产生正确方向响应；
2. 判断 full window 和 steady window 的实现率；
3. 判断 displacement/transport 是否与指令方向一致；
4. 响应 gate 通过后，才评价运动中的 tilt、body-rate、action smoothness、
   contact 和 survival。

静止轨迹即使 tilt 小、action 平滑，也不能通过 moving-command eligibility。

### 5.3 Combined command

对 `xy`、`x_yaw`、`y_yaw`、`xy_yaw` 等组合指令，必须保留逐分量指标，并至少提供：

```text
minimum_active_axis_response_ratio
minimum_active_axis_transport_ratio
```

排序使用最弱激活分量，禁止用“前进完成得好”掩盖横移或转向完全没有响应。

### 5.4 必需字段类别

summary 至少包含：

- stable `candidate_index`、`start_state_id`、candidate group；
- command `[x, y, yaw]`、active masks 和八种 command mode；
- full/steady window 的 linear/yaw realization ratio；
- direction correctness 和 signed displacement/transport；
- linear/yaw tracking error；
- survival、terminal、finite 和 reset reconstruction；
- tilt、roll/pitch-rate RMS；
- action saturation、action delta；
- 足端/contact/gait stability（若 collector 可可靠提供）。

禁止将下列内容作为 learned scorer 的输入：

```text
simulator_return
reward_mean/std/min/trend
RWM reward or uncertainty
critic Q/value
world-model loss
expert collector label
最终 MJLab 评测结果
```

这些量可以保留在独立诊断文件中，但不得进入 scorer feature schema。

## 6. 第一阶段：规则式筛选验证

必须先验证 scorer-independent TRACE core，不得直接以历史 learned scorer
代替这一阶段。

这里的 rule arm 是 scorer 上线前的资格验证和诊断对照，不是第七条正式训练路线，
也不改变权威路线文档中最终各路线统一使用同一 scorer/selection protocol 的要求。
主验收是注入 rule-selected simbuffer 后的最终 policy 是否优于对应的无 TRACE
父基线；matched random 是辅助归因对照，不是 rule 的强制晋级门槛。只有 rule
注入相对父基线有效后，才进入 learned scorer 验证。

### 6.1 Hard eligibility

至少 fail closed 检查：

- snapshot/reset reconstruction；
- finite state/action；
- terminal/valid boundary；
- action saturation/safety；
- moving command 的每个 active component 均有正确响应；
- stand command 使用单独静止稳定 gate。

### 6.2 Rule score

在 eligible candidates 内：

```text
weakest-component tracking/response: 70%
stability while actually moving:      30%
```

70/30 是 eligibility 之后的连续排序。这里的 stability 特指运动中的稳定性：
非零 command 必须先产生正确方向的真实运动，再评价姿态、base height、存活、
动作和步态平滑性。不得让站着不动、原地扭动或四足贴地因为姿态波动较小获得高分，
也不得用 stability 抵消“完全没运动”或 active command component 没有响应。

### 6.3 公平对照

每个 immutable candidate pool 同时构建：

1. same-size、command-distribution-matched random selection；
2. motion-first rule selection。

两者必须具有相同：

```text
candidate pool
selected trajectory/transition count
command-mode/sign/magnitude distribution
replay ratio
reward materializer
initial checkpoint
optimizer state
seed
gradient update budget
evaluation protocol
```

规则式筛选的主对照是对应的无 TRACE 父基线。保持公共 reward、buffer 比例、
初始化、optimizer state、seed、gradient update budget 和评测协议不变后，
rule-selected simbuffer 的注入必须使最终 policy 的统一闭环综合结果优于父基线；
velocity tracking、正确方向净运动和运动中稳定性均不得出现实质性退化，也不能
出现隐藏的 command-mode 大幅回归，才能进入 learned scorer 验证。

matched random 只用于进一步判断增益来自“simulator 数据注入”还是“规则选择”
本身。rule 不必强制超过 random；若未超过，只能收窄可宣称的结论。

## 7. 第二阶段：learned scorer 验证

规则式筛选通过后，在同一批 immutable summaries 上接入
`zkq/trace-scorer-clean`。

加载 scorer 时必须 fail closed 验证：

- checkpoint format 和 feature schema；
- normalization width 和有限正标准差；
- strict state-dict loading；
- required command/response/survival/posture/action-safety context；
- 每行缺失比例不超过模块阈值；
- 不含 reward/return shortcut。

历史 V11.1 seed-44 checkpoint 的 portability audit 只证明软件等价与可移植，
不证明它在 V13 上有效。V13 上必须比较：

```text
same-size random
motion-first rule
learned scorer
```

learned scorer 的主验收是：在新 baseline 的冻结 reward、比例、初始化、训练预算
和评测下，注入 learned-scorer-selected simbuffer 的最终 policy 优于对应的无
TRACE 父基线。它不强制必须超过 random 或 rule。

random 与 rule 仍应尽可能保留为辅助归因结果：

- learned scorer 优于父基线但不优于 random：证明 TRACE simulator 数据注入有效，
  但不能声明 learned selection 相对随机选择有独立优势；
- learned scorer 进一步优于 random：才可声明 scorer selection 本身有独立增益；
- learned scorer 不优于父基线：无论离线 pair accuracy 多高，都不能视为有效。

## 8. simbuffer reward、observation 与 n-step

### 8.1 Reward

simbuffer 在 selection 后统一使用当前冻结的
`compute_go2_imagination_reward()` 重算 reward：

```text
physical simulator next state
physical contact
current command
action/previous action history
epistemic uncertainty = 0
完整 reward config/hash
```

禁止额外加入：

```text
terminal reward override
failure backprop
action-delta penalty
saturation penalty
selector/scorer score
```

reward config hash 必须与 realbuffer、RWM环境和 frozen normalizer 一致。

### 8.2 Observation

```text
critic observation: 48D，保留 simulator 的真实 base_lin_vel
actor observation:  48D，与 critic 使用相同字段顺序
actor observation 中必须保留 base_lin_vel 和 3D command
```

禁止只对 TRACE rows 把 base linear velocity 置零，这会制造新的 source-specific
编码。正式训练前应运行 primary-RWM 与 TRACE critic-observation
source-leakage audit；如泄漏过强，projection 只能作为明确命名的消融。

### 8.3 n-step

```text
gamma = 0.99
n_step = 3
```

- real terminal 必须停止 bootstrap；
- candidate window 边界是 bootstrap boundary，不得伪装为环境 terminal；
- 不得跨 candidate、episode、reset 或 invalid mask；
- replay artifact 必须包含六个标准字段和完整 metadata/hash。

## 9. 接入当前两路线

总 batch 2048，realbuffer 固定 5%：

| 路线 | real | RWM | sim |
|---|---:|---:|---:|
| RWM-only | 102 | 1946 | 0 |
| RWM+TRACE r10 | 102 | 1752 | 194 |
| RWM+TRACE r25 消融 | 102 | 1460 | 486 |

注意：

- r10/r25 是剩余 95% synthetic 部分中的比例；
- 三来源先分别采样、校验，再 concatenate 和统一 shuffle；
- Actor/Critic 都直接消费同一顺序的48D full observation；
- V13 不允许 TRACE-only、BC 或任何 BC 组合路线；
- RWM+TRACE 在 refresh/resume 后不得清空 RWMbuffer；
- nominal sim ratio 尚未冻结时，不得启动正式实验或用未声明默认值生成主结果。

## 10. 两路线初始化不得被 TRACE 模块覆盖

TRACE 模块不得自行决定初始化。

```text
RWM baseline:
  actor/critic/target critic/temperature 随机初始化
  actor observation = 48D（含 BLV 和 3D command）
  critic observation = 48D
  update = 5% real + 95% RWM

RWM+TRACE:
  使用与父基线相同的随机初始化协议
  不加载 expert actor 或 BC checkpoint
  actor_bc_alpha = 0
  update = 5% real + 95% × (RWM + TRACE-selected sim)
```

baseline trainer 超参数完全服从权威路线文档，不得由 TRACE 模块覆盖。截至
2026-07-25，任何旧 BC、六路线和 proprioceptive 配置均只保留为历史记录。
Actor 48D / Critic 48D 是 V13 继续采用的标准 full-state policy observation
接口。正式学习率和 warmup 以权威路线
文档及 V13 配置的最新冻结状态为准，
任何 pilot 临时值都不得硬编码进 TRACE core 或 scorer。

## 11. Refresh、持久化和训练预算

每个 refresh 必须保存并恢复：

```text
actor
critic
target critic
temperature
全部 optimizer
agent update_step
frozen normalizer identity/hash
RWMbuffer（若使用）
simbuffer artifact 清单与策略
mixer/real/sim sampler RNG
其他可保存 RNG 状态
```

候选由 refresh 时的当前 actor 生成，但 collector 只加载 actor 做推理。完整 policy
训练恢复仍必须加载上述全部训练状态。

所有对照按相同 `num_policy_updates` 比较，不能按 simulator transition 数、
RWM env steps 或 wall-clock time 作为训练预算。

正式五条件训练预算为 `50M env steps`；`5M env steps` 只用于 pilot，不得写成
“全量”或混入正式主结果。对于不同数据源组合，公平比较仍以相同 policy-gradient
update 数为核心审计量。

## 12. 当前 V13 文件映射

以下 TRACE 文件属于历史实现参考或 Sim 侧生产路径。生产侧负责 candidate 采集、
筛选和不可变 artifact；消费侧负责 simbuffer artifact validator、replay loader、
mixer 和 hash/config 审计。具体由哪个分支实施，以主文档及对应子文档为准。

以下是目标项目侧已有入口，不代表它们已经满足 clean contract：

```text
candidate collector:
scripts/reinforcement_learning/rwm_dataset/
  collect_go2_expert_command_coverage_dataset.py

snapshot/reset helpers:
scripts/reinforcement_learning/rwm_trace/
  simulator_reset.py

current trajectory summarizer:
scripts/reinforcement_learning/rwm_trace/
  trajectory.py

current replay materializer/selector:
scripts/reinforcement_learning/rwm_trace/
  build_trace_replay_v4.py

current legacy scorer:
scripts/reinforcement_learning/rwm_trace/
  scorer.py
  train_go2_trace_scorer.py

current dynamic orchestration:
scripts/reinforcement_learning/go2_sim_gap_aligned/
  run_go2_trace_v5_dynamic_branch.sh

three-buffer sampler/mixer:
scripts/reinforcement_learning/rwm_flashsac/
  replay_mixer.py

trainer/agent:
scripts/reinforcement_learning/rwm_flashsac/
  train_flashsac_world_model_go2.py
  agent.py
```

## 13. 当前已知缺口：不得直接宣称完成

截至本文更新时，V12 rollout 骨架已具备 snapshot reset、四分支、T=1、H=20、
nominal simulator、actor-only inference、public reward 对齐、reward-independent
behavior summary V2，以及独立的 rule selector V1。

上述产物是 V13 的迁移基础，不等于满足 V13 的最终接口。迁移时 replay 必须改为
Critic 48D / Actor 48D（均含 BLV 和3D command），同时必须确保 RWM 是标准
full-state 45D 输入、45D输出且 BLV 参与输入和监督。

Rule selector V1 已实现：

```text
hard eligibility
-> weakest active-component motion validity
-> real foot swing validity
-> eligible 内 70% response/tracking + 30% moving stability
-> 每个 start-state 四分支最多 top1
-> top25 仅为上限
-> no fallback / no duplicate fill
```

对应实现与审计：

```text
scripts/reinforcement_learning/rwm_trace/rule_selection.py
tests/test_trace_rule_selection.py
logs/audits/v12_trace_rule_selection_v1_20260724/
```

RR05 小规模 controlled smoke 为 `8 starts × 4 branches × H20`，32 条中 25 条
eligible，最多可选 8 条，实际选择 7 条；一组无有效运动候选时选择 0 条，没有
补位。`state[0:33]` reset 最大误差为 `1.67413e-7`。

同时明确了跨动力学 identity：snapshot reset 检查物理可恢复的 `state[0:33]`；
actuator force `state[33:45]` 由 rollout 动力学重新产生。相同 rollout simulator
中的 repeated snapshot/action identity 仍必须通过；只有 source 与 rollout
动力学相同时才要求 dataset next_state identity。gap source 进入 nominal
simulator 时的一步分叉被记录但不误判为 reset 失败。

Rule manifest 到 n-step simbuffer 的 materialization 已接通并完成 RR05 小规模
审计：

```text
selection=rule
selected trajectories=7
H=20
n-step=3
materialized transitions=126
critic observation=48D
actor observation=48D（含 BLV 和3D command）
action=12D
reward=v1_1_dense_progress
gamma=.99
```

materializer 会校验 candidate/summary/manifest SHA256、selected identities、
70/30 权重、top25 cap 和 no-fallback/no-duplicate contract。六个输出 tensor
与独立重算结果逐位一致，最大误差均为 0；`ExternalReplaySampler` 已成功加载。

注意：首次生成的 48D observation 曾错误使用
`state[:33] + command + last_action`，使 command 落在 joint state 后面。该
artifact 已重命名为 `simbuffer_rule_n3.INVALID_observation_order.pt`，不得使用。
当前实现调用公共 `make_go2_policy_obs()`，正确顺序为：

```text
critic: base_lin_vel, base_ang_vel, gravity, command, joint_pos, joint_vel, last_action
actor:  与 critic 相同的48D full observation
```

修复后的 `simbuffer_rule_n3.pt` 已逐段核对 command/joint/last-action，并验证
n-step next observation 使用最终 next_state、下一 command 与最终 action；
全部检查通过。
审计目录：

```text
logs/audits/v12_trace_rule_replay_materialization_20260724/
```

V13 当前仍未完成：

1. V12 rule simbuffer 的 10-update 与 1000-update three-buffer smoke 均已通过，
   但 V13 两路线、RWM BLV 全监督接口仍必须重新执行；
2. 尚未完成五组 immutable candidate pools 与 rule downstream pilot；
3. `scorer.py` 的历史 reward-derived schema 不得复用，learned scorer 尚未按
   behavior summary V2 重建；
4. 历史 scorer 尚未在当前 V13 两路线基线上证明相对 RWM 父基线有效。

这些问题修复前，只能说 candidate rollout infrastructure 已接通，不能说完整
TRACE 或 TRACE scorer 已在 V13 上验证有效。

### 13.1 10-update three-buffer smoke 补充

正式入口为：

```text
scripts/reinforcement_learning/rwm_flashsac/run_v12_rule_three_buffer_smoke.sh
```

有效审计产物：

```text
logs/audits/v12_trace_rule_three_buffer_smoke_10updates_20260724/
```

保存 checkpoint 中的真实 `update_step=10`；每批固定为
`102 real + 1752 RWM + 194 sim`，actor/critic observation 为 `45D/48D`，
reward normalizer 冻结且无 fallback。旧 V1/V3 replay builder 的 observation
已同步为 canonical layout；旧 V3/T4 pipeline 默认禁用。

以上为 V12 proprioceptive 历史 smoke，不得作为 V13 验收。V13 必须用
Actor 48D / Critic 48D、两路线和新的45→45 RWM artifact 重新生成 replay并重跑
相同层级的 smoke。

同一入口的 1000-update 数值 smoke 也已通过：

```text
logs/audits/v12_trace_rule_three_buffer_smoke_1000updates_20260724/
```

它完成 502 interaction / 514048 env steps，保存 `step502`，其中
`agent_state.pt["update_step"]=1000`。500 组 actor/critic loss、全部日志数值
和最终网络 tensor 均为有限值；三路 reward hash 一致，无额外 sim reward
shaping。由于该 smoke 只重复使用 126 条 sim transition，它只证明训练接线和
数值稳定，不证明策略效果。下一暂停点是生成五条件 immutable candidate pools
并完成 rule downstream pilot；未通过 rule pilot 前不得进入 learned scorer。

## 14. 实施顺序与暂停点

### Phase A：adapter 与静态契约

1. 从两个 clean 分支读取 contract；
2. 填写 V13 两路线 baseline/pipeline adapter；
3. dry-run；
4. summary schema 和 reward shortcut fail-closed 测试；
5. 暂停审计。

### Phase B：候选与规则式筛选

1. g0 重采五组候选；
2. 审计 1024×4×20、T=1、nominal simulator、snapshot identity；
3. 生成 immutable summaries；
4. 构建 command-matched random 与 rule selections；
5. 审计 selection manifest 和 replay artifact；
6. 暂停审计。

### Phase C：规则式 downstream 验证

1. 10 update 单元 smoke；
2. 1000 update 数值 smoke；
3. 短 pilot；
4. 统一 256 env、1000 steps、seed 400 MJLab 评测；
5. 汇总五组 success、velocity error、survival 及 command-mode 指标；
6. 未同时改善三项则停止，不进入 scorer。

### Phase D：learned scorer

1. 在同一 immutable candidate pools 上运行 scorer；
2. schema/missingness/reward-leakage audit；
3. scorer、rule、random 等量 replay；
4. 相同初始化、seed 和 update budget；
5. scorer 未优于 random 与 rule则不得晋升。

### Phase E：扩展 source/condition

Sim 侧按主文档冻结的 `g0/rr05/rr03/p5/p75` 五个 condition 分别构建 mixed
source、candidate pools 和 simbuffer，并交付配置、manifest 和 SHA256。任何一组
都不得回退到旧 pure-expert manifest，也不得从其他 condition 复制或替换数据。

## 15. 最终验收清单

### Candidate

- [ ] Sim 侧 source dataset/config/hash 已明确；
- [ ] source 来自新 active mixed manifest，旧 pure-expert manifest 未被使用；
- [ ] 每组实际 collector counts/ratios 已记录并符合主文档协议；
- [ ] 独立候选组数量已在 Sim 侧 manifest 中冻结；
- [ ] 每组 1024 start states；
- [ ] 每个 start 恰好四个相同初始 branch；
- [ ] T=1、H=20；
- [ ] nominal simulator；
- [ ] 当前 actor-only；
- [ ] terminal 后停止，无 auto-reset 拼接；
- [ ] snapshot/one-step identity PASS。
- [ ] 所有正式 Sim source 均有独立 simulator restore 审计；
- [ ] restore 后 `state[0:33]` reconstruction error 通过阈值；
- [ ] 普通采集中的 NaN reset-error 字段未被冒充为通过证据；

### Summary/selection

- [ ] moving command 先响应再稳定；
- [ ] stand 独立稳定性；
- [ ] weakest active component；
- [ ] 八类 command mode；
- [ ] 无 reward/Q/RWM shortcut；
- [ ] saturation 字段名和 gate 一致；
- [ ] immutable candidate/summary/selection manifest；
- [ ] command-matched random 与 rule 等量。

### Replay/training

- [ ] common reward/hash，uncertainty=0；
- [ ] 无额外 reward shaping；
- [ ] critic 48D、actor 48D，且两者都含 BLV 和3D command；
- [ ] gamma=.99、n-step=3；
- [ ] realbuffer 固定 5%；
- [ ] 已冻结并记录 synthetic 内部 sim ratio；
- [ ] frozen normalizer；
- [ ] TRACE-only 与所有 BC 路线均被禁用；
- [ ] RWMbuffer refresh/resume 连续；
- [ ] 相同 gradient update budget；
- [ ] manifest/hash/log 完整。
- [ ] Sim 侧 source commit/config 和 simbuffer SHA256 已冻结；
- [ ] 消费侧只使用已验收 artifact，不静默重新生成或改写 simbuffer。

### Evaluation

- [ ] success 为达到完整评测 horizon 的比例；
- [ ] survival 以步数报告；
- [ ] velocity error 与权威评测定义一致；
- [ ] success 上升；
- [ ] velocity error 下降；
- [ ] survival 上升；
- [ ] 无 command-mode/sign/magnitude 大幅回归；
- [ ] MuJoCo/MJLab 人工确认真实运动而非原地扭动或 reward exploit。

## 16. 一句话交接原则

> 先用解耦的 TRACE core 证明“注入运动优先规则筛选出的 simulator replay 后，
> 最终 policy 优于对应无 TRACE 父基线”；再用独立 scorer 验证 learned selection
> 注入后同样优于父基线。command-matched random 与 rule 是辅助归因对照，不是
> learned scorer 的强制超越门槛。生产侧负责生成和筛选 simulator replay，
> 消费侧负责验收并注入对应训练路线。整个过程不得改变 V13 的 5% real、
> Actor 48D / Critic 48D observation、初始化、训练预算和评测协议。
