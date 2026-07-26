# V13 Real 侧两路线实验实施规范（权威版）

更新时间：2026-07-24

状态：V13 Real 侧路线语义、五组实机源数据矩阵及“同一台实机”身份已由用户确认；
五组 collector/condition artifact、CENet checkpoint、重算后新 dataset 路径、
BLV 估算验收阈值、TRACE simbuffer 比例和正式训练预算仍需按直接证据逐项冻结。
在这些配置冻结并完成启动前审计之前，禁止启动正式实验。

父文档：`V13_TWO_ROUTE_EXPERIMENT_DESIGN_ZH.md`。本文由主文档副本按 Real 侧
数据流程扩展而来，只规定 Real 侧专用的数据合同和实施细节；共同的路线、维度、
buffer、公平性与评测语义以父文档为准，本文不得覆盖父文档。

## 1. V13 Real 侧负责什么

V13 已拆分为两个并行方向：

- Sim 侧：由另一个分支继续负责 simulator 数据和 TRACE simbuffer；
- Real 侧：由本分支负责 expert 实机数据、BLV 估算、real dataset、RWM 和最终
  两路线 policy。

本分支不重新定义 Sim 侧实现，只通过带 hash 的 artifact contract 接收
TRACE-selected simbuffer。

V13 Real 侧只保留两条正式路线：

```text
路线一：5% realbuffer + 95% RWMbuffer

路线二：5% realbuffer
      + 95% × (RWMbuffer + TRACE-selected simbuffer)
```

V13 不使用 BC，不加载 expert actor 初始化，也不持续加入 expert-action imitation
loss。expert actor 只负责部署到实机并采集原始数据。

V13 Real 侧必须同时处理五组 condition：

```text
g0
rr05
rr03
p5
p75
```

“两条路线”指每个 condition 内都有同样的两条路线。因此正式矩阵是五组独立数据
流程、五个 CENet、五个 RWM，以及每组两个 policy arm，共十个最终 policy 实验。
不同 condition 的 dataset、CENet、RWM、normalizer、replay 和 checkpoint 禁止
混用。

## 2. 正确的维度定义

必须区分“RWM dynamics state”和“Policy observation”，二者不是同一个 tensor。

### 2.1 RWM dynamics

```text
RWM full state:       45 维
RWM input:            45 维，包含 base_lin_vel
RWM output:           45 维，包含 base_lin_vel
Action:               12 维
```

RWM full state 的冻结顺序为：

```text
base_lin_vel[3]
base_ang_vel[3]
projected_gravity[3]
joint_pos_rel[12]
joint_vel[12]
actuator_force[12]
```

因此：

```text
RWM input  = state[0:45]     # 45D
RWM output = next_state      # 45D
```

### 2.2 Policy observation

```text
Actor observation:   48 维，包含 base_lin_vel
Critic observation:  48 维，包含 base_lin_vel
```

Actor 和 Critic 共享的冻结顺序为：

```text
base_lin_vel[3]
base_ang_vel[3]
projected_gravity[3]
command[3]
joint_pos_rel[12]
joint_vel[12]
last_action[12]
```

统一断言：

```text
input_state_dim=45
output_state_dim=45
actor_observation_dim=48
critic_observation_dim=48
command_dim=3
action_dim=12
```

禁止切回 proprioceptive 42→45 RWM。Actor 和 Critic 必须保持相同的48D
full-state observation，均使用估算 BLV 并保留三维 command；禁止只从 Actor
输入中删除前三维 BLV。

## 3. Real 侧 25K dataset

### 3.1 数据来源

当前接受并冻结以下五组实机源数据/selection artifact，主副本均在3090：

```text
g0:
  /data1/xjy/unitree_rl_mjlab_normal_fixstand_v2/logs/rwm_datasets/go2_real_g0_go2sun_recollect_25k_20260719
  dataset.pt SHA256:
  3e267087bcdc0b79b6e3f13745f0ab8ed71005c28478665252946cb13b31d441

rr05:
  /data1/xjy/unitree_rl_mjlab_normal_fixstand_v2/logs/rwm_datasets/go2_real_g0_d0_rrcalf_0p5_25k_20260714
  dataset.pt SHA256:
  0b8dda4dee13fdc9836aa4130ed25616d674ad70fe0cc807c315243ccd315aad

rr03:
  /data1/xjy/unitree_rl_mjlab_normal_fixstand_v2/logs/rwm_datasets/go2_real_g0_d0_rrcalf_0p3_25k_20260715
  dataset.pt SHA256:
  42382009099aac30aae01b217bb5289e649c97ce57c9bac7d8eb74465245ba02

p5:
  /data1/xjy/unitree_rl_mjlab_normal_fixstand_v2/logs/rwm_datasets/go2_real_g0_d0_p5_25k_20260715
  dataset.pt SHA256:
  ad218a2238720b0240d3739c0de190f3cebeffaad4d72f458cbab4d6d8241398

p75:
  /data1/xjy/unitree_rl_mjlab_normal_fixstand_v2/logs/rwm_datasets/go2_real_g0_d0_p75_25k_20260715
  dataset.pt SHA256:
  bb3b29e713f93b3de260fcc4ebd9247d30f2442675479974979d04192ab25171
```

五组筛选和序列统计为：

```text
group  raw transitions  selected  selected runs  valid 40-step starts
g0              48,002    25,000            218                16,940
rr05            56,433    25,000            457                 9,164
rr03           107,658    25,000            226                17,457
p5              47,884    25,000            343                13,597
p75             50,797    25,000            315                14,416
```

用户于2026-07-24明确确认：G0 的2026-07-19 `go2sun` 重采数据与其余四组
2026-07-14至15数据来自同一台实机，并接受它们共同组成 V13 Real 五组矩阵。
因此采集日期和目录命名差异不再构成“是否同一实机”的暂停项。五组各自的
condition 仍必须由对应配置、物理设置记录和 collector artifact 独立绑定。

G0 在4090上有一个已核验的同 hash 镜像：

```text
/data/xiejiayi/unitree_rl_mjlab_normal_fixstand_v2/logs/rwm_datasets/go2_real_g0_go2sun_recollect_25k_20260719
```

该镜像的 `dataset.pt` SHA256 与3090一致。4090上未找到其余四组对应目录，
4090_132上未找到这五组数据。正式转换 manifest 必须绑定实际读取的绝对路径，
禁止根据组名静默替换数据。

G0 原始 CSV 为：

```text
/data1/xjy/unitree_rl_mjlab_normal_fixstand_v2/logs/rwm_datasets/go2_real_g0_go2sun_recollect_25k_20260719/source/logs/run_data_2026-07-19_11-18-42.csv

SHA256:
8fdcb15e2162b409ca47c31dae73775375a5e631b4f72955937de6d77bcf017e
```

其余四组包含的原始 CSV 数量分别为 `rr05=2`、`rr03=5`、`p5=2`、`p75=3`。
四组缺失的 source-row 映射已于2026-07-24从原始 CSV 重建，并逐字段复现现有
`dataset_all.pt`、selection 和最终 `dataset.pt`。重建没有覆盖任何原始数据，
只在各组 `reports/` 下新增以下 manifest：

```text
rr05:
  reports/v13_cenet_source_mapping.json
  manifest SHA256:
  7eb443b6b7b8cb6a59f1764075f275c7ac63ffb5add73c7a1db70b4b68db119b
  source mapping SHA256:
  ebab8b32b03e749775b0d28dabdc18e0e937ade5f85169a115adc1acc2a1cb57

rr03:
  reports/v13_cenet_source_mapping.json
  manifest SHA256:
  bc5de7f219c22b6b74ad377fe9489ace60d1e954ce7482a42d37340ea7655630
  source mapping SHA256:
  fd94216b15b2d25133c343eb3b394a44ba94ee10d9b43ec87169aa7bf371b1c9

p5:
  reports/v13_cenet_source_mapping.json
  manifest SHA256:
  b7fecd86de2cb279ca72e07376ab22813c0015c008e3dc9a2e707770fdb43029
  source mapping SHA256:
  f388544cbd3b0e73960a13de479373a2c0ef48ca390461c8116687b88e318b7d

p75:
  reports/v13_cenet_source_mapping.json
  manifest SHA256:
  0b43695d6aa898c82bc92b49cbcce5fce2ffa4dc884caf9100229ae397b0cbef
  source mapping SHA256:
  3863f8cfc86357331584a9a11e210a63378027d38d899cdd40514a886c98039a
```

这些 manifest 同时冻结每个原始 CSV 的绝对路径和 SHA256、25K
`source_file_ids/source_rows`、现有 `dataset_all.pt` 和 `dataset.pt` SHA256。
四组均满足：25K 映射无重复、原始 CSV hash 当前一致、完整非 BLV 数据流逐字段
一致、selector 输出与最终 25K dataset 逐字段一致。

G0 的 `dataset.pt` 已内嵌 `source_file_ids/source_rows`；共25,000条且无重复，
其映射 SHA256 为：

```text
781aa476fe69b92f4aa559d31a68f006f43dee4b2a0d5281bf5041b9ac5046ed
```

五组现有 25K source-row 映射现已冻结。它们只用于在完整连续 real segment 上
完成 CENet 推理后，按原选择提取对应 BLV；不得先按映射抽出离散行再构造 history。

该数据来自：

```text
expert policy
-> 部署到真实 Go2
-> 采集若干真实 run/segment
-> 完整性与安全审计
-> 筛选得到 25,000 条合法 real transitions
```

25K 只表示最终合法 transition 总数为 25,000，不表示：

```text
25 episodes × 1000 steps
[1000, 25] 矩形 tensor
五组数据合并后的总数
每条 trajectory 等长
```

真实数据可以由不同数量、不同长度的连续片段组成。必须保存每条 transition 所属的
原始文件、run、segment、时间戳和片段内 timestep。

五组 selection report 与 validation report 的八类 command 计数一致，均精确命中
以下目标；G0 还已从最终 `dataset.pt` 独立重算确认：

```text
pure_x=25%, pure_y=10%, pure_yaw=8%, stand=8%
x_yaw=17%, xy=14%, xy_yaw=13%, y_yaw=5%
```

每组后续 CENet 重算都不得改变该组 25,000 条 transition 的
selection/source-row 映射和 command。若因 CENet history 或质量门槛必须排除任何
transition，则该组原 25K selection 立即失效，必须重新生成 selection manifest
并再次精确审计指令比例，不得静默补行。

### 3.2 禁止跨越的边界

CENet history、RWM sequence、n-step replay 和 train/validation split 都不得跨越：

```text
CSV 文件边界
controller 或 policy 重启
policy state 切换
episode_step 回退
时间戳倒退或超阈值跳变
丢帧/重复帧
急停与 safety event
人工删除或筛选造成的非连续切口
```

真实片段必须先完成连续性审计，再生成 sequence。禁止为了凑满 25K 跨边界拼接。

### 3.3 原始数据没有 BLV

实机 expert actor 的 45D observation 不包含 `base_lin_vel`。因此原始 real CSV
不能直接构成带 BLV 的 RWM 45D target。

五组当前 `dataset.pt` 虽然含有三维 BLV 字段，但其来源全部是旧的：

```text
base_lin_vel_source = contact_kinematics_affine_ema_estimate
```

该旧 BLV 不属于 V13 的正式监督标签，禁止直接用于 V13 RWM、realbuffer、reward
或 Actor/Critic observation。`dataset.pt` 当前只能作为已冻结 transition
selection、command 比例和 source-row 映射的参考 artifact。

Real 侧必须使用冻结的 CENet，从连续 observation history 估算：

```text
estimated_base_lin_vel[3]
```

这个值是估算标签或伪监督，不是真实传感器测得的 ground truth。所有 dataset、
checkpoint 和结果报告必须使用 `estimated`/`pseudo_target` 语义，禁止把它写成
`true_base_lin_vel`。

## 4. CENet BLV 估算协议

当前参考实现来自 expert 分支提交：

```text
commit: 4c2fe9782a144030ec0d2cee8b6394fac2356bf5
message: add cenet
```

其基本接口为：

```text
过去 10 帧 × 45D 标准 expert 部署 observation
-> CENet
-> estimated body-frame base_lin_vel[3]
```

CENet 是隐变量速度估计器，不是预测 next state 的完整动力学模型。这里“贴近实机”
表示它在对应实机 observation 分布上能准确估算 BLV；不能把 simulator held-out
误差直接写成实机动力学已经匹配，仍必须保留 real-side 独立有效性验证。

CENet 本身必须先在具有真实 simulator BLV 的数据上训练和验证，然后冻结，再用于
real CSV 的离线估算。没有 BLV 标签的 real dataset 不能直接训练这个 velocity
estimator。

V13 必须为 `g0/rr05/rr03/p5/p75` 各训练一个独立 CENet。每个 CENet 使用对应
condition 的 simulator dataset 中真实 BLV 进行监督，再处理同组的原始连续实机
记录。禁止五组共享一个 checkpoint，也禁止拿某组 CENet 给另一组生成标签。

最新 expert 分支 `4c2fe9782a144030ec0d2cee8b6394fac2356bf5` 提供的是联合
`CENet + BC actor` 参考实现：训练 loss 含 action imitation，并按 validation
action MSE 选择 checkpoint。该实现只作为 CENet 网络和评测代码参考，V13 明确
禁用其中的 BC actor、action target、action imitation loss 和按 action MSE 选
checkpoint 的语义。

V13 必须拆出 CENet-only 训练入口：

```text
input:  previous 10 × 标准 expert 部署 observation[45]
output: estimated body-frame base_lin_vel[3]
target: simulator true body-frame base_lin_vel[3]
loss:   velocity-only loss
BC actor/action target/action loss: absent
```

### 4.1 单一标准 expert 与五个 CENet 的对齐路线

用户于2026-07-25明确：五组真机数据不是由五个不同 expert 采集，而是先训练一个
标准 expert，再将同一个 expert 部署到五种实机状态下分别采集。因此 CENet 的
simulator 数据也必须冻结同一个标准 expert，只改变对应物理 condition。禁止为
五组分别训练不同 expert，否则同时改变 policy behavior 和 dynamics，无法判断
CENet 差异来自哪一项，也不再匹配真实采集分布。

```text
唯一冻结的标准 expert（必须与实机采集使用的 checkpoint 完全一致）
├─ g0 simulator   -> g0 final-train/held-out   -> g0 CENet
├─ rr05 simulator -> rr05 final-train/held-out -> rr05 CENet
├─ rr03 simulator -> rr03 final-train/held-out -> rr03 CENet
├─ p5 simulator   -> p5 final-train/held-out   -> p5 CENet
└─ p75 simulator  -> p75 final-train/held-out  -> p75 CENet
```

简单执行流程为：

```text
冻结同一个标准 expert，不再训练 expert
-> 分别放入 g0 / rr05 / rr03 / p5 / p75 五种 simulator 状态
-> 每组采集标准 expert 部署 observation[45] 与 simulator true BLV[3]
-> 每组构造 previous 10 × 标准 expert 部署 observation[45] -> current true BLV[3]
-> 每组独立训练一个 velocity-only CENet
-> 得到五个分别对应五种状态的 CENet
```

必须满足：

- 五组加载同一个标准 expert checkpoint 和相同 observation/action preprocessing；
- expert 参数、normalizer 和行为模式全部冻结；
- 五组之间唯一计划变化是对应 simulator physical condition；
- CENet 输出不作为 expert actor 输入，避免两个网络共同适应仿真误差；
- CENet 始终只有 simulator true BLV 的 velocity loss；
- 每组独立训练一个 CENet，禁止跨 condition 共用 checkpoint；
- 每条 trajectory 必须记录标准 expert checkpoint/path/SHA256；
- 在标准 expert 的实机采集 checkpoint 尚未直接绑定前，不得启动正式 Sim 采集。

当前已找到的标准 expert 直接证据为：

```text
checkpoint:
/data1/xjy/unitree_rl_mjlab_normal_fixstand_v2/logs/experiments/go2_gap_experts/payload_20260713_payload_0_vs_5to10kg/G0_D0/run/step97656

actor.pt SHA256:
b53a3bda4b65199118c4c5112cdedd33b26274263f676aff0eb80a37f50d6b92

flashsac_config.yaml SHA256:
61c6c78f713a641f5c8652ba8ffa85a7339854dd326264593de4cdb0f5cdcb85

exported policy.onnx SHA256:
e027bfb3b530cdd07a34f0dcd6b5249187d50294de7a6654296c930196967d69
```

G0 与 P75 的 `source/metadata.json` 直接记录了同一 checkpoint path；用户明确确认
五组使用同一个标准 expert。RR05/RR03/P5 的现有 dataset 目录没有全部保留可独立
复核的 checkpoint/hash metadata，因此跨五组 artifact 绑定仍标为部分完成，
不得把用户语义确认误写成五份 hash 证据都已落盘。

每个 condition 都使用该冻结标准 expert 采集数据，并按完整 trajectory 或
environment stream 划分互斥的 train/held-out，禁止随机切行造成 history 或
trajectory 泄漏：

```text
train:
  允许参与 CENet gradient update

held-out:
  不参与任何 gradient update、normalizer fit 或超参数选择
  只用于 velocity checkpoint selection 和最终 sim 验收
```

禁止使用 expert 训练过程中的非平稳 rollout 作为正式数据。train/held-out 都必须
绑定标准 expert、物理状态、seed/environment stream、数据文件和 SHA256。

### 4.2 当前 mixed 25K 的 CENet 排除状态

此前审计的 V13 Sim 侧五组 condition-specific mixed 25K 不再是正式 CENet
训练源，也不得用于 CENet 预训练、补充训练或 held-out 验证。状态和审计证据
记录在：

```text
/home/xjy/Go2/.v13_doc_sync/V13_CENET_SIM_DATASET_MANIFEST.json
status: EXCLUDED_FROM_CENET_POLICY_MISMATCH
```

这批 mixed 25K 使用的 actor SHA256 是
`55ff501f997e70fc049963156b04ba7671a4d7a104759071949383f8ee32e55e`，
与实机标准 expert 的 `b53a3b...` 不同，因此即使字段和 condition 审计通过，
也不能升级为正式对齐数据。

五组候选均位于3090：

```text
root:
/data1/xjy/unitree_rl_mjlab_v13/logs/rwm_datasets_v13_mixed_full1000_snapshots_5conditions_realrange_20260724_v2

group  dataset SHA256                                                    valid H10
g0     39699589c47cf056df26fa9ea651ec78e3692b2099b4333f9678058051af96f2     24,730
rr05   928b70fd6133201824fb5c7283c1f43fc5dc2c238735004a27c7f601704bc156     24,700
rr03   3a2cb6487a5cf79b94cd02bba9037c95124ada85d2b629974c4b15facf5938f2     22,737
p5     e102ee455e7d512289f3f00359fc2c703f181ec402a8e169dbb3d5909641622d     24,700
p75    d152a35d4917b88f01008de5d532d918346bffad1fab5b4a3b8f25d8ffd88a59     24,670
```

每组 dataset 都是 `[1000,25]` 的完整环境流存储。此前计算的 `valid H10` 和
20/5环境流划分只保留为审计证据，不能进入任何 CENet gradient update，也不能
替代新的冻结标准 expert held-out 数据。

五组审计均确认：

- dataset condition 与 `g0/rr05/rr03/p5/p75` 物理设置逐组匹配；
- 45D 标准 expert 部署 observation 可由
  `state[3:9] + command[3] + state[9:33] + prev_action[12]` 零误差重建；
- `state[0:3]` 是有限且三轴非恒定的 simulator true BLV；
- history 不跨 `episode_id/timestep/done/timeout` 边界；
- launch code、expert artifact、dataset、独立审计和 CENet 审计均由 SHA256 绑定。

这里审计的是各组 `selected_25k/dataset.pt`，不是1,024,000条 raw pool。二者都
不能进入 CENet，且禁止把不同 condition 合并训练。它们在 Sim 侧是否继续用于
RWM/TRACE 由 Sim 侧正式路线和 active manifest 单独决定，本节不改变该用途。

正式 checkpoint 必须按 held-out velocity 指标选择并保存为独立 CENet artifact。
禁止用全数据 refit 结果覆盖 validation-best checkpoint；若另做 refit，只能作为
独立候选且不能替代具有 held-out 证据的正式 checkpoint。

对每组都从上述目录中的原始连续实机记录出发，用该组冻结的 CENet 重新计算全部
三维 `base_lin_vel`，再按该组当前 25K 的 source-row 映射提取对应 label，生成
新的 V13 Real dataset。新产物必须使用新路径和新 SHA256，禁止覆盖或原地修改
现有 `dataset.pt`。

### 4.3 离线估算顺序

必须按以下顺序执行：

```text
上述目录中的原始完整 real runs
-> 识别连续 segment 和无效边界
-> 在每个连续 segment 内按时间顺序运行 CENet
-> 生成 estimated BLV 与 estimator metadata
-> 按冻结的 source-row/selection 映射提取当前 25K
-> 重新验收 transition、history 和 command 比例
-> 另存并冻结 CENet 重算后的 V13 Real 25K dataset
```

不能先把不连续的 transition 随机筛成 25K，再在筛选结果上建立 10 帧 history。

每个 segment 开头没有足够历史的 transition 默认不得产生正式 BLV 标签。具体
warmup 长度必须与冻结 CENet 的 `history_length` 一致。

simulator CENet 训练 history 同样必须根据 `episode_ids/timesteps/done` 切段。
最新参考实现使用 `clamp_min(0)` 重复第一帧，并且不检查 episode/reset 边界；
V13 禁止沿用这两个行为。训练与实机推理都必须排除每段最前面的
`history_length` 个无完整历史样本，禁止跨 reset 拼接 history。

### 4.4 CENet artifact 必须记录

```text
CENet source commit
checkpoint absolute path
checkpoint SHA256
training dataset path/hash
expert checkpoint path/hash
frozen standard expert deployment/config/normalizer provenance
frozen-expert final-train dataset path/hash
frozen-expert held-out dataset path/hash
observation_dim
history_length
observation order
control period
sim validation metrics
real-side validation evidence
```

当前 expert 提交没有在 Git 中提供可直接使用的 CENet checkpoint。没有冻结
checkpoint 和验证报告前，不得生成正式 V13 BLV labels。

### 4.5 CENet 验收

正式采用前至少必须确认：

- 45D 标准 expert 部署 observation 顺序与实机 CSV 完全一致；
- control period 与训练数据一致，或有明确的重采样策略；
- history 不跨真实 segment 边界；
- checkpoint 只按 held-out velocity 指标选择，不读取 action imitation 指标；
- checkpoint 选择与训练流程中不存在 action imitation loss；
- 仿真 held-out 数据报告三轴 RMSE、MAE、bias、相关性和 R²；
- real 数据至少完成独立速度参考或其他直接有效性检查；
- 输出无 NaN/Inf，异常值、饱和值和低置信度 transition 可追溯；
- 估算结果与原始 CSV、CENet checkpoint、转换代码全部用 SHA256 绑定。

如果没有 real-side 有效性证据，只能生成候选/试验 dataset，不得将其标记为正式
V13 dataset。

## 5. Real RWM dataset 的构造

每条合法 transition 必须能够构造：

```text
state_t[45]
action_t[12]
next_state_t[45]
command_t[3]
actor_observation_t[48]
actor_next_observation_t[48]
critic_observation_t[48]
critic_next_observation_t[48]
contacts
terminated / truncated
run_id / segment_id / timestep
CENet estimated BLV metadata
```

其中：

```text
state_t[0:3]      = CENet estimated BLV at t
next_state_t[0:3] = CENet estimated BLV at t+1
```

其余 state 字段必须从实机记录中按冻结的关节顺序、单位和坐标系重建。
`tau_est` 是否可直接作为 `actuator_force` 必须用当前 collector contract 和单位
证据确认，禁止只因维度都是 12 就直接等同。

最终 dataset manifest 必须记录：

```text
format_version
num_transitions=25000
num_runs
num_segments
segment length distribution
source CSV absolute paths and SHA256
selection manifest and SHA256
CENet checkpoint and SHA256
converter code commit/hash
field order, units and coordinate frames
boundary rules
invalid/rejected transition counts and reasons
```

## 6. RWM 监督语义

RWM 输入包含 BLV：

```text
input_dropped_state_indices:  []
input_state_dim:              45
```

RWM 输出必须预测并监督完整 45D next state：

```text
output_dropped_state_indices: []
state_loss_ignored_indices:   []
output_state_dim:             45
```

BLV 三维必须具有非零 loss 权重：

\[
L_{\mathrm{state}}
=
\sum_{i=0}^{44} w_i
\ell(\hat{s}_{t+1,i},s_{t+1,i}),
\qquad
w_0,w_1,w_2 > 0
\]

训练日志必须单独报告：

```text
estimated-BLV target loss
non-BLV state loss
total state loss
BLV target confidence/validity statistics
```

这里的监督 target 来自冻结 CENet。当前估算 BLV 同时进入 RWM current-state input
和 Actor/Critic full observation；必须绑定 estimator/hash 并报告其置信度，禁止
把估算值声称为真实传感器值。

## 7. 三种 buffer 的 observation

三种 buffer 均保存同一字段顺序的48D full observation，Actor/Critic直接消费，
不得执行 `[..., 3:48]` 投影。

BLV 来源分别为：

```text
realbuffer: CENet 估算的 real BLV
RWMbuffer:  RWM 预测的 BLV
simbuffer:  simulator 的真实 BLV
```

Actor 对三种来源始终使用相同顺序的48D observation，并看到对应来源的 BLV。

### 7.1 realbuffer

realbuffer 来自冻结的 real 25K dataset。它参与 SAC update，但不是 BC：

- 不加载 expert actor；
- 不加入 expert-action supervised loss；
- `actor_bc_alpha=0`。

realbuffer reward 中如果使用线速度项，必须明确使用 CENet estimated next BLV，并
绑定 estimator checkpoint/hash。

### 7.2 RWMbuffer

RWMbuffer 来自当前 policy 与 frozen real-data RWM 的 rollout。RWM 必须由上述
real 25K dataset 独立训练，并在 policy training 期间冻结。

### 7.3 simbuffer

simbuffer 来自 Sim 侧分支，经 TRACE 筛选后以冻结 artifact 交给 Real 侧。

Real 侧接收前必须校验：

```text
artifact path/hash
sim source commit/config
selector type/checkpoint/hash
reward config/hash
Actor 48D / Critic 48D observation order
action order and units
gamma / n_step
selected transition count
```

Real 侧不得静默重新解释或重排 Sim 侧字段。

## 8. 两条正式路线

以下两条路线必须在五个 condition 内分别执行。不同 condition 之间不混合 dataset
或 replay，也不互相充当对照；每组路线一只与同组路线二配对比较。

### 8.1 路线一：Real RWM baseline

```text
对应 condition 的 expert 实机原始数据
-> 连续片段审计
-> 该 condition 的 CENet BLV 估算
-> 筛选并冻结 real 25K dataset
-> 训练该 condition 的 RWM（45输入、45输出、estimated BLV 参与输入和监督）
-> 冻结 RWM
-> 随机初始化 Actor(48)、Critic(48)、temperature
-> update = 5% realbuffer + 95% RWMbuffer
-> final policy
```

### 8.2 路线二：Real RWM + TRACE

前半段与路线一完全相同，只增加 Sim 侧交付的 TRACE-selected simbuffer：

```text
update = 5% realbuffer
       + 95% × (RWMbuffer + TRACE-selected simbuffer)
```

定义 \(\rho_{\mathrm{sim}}\) 为 95% synthetic 内部的 sim 比例：

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

无论比例为何，都必须满足：

```text
realbuffer = 5%
RWMbuffer + simbuffer = 95%
```

当前 \(\rho_{\mathrm{sim}}\) 尚未冻结，禁止使用未声明默认值启动正式实验。

## 9. 公平性约束

两条路线必须共享：

```text
同一 condition
同组 real 25K dataset/hash
同组 CENet checkpoint/hash 和 BLV labels
同组 frozen RWM checkpoint/hash
同组 realbuffer
同一 reward 与 reward normalizer
同一 Actor/Critic/temperature 初始化 seed
同一 optimizer、batch size、gamma、n_step
同一 gradient-update budget
同一评测协议
```

每个 condition 内的唯一主变量是 95% synthetic 中是否加入冻结比例的
TRACE-selected simbuffer。五组使用同一 protocol，但五组 artifact 必须独立。

## 10. 强制日志

每个 RWM run 必须记录：

```text
version=V13-real
real dataset path/hash
CENet checkpoint/hash
input_state_dim=45
output_state_dim=45
state_loss_ignored_indices=[]
estimated BLV loss
non-BLV loss
BLV validity/confidence statistics
```

每个 policy run 必须记录：

```text
route_name
actor_observation_dim=48
critic_observation_dim=48
BC enabled=false
expert actor loaded=false
real/RWM/sim requested and actual counts/ratios
dataset/CENet/RWM/replay hashes
reward/normalizer hashes
gradient update count
```

## 11. 分阶段执行

### Step 1：Real 原始数据合同

- 为五组分别绑定第 3.1 节已冻结的3090源数据目录；
- 为每组冻结真实 CSV、现有 `dataset.pt`、selection report 和 collector commit；
- 逐组审计机器人、condition、字段、顺序、单位、控制周期和连续边界；
- 使用本节已冻结的五组 25K source-row 映射，并再次核对 command 比例；
- 不生成训练 dataset。

### Step 2：CENet estimator

- 同步并审计 expert 分支 CENet 实现；
- 实现不含 BC/action imitation loss 的 CENet-only 训练入口；
- 直接绑定五组实机采集共同使用的唯一标准 expert checkpoint/path/SHA256；
- 冻结其 observation/action preprocessing、normalizer、control period 和部署配置；
- 固定同一个 expert，分别在 `g0/rr05/rr03/p5/p75` simulator condition 中采集
  互斥的 final-train 与 held-out trajectories；
- 当前 mixed 25K 因 expert hash 不匹配而完全排除出 CENet；
- 使用五组 condition-specific 数据分别训练五个 CENet；
- 逐组按 velocity 指标选择 checkpoint；
- 逐组完成 sim held-out 与 real-side 有效性验收；
- 分别冻结五个 checkpoint/hash。

### Step 3：生成 Real 25K dataset

- 每组在完整连续 segment 上运行对应 CENet；
- 每组按现有 selection/source-row 映射提取对应 25K；
- 分别构造 state/observation/transition；
- 使用五个新路径输出 dataset 与完整 manifest，禁止覆盖旧 `dataset.pt`；
- 逐组审计 25K 数量、边界、维度、NaN/Inf、BLV 分布和 SHA256。

### Step 4：RWM

- 逐组验证45→45、BLV输入和非零监督；
- 逐组 smoke forward/backward；
- 分别训练并验收五个 frozen real-data RWM。

### Step 5：两路线 replay

- 构建 5% realbuffer；
- 验收 RWMbuffer；
- 接收并验收 Sim 侧 TRACE simbuffer；
- 冻结 synthetic 内部 sim ratio。

### Step 6：Policy

- 五组各执行两路 10-update smoke；
- 五组各执行配对短 pilot；
- 配置 diff 无非预期变化后再启动正式训练；
- 使用统一协议评测。

## 12. 当前证据缺口

截至 2026-07-25，以下内容尚未冻结：

- 五组 collector commit、condition 配置和对应物理设置记录的完整绑定证据；
- 五组实机采集共同使用的标准 expert checkpoint、部署配置、normalizer 和 SHA256
  的直接绑定证据；
- 同一冻结标准 expert 在五个 simulator condition 下的 final-train/held-out
  数据路径、互斥证明和 SHA256；
- 五组 CENet 重算后新 V13 Real 25K 的输出路径和 SHA256；
- 五个 CENet checkpoint、SHA256 和 velocity 验收阈值；
- 五组 real-side 独立速度验证证据；
- `tau_est -> actuator_force` 的字段、顺序和单位合同；
- TRACE simbuffer artifact 与 synthetic 内部比例；
- V13 正式训练预算和完整评测协议。

这些缺口关闭前，只允许做代码审计、数据审计和 fail-closed smoke，不得启动正式
RWM 或 policy 训练。
