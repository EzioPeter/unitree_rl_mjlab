# V13 建议与审计跟踪

更新时间：2026-07-24

状态：持续更新；当前总体结论为 `BLOCKED_FOR_FORMAL_TRAINING`

适用目录：

```text
/data1/xjy/unitree_rl_mjlab_v13
```

## 1. 文档用途

本文记录 V13 目录、权威文档、数据产物、验证结果和运行环境中发现的问题，并跟踪
建议动作及关闭证据。本文是审计台账，不覆盖以下权威实验文档：

```text
V13_TWO_ROUTE_EXPERIMENT_DESIGN_ZH.md
V13_REAL_SIDE_EXPERIMENT_DESIGN_ZH.md
V13_TRACE_HANDOFF_ZH.md
EXPERIMENT_INTEGRITY_FIRST.md
```

如果本文与权威实验文档冲突，应停止执行并修正源文档；不得只修改本文来掩盖冲突。

状态定义：

```text
已完成：已有直接证据，且证据路径/hash/结果已记录
待处理：问题明确，但尚未关闭
阻塞：在关闭前禁止启动相关正式实验
观察：当前不阻塞，但需防止演变为配置或复现问题
```

## 2. 当前结论

V13 已明确收敛为两条正式路线：

```text
RWM baseline
RWM + TRACE
```

当前可以继续进行代码审计、数据审计和 fail-closed smoke，但不能启动正式 RWM 或
policy 训练。主要原因是：

- active Sim mixed-data manifest 尚未形成；
- 五条件数据未全部通过，`rr03` 已明确无效，`p5/p75` 尚未生成；
- 旧 pure-expert manifest 在不同文档中的有效性描述互相冲突；
- 正式训练预算、TRACE 比例以及部分 reward/gamma/n-step 权威状态尚未统一；
- Real 侧 CENet checkpoint、BLV 验收和新 Real dataset 尚未冻结。

## 3. 已完成项目

### C-001：V13 根目录与三份核心文档审阅

状态：**已完成**

已审阅：

```text
AGENTS.md
EXPERIMENT_INTEGRITY_FIRST.md
V13_TWO_ROUTE_EXPERIMENT_DESIGN_ZH.md
V13_REAL_SIDE_EXPERIMENT_DESIGN_ZH.md
V13_TRACE_HANDOFF_ZH.md
V13_SIM_DATASET_MANIFEST.json
```

审阅时文件 SHA256：

```text
V13_TWO_ROUTE_EXPERIMENT_DESIGN_ZH.md
  368dc981eac5131f5d9229c8949895b5cf16b2862a9061c768be6f60d4784aa0
V13_REAL_SIDE_EXPERIMENT_DESIGN_ZH.md
  452dc2fb7d6aae34a0ae1dc05bbe1fc86339ff0eca53ff83b1f8c2b2c20dfef7
V13_TRACE_HANDOFF_ZH.md
  0b59f11aad495cfdb7bba33f85c0a1160cdf344d216d63a3ee181a55dc6d5a2e
V13_SIM_DATASET_MANIFEST.json
  f809a40a61d488b148c75ae5fe96793ef6dc0d27ee7bd065da32e6eb23e1e60f
```

### C-002：V13 mixed g0 exact-reset smoke

状态：**已完成**

证据：

```text
logs/v13_validation_20260724/mixed_exact_reset_smoke_g0_v2/result.json
```

结果：

```text
status: passed
reset reconstruction max error: 2.1336893496481935e-07
repeated one-step max error:     2.6702880859375e-04
source transition max error:     3.7384033203125e-04
actor observation rebuild error: 0
tolerance:                       0.005
formal_training_input:           false
```

该结果只证明 g0 小规模 mixed snapshot/reset 链路可工作，不替代五条件正式数据审计。

### C-003：V13 RWM 42→45 与 BLV 监督单元检查

状态：**已完成**

执行：

```text
PYTHONPATH=. .venv/bin/python tests/test_v13_rwm_blv_supervision.py
```

结果：

```text
PASS test_v13_config_is_42_to_45_with_blv_supervision
PASS test_v13_semantics_reject_blv_loss_mask_but_allow_input_drop
PASS test_blv_target_has_gradient_and_input_blv_is_invisible
```

这证明当前被测配置和模型代码满足：

- RWM 输入丢弃 BLV；
- RWM 输出保留 45D；
- BLV target 参与梯度；
- 修改 current-state BLV 不改变模型输入预测。

### C-004：g0 与 rr05 mixed 25K 静态审计

状态：**已完成**

```text
g0:
  audit status: passed
  dataset SHA256:
    16043033b37db85a53d88a017175d976fe663d7394382b1baea9eb5908c3e743
  max collector-ratio error: 0.07412

rr05:
  audit status: passed
  dataset SHA256:
    99f8ba9ccbe510c13d60be7138d501cb6458d11acaadc14f13447c872fb285a3
  max collector-ratio error: 0.06224
```

两者通过的是当前 `0.08` collector-ratio 容差下的静态审计；仍需正式
per-condition exact-reset 审计和最终 active manifest 绑定。

## 4. 阻塞与待处理问题

### A-001：旧 Sim manifest 的有效性描述冲突

优先级：P0
状态：**阻塞**

证据：

- `V13_TWO_ROUTE_EXPERIMENT_DESIGN_ZH.md` 明确将
  `V13_SIM_DATASET_MANIFEST.json` 及其 SHA256 标为
  `SUPERSEDED_DO_NOT_TRAIN`；
- `V13_TRACE_HANDOFF_ZH.md` 开头仍称该 manifest 是当前冻结输入，并要求 Sim RWM、
  TRACE reset source 和 replay 使用其中 pure-expert 25K；
- `V13_SIM_DATASET_MANIFEST.json` 自身仍声明 `role` 为 V13 正式 RWM/TRACE source，
  且包含 `recollection_required=false`。

风险：

launcher 或消费方若只读取 TRACE 文档或 JSON manifest，可能把已废止 pure-expert
25K 当成正式输入。

建议：

1. 在旧 JSON 顶层加入机器可读状态：
   `status=SUPERSEDED_DO_NOT_TRAIN`；
2. 删除或改写 TRACE 文档开头对旧 manifest 的正式绑定；
3. launcher、RWM trainer、TRACE pipeline 对旧 manifest SHA256 fail closed；
4. 新 mixed 五条件通过后发布全新的 active manifest 和 SHA256。

关闭条件：

- 三份文档和 JSON 对旧 manifest 的状态一致；
- 自动测试证明旧 SHA256 会被正式 launcher 拒绝；
- 新 active manifest 已绑定五条件 mixed dataset。

### A-002：五条件 mixed 数据链未完成

优先级：P0
状态：**阻塞**

当前状态：

```text
g0:   passed
rr05: passed
rr03: INVALID_DO_NOT_USE
p5:   not generated
p75:  not generated
```

`rr03` selection 中：

```text
expert requested: 0.45
expert actual:    0.564
absolute error:   0.114
audit threshold:  0.08
```

证据：

```text
logs/rwm_datasets_v13_mixed_full1000_snapshots_5conditions_realrange_20260724/
  rr03/INVALID_DO_NOT_USE.txt
  rr03/selected_25k/selection_report.json
```

当前没有 V13 数据采集进程，流水线在 rr03 审计失败后停止。

建议：

- 保留 rr03 无效标记和现有产物用于审计，不得训练；
- 检查按完整 env columns 选择时，transition-level collector mix 为何偏离；
- 修改 selection 算法或明确重新冻结合理容差，禁止只为通过而放宽阈值；
- 重新生成并审计 rr03；
- 完成 p5、p75；
- 五条件全部通过后再生成 active Sim manifest。

关闭条件：

- 五条件均有独立 dataset/hash、selection report 和 independent audit；
- 不存在 `INVALID_DO_NOT_USE.txt` 对应的正式输入；
- 五条件 exact-reset 与边界审计完成；
- active manifest 发布并由消费者 fail-closed 校验。

### A-003：AGENTS.md 指向不存在的强制文件

优先级：P0
状态：**阻塞**

`AGENTS.md` 要求完整读取：

```text
/home/xjy/Go2/EXPERIMENT_INTEGRITY_FIRST.md
```

该路径当前不存在；实际文件位于：

```text
/data1/xjy/unitree_rl_mjlab_v13/EXPERIMENT_INTEGRITY_FIRST.md
```

建议：

- 将 `AGENTS.md` 改为仓库内相对路径，或在要求的绝对路径部署同 hash 文件；
- 增加 preflight，验证强制文档存在且 SHA256 符合预期。

关闭条件：

- 新会话可严格按 `AGENTS.md` 路径读取完整原则文件；
- 路径和 hash 由 preflight 自动检查。

### A-004：V13 目录缺少 Git 元数据

优先级：P1
状态：**待处理**

当前：

```text
NO_GIT_METADATA
launch manifest:
  git.available=false
  git.reason=not_a_git_worktree
```

风险：

仅记录部分脚本 SHA256 不能完整表达代码版本、未跟踪文件和非预期修改，不满足完整
配置 diff 与可复现性要求。

建议：

- 优先把 V13 建立为明确 commit/branch 的 Git worktree；
- 若暂时不能恢复 Git 元数据，至少生成覆盖所有训练相关代码/配置的 immutable
  source manifest，并记录来源 commit、完整文件清单和 hash。

关闭条件：

- 每个正式 run 能记录 commit、工作区状态和目标配置 diff；
- 或存在经过审计的等价 immutable source manifest。

### A-005：learned scorer 晋级标准自相矛盾

优先级：P1
状态：**待处理**

`V13_TRACE_HANDOFF_ZH.md` 第 7 节说明 learned scorer 不强制超过 random 或 rule；
Phase D 第 5 步又要求 scorer 未优于 random 与 rule则不得晋升。

建议：

统一为一个明确的声明层级，例如：

```text
最低有效性：优于无 TRACE 父基线
选择器独立增益：进一步优于 matched random
相对 rule 增益：进一步优于 motion-first rule
```

并明确哪个层级允许进入正式两路线、哪个层级只影响论文中可宣称的结论。

关闭条件：

- 第 7 节、Phase D 和最终一句话原则使用同一晋级规则；
- 评测脚本能根据该规则给出机器可读结论。

### A-006：正式训练预算与公共超参数权威状态未统一

优先级：P1
状态：**待处理**

冲突包括：

- 主文档和 Real 子文档将正式训练预算列为未冻结；
- TRACE 文档写正式五条件预算为 `50M env steps`；
- 主文档要求正式 `gamma/n_step/reward` 启动前冻结；
- TRACE 文档直接指定 `gamma=0.99`、`n_step=3` 及历史 reward 信息。

建议：

- 在主文档建立唯一的“已冻结正式配置”表；
- 未冻结字段显式写 `TBD/BLOCKED`，禁止子文档提前赋值；
- 历史 smoke 值使用 `historical_validation_only` 标记；
- launcher 只读取冻结配置或带 hash 的 run manifest。

关闭条件：

- 主文档、Real 子文档、TRACE 文档无不同值；
- 正式预算、reward、gamma、n-step、sim ratio 和评测协议均有直接证据和 hash。

### A-007：已完成采集的 launch manifest 状态未闭环

优先级：P1
状态：**待处理**

g0、rr05 已生成并通过 selection/audit，但其 `launch_manifest.json` 仍记录：

```text
status: planned_pre_collection
```

建议：

- 保持原始 launch manifest 不可变；
- 另写 completion manifest，记录 completed/failed 状态、退出码、起止时间、产物
  SHA256、selection report 和 audit report；
- 总 manifest 只绑定 completion manifest，不根据目录存在推断成功。

关闭条件：

- 每个 condition 都有机器可读的最终状态；
- planned、completed、failed/invalid 的 artifact 不会被混淆。

### A-008：共享环境与测试入口不完整

优先级：P2
状态：**观察**

当前 `.venv` 最终解析到共享环境：

```text
/data1/xjy/mjlab_ws/unitree_rl_mjlab/.venv
```

该环境没有 `pytest`。V13 的部分测试可作为独立 Python 脚本运行，但标准
`python -m pytest` 不可用。

建议：

- 冻结 Python、Torch、MJLab 和关键依赖版本；
- 明确测试入口是独立脚本还是 pytest；
- 给正式 preflight 提供一条可重复执行的完整测试命令；
- 避免共享环境被其他项目更新后静默改变 V13 行为。

关闭条件：

- 环境 lock/hash 已记录；
- 完整测试入口可执行并保存结果；
- 正式 run manifest 记录解析后的环境版本。

## 5. Real 侧已知阻塞

以下内容来自 `V13_REAL_SIDE_EXPERIMENT_DESIGN_ZH.md`，尚未完成：

- CENet checkpoint 和 SHA256；
- CENet sim held-out 与 real-side 速度有效性验收；
- CENet 重算后的新 V13 Real 25K 路径和 SHA256；
- 现有 25K source-row/selection 独立 manifest；
- `tau_est -> actuator_force` 的字段、顺序、单位和坐标系合同；
- TRACE simbuffer artifact 与 synthetic 内 sim ratio；
- 正式训练预算和完整统一评测协议。

这些项目关闭前，不得启动正式 Real RWM 或 policy 训练。

## 6. 建议执行顺序

```text
1. 修复 A-003 强制原则文档路径
2. 统一 A-001 旧 manifest 的废止状态并加入自动拒绝
3. 统一 A-005/A-006 的权威实验语义
4. 修复 rr03 selection，完成 p5/p75 和五条件 exact-reset 审计
5. 发布新的 active Sim mixed-data manifest
6. 建立可追溯代码版本与最终 completion manifest
7. 冻结 Real CENet、Real dataset 和公共超参数
8. 两路线 smoke、配对 pilot、配置 diff 审计
9. 证据全部闭环后再启动正式训练
```

## 7. 更新要求

每次关闭问题时，必须同时补充：

```text
完成时间
修改文件
直接证据路径
artifact/config SHA256
执行命令
测试或审计结果
剩余限制
```

不得只把状态改为“已完成”而不提供证据。无效产物必须保留
`INVALID_DO_NOT_USE` 标识，并确保 launcher fail closed。
