# Go2 Online TRACE（V13）

当前目录实现两阶段 Go2 TRACE：先用 token-free rule bootstrap 验证整条工程链和策略收益，再切换到 `new_prompt` 的 LLM feedback + learned scorer 最终算法。RWM/FlashSAC 继续拥有 reward、n-step、actor/critic 与 loss。

```text
V13 active dataset V1 snapshots + current FlashSAC actor
  -> normal-dog imperfect auxiliary MJLab rollout
  -> Go2 V3 velocity/posture/control/contact/reward summary
  -> Stage 1: tracking/stability/normalized-return each 1/3, zero LLM calls
  -> Stage 2: fixed 80% within / 20% cross command-region LLM pairs
              + per-dataset BT scorer
  -> rule: command-mode-stratified top-alpha; learned: one global top-alpha
  -> canonical V13 reward + n-step
  -> mutable TRACE buffer
  -> TRACE only within the synthetic replay sub-batch
```

Stage 1 的规则只用于在花费 LLM token 前验证 reset、rollout、筛选、replay 和训练能否工作及是否有提升；Stage 2 的最终 selector 仍是 learned scorer。两阶段都不包含 same-state advantage、uncertainty/conflict/mixed active mode、random selector、per-start top-k 或 failure quota，score 也不会进入 reward、Q target、sample weight 或 loss。`gravity`、`friction`、`joint_noise` 不属于正式真机 mismatch：target/perfect 是目标狗，imperfect auxiliary simulator 是正常狗。

## 主要文件

| 文件 | 职责 |
|---|---|
| `simulator_reset.py` | V1 snapshot、buffer manual reset、native random test-only reset、no-auto-reset step、状态捕获。 |
| `source_sampler.py` | 从 hash/condition 匹配的 active V13 dataset 选择 V1 snapshot；保存独立 RNG。 |
| `proposal.py` | 从 buffer snapshot 恢复；只用 current actor distribution、`T=1` 采 branch；terminal 后停止。 |
| `trajectory.py` | 固定 V3 summary：三轴速度跟踪为首要信号，姿态/生存为 veto，reward/return 可见。 |
| `rule_bootstrap.py` | 规则阶段：response/tracking、stability、归一化 simulator return 各占 1/3；不调用 LLM。 |
| `go2_feedback_prompt.py` | offline/same/cross source-aware Go2 prompt 与严格 label contract。 |
| `feedback_pairs.py` | 不读 score/return；按前/后/左/右/纯转向/站立六区域构造 80% 区域内、20% 跨区域 pair，不在区域间均分。 |
| `label_feedback_with_codex.py` | batch label、合法 pair 部分保留、只 repair 缺失/非法 pair、可恢复落盘。 |
| `feedback_manager.py` | 在线 cumulative labels、去重、原子写、Codex provider。 |
| `scorer.py` | 固定 ordered feature + missing mask、2×256 ReLU BT scorer、强 checkpoint binding。 |
| `online_scorer_update.py` | cumulative BT refresh、connected-component split、CPU 训练、原子 checkpoint/switch。 |
| `selection.py` | 规则版按八种 command mode 分层做 top-alpha；learned 版保留 global top-alpha。 |
| `v13_replay_adapter.py` | 直接调用 canonical V13 的 `compute_go2_imagination_reward` 与 `_build_n_step`。 |
| `materializer.py` | selected trajectories 经注入的 V13 reward/n-step adapter 写 replay；score 仅审计 metadata。 |
| `replay.py` | mutable FIFO/ring TRACE buffer、with-replacement sampler、within-synthetic mixing、state_dict。 |
| `online_trace_manager.py` | proposal/feedback/rescore/select/materialize 生命周期和原子 checkpoint/resume。 |
| `v13_runtime_factory.py` | 创建 normal imperfect MJLab env，并连接 V13 trainer/current actor/replay mixer。 |
| `go2_trace_online.template.yaml` | Stage 1 rule-bootstrap 模板。 |
| `go2_trace_online_learned.template.yaml` | Stage 2 learned-scorer/LLM 模板。 |

旧 `rule_selection.py` 依赖过时 summary schema，仍为 legacy-only；正式 Stage 1 使用当前-schema `rule_bootstrap.py`，并按最新要求加入等权 reward。`build_trace_replay*.py` 和 `merge_trace_replays_v5.py` 不被正式模块导入。

## Canonical V13 绑定

当前适配源码：

```text
repo:   https://github.com/EzioPeter/unitree_rl_mjlab
branch: v13/fullstate-widecmd-25k
commit: b704130b848db1835b6a0102043f3b6cb0e46e03
```

选择的 V13 50M config 实际解析为 `gamma=0.99, n_step=3`。adapter smoke 已验证输出为 48D critic observation 和 normal condition 的 12D action；这些值从 config 读取，不在 TRACE 中另写公式。

## 训练入口

先填写 Stage 1 模板并跑 rule bootstrap。该模式强制 `initial_scorer_checkpoint=null`、`online_feedback_enabled=false`，不会创建 Codex provider。确认代码可运行且相对相同 baseline 有提升后，再为当前 task/dataset 生成 LLM labels/initial scorer，改用 Stage 2 模板。V13 trainer 统一通过：

```bash
python scripts/reinforcement_learning/rwm_flashsac/train_flashsac_world_model_go2.py \
  --config_path <V13_BASE_CONFIG> \
  --trace_config_path scripts/reinforcement_learning/rwm_trace/go2_trace_online.yaml
```

启动时会核对 dataset/reset/scorer/config hashes。TRACE buffer 为空时，synthetic 子批明确记录 `Replay/trace_warmup=1` 并使用原 RWM synthetic rows；一旦 buffer 非空，恢复配置的离散 TRACE 数量。checkpoint 同时保存 agent/RWM 状态、mutable buffer、scorer/schema、source/actor/mix RNG、feedback cursor 与 `COMPLETE` marker。

## 验证

```text
workspace regression: 93 passed
Stage B + V13 replay-mixer integration: 22 passed
canonical V13 reward/n-step adapter smoke: passed
```

没有自动启动长训练。正式 run 仍需填写模板中未冻结的 task-specific 超参数和真实 artifact 路径/hash。
