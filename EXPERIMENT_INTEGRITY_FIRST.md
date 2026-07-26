# Unitree／Go2 大项目实验最高原则：正确、有效、可复现

更新时间：2026-07-24

本文件适用于整个 Unitree／Go2 大项目，包括但不限于数据采集、RWM、BC、TRACE、
sim、real、MJLab、MuJoCo、部署以及 V9、V11、V12 等版本。实验结果的正确性、
有效性、可解释性和可复现性，永远优先于启动速度、GPU 利用率、并行数量、训练
进度和尽快产出结果。

## 1. 禁止猜测关键配置

以下任一设置没有从当前权威文档、目标对照实验的落盘配置或 artifact manifest
得到直接证据时，必须停止，不得根据历史会话、目录名、经验或“应该是”进行推断：

- dataset 路径、规模、condition 与 hash；
- RWM checkpoint、输入输出维度与 loss 监督语义；
- reward 版本及全部权重；
- policy 初始化、resume 模式与 optimizer 状态；
- imagination horizon；
- real/RWM/sim buffer 比例；
- BC、RWM、TRACE 是否启用；
- seed、训练预算、学习率、评测协议；
- simulator condition、DR、摩擦、关节强度和 payload。

## 2. 启动前必须完成配置审计

每个实验启动前必须生成并检查：

1. 目标问题和对照实验的唯一标识；
2. 参考实验的原始落盘配置；
3. 新实验相对参考实验的完整逐项 diff；
4. 预期唯一变量，以及所有必须保持不变的变量；
5. dataset、RWM、policy、replay 等 artifact 的绝对路径和 SHA256；
6. 维度、监督语义、reward、初始化、训练预算和设备的 fail-closed 断言；
7. 输出目录不存在或为空，禁止覆盖既有结果。

如果 diff 中出现任何未计划变化，禁止启动。

## 3. 禁止把“复用 artifact”扩展成“继承整套配置”

复用 RWM checkpoint 不代表允许继承其历史 policy reward、horizon、初始化或其他
训练配置。每个字段必须分别确认。尤其禁止因为复用旧 RWM 而静默继承旧 reward。

## 4. 证据优先级

发生冲突时，按以下顺序判定：

1. 用户对当前实验的最新明确指令；
2. 当前权威路线文档；
3. 目标参考实验实际落盘的 config、manifest 和 hash；
4. 代码默认值；
5. 历史会话和记忆。

低优先级信息不得覆盖高优先级信息。历史决定在路线发生变化后不得自动沿用。

## 5. 运行与结果审计

- 每个 run 必须保存 launch manifest、最终解析配置、代码版本/hash 和 artifact hash；
- 启动后必须再次检查实际进程参数和落盘配置，而不能只检查 launcher；
- 错误实验必须立即停止并写入 `INVALID_DO_NOT_USE.txt`，不得进入汇总或评测；
- 没有完成统一评测与行为检查的 checkpoint，不得宣称“有效”或“改进”；
- 训练指标、MJLab、MuJoCo 和实机结果必须明确区分，禁止互相替代。

## 6. 默认行为

当正确性与继续执行发生冲突时，默认停止并报告证据缺口。宁可晚启动，也不运行
配置不确定的实验；宁可没有结果，也不保留不可解释的结果。
