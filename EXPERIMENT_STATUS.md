# EXPERIMENT_STATUS

> 用途：记录“已经有证据完成了什么”，避免把代码存在、脚本预设或未来计划误当成实验结果。更新时间：2026-07-10。

## 当前结论摘要

代码层面的 FlexBond-4D Adapter 主线已经具备：数据身份校验、轻量 EGNN、Cartesian/4D/hybrid 三种模式、训练期 `q_star` 伪标签、无标签推理、稳定性检查、公平 cohort 评估以及 alpha/checkpoint 扫描。

但是，当前项目工作区没有 `data/`、`logs_*`、`diagnostics/`、checkpoint、`metrics.csv`、`summary.csv` 或 `sweep_summary.csv`。因此目前没有可复现、可引用的训练 loss 数值或 RMSD/COV/MAT 数值，也没有证据支持“FlexBond-4D 已超过 upstream/Cartesian”。

## 当前模型与配置状态

主实验配置是 `configs/flexbond_optimizer_egnn.yaml`：

| 项目 | 当前值 |
|---|---:|
| 默认模式 | `flexbond4d_hybrid_optimizer` |
| EGNN 层数 / hidden | 6 / 128 |
| time embedding | 64 |
| cutoff | 10.0 |
| `correction_scale` | 0.01 |
| `q_loss_weight` | 0.001 |
| `corr_reg_weight` | 0.0001 |
| `ridge_eps` | 1e-5 |
| 每分子最多目标键 | 16 |
| batch size / grad accumulation | 4 / 2 |
| optimizer / lr | AdamW / 2e-4 |
| 默认训练步数 | 5,000 |
| 默认 refinement steps | 10 |
| 默认 alpha | 1.0 |

主 loss 为 `L_final + 0.001 L_q + 0.0001 L_reg`。其中 `L_final=MSE(v_cart+0.01*v_4d, x_ref-x_init)`；`L_q` 监督 `q_head` 拟合 detached ridge solve 的 `q_star`；`L_reg=mean(v_4d^2)`。

## 数据状态与规模

当前工作区没有实际 cache，无法从文件核验样本条数、有效可旋转键分布或 train/val/test 去重情况。仓库中可执行的预设规模为：

| 层级 | train | val | test | 训练步数 | 状态 |
|---|---:|---:|---:|---:|---|
| formal-small | 100 | 20 | 100 | 未在准备脚本中固定 | 脚本已定义，产物缺失 |
| smoke | 最多 100 | cache 内 val | 最多 100 | 5,000 | 流程已定义，产物缺失 |
| small-20k | 默认 1,000 | cache 内 val | 未自动评估 | 20,000 | 手动脚本已定义，产物缺失 |
| formal-medium | 500 | 100 | 200 | progressive 默认 3,000 | 脚本已定义，产物缺失 |

不要把上述预设写成“已经训练了 N 个分子”。只有 cache summary、run provenance 和日志存在后才能更新为实际规模。

## 实验结果

### 已确认的工程结果

- 训练和推理职责分离：`sample_flexbond_optimizer.py` 从 label-free `FlexBondInferenceDataset` 读取数据，不调用 `q_star` least-squares。
- 公平比较协议已编码：manifest、sample id、`x_init_hash` 和 reference-set 一致性会被检查。
- 三个 ablation mode 已实现：Cartesian-only、FlexBond-4D-only、Hybrid。
- 训练会保存 resolved config、run provenance、top-3 checkpoint、last checkpoint 和 CSV metrics。
- 测试目录覆盖 data contract、loss contract、无标签推理、Jacobian、等变性相关几何、评估公平性和 optimizer 行为。

### 尚无数值可报告的指标

以下项目在当前工作区均为“未获得/未提交”，不是 0：

| 指标 | upstream | Cartesian adapter | FlexBond-4D hybrid |
|---|---:|---:|---:|
| mean RMSD | 未获得 | 未获得 | 未获得 |
| COV-R / COV-P | 未获得 | 未获得 | 未获得 |
| MAT-R / MAT-P | 未获得 | 未获得 | 未获得 |
| fraction improved | 基线不适用 | 未获得 | 未获得 |
| fraction worsened | 基线不适用 | 未获得 | 未获得 |
| failure rate | 未获得 | 未获得 | 未获得 |
| best `val/final_loss` | 不适用 | 未获得 | 未获得 |

## 已知问题

1. 缺少实际数据、日志和 checkpoint，是当前最直接的实验阻塞项。
2. Kabsch 评估没有 symmetry permutation，正式 RMSD/COV/MAT 可能与标准 GEOM/RDKit evaluator 有偏差。
3. `correction_scale=0.01` 与 rollout `alpha=1.0` 都是默认值，尚无本工作区结果证明其最优。
4. 全时间区间 flow matching 可能与局部 refinement 推理分布不一致，需要比较 `t_max=0.25`、`0.5`、`1.0`。
5. 需要报告整体以及 `rotatable_ge_3/5/6` 子集；只报对 FlexBond 有利的高可旋转键子集会造成选择性结论。
6. 需要同时观察 `fraction_improved` 与 `fraction_worsened`，平均 RMSD 改善可能掩盖少量严重退化。
7. 退化 frame、affected side 太小、Jacobian 秩亏和每分子 16 键上限都会跳过部分键，必须从日志统计其频率。
8. 当前轻量 EGNN 只用标量特征、距离和相对坐标；是否有足够表达能力需要用 Cartesian baseline 和 4D-only ablation 验证。

## 结果更新规则

后续每次实验至少记录：git commit、`config.resolved.yaml`、cache summary/hash、manifest、seed、checkpoint、训练步数、最佳 `val/final_loss`、alpha、clipping 参数、整体与分组 RMSD/COV/MAT、改善/恶化比例、failure rate。若缺任一关键 provenance，不把该次结果作为正式结论。

