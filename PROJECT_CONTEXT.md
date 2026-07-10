# PROJECT_CONTEXT

> 用途：新开对话时，先把本文复制给模型。本文描述项目的稳定工程上下文；阶段性结果看 `EXPERIMENT_STATUS.md`，执行清单看 `NEXT_ACTIONS.md`。
>
> 更新时间：2026-07-10。项目实际根目录为 `E:\3dconformergenerationcode\4dadapter`；用户侧常用路径为 `~/Experiment/qdl/4dadapter`，下文均使用相对项目根目录的路径。

## 项目名称

FlexBond-4D Adapter / bond-local conformer refinement。

## 研究任务

本项目做的是 3D molecular conformer generation 的二次优化（refinement），不是 de novo 分子生成。输入是 upstream generator 已经生成好的 conformer，再用轻量 adapter 做二次修正，输出 refined conformer。当前 upstream 主要按 ET-Flow/GEOM-DRUGS 的数据格式接入，但 adapter 设计为 generator-agnostic。

```text
x_init = upstream generated conformer
G = molecular graph
x_ref = nearest reference conformer（仅训练/评估使用）
adapter predicts correction
x_new = x_old + alpha * correction
```

训练阶段先对每个 `x_init` 从同一分子的参考构象集合中选择 Kabsch 对齐后 RMSD 最小的 `x_ref`。推理阶段的 cache 不含 `x_ref`，不能使用真值或最小二乘伪标签。

## 当前主线模型结构

当前主线不是完整 ET-Flow Transformer，而是 `etflow/models/flexbond_optimizer.py` 中的 `FlexBondOptimizerLightningModule`，骨干网络位于 `etflow/models/components/light_egnn_refiner.py`。

- 输入：原子特征 `node_attr`（默认 10 维）、分子图 `edge_index`、边特征 `edge_attr`（默认 1 维）、当前坐标 `pos`、连续 refinement time `t`。
- 骨干：6 层轻量 E(n)-equivariant message passing；`hidden_dim=128`、`edge_hidden_dim=128`、time embedding 64 维、cutoff 10 Å、dropout 0。
- Cartesian 分支：每层产生基于相对坐标的等变向量，经过可学习 softmax 层权重合并为 `v_cart`。
- FlexBond-4D 分支：对符合条件的可旋转键，`q_head` 预测四个标量 `q_b=[s,w1,w2,w3]`；`etflow/commons/flexbond_jacobian.py` 用纯几何 Jacobian 将其映射为受影响原子的 Cartesian 修正 `v_4d`。
- 默认只处理 rotatable bonds；确定性选择较小一侧作为 affected side；每个分子最多 16 根键；affected atoms 少于 2 的键跳过；重叠键修正取平均。
- 默认主模式 `flexbond4d_hybrid_optimizer`：

```text
v_final = v_cart + correction_scale * v_4d
correction_scale = 0.01
```

- 对照模式：`cartesian_optimizer` 只用 `v_cart`；`flexbond4d_only_optimizer` 只用 `v_4d`。
- 推理：默认 10 步 Euler rollout；完整 rollout 后执行 `x_refined = x_init + update_scale * (x_rollout - x_init)`。命令行中 `--update_scale` 与 `--alpha` 是同一参数；可用 `--max_displacement` 做逐原子单步裁剪。

默认配置：`configs/flexbond_optimizer_egnn.yaml`。训练入口：`scripts/train_flexbond_optimizer.py`。推理入口：`scripts/sample_flexbond_optimizer.py`。

## 训练目标与 loss

训练使用直线路径 flow matching：

```text
t ~ Uniform(t_min, t_max)
x_t = (1-t) * x_init + t * x_ref
u_t = x_ref - x_init
L_final = MSE(v_final, u_t)
L_cart  = MSE(v_cart,  u_t)       # 仅诊断
```

Hybrid 模式中，先对真实 residual `u_t - stopgrad(v_cart)` 做 detached ridge least-squares，得到训练期伪标签 `q_b_star`：

```text
L_q   = MSE(q_b, q_b_star)（仅对有效键）
L_reg = mean(v_4d^2)
L = L_final + q_loss_weight * L_q + corr_reg_weight * L_reg
```

默认 `q_loss_weight=0.001`、`corr_reg_weight=0.0001`、`ridge_eps=1e-5`、`max_q_norm=10`、`max_condition=1e6`。`q_b_star` 只用于训练，推理严格只调用学习到的 `q_head`。Cartesian 模式的总 loss 就是 `L_final`；4D-only 模式为 `L_final + corr_reg_weight * L_reg`。

优化器为 AdamW，默认 `lr=2e-4`、`weight_decay=1e-6`、gradient norm clip 1.0。checkpoint 监控 `val/final_loss`。

## 数据与数据规模

数据集为 GEOM-DRUGS 风格的 upstream 生成构象和参考构象。每条训练 cache 保存分子身份、原子顺序与拓扑签名、`x_init`、全部参考构象、最近的 Kabsch 对齐参考 `x_ref_aligned`、可旋转键信息和 upstream provenance。跨文件匹配必须依赖 molecule id、精确有序 SMILES 等显式身份，禁止按列表位置配对。

当前仓库只包含代码、配置与测试，没有提交实际 `data/`、checkpoint、训练日志或评估 summary。因此应区分以下“脚本定义规模”和“已验证实际规模”：

- formal-small：`scripts/prepare_flexbond_formal_small.sh` 定义 train/val/test = 100/20/100 molecules，`sample_seed=12`。
- smoke：`scripts/run_flexbond_optimizer_smoke.sh` 使用最多 100 个训练分子、5,000 steps，并冻结最多 100 个测试分子的 manifest。
- manual small-20k：`scripts/run_flexbond_optimizer_20k.sh` 默认 1,000 个分子、20,000 steps。
- formal-medium：`scripts/prepare_flexbond_formal_medium.sh` 默认 train/val/test = 500/100/200 molecules。
- progressive 默认：`scripts/run_flexbond_progressive_long.sh` 使用 500/100/200，small 2,000 steps、medium 3,000 steps，并扫描 alpha `0.1 0.2 0.5 1.0`。

这些数字目前是脚本参数/计划规模，不代表本工作区中已有对应实验产物。

## 评估协议

- 先由 `scripts/export_flexbond_inference_cache.py` 导出无标签 test cache，再用 `scripts/build_flexbond_eval_manifest.py` 冻结同一 cohort。
- upstream、Cartesian adapter、FlexBond-4D adapter 必须使用完全相同的 manifest 和 `x_init_hash`。
- `scripts/eval_flexbond_optimizer.py` 报告 Kabsch RMSD、COV-R/COV-P、MAT-R/MAT-P、`fraction_improved`、`fraction_worsened`、更新范数和 failure rate，并按可旋转键数、初始 RMSD、更新范数、alpha、checkpoint 分组。
- 当前轻量 evaluator 不做原子对称置换；正式报告还应运行仓库既有的 RDKit/GEOM COV/MAT benchmark。
- 采样稳定性额外检查坐标有限性、坐标范数上限和键长比例（默认要求 refined/original 在 0.3 到 3.0 之间）。

## 已知问题与边界

- 当前工作区没有可核验的真实数值结果，不能声称 adapter 已优于 upstream 或 Cartesian baseline。
- 最近参考构象选择和轻量评估只做 Kabsch 对齐，不做 atom-symmetry matching，可能高估 RMSD。
- 当前仅 Euler integration；只显式建模可旋转键且固定影响较小一侧。
- 多根键影响同一原子时简单平均；退化几何和秩亏键会被跳过。
- Hybrid 的 `q_b_star` 是依赖真值 residual 的训练期伪标签；任何推理路径调用它都属于标签泄漏。
- 全区间 `t in [0,1]` 的训练分布未必匹配“从 `x_init` 附近开始的小幅 refinement”；仓库已提供局部时间窗 `t_max=0.25/0.5` 的对照计划。
- update scale 过大可能令 RMSD、键长或坐标稳定性恶化，需要在冻结 cohort 上扫描 alpha，并同时报告失败率和改善/恶化比例。
- 旧的 `scripts/train_jacobian_4d.py`、`configs/drugs-so3-jacobian-4d-bs4.yaml` 属于基于完整 ET-Flow velocity field 的另一条分支，不应与当前轻量 FlexBond optimizer 的默认参数混写。

## 关键文件

- `configs/flexbond_optimizer_egnn.yaml`：当前主配置。
- `etflow/models/flexbond_optimizer.py`：三种 optimizer mode、loss、Euler refinement。
- `etflow/models/components/light_egnn_refiner.py`：轻量 EGNN 与 `q_head`。
- `etflow/commons/flexbond_jacobian.py`：目标键选择、4D Jacobian、`q_star` ridge solve。
- `scripts/build_flexbond_init_cache.py`：训练 cache 构建。
- `scripts/train_flexbond_optimizer.py`：训练。
- `scripts/sample_flexbond_optimizer.py`：无标签 refinement。
- `scripts/eval_flexbond_optimizer.py`：公平三路评估。
- `scripts/sweep_flexbond_update_scale.py`、`scripts/sweep_flexbond_checkpoints.py`：alpha/checkpoint 扫描。
- `scripts/run_flexbond_progressive_long.sh`：分阶段实验编排。

