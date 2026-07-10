# NEXT_ACTIONS

> 用途：新对话中可直接要求模型按优先级继续执行。更新时间：2026-07-10。所有动作都应保留 upstream baseline，禁止修改或覆盖旧日志与 checkpoint。

## 当前最高优先级

先补齐一条可复现的 formal-small 闭环，再决定是否扩大数据和训练步数。当前不能直接讨论模型是否有效，因为工作区没有数值产物。

## P0：确认外部输入与建立 formal-small 数据

需要提供或确认三个实际路径：upstream ET-Flow config、upstream checkpoint、processed GEOM 数据根目录（其下应有 `drugs/train`、`drugs/val`、`drugs/test`）。然后执行：

```bash
bash scripts/prepare_flexbond_formal_small.sh \
  <UPSTREAM_CONFIG> <UPSTREAM_CHECKPOINT> <PROCESSED_DATA_DIR>
```

预期生成：

- `data/upstream_formal_small/{train,val,test}/generated_files.pkl`
- `data/flexbond_cache_formal_small/{train,val,test}/`
- `data/flexbond_inference_formal_small/test/`
- `eval_manifest_formal_small.json`
- `data/formal_small_cache_summary.json`

验收条件：graph consistency 全部通过；inference cache 无 `x_ref`/target 标签；summary 中实际条数与 100/20/100 预设一致；分子身份与拓扑无位置配对。

## P1：运行数据、Jacobian 与等变性检查

在正式训练前至少执行：

```bash
python scripts/check_flexbond_data_pairs.py \
  --cache_dir data/flexbond_cache_formal_small --split train --num_samples 20
python scripts/check_flexbond_jacobian.py \
  --cache_dir data/flexbond_cache_formal_small --split train
python scripts/check_flexbond_graph_consistency.py \
  --cache_dir data/flexbond_cache_formal_small --split train
python scripts/check_flexbond_inference_no_labels.py \
  --cache_dir data/flexbond_inference_formal_small --split test
```

同时运行仓库测试：

```bash
pytest -q
```

验收条件：无 NaN；Jacobian 数值检查和梯度检查通过；label-free contract 通过；记录 skipped-too-small、rank-deficient、skipped-by-cap 的比例。

## P2：先做公平的 small baseline

在同一 train cache、seed 和 frozen manifest 上训练：

1. `cartesian_optimizer`
2. `flexbond4d_only_optimizer`（结构消融）
3. `flexbond4d_hybrid_optimizer`

先使用 `configs/flexbond_optimizer_egnn.yaml` 的 5,000-step 设置。可用 `scripts/run_flexbond_optimizer_smoke.sh` 自动跑 Cartesian 与 Hybrid；4D-only 需单独调用 `scripts/train_flexbond_optimizer.py --mode flexbond4d_only_optimizer`。

每个 run 必须保留 `config.resolved.yaml`、`run_provenance.json`、`metrics.csv`、`checkpoints/last.ckpt` 和最佳 checkpoint。训练结束先比较：

- `val/final_loss`、`val/cartesian_loss`
- `val/flexbond/q_loss`、`val/flexbond/corr_reg_loss`
- `corr_to_target_ratio`、`corr_to_residual_ratio`
- 有效键数、各类跳过计数、`q_star_nan_count`

停止条件：若出现 NaN、梯度爆炸或 4D correction 远大于 residual，先诊断，不直接扩大规模。

## P3：冻结 cohort 后扫描 checkpoint 与 alpha

不要只评估 `last.ckpt + alpha=1.0`。在同一个 `eval_manifest_formal_small.json` 上，使用：

- `scripts/sweep_flexbond_checkpoints.py`
- `scripts/sweep_flexbond_update_scale.py`

首轮 alpha 建议沿用已有 progressive 脚本：`0.1 0.2 0.5 1.0`；若出现明显坐标或键长不稳定，再加入 `--max_displacement 0.1` 对照。选择最佳设置时以低 `rmsd_mean` 为主、`failure_rate` 为次，并完整报告：

- upstream / Cartesian / FlexBond-4D 的 RMSD、COV-R/P、MAT-R/P
- `fraction_improved`、`fraction_worsened`、mean/median `delta_rmsd`
- update norm、clipping fraction、failure rate
- all、rotatable ≥3/5/6、初始 RMSD bins 的分组结果

## P4：验证训练时间窗是否匹配 refinement

比较同样训练步数和 seed 的：

- `t_min=0, t_max=1.0`（当前默认）
- `t_min=0, t_max=0.5`
- `t_min=0, t_max=0.25`

可复用 `scripts/run_flexbond_progressive_long.sh` 中 stage 3/4 的命名和命令。判断标准必须是冻结 test cohort 的 rollout 指标，不只看 validation flow loss。

## P5：通过 small gate 后扩大规模

只有同时满足以下条件才进入 formal-medium（500/100/200）或 20k：

- 数据和 label-free 检查通过。
- 至少两个独立 seed 趋势一致；正式结论建议 3 seeds。
- Hybrid 相对 upstream 与 Cartesian 的整体指标有稳定收益，且 `fraction_worsened` 和 failure rate 可接受。
- 收益不只来自单一高可旋转键子集。
- alpha/checkpoint 选择没有使用 test labels 做过度调参；最好另设 validation selection cohort。

formal-medium 准备入口：

```bash
bash scripts/prepare_flexbond_formal_medium.sh \
  <UPSTREAM_CONFIG> <UPSTREAM_CHECKPOINT> <PROCESSED_DATA_DIR>
```

## P6：正式评估与方法改进候选

规模扩大后，补跑带 atom-symmetry matching 的标准 RDKit/GEOM COV/MAT evaluator。若 Hybrid 仍无收益，按下列顺序排查：

1. `v_4d` 的方向是否与 Cartesian residual 对齐，而不只是范数过小/过大。
2. `q_star` 有效率、condition number 和键 frame 退化率。
3. `correction_scale` 与 alpha 是否重复缩小/放大更新。
4. 固定较小 affected side 和多键平均是否损失正确运动方向。
5. 局部时间窗是否优于全区间训练。
6. 最后才考虑扩大 backbone、改变 Jacobian 参数化或引入更复杂积分器。

## 每次新对话可直接使用的任务描述

```text
请先阅读 PROJECT_CONTEXT.md、EXPERIMENT_STATUS.md 和 NEXT_ACTIONS.md。
以仓库中的真实文件和实验产物为准，不要把脚本预设当成已完成结果。
当前先完成 NEXT_ACTIONS.md 中最高优先级且尚未完成的项目；保留 provenance，
不要覆盖旧日志/checkpoint，不要让推理接触 x_ref 或 q_star。
完成后更新 EXPERIMENT_STATUS.md 的实际规模、命令、checkpoint 和数值结果，
并同步勾清 NEXT_ACTIONS.md 中已完成项与下一阻塞项。
```
