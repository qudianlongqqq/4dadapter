# Gated Molecular Kinematic Flow 工程说明

## 数据流

```text
(G, x_t, t)
  -> GatedKinematicBackbone
  -> v_cart_raw + [gate_logit, raw_rate] per rooted joint
  -> gate = sigmoid(gate_logit)
  -> bounded_rate = torsion_rate_scale * tanh(raw_rate)
  -> effective_rate = gate * bounded_rate
  -> v_kin = J_tau(x_t) effective_rate
  -> v_residual = (I - P_J) v_cart_raw
  -> v_final = v_residual + v_kin
```

训练标签使用 `build_dense_jacobian()` 和 exact SVD column basis；稳定 rate target 使用全局 damped solve。二者在代码和日志含义上严格区分。无标签推理使用 matrix-free `Jz/J^T v` 和 CG 投影，不构造 dense Jacobian，也不读取 reference、rate target 或 gate target。

## 数学到函数映射

| 数学对象 | 实现 |
|---|---|
| rigid fragments / rooted joint tree | `build_molecular_kinematic_topology` |
| `J_tau z` | `apply_jacobian` |
| `J_tau^T v` | `apply_jacobian_transpose` |
| dense `J_tau` | `build_dense_jacobian` |
| `Q`, rank, singular values | `compute_column_basis` |
| exact `u_kin_star/u_res_star` | `decompose_target` |
| damped global rate target | `damped_global_rate_target` |
| soft activity target | `soft_gate_target` |
| gate/rate/final velocity | `GatedKinematicFlowLightningModule.forward` |

## 模式兼容

`etflow.models.motion_factory.build_motion_model()` 支持：

- `motion_mode: gated_global_torsion_kinematic`
- `motion_mode: legacy_flexbond4d`
- `motion_mode: cartesian`

旧 `FlexBondOptimizerLightningModule`、旧配置的 `mode` 和旧 checkpoint 参数键没有删除或改名。新 checkpoint 使用独立的 `GatedKinematicFlowLightningModule`，避免被 legacy 类静默误加载。

## 复杂度

- legacy 4D head：每 joint 输出 4 个标量；新 head：每 joint 输出 2 个标量。
- matrix-free Jacobian：`O(sum_b |affected_b|)` 时间，`O(N+M)` 主要工作内存。
- inference CG projection：上述成本乘 `projection_cg_iterations`，不创建 `[3N,M]`。
- training exact SVD：需要 `[3N,M]`，仅用于带标签 target；近似复杂度 `O(3NM + min(3N,M)M^2)`。
- 可用 `scripts/benchmark_gated_kinematic.py` 在目标机器生成实际参数量和 synthetic forward timing。

## 安全边界

非树 fragment graph、disconnected graph、缺失 rotatable edge、过少原子都会返回明确 topology status，并令运动学分支退化为零。极短 bond 的 geometry valid mask 为 false。多个 joint 的原子速度使用 scatter-add，不做 overlap 平均。
