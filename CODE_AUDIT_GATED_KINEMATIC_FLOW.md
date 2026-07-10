# Gated Molecular Kinematic Flow 严格代码审查

审查日期：2026-07-11。范围为本次新方法代码；未运行正式数据。`severity=none` 表示已检查且当前实现有明确防护，`low/medium/high` 表示仍需在 Linux 完整依赖环境或真实数据上验证的风险。

| # | 检查项 | severity | file:line | trigger | consequence | minimal fix | 是否使实验结论失效 |
|---:|---|---|---|---|---|---|---|
| 1 | 数学公式与代码一致 | none | `etflow/models/gated_kinematic_flow.py:133-150` | 正常 forward | `sigmoid`、`tanh`、乘积、`v_residual+v_kin` 与规格一致 | 无 | 否 |
| 2 | joint 顺序与 head 顺序 | none | `etflow/commons/molecular_kinematics.py:150-180` | BFS 构造 joint | topology 和 head 均使用同一个 BFS 顺序 | 无 | 否 |
| 3 | parent-child 方向 | none | `etflow/commons/molecular_kinematics.py:156-163` | 原始 bond 顺序反转 | 方向由 rooted fragment tree 决定，不依赖输入顺序 | 保留 reversal 单测 | 否 |
| 4 | torsion 符号 | medium | `etflow/commons/molecular_kinematics.py:172` | 原子重编号导致等大 root tie 改变 | `orientation_sign` 被记录，但模型物理方向以 rooted axis 为准；等大片段重编号可能换 root | 正式数据前增加 RDKit atom permutation property test，必要时引入稳定 atom-map root key | 是，若不同预处理得到相反运动 |
| 5 | overlap 平均 | none | `etflow/commons/torsion_kinematic_jacobian.py:69` | 多 joint 影响同一原子 | 使用 `index_add_` 求和，无 count 除法 | 无 | 否 |
| 6 | batch atom offset | low | `etflow/models/gated_kinematic_flow.py:89-94` | 非标准 batch 节点不连续 | 当前假设 PyG 按图连续拼接；自定义 interleaved batch 会错误 | 若未来支持自定义 batch，改为显式 local/global index map | 是，仅对非标准 batch |
| 7 | batch joint offset | none | `etflow/models/gated_kinematic_flow.py:109-149` | 多分子 batch | 每个图独立 topology，head 输出按图顺序 concat，不共享 joint index | 无 | 否 |
| 8 | 分子间索引串联 | none | `etflow/models/gated_kinematic_flow.py:90-94` | 多图边集合 | edge/rotatable bond 先按 atom batch 过滤再局部化 | 保留多图测试作为后续补充 | 否 |
| 9 | SVD 梯度与 detach | none | `etflow/models/gated_kinematic_flow.py:176-188` | 训练 target 构造 | exact/damped target 均 detach；监督 target 不反传 SVD | 无 | 否 |
| 10 | gate/rate 不可辨识 | low | `etflow/models/gated_kinematic_flow.py:195-204` | gate 和 rate 互相补偿 | 有 supervised gate、sparse、binary 和 bounded-rate 正则，但不可辨识不能完全消除 | 监控 gate/rate 联合分布；必要时调权重但不要事后挑 test | 可能 |
| 11 | gate 全塌缩到 0 | medium | `etflow/models/gated_kinematic_flow.py:207-215` | sparse 权重压过监督 | 已记录 mean/median/active/near-zero；默认监督权重大于 sparse | 设置训练中告警阈值并检查早期曲线 | 是，运动学分支将无效 |
| 12 | gate 全塌缩到 1 | medium | `etflow/models/gated_kinematic_flow.py:207-215` | gate supervision 或 rate target 普遍偏大 | 已记录 near-one 和 active fraction | 用 basis oracle 校准 threshold/temperature | 可能，稀疏性结论失效 |
| 13 | rate 补偿 gate 而爆炸 | low | `etflow/models/gated_kinematic_flow.py:138` | gate 很小 | `tanh` 严格限制 bounded rate，另有 rate regularizer | 无 | 否 |
| 14 | inference 标签泄漏 | none | `scripts/sample_gated_kinematic_flow.py:40-45` | 正式采样 | 只实例化 `FlexBondInferenceDataset`；forward/refine 无 reference 参数 | 保留 label-free contract 测试 | 否 |
| 15 | topology cache 复用几何 | none | `etflow/models/gated_kinematic_flow.py:222-248` | 多步 rollout | 当前甚至每步重建 topology；坐标和 Jacobian必然重算 | 后续若加 cache，只缓存 topology dataclass | 否 |
| 16 | rollout 使用旧 Jacobian | none | `etflow/models/gated_kinematic_flow.py:228-232` | 每个 Euler step | forward 接收当前 `x` 并重新调用 matrix-free Jacobian | 无 | 否 |
| 17 | M=0 | none | `etflow/models/gated_kinematic_flow.py:120-123` | 无可旋转键/不安全 topology | projection、kinematic velocity 为零，全部交给 Cartesian residual | 无 | 否 |
| 18 | rank deficient | low | `etflow/commons/kinematic_projection.py:24-34` | 重复/退化 Jacobian 列 | exact SVD rank-aware；damped target 加 ridge | 真实数据报告 rank 分布 | 否，若正确分组报告 |
| 19 | macrocycle | none | `etflow/commons/molecular_kinematics.py:126-134` | 切键后 fragment graph 非树 | 明确 `non_tree_fragment_graph`，无 joint，退化 Cartesian | 无 | 否 |
| 20 | branch | none | `etflow/commons/molecular_kinematics.py:142-168` | 多子 fragment | rooted tree descendant 集合逐支计算 | 保留 branch 单测 | 否 |
| 21 | disconnected molecule | none | `etflow/commons/molecular_kinematics.py:105-108` | 原始图多 component | 返回 `disconnected` 且无 joint | 无 | 否 |
| 22 | mixed precision | low | `etflow/commons/kinematic_projection.py:28-34` | fp16/bf16 SVD/solve | 已提升到 fp32 求解再转回；极病态系统仍可能精度不足 | basis diagnostic 可选择 fp64 CPU 复核尾部样本 | 可能影响极病态子集 |
| 23 | NaN/Inf | low | `etflow/models/gated_kinematic_flow.py:240-248` | 网络或坐标非有限 | rollout 会停止并报告；训练尚未主动 skip 非有限 loss | 增加 trainer finite-loss callback 或 fail-fast | 是，若训练已出现非有限值 |
| 24 | 旧 checkpoint 兼容 | none | `etflow/models/components/light_egnn_refiner.py:99-145` | 加载旧 FlexBond checkpoint | 只新增无参数 `encode()`，旧参数名/形状未改变；新类独立 | 在原训练环境实际加载一份旧 ckpt | 否，待实测确认 |
| 25 | partial 被当 completed | none | `etflow/commons/run_state.py:20-34` | 中断/恢复/失败 | JSON 原子写；互斥 marker；只有 expected outputs 存在才 completed | 无 | 否 |

## 额外发现

| severity | file:line | trigger | consequence | minimal fix | 是否使实验结论失效 |
|---|---|---|---|---|---|
| medium | `etflow/commons/torsion_kinematic_jacobian.py:91-125` | inference 中 CG 提前停止 | `v_residual` 对运动学空间只是数值近似正交，不应宣称机器精度 exact | 报告 CG residual；对诊断使用 dense exact projector | 若把 inference 分解称作 exact，则是 |
| medium | `etflow/commons/torsion_kinematic_jacobian.py:20-26` | uniform Cartesian translation | rooted child-side torsion column可能带非零整体 COM 速度，因此 translation 与列空间不保证严格正交 | 正式数据已 Kabsch 对齐；若要理论刚体分离，需明确引入 COM-constrained basis，不能静默改公式 | 对未对齐数据是 |
| low | `scripts/diagnose_gated_kinematic_basis.py:37-42` | 大 M 的 global-4D oracle | 逐列构造 dense basis，时间/内存较高 | 仅诊断小批使用，增加内存阈值 | 否 |
| low | `etflow/models/gated_kinematic_flow.py:106-149` | 长训练 | topology 每次 forward 重建，未利用允许的 topology cache | 按 graph topology signature 缓存 dataclass，绝不缓存坐标/Jacobian | 否，只影响性能 |

## 审查结论

核心公式、全局 joint solve、无 overlap 平均、标签隔离和 fail-closed topology 已落实。当前不得直接宣称方法有效；在接受实验结论前，必须在完整 Linux 环境通过旧 checkpoint 加载、atom permutation、multi-graph batch、mixed-precision、synthetic backward 和真实 cache basis oracle。尤其应在报告中把训练 exact SVD target 与 inference matrix-free CG 投影明确区分。
