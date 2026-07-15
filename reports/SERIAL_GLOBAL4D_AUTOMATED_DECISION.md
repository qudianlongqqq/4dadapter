# Serial Global4D 自动决策

最终结论：**DECISION_D — Confirm30 pilot 有效，可以扩大 validation 和训练规模。**

1. Oracle 明确证明 Cartesian 剩余误差中存在可利用的 Global4D 结构；最佳
   lambda 为 1.0，RMSD 从 1.394668 理论下降到 0.917641。
2. Oracle 的 60/60 record、30/30 molecule 都改善，不是少数异常样本驱动；
   high-flex 同样稳定获益。
3. projection energy ratio 为 0.4259；Oracle 分解中 torsion 贡献最大，其次
   bending，stretch 最小。pilot 学习结果中 stretch-only 当前最强，说明 learned
   head 尚未复现完整 Oracle 模式分配。
4. Stage 2 cache 包含 5,001 train records / 1,667 molecules 和固定 Confirm30
   60 records / 30 molecules；train/val/test 身份交叉均为零，且有原子完成门禁。
5. RTX 5080 的 MAX_OOM_FREE/MAX_SAFE/MAX_THROUGHPUT 都是 128；吞吐 95%
   规则推荐 batch 96、accumulation 1、effective batch 96、lr 2e-4。
6. bs8 与 auto96 都出现 Phase A 合格 checkpoint；auto96 step1500 的 positive
   gain 为 60%，internal MSE 低于 zero predictor，故被 validation 选中。
7. learned `Jq` 仅达到 target norm 的约 12%，距 Oracle 仍远；这是扩大训练的
   主要空间与风险。
8. Gate 把 negative-gain fraction 从 40% 降到 35%，没有塌缩；high-flex gate
   响应最高。
9. trust region、backtracking、reject 完全 label-free。Confirm30 无 reject/failure；
   两步中 20% record 发生 clipping、10% 发生 backtracking。
10. 完整安全的一步 RMSD 为 1.394399，两步为 1.393855；两步更好且 high-flex
    未恶化，因此推荐两步。
11. 两步优于 Cartesian，但幅度很小，不能声称已经接近 Oracle。下一阶段应扩大
    train/validation 并改善 batched Jacobian/CPU topology 吞吐，不应直接运行 test。

本轮没有启动 formal-large Serial 或 200k Serial 训练，也没有使用 test 选参。
