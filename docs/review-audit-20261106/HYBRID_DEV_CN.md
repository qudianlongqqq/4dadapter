# PINA 与 MMFF 混合动作开发集结果

2026-11-08。在同一条 Source 上分别对 PINA 解析方向和 MMFF94s 负梯度去除刚体分量、按 RMS 归一化。混合方向是 25% PINA 加 75% MMFF，随后再次归一化。三个冻结 PINA seed 分别参与比较。权重和主位移预算 0.01 Å 均在查看结果前固定。

从原 ETFlow DEV 按分子身份哈希排序固定 256 个分子，每分子保留两条 Source。三组使用相同的 RMS 预算、0.10 Å 单原子限位、立体化学回退和四位小数坐标精度。预留 0.00018 Å 量化余量。512 条 Source 全部支持 MMFF，11,264 个 V3D 评估无失败。Source V3D 为 45.12%。

| RMS 预算 | 原 PINA 方向 | MMFF94s 单步 | 混合方向 | 混合减 MMFF，百分点和 95% 区间 |
|---:|---:|---:|---:|---:|
| 0.00432154 Å | 54.82% | 58.98% | 59.44% | +0.46 [−0.91, +1.89] |
| 0.010 Å | 51.43% | 60.94% | 63.28% | +2.34 [+0.39, +4.36] |
| 0.020 Å | 15.36% | 16.60% | 19.08% | +2.47 [+0.07, +4.95] |

区间来自 10,000 次分子聚类 bootstrap，先在每个分子内平均两条 Source 和固定 seed。0.01 Å 下三个混合 seed 分别达到 63.09%、63.28%、63.48%。三组实际平均 RMS 均约 0.00982 Å。混合组原来通过后变为失败的构象占全体约 4.88%，MMFF 组约 6.05%。三组各预算均未观察到所检查的四面体或 E/Z 翻转。0.02 Å 下三组 V3D 均低于 Source，表现出明显过冲。

0.01 Å 下，混合、MMFF、PINA 的 MMFF94s 能量变化中位数分别为 −4.34、−4.59、−1.20 kcal/mol。该能量与 MMFF 对照使用相同势函数。对应的单构象 CPU 动作耗时中位数分别为 6.48、2.02、4.98 ms，包含已有图上的推理或力场初始化和保护，排除文件读取和 V3D 评估。混合方向通过预设的开发集 V3D 门槛，速度与 MMFF 能量指标仍有代价。适合在独立队列复验固定原型，再决定是否训练自适应动作权重。

在现有资产根目录和冻结 GenBench3D 环境下执行以下三个阶段。服务器通过 `PINA_ASSET_ROOT` 和 `PINA_GENBENCH_REPO` 配置原有资产位置。

```bash
# 模型环境
python scripts/run_hybrid_action_dev.py prepare --out review_outputs/hybrid_dev_v1
# 冻结 V3D 环境
python scripts/run_hybrid_action_dev.py evaluate --out review_outputs/hybrid_dev_v1
# 模型环境
python scripts/run_hybrid_action_dev.py summarize --out review_outputs/hybrid_dev_v1
```

输出包括 `PROTOCOL.json`、`METHOD_SUMMARY.csv`、`SEED_V3D.csv`、`PAIRED_V3D.csv`、`RESULT.json` 和逐条 parquet。开发结果以固定数据和权重为准，CPU 耗时会随机器与负载变化。
