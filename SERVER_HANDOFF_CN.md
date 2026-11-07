# 5090服务器交接

本分支提交审计代码、测试、已有实验清单和轻量结果。没有启动训练；也没有要求重新训练全部历史模型。

## 获取代码

```bash
git clone --branch audit/review-gaps-20261106 --single-branch git@github.com:qudianlongqqq/4dadapter.git pina-review-audit
cd pina-review-audit
```

已有clone可执行 `git fetch origin` 后为远端分支建立独立worktree，保留原工作区。

## 检查现有服务器数据

继续使用已有环境，不自动安装/升级CUDA、PyTorch或RDKit。先设置真实资产根目录和冻结GenBench checkout：

```bash
export PINA_ASSET_ROOT=/absolute/path/to/3dconformergenerationcode
export PINA_GENBENCH_REPO=/absolute/path/to/genbench3d
mkdir -p review_outputs
python scripts/review_server_preflight.py --out review_outputs/server_inventory.json
python -m pytest tests/test_review_interventions.py tests/test_review_reference_validation.py tests/test_review_asset_paths.py -q
```

`PINA_ASSET_ROOT`下需要保留原逻辑相对路径。新入口会在读取时重定位冻结manifest里的Windows绝对路径，不改写原文件，不伪造原hash。若服务器原有目录结构不同，先根据预检输出建立清晰的资产映射；不要替换数据后沿用旧hash。

可用 `--hash-check` 对现有关键大文件核验。模型环境与外部V3D环境可以分开。正式V3D运行还需要冻结适配器、LigBoundConf、KDE、GenBench checkout；只有Git代码不等于这些数据和权重已经上传。

## 执行顺序

1. 先读 `REVIEW_EXPERIMENT_INVENTORY_CN.md`，排除重复训练。
2. 预检服务器已有资产。核心TRAIN/VAL prepared约1.97GB、Source binding约175MB；它们不包含在Git提交中。权重和其他Source/Reference文件也继续复用原资产。
3. 核实TRAIN Reference来源，以及raw/standardized cohort标签。当前预检只核对工程资产，不能自动证明原始几何身份。
4. 冻结训练矩阵与评估协议后再启动必要的GPU实验。这个分支不是一键重训全部历史实验的调度器。

现有审计脚本的完整分阶段命令见 `REVIEW_AUDIT_README.md`。先复现小样本原生结果，再运行新干预；不要跳过V3D parity直接汇报跨平台数值。
