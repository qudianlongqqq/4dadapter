#!/usr/bin/env python
from __future__ import annotations
import argparse, hashlib, importlib, json, os, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import torch

PACKAGES={"torch_geometric":"torch_geometric","lightning":"lightning","pytorch_lightning":"pytorch_lightning","rdkit":"rdkit","numpy":"numpy","scipy":"scipy","pandas":"pandas","pyarrow":"pyarrow","scikit_learn":"sklearn","pyyaml":"yaml","networkx":"networkx","pytest":"pytest"}
def version(name):
    try:
        module=importlib.import_module(name); return str(getattr(module,"__version__",getattr(module,"VERSION","unknown")))
    except Exception as exc: return f"UNAVAILABLE:{type(exc).__name__}"
def git(*args): return subprocess.run(["git",*args],capture_output=True,text=True,check=True).stdout.strip()
def capture(output:Path,freeze_path:Path|None=None):
    freeze=subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True,check=True).stdout
    if freeze_path: freeze_path.parent.mkdir(parents=True,exist_ok=True); freeze_path.write_text(freeze,encoding="utf-8")
    gpus=[{"index":i,"name":torch.cuda.get_device_name(i),"compute_capability":list(torch.cuda.get_device_capability(i))} for i in range(torch.cuda.device_count())]
    report={"schema_version":"mcvr-environment-v1","timestamp":datetime.now(timezone.utc).isoformat(),"platform":platform.system(),"os":platform.platform(),"python":platform.python_version(),"python_full":sys.version,"executable":Path(sys.executable).name,"git":{"branch":git("branch","--show-current"),"commit":git("rev-parse","HEAD"),"dirty":bool(git("status","--porcelain"))},"torch":torch.__version__,"torch_cuda_runtime":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"cudnn_version":torch.backends.cudnn.version(),"gpu_count":len(gpus),"gpus":gpus,"packages":{key:version(module) for key,module in PACKAGES.items()},"environment_variables":{key:os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES","CUBLAS_WORKSPACE_CONFIG","PYTHONHASHSEED","OMP_NUM_THREADS","MKL_NUM_THREADS")},"pip_freeze_sha256":hashlib.sha256(freeze.encode()).hexdigest(),"default_dtype":str(torch.get_default_dtype()),"deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),"cudnn_deterministic":torch.backends.cudnn.deterministic,"cudnn_benchmark":torch.backends.cudnn.benchmark,"tf32_matmul":torch.backends.cuda.matmul.allow_tf32,"tf32_cudnn":torch.backends.cudnn.allow_tf32}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(report,indent=2),encoding="utf-8"); return report
def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--pip-freeze-output",type=Path);a=p.parse_args();capture(a.output,a.pip_freeze_output)
if __name__=="__main__":main()
