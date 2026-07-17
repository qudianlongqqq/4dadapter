#!/usr/bin/env python
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap
ROOT = bootstrap()
import torch,yaml
from scripts.verify_mcvr_stage_h0_assets import verify
def run(config,environment_comparison,smoke_comparison,output,allow_dirty=False):
    cfg=yaml.safe_load(config.read_text());branch=subprocess.run(["git","branch","--show-current"],capture_output=True,text=True,check=True).stdout.strip();commit=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip();dirty=bool(subprocess.run(["git","status","--porcelain"],capture_output=True,text=True,check=True).stdout)
    assets=verify(config,output.with_name("stage_h0_asset_verification.json")); env=json.loads(environment_comparison.read_text()); smoke=json.loads(smoke_comparison.read_text())
    tests=subprocess.run([str(Path(__import__("sys").executable)),"-m","pytest","-q","tests/test_ecir_mvr_stage_h0.py"],capture_output=True,text=True)
    checks={"branch":branch=="feat/mcvr-bond-explicit-proposal","commit":bool(commit),"clean_worktree":allow_dirty or not dirty,"environment":env["status"]!="INCOMPATIBLE","assets":not assets["missing"],"cuda":torch.cuda.is_available(),"gpu_visible":torch.cuda.device_count()>0,"targeted_tests":tests.returncode==0,"smoke":smoke["status"] in {"MATCH","MATCH_WITH_NUMERICAL_TOLERANCE"},"validation_test_isolation":cfg["validation_only"] and cfg["test_records_read"]==0}
    result={"status":"STAGE_H0_READY" if all(checks.values()) else "STAGE_H0_NOT_READY","branch":branch,"commit":commit,"dirty":dirty,"checks":checks,"test_output":tests.stdout[-2000:]}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2),encoding="utf-8");return result
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=Path("configs/ecir_mvr_stage_h0_conflict_fusion.yaml"));p.add_argument("--environment-comparison",type=Path,required=True);p.add_argument("--smoke-comparison",type=Path,required=True);p.add_argument("--output",type=Path,default=Path("reports/environment/stage_h0_preflight.json"));p.add_argument("--allow-dirty",action="store_true");a=p.parse_args();r=run(a.config,a.environment_comparison,a.smoke_comparison,a.output,a.allow_dirty);print(r["status"]);raise SystemExit(r["status"]!="STAGE_H0_READY")
if __name__=="__main__":main()
