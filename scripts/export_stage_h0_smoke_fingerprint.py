#!/usr/bin/env python
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
import pandas as pd,yaml
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical_csv(path):
    frame=pd.read_csv(path).sort_values(list(pd.read_csv(path,nrows=0).columns),kind="stable").reset_index(drop=True)
    return hashlib.sha256(frame.to_csv(index=False,float_format="%.12g").encode()).hexdigest()
def export(source:Path,environment:Path,config:Path,output:Path):
    result=json.loads((source/"validation_result.json").read_text()); conflict=pd.read_csv(source/"conflict_summary.csv"); molecules=pd.read_csv(source/"per_molecule_conflict.csv");cfg=yaml.safe_load(config.read_text())
    methods={}
    for row in conflict.to_dict("records"):
        name=row.pop("method"); methods[name]={k:(v.item() if hasattr(v,"item") else v) for k,v in row.items()}
    report={"schema_version":"stage-h0-smoke-fingerprint-v1","git_commit":subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip(),"environment_report_sha256":sha(environment),"checkpoint_sha256":cfg["checkpoint"]["sha256"],"validation_source_sha256":sha(Path(cfg["data"]["val_sources"])),"validation_target_sha256":sha(Path(cfg["data"]["val_targets"])),"record_ids":sorted(molecules.record_id.astype(str).unique()),"molecule_ids":sorted(molecules.molecule_id.astype(str).unique()),"validation_records":result["validation_records_read"],"test_records":result["test_records_read"],"variants":sorted(result["methods"]),"methods":methods,"clean_identity":{row["method"]:row["clean_identity"] for row in result["metrics"]},"normalized_csv_sha256":{name:canonical_csv(source/name) for name in ("method_summary.csv","conflict_summary.csv","per_molecule_conflict.csv")}}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2),encoding="utf-8");return report
def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,default=Path("diagnostics/ecir_mvr/stage_h0/smoke"));p.add_argument("--environment",type=Path,required=True);p.add_argument("--config",type=Path,default=Path("configs/ecir_mvr_stage_h0_conflict_fusion.yaml"));p.add_argument("--output",type=Path,required=True);a=p.parse_args();export(a.source,a.environment,a.config,a.output)
if __name__=="__main__":main()
