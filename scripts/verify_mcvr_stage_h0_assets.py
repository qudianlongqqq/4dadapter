#!/usr/bin/env python
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap
ROOT = bootstrap()
import pandas as pd,torch,yaml
from etflow.ecir.confidence_calibration import strict_load_frozen_model
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()
def verify(config_path:Path,output:Path):
    cfg=yaml.safe_load(config_path.read_text()); root=Path("data/ecir_mvr/medium")
    checkpoint_path=Path(cfg["checkpoint"]["path"])
    paths=[root/"split_metadata.json",root/"split_train.json",root/"split_val.json",root/"real_sources/train.parquet",root/"real_sources/val.parquet",root/"minimal_targets/train.parquet",root/"minimal_targets/val.parquet",Path(cfg["data"]["validity_statistics"]),checkpoint_path,checkpoint_path.parent.parent/"config.resolved.yaml"]
    records=[];missing=[];extra=[]
    for path in paths:
        item={"path":path.as_posix(),"exists":path.is_file()}
        if not path.is_file():missing.append(str(path));records.append(item);continue
        item.update(size=path.stat().st_size,sha256=sha(path))
        if path.suffix==".parquet":
            frame=pd.read_parquet(path);item.update(rows=len(frame),schema={c:str(t) for c,t in frame.dtypes.items()})
            for column in frame.columns:
                if "path" in column.lower():
                    for value in frame[column].dropna().astype(str).unique():
                        candidate=Path(value); candidate=candidate if candidate.is_absolute() else Path(value)
                        if candidate.exists():extra.append(candidate)
        records.append(item)
    if not missing:
        payload=torch.load(cfg["checkpoint"]["path"],map_location="cpu",weights_only=False)
        records[-2]["checkpoint_metadata"]={"step":payload.get("step"),"keys":sorted(payload)}
        strict_load_frozen_model(cfg["checkpoint"]["path"],expected_sha256=cfg["checkpoint"]["sha256"],device=torch.device("cpu"))
    report={"schema_version":"mcvr-stage-h0-assets-v1","status":"COMPLETE" if not missing else "MISSING_ASSETS","config":config_path.as_posix(),"assets":records,"referenced_files":sorted({p.as_posix() for p in extra}),"test_paths_accessed":[],"test_records_read":0,"missing":missing}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    required=output.with_name("stage_h0_required_assets.txt");required.write_text("\n".join([r["path"] for r in records]+report["referenced_files"])+"\n",encoding="utf-8")
    return report
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();r=verify(a.config,a.output);print(r["status"]);raise SystemExit(bool(r["missing"]))
if __name__=="__main__":main()
