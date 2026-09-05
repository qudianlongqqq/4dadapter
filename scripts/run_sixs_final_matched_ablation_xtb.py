#!/usr/bin/env python
"""Frozen 750-molecule xTB subset for priority final matched ablations."""

from __future__ import annotations

import argparse, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap
ROOT = bootstrap()
import scripts.run_sixs_j1r1_full_joint_xtb_dev as engine

PRIMARY_ASSET = Path(r"E:\3dconformergenerationcode\dataset\sixs_primary_final_evaluation_v1")

def atomic_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp,index=False); os.replace(tmp,path)

def atomic_json(path,value):
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False,default=str)+"\n",encoding="utf-8"); os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--protocol",type=Path,required=True); p.add_argument("--coordinate-dir",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--report-dir",type=Path,required=True); a=p.parse_args()
    protocol=json.loads(a.protocol.read_text(encoding="utf-8")); settings=protocol["xtb_single_point"]
    engine.OUT=a.output_dir; engine.SETTINGS={"version":settings["version"],"wsl":settings["wsl_executable"],"distribution":settings["wsl_distribution"],"executable":settings["executable"],"executable_sha256":settings["executable_sha256"],"gfn":2,"threads":1,"workers":int(settings["workers"]),"timeout_seconds":int(settings["timeout_seconds"]),"solvent":None,"geometry_optimization":False}
    if engine.sha_file(engine.SETTINGS["executable"]) != engine.SETTINGS["executable_sha256"]: raise RuntimeError("xTB executable SHA mismatch")
    selected=set(protocol["xtb_subset"]["record_ids"]); expected=len(selected)
    if expected != 1500: raise RuntimeError("frozen xTB subset must contain 1500 records")
    source_records=pd.read_parquet(PRIMARY_ASSET/"methods/source/PER_RECORD.parquet")
    source_records=source_records[source_records.record_id.astype(str).isin(selected)].copy()
    if len(source_records)!=expected: raise RuntimeError("source subset mismatch")
    ids=source_records.record_id.astype(str).tolist()
    primary_source=pd.read_csv(PRIMARY_ASSET/"xtb_single_point/SOURCE_XTB.csv")
    source=source_records[["record_id","molecule_id"]].merge(primary_source,on="record_id",validate="one_to_one")
    source=source.set_index("record_id").loc[ids].reset_index(); atomic_csv(a.output_dir/"SOURCE_XTB.csv",source)
    source_mols=list(Chem.ForwardSDMolSupplier(str(PRIMARY_ASSET/"methods/source/COORDINATES.sdf"),sanitize=False,removeHs=False))
    source_ids=pd.read_parquet(PRIMARY_ASSET/"methods/source/PER_RECORD.parquet").record_id.astype(str).tolist()
    source_by={rid:mol for rid,mol in zip(source_ids,source_mols,strict=True) if rid in selected}
    methods=[m for m in protocol["model_methods"] if m["variant"] in {"reliability_off","equal_ba","fixed_tau"}]
    summaries=[]
    for method in methods:
        mid=method["id"]; result_path=a.output_dir/f"{mid.upper()}_XTB.csv"
        if result_path.is_file():
            frame=pd.read_csv(result_path)
            if len(frame)!=expected or frame.record_id.astype(str).tolist()!=ids: raise RuntimeError(f"stale xTB result {mid}")
        else:
            all_records=pd.read_parquet(a.coordinate_dir/f"methods/{mid}/PER_RECORD.parquet"); all_ids=all_records.record_id.astype(str).tolist()
            all_mols=list(Chem.ForwardSDMolSupplier(str(a.coordinate_dir/f"methods/{mid}/COORDINATES.sdf"),sanitize=False,removeHs=False))
            candidate_by={rid:mol for rid,mol in zip(all_ids,all_mols,strict=True) if rid in selected}
            tasks=[]
            for index,(rid,molecule_id) in enumerate(zip(ids,source_records.molecule_id.astype(str),strict=True)):
                mol=candidate_by[rid]; src=source_by[rid]
                if engine.elements(mol)!=engine.elements(src) or engine.formal_charge(mol)!=engine.formal_charge(src) or engine.radical_electrons(mol)!=engine.radical_electrons(src): raise RuntimeError(f"topology mismatch {mid}/{rid}")
                coordinates=np.ascontiguousarray(engine.coordinates(mol),dtype=np.float64)
                identity={"elements":engine.elements(mol),"charge":engine.formal_charge(mol),"uhf":engine.radical_electrons(mol),"coordinate_sha256":engine.sha_bytes(f"float64|{coordinates.shape}|".encode()+coordinates.tobytes()),"settings":{k:engine.SETTINGS[k] for k in engine.SETTINGS if k!="workers"}}
                tasks.append({"method":mid,"record_index":index,"record_id":rid,"molecule_id":molecule_id,"coordinates":coordinates,"identity":identity,"identity_sha256":engine.canonical_sha(identity)})
            rows=[]
            with ThreadPoolExecutor(max_workers=engine.SETTINGS["workers"]) as pool:
                futures=[pool.submit(engine.execute,t) for t in tasks]
                for future in as_completed(futures): rows.append(future.result())
            frame=pd.DataFrame(rows).sort_values("record_index",kind="stable").reset_index(drop=True); atomic_csv(result_path,frame)
        delta=frame.merge(source[["record_id","energy_hartree","success"]].rename(columns={"energy_hartree":"source_energy_hartree","success":"source_success"}),on="record_id",validate="one_to_one")
        delta["delta_e_kcal_mol"]=(delta.energy_hartree-delta.source_energy_hartree)*627.509474
        delta["matched_success"]=delta.success.astype(bool)&delta.source_success.astype(bool)&np.isfinite(delta.delta_e_kcal_mol)
        atomic_csv(a.output_dir/f"{mid.upper()}_DELTA_VS_SOURCE.csv",delta)
        summaries.append({"method":mid,"attempted":len(frame),"success":int(frame.success.sum()),"failures":int((~frame.success.astype(bool)).sum())})
    atomic_json(a.report_dir/"XTB_FINAL_STATUS.json",{"status":"PASS","molecules":750,"records":1500,"methods":summaries,"source_energy_reused_from_frozen_primary":True,"geometry_optimization_performed":False})
    return 0
if __name__=="__main__": raise SystemExit(main())
