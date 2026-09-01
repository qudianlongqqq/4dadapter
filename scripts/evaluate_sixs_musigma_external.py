#!/usr/bin/env python3
"""Pure external-validity worker for one frozen factorial Proposal SDF."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import yaml
from rdkit import Chem

OFFICIAL_ADAPTER=Path(r"E:\3dconformergenerationcode\4dadapter-lsgo-v2\scripts\run_lsgo_standard_genbench3d.py")
GENBENCH_REPO=Path(r"E:\miniconda\envs\external-validity\src\genbench3d")
LIGBOUNDCONF=Path(r"E:\3dconformergenerationcode\external_data\genbench3d_official\ligboundconf_minimized\S2_LigBoundConf_minimized.sdf")
REFERENCE_ROOT=Path(r"E:\3dconformergenerationcode\4dadapter-lsgo-v2\reports\ecir_mvr\lsgo_standard_eval\genbench3d_reference_cache")
GENBENCH_COMMIT="0926bc6614509aa10ccf6f69da0405d4be6af6b3"
GENBENCH_ADAPTER_SHA="6ebc450b4f841a9f6a3b463b7838e50bc2951e92570146cd0a473abbfd970450"
REFERENCE_SDF_SHA="15e8e4635525f3d9452292e86995d28f6c24eb50baad661eb4c2665274d00fe2"
REFERENCE_VALUE_SHA="63659acddd04017a4b8fc5f2df767540e48cd36a7849e578ddb6caf6130deadc"
REFERENCE_KERNEL_SHA="6c098fa5b10c85f12db49df3a35efa33963fc222754e5d2a9d0b64e61c604a19"


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1<<20),b""): digest.update(block)
    return digest.hexdigest()


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f".tmp.{os.getpid()}");frame.to_parquet(tmp,index=False);os.replace(tmp,path)


def selected_columns(installed: dict) -> list[str]:
    result=[]
    for module in installed.get("modules",[]):
        renames=module.get("rename_outputs",{})
        for raw in module.get("chosen_binary_test_output",[]):
            name=str(renames.get(raw,raw)).lower().replace(" ","_")
            if name not in result:result.append(name)
    return result


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--arm",required=True);parser.add_argument("--sdf",type=Path,required=True);parser.add_argument("--records",type=Path,required=True);parser.add_argument("--pb",type=Path,required=True);parser.add_argument("--v3d",type=Path,required=True);args=parser.parse_args()
    records=pd.read_parquet(args.records);ids=records.record_id.astype(str).tolist()
    if len(ids)!=5000 or len(set(ids))!=5000:raise RuntimeError("external worker record identity changed")
    import posebusters
    pb_root=Path(posebusters.__file__).parent;installed=yaml.safe_load((pb_root/"config/mol_fast.yml").read_text(encoding="utf-8"));runtime=copy.deepcopy(installed)
    for module in runtime.get("modules",[]):
        if module.get("function")=="energy_ratio":module.setdefault("parameters",{})["num_threads"]=1
    from posebusters import PoseBusters
    chosen=selected_columns(installed);raw=PoseBusters(config=runtime,max_workers=1,chunk_size=50).bust(str(args.sdf),None,None,full_report=True).reset_index();order={sample:index for index,sample in enumerate(ids)};raw["_order"]=raw.molecule.astype(str).map(order);raw=raw.sort_values("_order",kind="stable").reset_index(drop=True)
    if len(raw)!=len(ids) or raw.molecule.astype(str).tolist()!=ids or not set(chosen).issubset(raw.columns):raise RuntimeError("PoseBusters identity/schema failure")
    raw["record_id"]=ids;raw["arm"]=args.arm;raw["PB"]=raw[chosen].fillna(False).astype(bool).all(axis=1);atomic_frame(args.pb,raw);print(f"POSEBUSTERS_COMPLETE {args.arm} {len(raw)}",flush=True)
    if sha256(OFFICIAL_ADAPTER)!=GENBENCH_ADAPTER_SHA or subprocess.check_output(["git","rev-parse","HEAD"],cwd=GENBENCH_REPO,text=True).strip()!=GENBENCH_COMMIT:raise RuntimeError("GenBench3D evaluator identity changed")
    for path,expected in ((LIGBOUNDCONF,REFERENCE_SDF_SHA),(REFERENCE_ROOT/"LigBoundConf_geometry_values.p",REFERENCE_VALUE_SHA),(REFERENCE_ROOT/"LigBoundConf_geometry_kernel_densities.p",REFERENCE_KERNEL_SHA)):
        if sha256(path)!=expected:raise RuntimeError(f"GenBench3D reference identity changed: {path}")
    sys.modules.setdefault("gemmi",types.ModuleType("gemmi"));spec=importlib.util.spec_from_file_location("factorial_genbench_external",OFFICIAL_ADAPTER);official=importlib.util.module_from_spec(spec);assert spec.loader is not None;spec.loader.exec_module(official);official.METHODS=(args.arm,);SDFSource,ReferenceGeometry,Validity3D=official.import_official_evaluator();reference=ReferenceGeometry(source=SDFSource(str(LIGBOUNDCONF),name="LigBoundConf",removeHs=False),root=str(REFERENCE_ROOT),minimum_pattern_values=50,use_generalized_patterns=False);validity=Validity3D(reference_geometry=reference,q_value_threshold=.001,steric_clash_safety_ratio=.75,maximum_ring_plane_distance=.1,include_torsions=False,consider_hydrogens=False)
    supplier=Chem.SDMolSupplier(str(args.sdf),removeHs=False,sanitize=True);rows=[]
    for index,sample_id in enumerate(ids):
        molecule=supplier[index]
        if molecule is None or molecule.GetProp("_Name")!=sample_id:raise RuntimeError("Validity3D SDF binding failure")
        output=official.evaluate_record_group(validity,[molecule],index)
        if len(output)!=1:raise RuntimeError("Validity3D group denominator changed")
        output[0]["arm"]=args.arm;output[0]["record_id"]=sample_id;rows.append(output[0])
        if (index+1)%100==0:print(f"VALIDITY3D_PROGRESS {args.arm} {index+1}/5000",flush=True)
    frame=pd.DataFrame(rows)
    if len(frame)!=5000 or frame.record_id.astype(str).tolist()!=ids:raise RuntimeError("Validity3D denominator changed")
    atomic_frame(args.v3d,frame);print(f"VALIDITY3D_COMPLETE {args.arm} {len(frame)}",flush=True);return 0


if __name__=="__main__":raise SystemExit(main())
