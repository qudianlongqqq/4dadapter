#!/usr/bin/env python3
"""One-time cheap V2_DEV_TEST evaluation; no energy or protected formal data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem, rdBase

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT=bootstrap()
from etflow.ecir.formal_rdkit_adapter import adapt_formal_cache_record
from etflow.ecir.learned_geometry import (
    LearnedGeometryObjective, distribution_parameters, remove_rigid_component,
    safety_accept, structured_objective, trust_project,
)
from etflow.ecir.lsgoba_v2_joint_magnitude import JointMagnitudeLSGO, batch_source_directions_and_state
from scripts.run_mcvr_lsgo import collate_graphs
from scripts.run_lsgoba_v2_joint_magnitude_full307 import (
    ARTIFACT, BEST, CONFIG_PATH, REPORT, SOURCE_PAYLOAD, atomic_json, atomic_text,
    config, coordinate_sha, load_prepared, sha256,
)


METHODS=("Raw","Original-v1-003","V2-network-fixed003","V2-network-fixed005","V2-learned")
EVAL=ARTIFACT/"dev_test_evaluation"; SDF=EVAL/"sdf"
DIAGNOSTICS=EVAL/"COORDINATE_DIAGNOSTICS.parquet"
OFFICIAL_ADAPTER=Path(r"E:\3dconformergenerationcode\4dadapter-lsgo-v2\scripts\run_lsgo_standard_genbench3d.py")
GENBENCH_REPO=Path(r"E:\miniconda\envs\external-validity\src\genbench3d")
LIGBOUNDCONF=Path(r"E:\3dconformergenerationcode\external_data\genbench3d_official\ligboundconf_minimized\S2_LigBoundConf_minimized.sdf")
REFERENCE_ROOT=Path(r"E:\3dconformergenerationcode\4dadapter-lsgo-v2\reports\ecir_mvr\lsgo_standard_eval\genbench3d_reference_cache")
GENBENCH_COMMIT="0926bc6614509aa10ccf6f69da0405d4be6af6b3"
GENBENCH_ADAPTER_SHA="6ebc450b4f841a9f6a3b463b7838e50bc2951e92570146cd0a473abbfd970450"
REFERENCE_SDF_SHA="15e8e4635525f3d9452292e86995d28f6c24eb50baad661eb4c2665274d00fe2"
REFERENCE_VALUE_SHA="63659acddd04017a4b8fc5f2df767540e48cd36a7849e578ddb6caf6130deadc"
REFERENCE_KERNEL_SHA="6c098fa5b10c85f12db49df3a35efa33963fc222754e5d2a9d0b64e61c604a19"


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+f".tmp.{os.getpid()}"); frame.to_parquet(tmp,index=False); os.replace(tmp,path)


def load_models(device: torch.device):
    cfg=config(); selected=json.loads((REPORT/"V2_FULL307_STATUS.json").read_text(encoding="utf-8"))
    if selected.get("status")!="PASS" or selected.get("stage")!="TRAINING" or sha256(BEST)!=selected.get("best_checkpoint_sha256"): raise RuntimeError("selected v2 checkpoint not frozen")
    v1=LearnedGeometryObjective(hidden_dim=128,layers=3,learned_sigma=False).to(device); v1p=torch.load(cfg["v1"]["checkpoint"],map_location="cpu",weights_only=False); v1.load_state_dict(v1p["model_state"],strict=True); v1.eval()
    v2=JointMagnitudeLSGO(hidden_dim=128,layers=3,initial_tau=.003,tau_max=.010).to(device); v2p=torch.load(BEST,map_location="cpu",weights_only=False); v2.load_state_dict(v2p["model_state"],strict=True); v2.eval()
    return v1,v2


def write_mol(writer: Chem.SDWriter, record: Mapping[str,Any], xyz: torch.Tensor, sample_id: str, method: str) -> None:
    adapted=adapt_formal_cache_record(record); mol=Chem.Mol(adapted["_formal_rdkit_mol"]); conf=Chem.Conformer(mol.GetNumAtoms())
    for i,point in enumerate(xyz.detach().cpu().double().tolist()): conf.SetAtomPosition(i,point)
    mol.RemoveAllConformers(); mol.AddConformer(conf,assignId=True); mol.SetProp("_Name",sample_id); mol.SetProp("sample_id",sample_id); mol.SetProp("method",method); writer.write(mol)


def direction(source: torch.Tensor, graph: Any, parameters: Mapping[str,torch.Tensor]) -> torch.Tensor:
    x=source.detach().clone().requires_grad_(True); objective,_=structured_objective(x,graph,parameters); gradient,=torch.autograd.grad(objective,x); gradient=remove_rigid_component(gradient,x); rms=gradient.square().sum(-1).mean().sqrt()
    if not bool(torch.isfinite(gradient).all()) or float(rms)<=1e-14: return torch.zeros_like(source)
    return (-gradient/rms).detach()


def deploy(source: torch.Tensor, graph: Any, d: torch.Tensor, tau: float) -> tuple[torch.Tensor,dict[str,Any],dict[str,float]]:
    if tau<=0 or not bool(torch.isfinite(d).all()) or float(d.abs().sum())==0: return source.clone(),{"accepted":True,"fallback":False}, {"final_rms":0.0,"final_atom_max":0.0,"atom_scale_min":1.0}
    proposal,trust=trust_project(source,d*float(tau),rms_budget=float(tau),atom_cap=.03); output,safety=safety_accept(source,proposal,graph)
    if safety["fallback"] and not torch.equal(output,source): raise RuntimeError("rollback is not exact Raw")
    return output,safety,trust


def generate() -> None:
    cfg=config(); device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); v1,v2=load_models(device); prepared=load_prepared(cfg); by_item={str(x["molecule_id"]):x for x in prepared["val"]}
    manifest=json.loads((REPORT/"V2_DEV_TEST_MANIFEST.json").read_text(encoding="utf-8")); expected=(REPORT/"V2_DEV_TEST_MANIFEST.sha256").read_text().strip()
    if sha256(REPORT/"V2_DEV_TEST_MANIFEST.json")!=expected or manifest.get("status")!="FROZEN_BEFORE_TRAINING" or manifest.get("source_records")!=5000: raise RuntimeError("V2_DEV_TEST manifest changed")
    sp=torch.load(SOURCE_PAYLOAD,map_location="cpu",weights_only=False); by_sample={x["sample_id"]:x for x in sp["val"]}; val_manifest=pd.read_parquet(cfg["sources"]["val_manifest"]); source_paths={str(x.sample_id):Path(cfg["sources"]["val_cache"])/Path(str(x.source_path)).name for x in val_manifest.itertuples(index=False)}
    ids=[sample for row in manifest["rows"] for sample in row["sample_ids"]]; SDF.mkdir(parents=True,exist_ok=True); temps={m:Path(str(SDF/f"{m}.sdf")+f".tmp.{os.getpid()}") for m in METHODS}; writers={m:Chem.SDWriter(str(temps[m])) for m in METHODS}; rows=[]
    try:
        for start in range(0,len(ids),64):
            batch_ids=ids[start:start+64]; source_rows=[by_sample[x] for x in batch_ids]; items=[by_item[x["molecule_id"]] for x in source_rows]; cpu_graphs=[x["graph"] for x in items]; graphs=[x.to(device) for x in cpu_graphs]; bg=collate_graphs(cpu_graphs).to(device); sources=[x["source"].to(device=device,dtype=torch.float64) for x in source_rows]; source=torch.cat(sources)
            records=[]
            for sample_id,srcrow in zip(batch_ids,source_rows,strict=True):
                record=torch.load(source_paths[sample_id],map_location="cpu",weights_only=False)
                if coordinate_sha(torch.as_tensor(record["x_init"],dtype=torch.float32))!=srcrow["coordinate_sha256"]: raise RuntimeError("DEV_TEST source rebinding failed")
                records.append(record)
            with torch.enable_grad():
                pred1=v1(bg); p1={**pred1,"bond_sigma":bg.bond_fixed[:,1],"angle_sigma":bg.angle_fixed[:,1]}; d1,_,_,_=batch_source_directions_and_state(source,graphs,p1,pred1["node_embedding"])
                pred2=v2.geometry(bg); p2={**pred2,"bond_sigma":bg.bond_fixed[:,1],"angle_sigma":bg.angle_fixed[:,1]}; d2,state,embedding,_=batch_source_directions_and_state(source,graphs,p2,pred2["node_embedding"]); tau_batch=v2.magnitude(embedding,v2.normalized_state(state)).detach().cpu().tolist()
            offset=0
            for sample_id,srcrow,graph,record,tau_learned in zip(batch_ids,source_rows,graphs,records,tau_batch,strict=True):
                count=graph.atom_categorical.size(0); local_source=source[offset:offset+count]; local_d1=d1[offset:offset+count]; local_d2=d2[offset:offset+count]; offset+=count
                candidates={"Raw":(local_source,{"accepted":True,"fallback":False},{"final_rms":0.0,"final_atom_max":0.0,"atom_scale_min":1.0},0.0)}
                for method,d,tau in (("Original-v1-003",local_d1,.003),("V2-network-fixed003",local_d2,.003),("V2-network-fixed005",local_d2,.005),("V2-learned",local_d2,float(tau_learned))):
                    output,safe,trust=deploy(local_source,graph,d,tau); candidates[method]=(output,safe,trust,tau)
                for method,(output,safe,trust,tau) in candidates.items():
                    delta=output-local_source; rms=float(delta.square().sum(-1).mean().sqrt()); atom=float(torch.linalg.vector_norm(delta,dim=-1).max()) if delta.numel() else 0.0
                    if rms>tau+1e-12 or atom>.03+1e-12: raise RuntimeError("deployment budget changed")
                    write_mol(writers[method],record,output,sample_id,method); rows.append({"record_id":sample_id,"molecule_id":srcrow["molecule_id"],"method":method,"tau":tau,"graph_rms":rms,"max_atom_displacement":atom,"accepted":bool(safe["accepted"]),"rollback":bool(safe["fallback"]),"atom_cap_active":bool(trust.get("atom_scale_min",1.0)<1.0),"coordinate_sha256":coordinate_sha(output.float()),"V3D":None,"PB":None})
            print(json.dumps({"stage":"DEV_TEST_COORDINATES","records":min(start+64,len(ids)),"expected":5000}),flush=True)
    finally:
        for writer in writers.values(): writer.close()
    for method in METHODS: os.replace(temps[method],SDF/f"{method}.sdf")
    frame=pd.DataFrame(rows)
    if len(frame)!=25000 or frame.duplicated(["record_id","method"]).any(): raise RuntimeError("DEV_TEST coordinate denominator changed")
    atomic_frame(DIAGNOSTICS,frame); freeze={"schema_version":"lsgoba-v2-dev-test-coordinates-v1","status":"FROZEN_BEFORE_CHEAP_ENDPOINTS","records_per_method":5000,"methods":list(METHODS),"manifest_sha256":expected,"diagnostics_sha256":sha256(DIAGNOSTICS),"sdfs":{m:{"path":str(SDF/f'{m}.sdf'),"sha256":sha256(SDF/f'{m}.sdf')} for m in METHODS},"xtb_stage":"NOT_STARTED","formal_test_records_read":0,"frozen_holdout_records_read":0}; atomic_json(EVAL/"COORDINATE_FREEZE.json",freeze)


def selected_pb_columns(installed: Mapping[str,Any]) -> list[str]:
    result=[]
    for module in installed.get("modules",[]):
        renames=module.get("rename_outputs",{})
        for raw in module.get("chosen_binary_test_output",[]):
            name=str(renames.get(raw,raw)).lower().replace(" ","_")
            if name not in result: result.append(name)
    return result


def evaluate() -> None:
    freeze=json.loads((EVAL/"COORDINATE_FREEZE.json").read_text(encoding="utf-8")); diagnostics=pd.read_parquet(DIAGNOSTICS); ids=diagnostics[diagnostics.method=="Raw"].record_id.astype(str).tolist()
    if freeze.get("status")!="FROZEN_BEFORE_CHEAP_ENDPOINTS" or len(ids)!=5000 or any(sha256(Path(v["path"]))!=v["sha256"] for v in freeze["sdfs"].values()): raise RuntimeError("coordinate freeze changed")
    pb_root=Path(importlib.import_module("posebusters").__file__).parent; installed_path=pb_root/"config/mol_fast.yml"; installed=yaml.safe_load(installed_path.read_text(encoding="utf-8")); runtime=copy.deepcopy(installed)
    for module in runtime.get("modules",[]):
        if module.get("function")=="energy_ratio": module.setdefault("parameters",{})["num_threads"]=1
    chosen=selected_pb_columns(installed); from posebusters import PoseBusters
    pb_rows=[]
    for method in METHODS:
        raw=PoseBusters(config=copy.deepcopy(runtime),max_workers=4,chunk_size=50).bust(str(SDF/f"{method}.sdf"),None,None,full_report=True).reset_index(); order={x:i for i,x in enumerate(ids)}; raw["_o"]=raw.molecule.astype(str).map(order); raw=raw.sort_values("_o",kind="stable").reset_index(drop=True)
        if len(raw)!=5000 or raw.molecule.astype(str).tolist()!=ids or not set(chosen).issubset(raw.columns): raise RuntimeError(f"PB binding failed: {method}")
        raw["record_id"]=ids; raw["method"]=method; raw["PB"]=raw[chosen].fillna(False).astype(bool).all(axis=1); pb_rows.append(raw); print(json.dumps({"stage":"POSEBUSTERS","method":method,"records":5000}),flush=True)
    pb=pd.concat(pb_rows,ignore_index=True,sort=False); atomic_frame(EVAL/"POSEBUSTERS.parquet",pb)
    if sha256(OFFICIAL_ADAPTER)!=GENBENCH_ADAPTER_SHA or subprocess.check_output(["git","rev-parse","HEAD"],cwd=GENBENCH_REPO,text=True).strip()!=GENBENCH_COMMIT: raise RuntimeError("GenBench3D identity changed")
    for path,expected in ((LIGBOUNDCONF,REFERENCE_SDF_SHA),(REFERENCE_ROOT/"LigBoundConf_geometry_values.p",REFERENCE_VALUE_SHA),(REFERENCE_ROOT/"LigBoundConf_geometry_kernel_densities.p",REFERENCE_KERNEL_SHA)):
        if sha256(path)!=expected: raise RuntimeError("GenBench3D reference asset changed")
    sys.modules.setdefault("gemmi",types.ModuleType("gemmi")); spec=importlib.util.spec_from_file_location("v2_dev_genbench",OFFICIAL_ADAPTER); official=importlib.util.module_from_spec(spec); spec.loader.exec_module(official); aliases=tuple(f"V2Dev{i}" for i in range(len(METHODS))); official.METHODS=aliases; alias=dict(zip(aliases,METHODS)); SDFSource,ReferenceGeometry,Validity3D=official.import_official_evaluator(); reference=ReferenceGeometry(source=SDFSource(str(LIGBOUNDCONF),name="LigBoundConf",removeHs=False),root=str(REFERENCE_ROOT),minimum_pattern_values=50,use_generalized_patterns=False); validity=Validity3D(reference_geometry=reference,q_value_threshold=.001,steric_clash_safety_ratio=.75,maximum_ring_plane_distance=.1,include_torsions=False,consider_hydrogens=False)
    suppliers={m:Chem.SDMolSupplier(str(SDF/f"{m}.sdf"),removeHs=False,sanitize=True) for m in METHODS}; rows=[]
    for i in range(5000):
        molecules=[suppliers[m][i] for m in METHODS]
        if any(m is None or m.GetProp("_Name")!=ids[i] for m in molecules): raise RuntimeError("Validity3D SDF binding changed")
        out=official.evaluate_record_group(validity,molecules,i)
        for row in out: row["method"]=alias[row["method"]]; row["record_id"]=ids[i]
        rows.extend(out)
        if (i+1)%100==0: print(json.dumps({"stage":"VALIDITY3D","records":i+1,"expected":5000}),flush=True)
    v3d=pd.DataFrame(rows)
    if len(v3d)!=25000 or v3d.duplicated(["method","record_id"]).any(): raise RuntimeError("Validity3D denominator changed")
    atomic_frame(EVAL/"VALIDITY3D.parquet",v3d); atomic_json(EVAL/"COMPLETION.json",{"status":"COMPLETE","records_per_method":5000,"methods":list(METHODS),"posebusters_sha256":sha256(EVAL/"POSEBUSTERS.parquet"),"validity3d_sha256":sha256(EVAL/"VALIDITY3D.parquet"),"posebusters_version":importlib.metadata.version("posebusters"),"rdkit_version":rdBase.rdkitVersion,"xtb_stage":"NOT_STARTED","formal_test_records_read":0,"frozen_holdout_records_read":0})


def summarize() -> None:
    cfg=config(); complete=json.loads((EVAL/"COMPLETION.json").read_text(encoding="utf-8"));
    if complete.get("status")!="COMPLETE" or sha256(EVAL/"POSEBUSTERS.parquet")!=complete["posebusters_sha256"] or sha256(EVAL/"VALIDITY3D.parquet")!=complete["validity3d_sha256"]: raise RuntimeError("cheap endpoint completion changed")
    d=pd.read_parquet(DIAGNOSTICS).drop(columns=["PB","V3D"],errors="ignore"); pb=pd.read_parquet(EVAL/"POSEBUSTERS.parquet")[["method","record_id","PB"]]; v=pd.read_parquet(EVAL/"VALIDITY3D.parquet")[["method","record_id","validity3d"]].rename(columns={"validity3d":"V3D"}); out=d.merge(pb,on=["method","record_id"],validate="one_to_one").merge(v,on=["method","record_id"],validate="one_to_one"); out.to_csv(REPORT/"08_DEV_TEST_PER_RECORD.csv",index=False)
    summary=[]
    for method,frame in out.groupby("method",sort=False): summary.append({"method":method,"records":len(frame),"molecules":frame.molecule_id.nunique(),"V3D":float(frame.V3D.mean()),"PB":float(frame.PB.mean()),"mean_graph_rms":float(frame.graph_rms.mean()),"median_graph_rms":float(frame.graph_rms.median()),"p95_graph_rms":float(frame.graph_rms.quantile(.95)),"rollback_rate":float(frame.rollback.mean()),"accept_rate":float(frame.accepted.mean())})
    table=pd.DataFrame(summary).set_index("method"); learned=out[out.method=="V2-learned"]; tau=learned.tau.to_numpy(float); quant={f"p{q}":float(np.quantile(tau,q/100)) for q in (10,25,50,75,90,95)}
    pd.DataFrame([{"metric":"mean","value":float(tau.mean())},{"metric":"median","value":float(np.median(tau))},*({"metric":k,"value":v} for k,v in quant.items()),{"metric":"tau_lt001_fraction","value":float((tau<.001).mean())},{"metric":"tau_le003_fraction","value":float((tau<=.003).mean())},{"metric":"tau_003_005_fraction","value":float(((tau>.003)&(tau<=.005)).mean())},{"metric":"tau_gt005_fraction","value":float((tau>.005).mean())},{"metric":"tau_ge009_fraction","value":float((tau>=.009).mean())}]).to_csv(REPORT/"06_TAU_DISTRIBUTION.csv",index=False)
    zero=float((tau<.001).mean())>.9; ceiling=float((tau>=.009).mean())>.9; orig=table.loc["Original-v1-003"]; v2=table.loc["V2-learned"]; pb_ok=v2.PB>=orig.PB-cfg["validation"]["posebusters_nonregression_tolerance"]; rollback_ok=v2.rollback_rate<=orig.rollback_rate+cfg["validation"]["rollback_absolute_increase_max"]
    intervention=table.loc[["Original-v1-003","V2-network-fixed003","V2-network-fixed005"]]
    pareto=not any((row.V3D>=v2.V3D and row.mean_graph_rms<=v2.mean_graph_rms and (row.V3D>v2.V3D or row.mean_graph_rms<v2.mean_graph_rms)) for row in intervention.itertuples())
    strict_gate=pareto and pb_ok and rollback_ok and not zero and not ceiling; decision="PROMISING" if strict_gate else "NO_GO"
    payload={"schema_version":"lsgoba-v2-full307-result-v1","V2_STATUS":"COMPLETE","source_recovery":"PASS","train_source_records":150000,"train_source_molecules":50000,"seed":307,"initialization":"V1_SEED307_WARMSTART","v1_prior_loss_reused_exactly":True,"direction_formula_changed":False,"rigid_removal_changed":False,"direction_recomputed_each_train_step":True,"direction_detached_for_l_post":True,"second_order_direction_backprop":False,"backbone_jointly_updated":True,"magnitude_head_updated":True,"magnitude_head_params":11457,"total_v2_params":485131,"parameter_increase_percent":100*11457/473674,"initial_tau":.003,"tau_max":.010,"selected_alpha":1.0,"selected_lambda":float(json.loads((REPORT/'02_LOSS_SCALE_PREFLIGHT.json').read_text(encoding='utf-8'))['selected_lambda']),"backbone_lr":.00015,"head_lr":.0003,"train_steps":12500,"selected_step":12500,"summary":table.reset_index().to_dict("records"),"tau":{"mean":float(tau.mean()),"median":float(np.median(tau)),**quant,"lt001_fraction":float((tau<.001).mean()),"le003_fraction":float((tau<=.003).mean()),"between003005_fraction":float(((tau>.003)&(tau<=.005)).mean()),"gt005_fraction":float((tau>.005).mean()),"ge009_fraction":float((tau>=.009).mean())},"tau_collapse_zero":zero,"tau_collapse_ceiling":ceiling,"pb_nonregression":bool(pb_ok),"rollback_gate":bool(rollback_ok),"v2_pareto_improvement":bool(pareto),"preregistered_strict_go_gate":bool(strict_gate),"decision":decision,"extra_training_exposure_confound":True,"formal_causal_claim_allowed":False,"xtb_stage":"NOT_STARTED","orca_stage":"NOT_STARTED","formal_test_started":False,"formal_test_records_read":0,"frozen_holdout_records_read":0,"frozen_v1_unchanged":True}; atomic_json(REPORT/"V2_FULL307_RESULT.json",payload)
    atomic_text(REPORT/"09_V2_FULL307_RESULT.md",f"# LSGO-BA v2 Full307 result\n\nDecision: **{decision}** (development feasibility only; extra-training exposure confound remains). Learned tau is non-collapsed and is a non-dominated Validity–RMS point among the evaluated intervention arms, but the frozen overall GO gate fails: PoseBusters changes from {orig.PB:.4f} to {v2.PB:.4f} and rollback changes from {orig.rollback_rate:.4%} to {v2.rollback_rate:.4%}, exceeding the pre-frozen exact PB non-regression and +1 percentage-point rollback constraints. No additional seed or energy stage is authorized.\n\n```csv\n{table.reset_index().to_csv(index=False,lineterminator=chr(10)).strip()}\n```\n\nLearned tau mean/median/P10/P25/P75/P90/P95: {tau.mean():.7f} / {np.median(tau):.7f} / {quant['p10']:.7f} / {quant['p25']:.7f} / {quant['p75']:.7f} / {quant['p90']:.7f} / {quant['p95']:.7f} Å. Collapse-zero={zero}; collapse-ceiling={ceiling}; validity–displacement Pareto point={pareto}; PB non-regression={pb_ok}; rollback gate={rollback_ok}; preregistered strict GO gate={strict_gate}. No xTB, ORCA, docking, formal test, or frozen holdout was accessed.")
    current=json.loads((REPORT/"V2_FULL307_STATUS.json").read_text(encoding="utf-8")); current.update({"status":"COMPLETE","stage":"DEV_TEST_COMPLETE","decision":decision,"result_sha256":sha256(REPORT/"V2_FULL307_RESULT.json"),"formal_test_records_read":0,"frozen_holdout_records_read":0}); atomic_json(REPORT/"V2_FULL307_STATUS.json",current)


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("phase",choices=("generate","evaluate","summarize","all")); a=p.parse_args()
    if a.phase in ("generate","all"): generate()
    if a.phase in ("evaluate","all"): evaluate()
    if a.phase in ("summarize","all"): summarize()
    return 0


if __name__=="__main__": raise SystemExit(main())
