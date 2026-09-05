#!/usr/bin/env python3
"""Confirmation-only J1-R0/R1 x fixed/joint-magnitude development experiment."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
# The imported factorial module is used only as the frozen J1 implementation
# and data/evaluator adapter.  Keep it bound to the completed frozen namespace.
os.environ.setdefault("SIXS_FACTORIAL_RUN_NAMESPACE", "sixs_musigma_reliability_factorial_cuda")
import scripts.run_sixs_musigma_reliability_factorial as frozen
import etflow.ecir.musigma_reliability as mr
from etflow.ecir.formal_rdkit_adapter import adapt_formal_cache_record
from etflow.ecir.learned_geometry import gaussian_nll, geometry_values, remove_rigid_component, safety_accept
from etflow.ecir.lsgoba_v2_joint_magnitude import (
    AdaptiveTrustMagnitudeHead,
    SOURCE_STATE_FEATURES,
    _family_features,
    scaled_proposal,
)
from etflow.ecir.musigma_reliability import ActionOutput, DirectMuSigmaModel, PrimitiveReliabilityHead, action_loss, action_proposal
from scripts.run_mcvr_lsgo import collate_graphs

CONFIG_PATH = ROOT / "configs/sixs_j1r1_joint_magnitude_interaction.json"
REPORT = ROOT / "reports/ecir_mvr/sixs_j1r1_joint_magnitude_interaction_seed307"
ARTIFACT = ROOT / "artifacts/ecir_mvr/sixs_j1r1_joint_magnitude_interaction_seed307"
STATUS = REPORT / "FINAL_STATUS.json"
ARMS = ("J1-R0-JOINT", "J1-R1-JOINT")
FIXED = {
    "J1-R0-FIXED": ROOT / "artifacts/ecir_mvr/sixs_musigma_reliability_factorial_cuda/J1_R0/dev_evaluation/PER_RECORD.parquet",
    "J1-R1-FIXED": ROOT / "artifacts/ecir_mvr/sixs_musigma_reliability_factorial_cuda/J1_R1/dev_evaluation/PER_RECORD.parquet",
}


def cfg() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_frame(path: Path, value: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    if path.suffix == ".csv": value.to_csv(tmp, index=False)
    else: value.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, tmp); os.replace(tmp, path)


def protected() -> dict[str, str]:
    return {
        "FORMAL_READ": "NO", "LARGE_HOLDOUT_READ": "NO", "XTB_STARTED": "NO",
        "SEED331_STARTED": "NO", "SEED353_STARTED": "NO", "BA_CHANGED": "NO",
        "SIGMA_TEACHER_CREATED": "NO", "RELIABILITY_TEACHER_CREATED": "NO",
        "MOVEMENT_V2_STARTED": "NO", "TRUST_REGION_STARTED": "NO",
        "NAIVE_MU_SIGMA_JOINT_REINTRODUCED": "NO", "JOINT_MEANS": "MAGNITUDE_ONLY",
        "HYPERPARAMETER_SWEEP": "NO", "CHECKPOINT_SELECTED_BY_DEV_OUTCOME": "NO",
        "SECOND_ORDER_REQUIRED": "NO", "REPEATED_POLLING": "NO",
    }


def status(stage: str, state: str = "RUNNING", **extra: Any) -> None:
    old = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    if state != "FAIL":
        old.pop("ERROR", None)
        old.pop("ERROR_TYPE", None)
    atomic_json(STATUS, {**old, "schema_version": "sixs-j1r1-joint-magnitude-status-v1",
                         "PIPELINE_STATUS": state, "CURRENT_STAGE": stage,
                         "UPDATED_AT_EPOCH": time.time(), **protected(), **extra})


def paths(arm: str) -> dict[str, Path]:
    root = ARTIFACT / arm.lower().replace("-", "_")
    report = REPORT / arm.lower().replace("-", "_")
    return {"root": root, "report": report, "recovery": root/"recovery.ckpt",
            "final": root/"final.ckpt", "log": report/"TRAIN_LOG.csv",
            "sdf": root/"dev_evaluation/PROPOSAL.sdf",
            "per_record": root/"dev_evaluation/PER_RECORD.parquet",
            "pb": root/"dev_evaluation/POSEBUSTERS.parquet",
            "v3d": root/"dev_evaluation/VALIDITY3D.parquet",
            "coordinates": root/"dev_evaluation/COORDINATES_READY.json",
            "result": report/"RESULT.json", "done": root/"DONE.json"}


def device() -> torch.device:
    mode = os.environ.get("SIXS_JOINT_DEVICE", "auto").lower()
    if mode == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA_REQUIRED_BUT_UNAVAILABLE")
    return torch.device("cuda:0" if mode != "cpu" and torch.cuda.is_available() else "cpu")


def seed_all() -> torch.Generator:
    seed = cfg()["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.Generator().manual_seed(seed + 91000)


def frozen_name(arm: str) -> str:
    return "J1-R1" if "R1" in arm else "J1-R0"


def load_frozen(arm: str, target: torch.device):
    name = frozen_name(arm); spec = cfg()["frozen_checkpoints"][name]
    checkpoint = ROOT / spec["path"]
    if sha256(checkpoint) != spec["sha256"]: raise RuntimeError(f"frozen checkpoint changed: {name}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = DirectMuSigmaModel(128, 3).to(target); model.load_state_dict(saved["model_state"], strict=True)
    reliability = None
    if name.endswith("R1"):
        reliability = PrimitiveReliabilityHead(128).to(target)
        reliability.load_state_dict(saved["reliability_state"], strict=True)
    model.eval(); model.requires_grad_(False)
    if reliability is not None: reliability.eval(); reliability.requires_grad_(False)
    return model, reliability


def state_normalization(target: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    spec = cfg()["magnitude"]; path = ROOT / spec["state_preflight_path"]
    if sha256(path) != spec["state_preflight_sha256"]: raise RuntimeError("magnitude state preflight changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if tuple(value["state_feature_names"]) != tuple(SOURCE_STATE_FEATURES): raise RuntimeError("17D state schema changed")
    return (torch.tensor(value["state_mean"], device=target),
            torch.tensor(value["state_std"], device=target).clamp_min(1e-6), value)


def direction_state(source: torch.Tensor, graphs: Sequence[Any], prediction: Mapping[str, torch.Tensor], reliability):
    """Exact frozen action direction plus the established 17D state semantics."""
    fixed = action_proposal(source, graphs, prediction, tau=1.0, atom_cap=1.0e9, reliability_head=reliability)
    atom_offsets, bond_offsets, angle_offsets = mr._offsets(graphs)
    view = mr._collated_geometry_view(graphs, source.device)
    bq, aq = geometry_values(source, view)
    bnll = gaussian_nll(bq, prediction["bond_mu"].to(bq), prediction["bond_sigma"].to(bq))
    anll = gaussian_nll(aq, prediction["angle_mu"].to(aq), prediction["angle_sigma"].to(aq))
    bz = (bq - prediction["bond_mu"].to(bq)) / prediction["bond_sigma"].to(bq)
    az = (aq - prediction["angle_mu"].to(aq)) / prediction["angle_sigma"].to(aq)
    states=[]; embeddings=[]
    for i, graph_cpu in enumerate(graphs):
        graph = graph_cpu.to(source.device); aa=slice(atom_offsets[i],atom_offsets[i+1]); bb=slice(bond_offsets[i],bond_offsets[i+1]); cc=slice(angle_offsets[i],angle_offsets[i+1])
        nb=max(1,bond_offsets[i+1]-bond_offsets[i]); na=max(1,angle_offsets[i+1]-angle_offsets[i]); x=source[aa]
        bc=(.5/nb)*fixed.bond_reliability[bb].to(x)*(bq[bb]-prediction["bond_mu"][bb].to(bq))/prediction["bond_sigma"][bb].to(bq).square()
        ac=(.5/na)*fixed.angle_reliability[cc].to(x)*(aq[cc]-prediction["angle_mu"][cc].to(aq))/prediction["angle_sigma"][cc].to(aq).square()
        raw=mr._bond_descent(x,graph,bc)+mr._angle_descent(x,graph,ac); projected=remove_rigid_component(raw,x)
        norms=torch.linalg.vector_norm(projected,dim=-1); rms=projected.square().sum(-1).mean().sqrt()
        eb=bnll[bb].mean() if bnll[bb].numel() else x.new_zeros(()); ea=anll[cc].mean() if anll[cc].numel() else x.new_zeros(()); e=.5*(eb+ea)
        local_bz=bz[bb]/math.sqrt(2.0*nb); local_az=az[cc]/math.sqrt(2.0*na)
        size=x.new_tensor((x.size(0),graph.bonds.size(1),graph.angles.size(0)))
        states.append(torch.cat((_family_features(local_bz),_family_features(local_az),torch.stack((eb,ea,e)),torch.stack((rms,norms.mean(),norms.max())),size)).to(prediction["node_embedding"]))
        embeddings.append(prediction["node_embedding"][aa].mean(0))
    return fixed, torch.stack(states).detach(), torch.stack(embeddings).detach()


def make_magnitude(target: torch.device) -> AdaptiveTrustMagnitudeHead:
    seed_all()
    spec=cfg()["magnitude"]
    return AdaptiveTrustMagnitudeHead(spec["graph_dim"],spec["state_dim"],initial_tau=spec["initial_tau_angstrom"],tau_max=spec["tau_max_angstrom"]).to(target)


def proposal_from_tau(source, graphs, fixed_action, tau):
    proposal, cap, rms = scaled_proposal(source, fixed_action.direction.detach(), tau, graphs, atom_cap=cfg()["magnitude"]["atom_cap_angstrom"])
    return ActionOutput(fixed_action.direction.detach(), proposal, source + torch.repeat_interleave(tau, torch.tensor([g.atom_categorical.size(0) for g in graphs],device=tau.device))[:,None]*fixed_action.direction.detach(), cap, rms, fixed_action.bond_reliability.detach(), fixed_action.angle_reliability.detach())


def load_data():
    prepared, source_payload = frozen.load_inputs()
    manifest_path=ROOT/cfg()["evaluation"]["dev_manifest"]
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("identity_sha256") != frozen.canonical_sha(manifest["rows"]): raise RuntimeError("DEV identity changed")
    return prepared, source_payload, manifest


def implementation_freeze() -> None:
    REPORT.mkdir(parents=True,exist_ok=True); ARTIFACT.mkdir(parents=True,exist_ok=True)
    config=cfg(); preflight=ROOT/config["magnitude"]["state_preflight_path"]
    for name,spec in config["frozen_checkpoints"].items():
        if sha256(ROOT/spec["path"]) != spec["sha256"]: raise RuntimeError(f"checkpoint hash fail {name}")
    _,_,manifest=load_data()
    payload={**config,"config_sha256":sha256(CONFIG_PATH),"dev_identity_sha256":manifest["identity_sha256"],"state_preflight_sha256":sha256(preflight),"frozen_per_record_sha256":{k:sha256(v) for k,v in FIXED.items()},**protected()}
    atomic_json(REPORT/"IMPLEMENTATION_CONFIG.json",payload)
    status("IMPLEMENTATION_FROZEN","PASS",IMPLEMENTATION_STATUS="PASS",J1_R0_JOINT_STARTED="NO",J1_R1_JOINT_STARTED="NO")


def gates() -> None:
    implementation_freeze(); target=device(); prepared,sources,_=load_data(); source_map=frozen.source_index(sources["train"]); generator=seed_all()
    graphs,bg,source,reference,_=frozen.sample_batch(prepared["train"],source_map,generator,64,target)
    means,stds,_=state_normalization(target); grad_rows=[]; repro_rows=[]
    for arm in ARMS:
        model,reliability=load_frozen(arm,target); magnitude=make_magnitude(target); magnitude.train()
        with torch.no_grad(): prediction=model(bg,detach_sigma_features=False)
        fixed_action,state,embedding=direction_state(source,graphs,prediction,reliability)
        tau=magnitude(embedding,(state-means)/stds); action=proposal_from_tau(source,graphs,fixed_action,tau)
        loss=action_loss(action,reference,graphs,prediction)+cfg()["magnitude"]["lambda_move"]*(tau/cfg()["magnitude"]["tau_max_angstrom"]).square().mean()
        loss.backward(); mag_norm=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in magnitude.parameters() if p.grad is not None))
        frozen_grad=sum(int(p.grad is not None and bool(torch.any(p.grad != 0))) for p in model.parameters())+sum(int(p.grad is not None and bool(torch.any(p.grad != 0))) for p in ([] if reliability is None else reliability.parameters()))
        grad_rows.append({"arm":arm,"loss":float(loss.detach()),"magnitude_grad_norm":mag_norm,"frozen_nonzero_grad_tensors":frozen_grad,"pass":mag_norm>0 and frozen_grad==0})
        # Fixed-path reproduction uses the first frozen 32 records, not TRAIN outcomes.
        del magnitude,model,reliability,prediction,fixed_action,state,embedding,tau,action
    grad_pass=all(x["pass"] for x in grad_rows)
    atomic_json(REPORT/"GRADIENT_GATE.json",{"schema_version":"sixs-joint-gradient-gate-v1","status":"PASS" if grad_pass else "FAIL","rows":grad_rows,"JOINT_MAGNITUDE_GRADIENT_PATH":"PASS" if grad_pass else "FAIL","FROZEN_DIRECTION_MODEL_GRADIENT_ISOLATION":"PASS" if grad_pass else "FAIL","SECOND_ORDER_REQUIRED":"NO",**protected()})
    if not grad_pass: raise RuntimeError("gradient gate failed")
    # Independent exact inference-path reproduction on 32 frozen DEV records.
    by_item={str(x["molecule_id"]):x for x in prepared["val"]}; by_sample={str(x["sample_id"]):x for x in sources["val"]}; _,_,manifest=load_data(); ids=[s for row in manifest["rows"] for s in row["sample_ids"]][:cfg()["evaluation"]["reproduction_records"]]
    for arm in ARMS:
        model,reliability=load_frozen(arm,target); rows=[by_sample[x] for x in ids]; items=[by_item[str(x["molecule_id"])] for x in rows]; gs=[x["graph"] for x in items]; batch=collate_graphs(gs).to(target); src=torch.cat([torch.as_tensor(x["source"],dtype=torch.float64) for x in rows]).to(target)
        with torch.no_grad(): pred=model(batch,detach_sigma_features=False)
        old=action_proposal(src,gs,pred,tau=cfg()["magnitude"]["fixed_reproduction_tau_angstrom"],atom_cap=cfg()["magnitude"]["atom_cap_angstrom"],reliability_head=reliability)
        exact,state,embedding=direction_state(src,gs,pred,reliability); tau=torch.full((len(gs),),cfg()["magnitude"]["fixed_reproduction_tau_angstrom"],device=target,dtype=src.dtype); new=proposal_from_tau(src,gs,exact,tau)
        old_b,old_a=geometry_values(old.proposal,batch); new_b,new_a=geometry_values(new.proposal,batch)
        fixed_frame=pd.read_parquet(FIXED[f"{frozen_name(arm)}-FIXED"])
        schema_ok={"record_id","molecule_id","V3D","PB","internal_post"}.issubset(fixed_frame.columns)
        row={"arm":arm,"records":len(ids),"direction_max_abs":float((old.direction-new.direction).abs().max()),"proposal_max_abs_angstrom":float((old.proposal-new.proposal).abs().max()),"bond_max_abs":float((old_b-new_b).abs().max()),"angle_max_abs":float((old_a-new_a).abs().max()),"tau_max_abs":0.0,"evaluator_input_schema_compatible":schema_ok}
        row["pass"]=max(row["direction_max_abs"],row["proposal_max_abs_angstrom"],row["bond_max_abs"],row["angle_max_abs"])<=1e-12 and schema_ok; repro_rows.append(row)
    repro_pass=all(x["pass"] for x in repro_rows)
    atomic_json(REPORT/"FIXED_MOVEMENT_REPRODUCTION.json",{"schema_version":"sixs-fixed-movement-reproduction-v1","status":"PASS" if repro_pass else "FAIL","tau":cfg()["magnitude"]["fixed_reproduction_tau_angstrom"],"rows":repro_rows,"FIXED_MOVEMENT_REPRODUCTION":"PASS" if repro_pass else "FAIL",**protected()})
    if not repro_pass: raise RuntimeError("fixed movement reproduction failed")
    status("GATES_PASSED","PASS",IMPLEMENTATION_STATUS="PASS",FIXED_MOVEMENT_REPRODUCTION="PASS",JOINT_MAGNITUDE_GRADIENT_PATH="PASS",FROZEN_DIRECTION_MODEL_GRADIENT_ISOLATION="PASS")


def train(arm: str) -> None:
    if arm not in ARMS: raise ValueError(arm)
    p=paths(arm); p["root"].mkdir(parents=True,exist_ok=True); p["report"].mkdir(parents=True,exist_ok=True)
    config=cfg(); train_cfg=config["training"]; target=device(); prepared,sources,_=load_data(); source_map=frozen.source_index(sources["train"]); generator=seed_all(); model,reliability=load_frozen(arm,target); magnitude=make_magnitude(target); means,stds,_=state_normalization(target)
    optimizer=torch.optim.AdamW(magnitude.parameters(),lr=train_cfg["head_learning_rate"],weight_decay=train_cfg["weight_decay"]); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=train_cfg["scheduler_horizon"]); start=0; logs=[]
    if p["final"].is_file(): return
    if p["recovery"].is_file():
        saved=torch.load(p["recovery"],map_location="cpu",weights_only=False)
        if saved["arm"]!=arm or saved["config_sha256"]!=sha256(CONFIG_PATH): raise RuntimeError("invalid recovery")
        magnitude.load_state_dict(saved["magnitude_state"]);optimizer.load_state_dict(saved["optimizer"]);scheduler.load_state_dict(saved["scheduler"]);generator.set_state(saved["generator_state"]);random.setstate(saved["python_rng"]);np.random.set_state(saved["numpy_rng"]);torch.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start=int(saved["step"]); logs=pd.read_csv(p["log"]).to_dict("records") if p["log"].is_file() else []
    status("TRAINING","RUNNING",CURRENT_ARM=arm,CURRENT_STEP=start,TOTAL_STEPS=train_cfg["optimizer_steps"],J1_R0_JOINT_STARTED="YES",J1_R1_JOINT_STARTED="YES" if arm==ARMS[1] else "NO")
    for step in range(start+1,train_cfg["optimizer_steps"]+1):
        tick=time.time(); graphs,bg,source,reference,_=frozen.sample_batch(prepared["train"],source_map,generator,train_cfg["batch_molecules"],target)
        with torch.no_grad(): pred=model(bg,detach_sigma_features=False)
        fixed_action,state,embedding=direction_state(source,graphs,pred,reliability); magnitude.train(); optimizer.zero_grad(set_to_none=True); tau=magnitude(embedding,(state-means)/stds); action=proposal_from_tau(source,graphs,fixed_action,tau); post=action_loss(action,reference,graphs,pred); move=(tau/config["magnitude"]["tau_max_angstrom"]).square().mean(); loss=post+config["magnitude"]["lambda_move"]*move
        if not bool(torch.isfinite(loss)): raise RuntimeError(f"nonfinite loss {arm} {step}")
        loss.backward(); grad=torch.nn.utils.clip_grad_norm_(magnitude.parameters(),train_cfg["gradient_clip"])
        if not bool(torch.isfinite(grad)): raise RuntimeError(f"nonfinite gradient {arm} {step}")
        optimizer.step();scheduler.step()
        if step==1 or step%train_cfg["log_interval"]==0:
            row={"step":step,"loss":float(loss.detach()),"post":float(post.detach()),"move":float(move.detach()),"tau_mean":float(tau.mean()),"tau_p05":float(torch.quantile(tau,.05)),"tau_median":float(torch.quantile(tau,.5)),"tau_p95":float(torch.quantile(tau,.95)),"grad_norm":float(grad),"step_seconds":time.time()-tick};logs.append(row);status("TRAINING","RUNNING",CURRENT_ARM=arm,CURRENT_STEP=step,TOTAL_STEPS=train_cfg["optimizer_steps"],**{k.upper():v for k,v in row.items() if k!="step"});print(json.dumps({"arm":arm,**row}),flush=True)
        if step%train_cfg["recovery_interval"]==0:
            atomic_torch(p["recovery"],{"schema_version":"sixs-joint-magnitude-recovery-v1","arm":arm,"step":step,"config_sha256":sha256(CONFIG_PATH),"magnitude_state":magnitude.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"generator_state":generator.get_state(),"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []});atomic_frame(p["log"],pd.DataFrame(logs))
    atomic_torch(p["final"],{"schema_version":"sixs-joint-magnitude-final-v1","arm":arm,"step":train_cfg["optimizer_steps"],"config_sha256":sha256(CONFIG_PATH),"frozen_checkpoint_sha256":config["frozen_checkpoints"][frozen_name(arm)]["sha256"],"magnitude_state":magnitude.state_dict(),**protected()});atomic_frame(p["log"],pd.DataFrame(logs));status("FINAL_CHECKPOINT_FROZEN","PASS",CURRENT_ARM=arm,CURRENT_STEP=train_cfg["optimizer_steps"],TOTAL_STEPS=train_cfg["optimizer_steps"],FINAL_CHECKPOINT_SHA256=sha256(p["final"]))


def coordinates(arm: str) -> None:
    p=paths(arm)
    if p["coordinates"].is_file(): return
    target=device(); prepared,sources,manifest=load_data(); by_item={str(x["molecule_id"]):x for x in prepared["val"]}; by_sample={str(x["sample_id"]):x for x in sources["val"]}; ids=[s for row in manifest["rows"] for s in row["sample_ids"]]; model,reliability=load_frozen(arm,target); magnitude=make_magnitude(target); magnitude.load_state_dict(torch.load(p["final"],map_location="cpu",weights_only=False)["magnitude_state"]);magnitude.eval();means,stds,_=state_normalization(target)
    val_manifest=pd.read_parquet(frozen.cfg()["data"]["val_manifest"]); manifest_by_sample={str(x.sample_id):x for x in val_manifest.itertuples(index=False)}; p["sdf"].parent.mkdir(parents=True,exist_ok=True);temp=Path(str(p["sdf"])+f".tmp.{os.getpid()}");writer=Chem.SDWriter(str(temp));out=[]
    try:
        for start in range(0,len(ids),64):
            batch_ids=ids[start:start+64]; source_rows=[by_sample[x] for x in batch_ids]; items=[by_item[str(x["molecule_id"])] for x in source_rows]; graphs=[x["graph"] for x in items]; bg=collate_graphs(graphs).to(target);source=torch.cat([torch.as_tensor(x["source"],dtype=torch.float64) for x in source_rows]).to(target);reference=torch.cat([torch.as_tensor(x["references"][0],dtype=torch.float64) for x in items]).to(target)
            with torch.no_grad(): pred=model(bg,detach_sigma_features=False)
            fixed_action,state,embedding=direction_state(source,graphs,pred,reliability)
            with torch.no_grad(): tau=magnitude(embedding,(state-means)/stds)
            action=proposal_from_tau(source,graphs,fixed_action,tau);ref_b,ref_a=geometry_values(reference,bg);src_b,src_a=geometry_values(source,bg);prop_b,prop_a=geometry_values(action.proposal,bg);bo=ao=atoms=0
            for local,(sample_id,source_row,item,graph) in enumerate(zip(batch_ids,source_rows,items,graphs,strict=True)):
                nb,na,nat=graph.bonds.size(1),graph.angles.size(0),graph.atom_categorical.size(0);bb=slice(bo,bo+nb);aa=slice(ao,ao+na);xx=slice(atoms,atoms+nat);bo+=nb;ao+=na;atoms+=nat
                yb=ref_b[bb].detach().cpu().numpy();ya=ref_a[aa].detach().cpu().numpy();sb=src_b[bb].detach().cpu().numpy();sa=src_a[aa].detach().cpu().numpy();pb=prop_b[bb].detach().cpu().numpy();pa=prop_a[aa].detach().cpu().numpy();bstat=bg.bond_fixed[bb,1].detach().cpu().numpy();astat=bg.angle_fixed[aa,1].detach().cpu().numpy();bs=pred["bond_sigma"][bb].detach().cpu().numpy();ass=pred["angle_sigma"][aa].detach().cpu().numpy();common_source=.5*(np.mean(((sb-yb)/bstat)**2)+np.mean(((sa-ya)/astat)**2));common_post=.5*(np.mean(((pb-yb)/bstat)**2)+np.mean(((pa-ya)/astat)**2));own_post=.5*(np.mean(((pb-yb)/bs)**2)+np.mean(((pa-ya)/ass)**2));delta=action.proposal[xx]-source[xx];movement=float(delta.square().sum(-1).mean().sqrt());_,safe=safety_accept(source[xx].detach().cpu(),action.proposal[xx].detach().cpu(),graph)
                out.append({"record_id":sample_id,"molecule_id":str(source_row["molecule_id"]),"arm":arm,"common_source_objective":common_source,"internal_post":common_post,"own_sigma_post":own_post,"direction_improvement":common_source-common_post,"bond_raw_mae":float(np.mean(np.abs(pb-yb))),"angle_raw_mae":float(np.mean(np.abs(pa-ya))),"source_rmsd":movement,"proposal_movement":movement,"tau":float(tau[local]),"atom_cap_active":bool(action.cap_active[local]),"rollback":bool(safe["fallback"])})
                meta=manifest_by_sample[sample_id];cache=Path(frozen.cfg()["data"]["val_cache"])/Path(str(meta.source_path)).name;raw=cache.read_bytes()
                if hashlib.sha256(raw).hexdigest()!=str(meta.source_file_sha256):raise RuntimeError("DEV cache hash changed")
                record=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=False);frozen.write_molecule(writer,record,action.proposal[xx],sample_id,arm)
            status("DEV_COORDINATES","RUNNING",CURRENT_ARM=arm,EVALUATION_RECORDS=min(start+64,len(ids)),EVALUATION_TOTAL=len(ids))
    finally: writer.close()
    os.replace(temp,p["sdf"]);atomic_frame(p["per_record"],pd.DataFrame(out));atomic_json(p["coordinates"],{"schema_version":"sixs-joint-coordinates-ready-v1","status":"PASS","arm":arm,"records":len(out),"record_ids_sha256":hashlib.sha256("\n".join(ids).encode()).hexdigest(),"checkpoint_sha256":sha256(p["final"]),"sdf_sha256":sha256(p["sdf"]),"per_record_sha256":sha256(p["per_record"]),**protected()});status("COORDINATES_READY","PASS",CURRENT_ARM=arm,EVALUATION_RECORDS=len(ids),EVALUATION_TOTAL=len(ids))


def evaluate(arm: str) -> None:
    p=paths(arm)
    if p["done"].is_file(): return
    if not p["coordinates"].is_file(): raise RuntimeError("coordinates not ready")
    records=pd.read_parquet(p["per_record"]);ids=records.record_id.astype(str).tolist();status("EXTERNAL_EVALUATION","RUNNING",CURRENT_ARM=arm)
    frozen.run_external_evaluators(arm,p,ids);pb=pd.read_parquet(p["pb"]);v3d=pd.read_parquet(p["v3d"]);records=records.merge(pb[["record_id","PB"]],on="record_id",validate="one_to_one").merge(v3d[["record_id","validity3d"]].rename(columns={"validity3d":"V3D"}),on="record_id",validate="one_to_one");atomic_frame(p["per_record"],records)
    summary={"arm":arm,"records":len(records),"molecules":records.molecule_id.nunique(),"proposal_V3D":float(records.V3D.mean()),"proposal_PB":float(records.PB.mean()),"internal_post":float(records.internal_post.mean()),"direction_improvement":float(records.direction_improvement.mean()),"bond_raw_mae":float(records.bond_raw_mae.mean()),"angle_raw_mae":float(records.angle_raw_mae.mean()),"source_rmsd":float(records.source_rmsd.mean()),"proposal_movement":float(records.proposal_movement.mean()),"tau_mean":float(records.tau.mean()),"tau_median":float(records.tau.median()),"tau_p05":float(records.tau.quantile(.05)),"tau_p95":float(records.tau.quantile(.95)),"atom_cap_fraction":float(records.atom_cap_active.mean()),"rollback_fraction":float(records.rollback.mean())}
    atomic_json(p["result"],{"schema_version":"sixs-joint-magnitude-result-v1","status":"COMPLETE","summary":summary,"artifact_hashes":{"final":sha256(p["final"]),"per_record":sha256(p["per_record"]),"sdf":sha256(p["sdf"]),"pb":sha256(p["pb"]),"v3d":sha256(p["v3d"])},**protected()});atomic_json(p["done"],{"schema_version":"sixs-joint-magnitude-done-v1","status":"COMPLETE","arm":arm,"result_sha256":sha256(p["result"]),**protected()});status("ARM_COMPLETE","PASS",CURRENT_ARM=arm,**{k.upper():v for k,v in summary.items() if isinstance(v,(int,float))})


def bootstrap_delta(left: pd.DataFrame,right: pd.DataFrame,metric: str,seed: int) -> dict[str,Any]:
    joined=left[["record_id","molecule_id",metric]].merge(right[["record_id",metric]],on="record_id",suffixes=("_left","_right"),validate="one_to_one");joined["delta"]=joined[f"{metric}_right"].astype(float)-joined[f"{metric}_left"].astype(float);clusters=joined.groupby("molecule_id",sort=True).delta.mean().to_numpy();rng=np.random.default_rng(seed);draws=np.empty(cfg()["evaluation"]["bootstrap_resamples"]);n=len(clusters)
    for start in range(0,len(draws),500):
        count=min(500,len(draws)-start);draws[start:start+count]=clusters[rng.integers(0,n,size=(count,n))].mean(1)
    return {"delta_candidate_minus_baseline":float(clusters.mean()),"ci95_low":float(np.percentile(draws,2.5)),"ci95_high":float(np.percentile(draws,97.5)),"clusters":n,"resamples":len(draws),"seed":seed}


def interaction(frames: Mapping[str,pd.DataFrame],metric: str,seed: int) -> dict[str,Any]:
    cols=["record_id","molecule_id",metric];x=frames["J1-R0-FIXED"][cols].rename(columns={metric:"r0f"})
    for name,col in (("J1-R1-FIXED","r1f"),("J1-R0-JOINT","r0j"),("J1-R1-JOINT","r1j")):
        x=x.merge(frames[name][["record_id",metric]].rename(columns={metric:col}),on="record_id",validate="one_to_one")
    for column in ("r0f", "r1f", "r0j", "r1j"):
        x[column] = x[column].astype(float)
    x["interaction"]=(x.r1j-x.r0j)-(x.r1f-x.r0f);cluster=x.groupby("molecule_id",sort=True).interaction.mean().to_numpy();rng=np.random.default_rng(seed);draws=np.empty(cfg()["evaluation"]["bootstrap_resamples"]);n=len(cluster)
    for start in range(0,len(draws),500):
        count=min(500,len(draws)-start);draws[start:start+count]=cluster[rng.integers(0,n,size=(count,n))].mean(1)
    return {"interaction":float(cluster.mean()),"ci95_low":float(np.percentile(draws,2.5)),"ci95_high":float(np.percentile(draws,97.5)),"clusters":n,"resamples":len(draws),"seed":seed}


def finalize() -> None:
    if not all(paths(a)["done"].is_file() for a in ARMS): raise RuntimeError("both joint arms must complete")
    frames={k:pd.read_parquet(v) for k,v in FIXED.items()};frames.update({a:pd.read_parquet(paths(a)["per_record"]) for a in ARMS})
    summary=[]
    for name,frame in frames.items():
        row={"arm":name,"molecules":frame.molecule_id.nunique(),"records":len(frame),"proposal_V3D":float(frame.V3D.mean()),"proposal_PB":float(frame.PB.mean()),"internal_post":float(frame.internal_post.mean()),"direction_improvement":float(frame.direction_improvement.mean()),"bond_raw_mae":float(frame.bond_raw_mae.mean()),"angle_raw_mae":float(frame.angle_raw_mae.mean()),"source_rmsd":float(frame.source_rmsd.mean()),"proposal_movement":float(frame.proposal_movement.mean()),"atom_cap_fraction":float(frame.atom_cap_active.mean()),"rollback_fraction":float(frame.rollback.mean())}
        if "tau" in frame: row.update({"tau_mean":float(frame.tau.mean()),"tau_median":float(frame.tau.median()),"tau_p05":float(frame.tau.quantile(.05)),"tau_p95":float(frame.tau.quantile(.95))})
        else: row.update({"tau_mean":cfg()["magnitude"]["fixed_reproduction_tau_angstrom"],"tau_median":cfg()["magnitude"]["fixed_reproduction_tau_angstrom"],"tau_p05":cfg()["magnitude"]["fixed_reproduction_tau_angstrom"],"tau_p95":cfg()["magnitude"]["fixed_reproduction_tau_angstrom"]})
        summary.append(row)
    atomic_frame(REPORT/"FOUR_CELL_SUMMARY.csv",pd.DataFrame(summary))
    comparisons=(("A_RELIABILITY_FIXED","J1-R0-FIXED","J1-R1-FIXED"),("B_RELIABILITY_JOINT","J1-R0-JOINT","J1-R1-JOINT"),("C_JOINT_R0","J1-R0-FIXED","J1-R0-JOINT"),("D_JOINT_R1","J1-R1-FIXED","J1-R1-JOINT"));metrics=("V3D","PB","source_rmsd","internal_post");rows=[]
    for ci,(label,left,right) in enumerate(comparisons):
        for mi,metric in enumerate(metrics): rows.append({"comparison":label,"baseline":left,"candidate":right,"metric":metric,**bootstrap_delta(frames[left],frames[right],metric,cfg()["evaluation"]["bootstrap_seed"]+ci*100+mi)})
    boot=pd.DataFrame(rows);atomic_frame(REPORT/"PAIRED_BOOTSTRAP.csv",boot)
    interactions={m:interaction(frames,m,cfg()["evaluation"]["bootstrap_seed"]+900+i) for i,m in enumerate(("V3D","internal_post","source_rmsd"))}
    get=lambda comp,metric:boot[(boot.comparison==comp)&(boot.metric==metric)].iloc[0]
    rj=get("B_RELIABILITY_JOINT","V3D");jr1=get("D_JOINT_R1","V3D");pb_rj=get("B_RELIABILITY_JOINT","PB");pb_jr1=get("D_JOINT_R1","PB");tol=cfg()["evaluation"]["pb_material_drop_tolerance"]
    if jr1.ci95_low>0 and rj.ci95_low>0 and pb_rj.delta_candidate_minus_baseline>=-tol and pb_jr1.delta_candidate_minus_baseline>=-tol: synergy="SUPPORTED";preserved="YES";interaction_class="POSITIVE_OR_COMPLEMENTARY"
    elif rj.ci95_low>0: synergy="NO";preserved="YES";interaction_class="PRESERVED_WITHOUT_ADDITIONAL_JOINT_GAIN"
    else: synergy="NO";preserved="NO";interaction_class="NEGATIVE_OR_REDUNDANT"
    next_candidate="DIRECT_J1-R1_JOINT_MAGNITUDE" if jr1.ci95_low>0 and rj.ci95_low>0 and pb_rj.delta_candidate_minus_baseline>=-tol and pb_jr1.delta_candidate_minus_baseline>=-tol else "DIRECT_J1-R1_FIXED_MOVEMENT"
    md="# J1-R1 x Joint-Magnitude interaction analysis\n\n"+pd.DataFrame(summary).to_markdown(index=False)+"\n\n## Interaction\n\n```text\n"+"\n".join([f"INTERACTION_{k.upper()} = {v['interaction']:.8f}, 95% CI [{v['ci95_low']:.8f}, {v['ci95_high']:.8f}]" for k,v in interactions.items()])+f"\nJOINT_RELIABILITY_SYNERGY = {synergy}\nRELIABILITY_EFFECT_PRESERVED = {preserved}\nJOINT_RELIABILITY_INTERACTION = {interaction_class}\n```\n"
    (REPORT/"INTERACTION_ANALYSIS.md").write_text(md,encoding="utf-8")
    decision={"JOINT_RELIABILITY_SYNERGY":synergy,"RELIABILITY_EFFECT_PRESERVED":preserved,"JOINT_RELIABILITY_INTERACTION":interaction_class,"NEXT_CANDIDATE":next_candidate,"interactions":interactions,"primary_comparisons":{"reliability_joint":rj.to_dict(),"joint_r1":jr1.to_dict()},"artifact_hashes":{"FOUR_CELL_SUMMARY.csv":sha256(REPORT/"FOUR_CELL_SUMMARY.csv"),"PAIRED_BOOTSTRAP.csv":sha256(REPORT/"PAIRED_BOOTSTRAP.csv"),"INTERACTION_ANALYSIS.md":sha256(REPORT/"INTERACTION_ANALYSIS.md")},**protected()};atomic_json(REPORT/"FINAL_DECISION.json",decision);(REPORT/"FINAL_DECISION.md").write_text("# SIXS J1-R1 x Joint-Magnitude final decision\n\n"+pd.DataFrame(summary).to_markdown(index=False)+"\n\n```text\n"+"\n".join(f"{k} = {v}" for k,v in decision.items() if isinstance(v,str))+"\n```\n",encoding="utf-8");status("FINAL_DECISION","COMPLETE",**decision)


def main() -> int:
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True);sub.add_parser("gates");p=sub.add_parser("train-coordinate");p.add_argument("arm",choices=ARMS);p=sub.add_parser("evaluate");p.add_argument("arm",choices=ARMS);sub.add_parser("finalize");args=parser.parse_args()
    try:
        if args.command=="gates": gates()
        elif args.command=="train-coordinate": train(args.arm);coordinates(args.arm)
        elif args.command=="evaluate": evaluate(args.arm)
        else: finalize()
        return 0
    except Exception as exc:
        status("FAIL_CLOSED","FAIL",ERROR_TYPE=type(exc).__name__,ERROR=str(exc));raise


if __name__=="__main__": raise SystemExit(main())
