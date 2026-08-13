#!/usr/bin/env python3
"""Audit, preflight and train the development-only LSGO-BA v2 seed307 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
from etflow.ecir.formal_rdkit_adapter import adapt_formal_cache_record
from etflow.ecir.learned_geometry import (
    distribution_parameters, geometry_values, gaussian_nll, parameter_count,
    safety_accept, structured_objective,
)
from etflow.ecir.lsgo_io import center_coordinates, validate_record_identity
from etflow.ecir.lsgoba_v2_joint_magnitude import (
    JointMagnitudeLSGO, SOURCE_STATE_FEATURES,
    batch_source_directions_and_state, scaled_proposal,
)
from scripts.run_mcvr_lsgo import collate_graphs


CONFIG_PATH = ROOT / "configs/ecir_mvr_lsgoba_v2_joint_magnitude_full307.json"
REPORT = ROOT / "reports/ecir_mvr/lsgoba_v2_joint_magnitude_full307"
ARTIFACT = ROOT / "artifacts/ecir_mvr/lsgoba_v2_joint_magnitude_full307"
STATUS = REPORT / "V2_FULL307_STATUS.json"
SOURCE_PAYLOAD = ARTIFACT / "SOURCE_BINDING.pt"
BEST = ARTIFACT / "checkpoints/best_v2_seed307.ckpt"
LAST = ARTIFACT / "checkpoints/last_v2_seed307.ckpt"


def config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def coordinate_sha(value: torch.Tensor) -> str:
    # Exact frozen formal-large manifest convention (formal_target_assets.tensor_sha256).
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def reference_sha(value: torch.Tensor) -> str:
    value = torch.as_tensor(value, dtype=torch.float32).cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8"); os.replace(tmp, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(value.rstrip() + "\n", encoding="utf-8"); os.replace(tmp, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, tmp); os.replace(tmp, path)


def status(stage: str, state: str = "RUNNING", **extra: Any) -> None:
    current = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    if state == "RUNNING":
        current.pop("error", None); current.pop("error_type", None)
    current.update({"schema_version": "lsgoba-v2-full307-status-v1", "status": state, "stage": stage, "worker_pid": os.getpid(), "heartbeat": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(), "formal_test_records_read": 0, "frozen_holdout_records_read": 0, "xtb_stage": "NOT_STARTED", "orca_stage": "NOT_STARTED", **extra})
    atomic_json(STATUS, current)


def seed_all(seed: int) -> torch.Generator:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.Generator().manual_seed(seed + 91000)


def load_prepared(cfg: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(cfg["v1"]["prepared_payload"])
    if sha256(path) != cfg["v1"]["prepared_sha256"]: raise RuntimeError("frozen v1 prepared payload SHA changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if len(payload["train"]) != 50000 or len(payload["val"]) != 5000 or payload.get("formal_test_records_read") != 0 or payload.get("frozen_holdout_records_read") != 0: raise RuntimeError("prepared payload protection/denominator failure")
    return payload


def load_record(path: Path, expected_sha: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha: raise RuntimeError(f"source file SHA changed: {path}")
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)


def audit_sources() -> None:
    cfg = config(); REPORT.mkdir(parents=True, exist_ok=True); ARTIFACT.mkdir(parents=True, exist_ok=True)
    status("SOURCE_RECOVERY", completed_records=0, expected_records=160000)
    prepared = load_prepared(cfg); summaries={}; split_payload={}
    for split in ("train", "val"):
        spec=cfg["sources"]; manifest_path=Path(spec[f"{split}_manifest"]); cache=Path(spec[f"{split}_cache"])
        if sha256(manifest_path) != spec[f"{split}_manifest_sha256"]: raise RuntimeError(f"{split} manifest SHA changed")
        frame=pd.read_parquet(manifest_path).sort_values(["molecule_id","sample_id"],kind="stable").reset_index(drop=True)
        expected_records=int(spec[f"expected_{split}_records"]); expected_molecules=int(spec[f"expected_{split}_molecules"]); expected_per=int(spec[f"expected_{split}_sources_per_molecule"])
        if len(frame)!=expected_records or frame.molecule_id.nunique()!=expected_molecules or set(frame.split.astype(str))!={split} or frame.test_record.fillna(False).astype(bool).any() or set(frame.groupby("molecule_id").size().astype(int))!={expected_per}: raise RuntimeError(f"{split} manifest denominator/protection failure")
        items=prepared[split]; by_id={str(item["molecule_id"]):item for item in items}
        if set(frame.molecule_id.astype(str)) != set(by_id): raise RuntimeError(f"{split} prepared/manifest molecule mismatch")
        rows=[]; binding=[]; atom_ok=mol_ok=graph_ok=reference_ok=True
        for pos,row in enumerate(frame.itertuples(index=False),start=1):
            molecule_id=str(row.molecule_id); item=by_id[molecule_id]; path=cache/Path(str(row.source_path)).name; record=load_record(path,str(row.source_file_sha256)); validate_record_identity(record)
            source=torch.as_tensor(record["x_init"],dtype=torch.float32)
            if coordinate_sha(source)!=str(row.coordinate_sha256): raise RuntimeError(f"coordinate SHA mismatch: {row.sample_id}")
            if str(record["sample_id"])!=str(row.sample_id) or str(record["source_mol_id"])!=molecule_id: mol_ok=False; raise RuntimeError(f"molecule binding mismatch: {row.sample_id}")
            graph=item["graph"]
            if source.shape!=(graph.atom_categorical.size(0),3) or int(record["num_atoms"])!=graph.atom_categorical.size(0) or not torch.equal(torch.as_tensor(record["edge_index"],dtype=torch.long),graph.edge_index): graph_ok=False; raise RuntimeError(f"graph/atom mismatch: {row.sample_id}")
            if not torch.equal(torch.as_tensor(record["atomic_numbers"]),torch.as_tensor(record["x_init_atomic_numbers"])) or not torch.equal(torch.as_tensor(record["atomic_numbers"]),torch.as_tensor(record["x_ref_atomic_numbers"])): atom_ok=False; raise RuntimeError(f"atom order mismatch: {row.sample_id}")
            hashes=[]
            for reference in torch.as_tensor(record["x_ref_candidates"],dtype=torch.float32):
                digest=reference_sha(center_coordinates(reference))
                if digest not in hashes: hashes.append(digest)
            if hashes != list(item["reference_coordinate_sha256"]): reference_ok=False; raise RuntimeError(f"Reference ensemble mismatch: {row.sample_id}")
            rows.append({"sample_id":str(row.sample_id),"molecule_id":molecule_id,"source":center_coordinates(source),"coordinate_sha256":str(row.coordinate_sha256),"source_file_sha256":str(row.source_file_sha256)})
            binding.append((str(row.sample_id),str(row.source_file_sha256),str(row.coordinate_sha256)))
            if pos%1000==0: status("SOURCE_RECOVERY",completed_records=(pos if split=="train" else 150000+pos),expected_records=160000,active_split=split); print(json.dumps({"stage":"SOURCE_RECOVERY","split":split,"records":pos,"expected":expected_records}),flush=True)
        split_payload[split]=rows
        summaries[split]={"records":len(rows),"molecules":len(by_id),"atom_order_audit":"PASS" if atom_ok else "FAIL","molecule_identity_audit":"PASS" if mol_ok else "FAIL","graph_identity_audit":"PASS" if graph_ok else "FAIL","reference_ensemble_audit":"PASS" if reference_ok else "FAIL","source_binding_sha256":canonical_sha(binding)}
    train_ids={r["molecule_id"] for r in split_payload["train"]}; val_ids={r["molecule_id"] for r in split_payload["val"]}
    if train_ids & val_ids: raise RuntimeError("TRAIN/VAL molecule leakage")
    atomic_torch(SOURCE_PAYLOAD,{"schema_version":"lsgoba-v2-source-binding-v1","train":split_payload["train"],"val":split_payload["val"],"summaries":summaries,"formal_test_records_read":0,"frozen_holdout_records_read":0})
    audit={"schema_version":"lsgoba-v2-source-recovery-audit-v1","status":"PASS","train":summaries["train"],"validation":summaries["val"],"train_val_overlap":0,"source_payload":str(SOURCE_PAYLOAD),"source_payload_sha256":sha256(SOURCE_PAYLOAD),"formal_test_used_for_training":False,"formal_test_records_read":0,"frozen_holdout_records_read":0}
    atomic_json(ARTIFACT/"SOURCE_BINDING_FREEZE.json",audit)
    atomic_text(REPORT/"00_SOURCE_RECOVERY_AUDIT.md",f"""# Source recovery audit

Status: **PASS**. Every formal-large TRAIN and legal development-VALIDATION source cache was verified against its frozen manifest file SHA, raw coordinate SHA, molecule/sample identity, explicit atom identity/order, graph edge identity, atom count, and the exact v1 Reference-ensemble hashes.

- TRAIN Source: {summaries['train']['records']:,} records / {summaries['train']['molecules']:,} molecules (3 uniformly sampleable Sources per molecule)
- legal development VALIDATION Source: {summaries['val']['records']:,} records / {summaries['val']['molecules']:,} molecules
- TRAIN/VALIDATION molecule overlap: 0
- synthetic/corrupted Source: none
- formal test / frozen holdout records read: 0 / 0
- frozen Source payload SHA256: `{audit['source_payload_sha256']}`
""")
    status("SOURCE_RECOVERY",state="PASS",completed_records=160000,expected_records=160000)


def freeze_manifests() -> None:
    cfg=config(); prepared=load_prepared(cfg); source=torch.load(SOURCE_PAYLOAD,map_location="cpu",weights_only=False)
    by_molecule={};
    for row in source["val"]: by_molecule.setdefault(row["molecule_id"],[]).append(row)
    tag=cfg["validation"]["split_tag"]
    ordered=sorted(by_molecule,key=lambda value:hashlib.sha256(f"{tag}|{value}".encode()).hexdigest())
    val_ids=ordered[:2500]; dev_ids=ordered[2500:]
    if len(val_ids)!=2500 or len(dev_ids)!=2500 or set(val_ids)&set(dev_ids): raise RuntimeError("V2 VAL/DEV_TEST split failure")
    prepared_ids={str(item["molecule_id"]) for item in prepared["val"]}
    if set(ordered)!=prepared_ids: raise RuntimeError("development split does not cover v1 validation molecules")
    def manifest(role: str, ids: list[str]) -> dict[str,Any]:
        rows=[{"molecule_id":mid,"sample_ids":[x["sample_id"] for x in by_molecule[mid]],"source_coordinate_sha256":[x["coordinate_sha256"] for x in by_molecule[mid]]} for mid in ids]
        return {"schema_version":"lsgoba-v2-development-split-v1","role":role,"status":"FROZEN_BEFORE_TRAINING","split_rule":"SHA256 rank by molecule identity","split_tag":tag,"molecules":len(ids),"source_records":sum(len(x["sample_ids"]) for x in rows),"rows":rows,"formal_test_records_read":0,"frozen_holdout_records_read":0}
    for name,ids in (("V2_VAL",val_ids),("V2_DEV_TEST",dev_ids)):
        path=REPORT/f"{name}_MANIFEST.json"; atomic_json(path,manifest(name,ids)); (REPORT/f"{name}_MANIFEST.sha256").write_text(sha256(path)+"\n",encoding="utf-8")
    rule={"schema_version":"lsgoba-v2-checkpoint-rule-v1","status":"FROZEN_BEFORE_TRAINING","candidate_steps":cfg["training"]["checkpoint_steps"],"prior_catastrophic_absolute_increase_max":cfg["validation"]["prior_catastrophic_absolute_increase_max"],"selection_rule":cfg["validation"]["selection_rule"],"external_metrics_used":False,"formal_test_records_read":0,"frozen_holdout_records_read":0}
    atomic_json(REPORT/"V2_CHECKPOINT_RULE.json",rule)
    atomic_text(REPORT/"04_DATA_MANIFEST_AUDIT.md",f"""# V2 data-manifest audit

The existing molecule-disjoint formal-large training validation partition was used only as legal development data and deterministically re-frozen before v2 training.

- V2_VAL: 2,500 molecules / 5,000 real Source records; SHA256 `{sha256(REPORT/'V2_VAL_MANIFEST.json')}`
- V2_DEV_TEST: 2,500 molecules / 5,000 real Source records; SHA256 `{sha256(REPORT/'V2_DEV_TEST_MANIFEST.json')}`
- V2_VAL/V2_DEV_TEST overlap: 0
- overlap with v2 TRAIN: 0 (inherited frozen formal-large TRAIN/VAL separation)
- V2_DEV_TEST outcomes read before training: no
- formal test / frozen holdout records read: 0 / 0
""")


def make_model(cfg: Mapping[str,Any],device: torch.device) -> JointMagnitudeLSGO:
    model=JointMagnitudeLSGO(hidden_dim=cfg["model"]["hidden_dim"],layers=cfg["model"]["layers"],initial_tau=cfg["model"]["initial_tau_angstrom"],tau_max=cfg["model"]["tau_max_angstrom"])
    checkpoint=Path(cfg["v1"]["checkpoint"])
    if sha256(checkpoint)!=cfg["v1"]["checkpoint_sha256"]: raise RuntimeError("v1 seed307 checkpoint SHA changed")
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False); model.geometry.load_state_dict(payload["model_state"],strict=True)
    if parameter_count(model.geometry)!=cfg["v1"]["parameter_count"]: raise RuntimeError("v1 parameter identity changed")
    return model.to(device)


def source_index(source_payload: Mapping[str,Any], split: str) -> dict[str,list[dict]]:
    grouped={}
    for row in source_payload[split]: grouped.setdefault(row["molecule_id"],[]).append(row)
    for rows in grouped.values(): rows.sort(key=lambda x:x["sample_id"])
    return grouped


def sample_batch(items: Sequence[dict], sources: Mapping[str,list[dict]], generator: torch.Generator, batch: int, device: torch.device):
    chosen=torch.randint(len(items),(batch,),generator=generator).tolist(); graphs=[]; source_values=[]; references=[]
    for index in chosen:
        item=items[index]; pool=sources[str(item["molecule_id"])]; si=int(torch.randint(len(pool),(1,),generator=generator)); ri=int(torch.randint(len(item["references"]),(1,),generator=generator))
        graphs.append(item["graph"]); source_values.append(pool[si]["source"]); references.append(item["references"][ri])
    batch_graph=collate_graphs(graphs).to(device)
    # Frozen v1 deployment/direction semantics explicitly enter the Cartesian
    # objective in float64. Reference prior supervision remains in the original
    # formal-training float32 path.
    return graphs, batch_graph, torch.cat(source_values).to(device=device, dtype=torch.float64), torch.cat(references).to(device), chosen


def loss_terms(model: JointMagnitudeLSGO, graphs: Sequence[Any], batch_graph: Any, source: torch.Tensor, reference: torch.Tensor, atom_cap: float=0.03):
    prediction=model.geometry(batch_graph); parameters={**prediction,"bond_sigma":batch_graph.bond_fixed[:,1],"angle_sigma":batch_graph.angle_fixed[:,1]}
    prior, prior_groups=structured_objective(reference,batch_graph,parameters)
    direction,state,graph_embedding,gradient_diag=batch_source_directions_and_state(source,graphs,parameters,prediction["node_embedding"])
    tau=model.magnitude(graph_embedding,model.normalized_state(state)); proposal,cap_active,proposal_graph_rms=scaled_proposal(source,direction,tau,graphs,atom_cap=atom_cap)
    prop_b,prop_a=geometry_values(proposal,batch_graph); ref_b,ref_a=geometry_values(reference,batch_graph)
    post_b=((prop_b-ref_b)/parameters["bond_sigma"]).square().mean(); post_a=((prop_a-ref_a)/parameters["angle_sigma"]).square().mean(); post=.5*(post_b+post_a)
    move=(tau/model.magnitude.tau_max).square().mean()
    return {"prior":prior,"prior_b":prior_groups["bond"],"prior_a":prior_groups["angle"],"post":post,"post_b":post_b,"post_a":post_a,"move":move,"tau":tau,"proposal":proposal,"direction":direction,"state":state,"cap_active":cap_active,"proposal_graph_rms":proposal_graph_rms,"gradient_diag":gradient_diag}


def preflight() -> None:
    cfg=config(); freeze_manifests(); device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); prepared=load_prepared(cfg); source_payload=torch.load(SOURCE_PAYLOAD,map_location="cpu",weights_only=False); sources=source_index(source_payload,"train"); generator=seed_all(cfg["seed"]+1); model=make_model(cfg,device); model.train()
    states=[]; sampled=[]
    for i in range(cfg["preflight"]["batches"]):
        graphs,batch_graph,source,reference,chosen=sample_batch(prepared["train"],sources,generator,cfg["preflight"]["batch_molecules"],device)
        prediction=model.geometry(batch_graph); params={**prediction,"bond_sigma":batch_graph.bond_fixed[:,1],"angle_sigma":batch_graph.angle_fixed[:,1]}
        _,state,_,_=batch_source_directions_and_state(source,graphs,params,prediction["node_embedding"]); states.append(state.detach().cpu()); sampled.extend(chosen)
        if (i+1)%16==0: status("LOSS_SCALE_PREFLIGHT_NORMALIZATION",completed_batches=i+1,expected_batches=cfg["preflight"]["batches"]); print(json.dumps({"stage":"PREFLIGHT_NORMALIZATION","batches":i+1}),flush=True)
    all_state=torch.cat(states); mean=all_state.mean(0); std=all_state.std(0,unbiased=False).clamp_min(cfg["preflight"]["normalization_std_floor"]); model.set_state_normalization(mean,std)
    generator=seed_all(cfg["seed"]+1); rows=[]
    for i in range(cfg["preflight"]["batches"]):
        graphs,batch_graph,source,reference,_=sample_batch(prepared["train"],sources,generator,cfg["preflight"]["batch_molecules"],device); terms=loss_terms(model,graphs,batch_graph,source,reference)
        prior_grad=torch.autograd.grad(terms["prior"],tuple(model.geometry.parameters()),retain_graph=True,allow_unused=True); post_phi=torch.autograd.grad(terms["post"],tuple(model.magnitude.parameters()),retain_graph=True,allow_unused=True)
        if i==0:
            post_theta=torch.autograd.grad(terms["post"],tuple(model.geometry.parameters()),retain_graph=False,allow_unused=True)
            post_theta_norm=float(torch.sqrt(sum((x.detach().square().sum() for x in post_theta if x is not None),terms["post"].new_zeros(()))))
        else: post_theta_norm=None
        norm=lambda values:float(torch.sqrt(sum((x.detach().square().sum() for x in values if x is not None),terms["post"].new_zeros(()))))
        rows.append({"prior":float(terms["prior"].detach()),"post":float(terms["post"].detach()),"move":float(terms["move"].detach()),"tau_mean":float(terms["tau"].mean().detach()),"grad_prior_theta":norm(prior_grad),"grad_post_phi":norm(post_phi),"grad_post_theta_tau_path":post_theta_norm})
        if (i+1)%16==0: status("LOSS_SCALE_PREFLIGHT",completed_batches=i+1,expected_batches=cfg["preflight"]["batches"]); print(json.dumps({"stage":"LOSS_SCALE_PREFLIGHT","batches":i+1}),flush=True)
    frame=pd.DataFrame(rows); med_post=float(frame.post.median()); med_move=float(frame.move.median()); selected_lambda=.05*med_post/max(med_move,1e-12)
    values={name:{"mean":float(frame[name].mean()),"median":float(frame[name].median()),"p90":float(frame[name].quantile(.9))} for name in ("prior","post","move")}
    finite=bool(np.isfinite(frame[["prior","post","move","tau_mean","grad_prior_theta","grad_post_phi"]].to_numpy()).all() and math.isfinite(selected_lambda) and selected_lambda>0)
    if not finite: status("LOSS_SCALE_PREFLIGHT",state="BLOCKED_LOSS_SCALE_PATHOLOGY"); raise RuntimeError("BLOCKED_LOSS_SCALE_PATHOLOGY")
    report={"schema_version":"lsgoba-v2-loss-scale-preflight-v1","status":"PASS","train_only":True,"batches":len(frame),"molecule_draws":len(frame)*cfg["preflight"]["batch_molecules"],"initial_tau_mean_empirical":float(frame.tau_mean.mean()),"statistics":values,"gradient_norms":{"prior_to_theta_mean":float(frame.grad_prior_theta.mean()),"post_to_phi_mean":float(frame.grad_post_phi.mean()),"post_to_theta_through_tau_path_first_batch":float(frame.grad_post_theta_tau_path.dropna().iloc[0])},"selected_alpha":1.0,"selected_lambda":selected_lambda,"lambda_selection_rule":"5 percent of initial median post-loss scale","state_feature_names":list(SOURCE_STATE_FEATURES),"state_mean":mean.tolist(),"state_std":std.tolist(),"sampled_molecule_index_sha256":canonical_sha(sampled),"formal_test_records_read":0,"frozen_holdout_records_read":0}
    atomic_json(REPORT/"02_LOSS_SCALE_PREFLIGHT.json",report); status("LOSS_SCALE_PREFLIGHT",state="PASS",selected_lambda=selected_lambda)


def architecture_report() -> None:
    cfg=config(); model=make_model(cfg,torch.device("cpu")); head=parameter_count(model.magnitude); total=parameter_count(model); increase=100*head/cfg["v1"]["parameter_count"]
    atomic_text(REPORT/"01_V2_ARCHITECTURE.md",f"""# LSGO-BA v2 architecture

The frozen v1 graph encoder and Bond/cosine-Angle conditional-mean heads are retained and warm-started from seed307. Frozen DRCSR sigmas remain buffers supplied by each graph. The existing 128-dimensional node embedding is mean-pooled per molecule and concatenated with 17 TRAIN-normalized Source-state scalars. The only added network is `Linear(145,64) -> GELU -> Linear(64,32) -> GELU -> Linear(32,1)`.

- magnitude-head parameters: {head:,}
- total v2 trainable parameters: {total:,}
- parameter increase: {increase:.6f}%
- tau: `0.010 * sigmoid(a)`; final weights zero; initial bias `{cfg['model']['initial_final_bias']}`; initial tau 0.003 Å
- structured direction / rigid removal / atom cap / deployment safety: unchanged
- direction recomputed from current theta each optimizer step and detached before post loss
- Cartesian Reference loss / learned sigma / second GNN / second-order direction backprop: absent
""")


def manifest_ids(name: str) -> set[str]:
    path=REPORT/f"{name}_MANIFEST.json"; expected=(REPORT/f"{name}_MANIFEST.sha256").read_text().strip()
    if sha256(path)!=expected: raise RuntimeError(f"{name} manifest changed")
    return {row["molecule_id"] for row in json.loads(path.read_text(encoding="utf-8"))["rows"]}


def deterministic_eval(model: JointMagnitudeLSGO, items: Sequence[dict], source_map: Mapping[str,list[dict]], device: torch.device, selected_ids: set[str], lam: float, batch_size: int=64) -> dict[str,float]:
    chosen=[item for item in items if str(item["molecule_id"]) in selected_ids]; rows=[]; model.eval()
    for start in range(0,len(chosen),batch_size):
        group=chosen[start:start+batch_size]; graphs=[item["graph"] for item in group]; bg=collate_graphs(graphs).to(device); src=torch.cat([source_map[str(item["molecule_id"])][0]["source"] for item in group]).to(device); ref=torch.cat([item["references"][0] for item in group]).to(device)
        with torch.enable_grad(): terms=loss_terms(model,graphs,bg,src,ref)
        rows.append((len(group),float(terms["prior"].detach()),float(terms["post"].detach()),float(terms["move"].detach()),float(terms["tau"].mean().detach())))
    total=sum(x[0] for x in rows); avg=lambda i:sum(x[0]*x[i] for x in rows)/total
    return {"molecules":total,"prior":avg(1),"post":avg(2),"move":avg(3),"selection_objective":avg(2)+lam*avg(3),"tau_mean":avg(4)}


def checkpoint_payload(model,optimizer,scheduler,generator,step,lam,normalization,validation,logs):
    return {"schema_version":"lsgoba-v2-full307-checkpoint-v1","seed":307,"global_step":step,"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),"sampler_generator_state":generator.get_state(),"python_rng_state":random.getstate(),"numpy_rng_state":np.random.get_state(),"torch_rng_state":torch.get_rng_state(),"cuda_rng_state":torch.cuda.get_rng_state_all(),"lambda":lam,"normalization":normalization,"validation":validation,"log_rows":len(logs),"v1_checkpoint_sha256":config()["v1"]["checkpoint_sha256"],"config_sha256":sha256(CONFIG_PATH),"formal_test_records_read":0,"frozen_holdout_records_read":0}


def train() -> None:
    cfg=config(); pre=json.loads((REPORT/"02_LOSS_SCALE_PREFLIGHT.json").read_text(encoding="utf-8")); lam=float(pre["selected_lambda"]); device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); prepared=load_prepared(cfg); sp=torch.load(SOURCE_PAYLOAD,map_location="cpu",weights_only=False); train_sources=source_index(sp,"train"); val_sources=source_index(sp,"val"); val_ids=manifest_ids("V2_VAL")
    generator=seed_all(cfg["seed"]); model=make_model(cfg,device); model.set_state_normalization(torch.tensor(pre["state_mean"]),torch.tensor(pre["state_std"])); head_params=list(model.magnitude.parameters()); backbone_params=list(model.geometry.parameters()); optimizer=torch.optim.AdamW([{"params":backbone_params,"lr":cfg["training"]["backbone_learning_rate"]},{"params":head_params,"lr":cfg["training"]["head_learning_rate"]}],weight_decay=cfg["training"]["weight_decay"]); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg["training"]["scheduler_horizon"])
    logs=[]; validations=[]; start=0
    if LAST.is_file():
        payload=torch.load(LAST,map_location="cpu",weights_only=False); model.load_state_dict(payload["model_state"],strict=True); optimizer.load_state_dict(payload["optimizer_state"]); scheduler.load_state_dict(payload["scheduler_state"]); generator.set_state(payload["sampler_generator_state"]); random.setstate(payload["python_rng_state"]); np.random.set_state(payload["numpy_rng_state"]); torch.set_rng_state(payload["torch_rng_state"]); torch.cuda.set_rng_state_all(payload["cuda_rng_state"]); start=int(payload["global_step"])
        if (REPORT/"05_TRAIN_LOG.csv").is_file(): logs=pd.read_csv(REPORT/"05_TRAIN_LOG.csv").to_dict("records")
        if (REPORT/"VALIDATION_CHECKPOINTS.csv").is_file(): validations=pd.read_csv(REPORT/"VALIDATION_CHECKPOINTS.csv").to_dict("records")
    initial=deterministic_eval(model,prepared["val"],val_sources,device,val_ids,lam) if start==0 else json.loads((REPORT/"INITIAL_VAL.json").read_text())
    if start==0: atomic_json(REPORT/"INITIAL_VAL.json",initial)
    checkpoints=set(cfg["training"]["checkpoint_steps"]); started=time.time()
    for step in range(start+1,cfg["training"]["optimizer_steps"]+1):
        tick=time.time(); model.train(); graphs,bg,src,ref,_=sample_batch(prepared["train"],train_sources,generator,cfg["training"]["batch_molecules"],device); optimizer.zero_grad(set_to_none=True); terms=loss_terms(model,graphs,bg,src,ref); total=terms["prior"]+terms["post"]+lam*terms["move"]
        if not bool(torch.isfinite(total)): raise RuntimeError(f"nonfinite total loss step {step}")
        total.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["training"]["gradient_clip"])
        if not bool(torch.isfinite(grad)): raise RuntimeError(f"nonfinite gradient step {step}")
        optimizer.step(); scheduler.step()
        if step==1 or step%cfg["training"]["log_interval"]==0:
            tau=terms["tau"].detach().cpu().numpy(); safety_rate=float("nan")
            if step%100==0:
                offset=0; failures=0
                for graph in graphs:
                    n=graph.atom_categorical.size(0); _,safe=safety_accept(src[offset:offset+n].detach().cpu().double(),terms["proposal"][offset:offset+n].detach().cpu().double(),graph); failures+=int(safe["fallback"]); offset+=n
                safety_rate=failures/len(graphs)
            logs.append({"step":step,"L_prior":float(terms["prior"].detach()),"L_post":float(terms["post"].detach()),"L_move":float(terms["move"].detach()),"L_total":float(total.detach()),"tau_mean":float(tau.mean()),"tau_median":float(np.median(tau)),"tau_p10":float(np.quantile(tau,.1)),"tau_p25":float(np.quantile(tau,.25)),"tau_p75":float(np.quantile(tau,.75)),"tau_p90":float(np.quantile(tau,.9)),"tau_p95":float(np.quantile(tau,.95)),"tau_lt001":float((tau<.001).mean()),"tau_le003":float((tau<=.003).mean()),"tau_003_005":float(((tau>.003)&(tau<=.005)).mean()),"tau_gt005":float((tau>.005).mean()),"tau_ge009":float((tau>=.009).mean()),"graph_rms_proposal_mean":float(terms["proposal_graph_rms"].mean().detach()),"atom_cap_activation_rate":float(terms["cap_active"].mean().detach()),"training_safety_rollback_rate":safety_rate,"gradient_norm":float(grad),"backbone_lr":scheduler.get_last_lr()[0],"head_lr":scheduler.get_last_lr()[1],"step_time_seconds":time.time()-tick})
        validation={}
        if step in checkpoints:
            validation={"step":step,**deterministic_eval(model,prepared["val"],val_sources,device,val_ids,lam)}; validation["prior_eligible"]=validation["prior"]<=initial["prior"]+cfg["validation"]["prior_catastrophic_absolute_increase_max"]; validations.append(validation); print(json.dumps({"stage":"CHECKPOINT","step":step,"validation":validation}),flush=True)
        if step%cfg["training"]["recovery_checkpoint_interval"]==0 or step in checkpoints:
            payload=checkpoint_payload(model,optimizer,scheduler,generator,step,lam,{"mean":pre["state_mean"],"std":pre["state_std"]},validation,logs)
            atomic_torch(LAST,payload)
            if step in checkpoints: atomic_torch(ARTIFACT/f"checkpoints/step{step:05d}.ckpt",payload)
            pd.DataFrame(logs).to_csv(REPORT/"05_TRAIN_LOG.csv",index=False); pd.DataFrame(validations).to_csv(REPORT/"VALIDATION_CHECKPOINTS.csv",index=False); status("TRAINING",completed_steps=step,expected_steps=12500,elapsed_seconds=time.time()-started)
        if step%25==0: print(json.dumps({"stage":"TRAINING","step":step,"expected":12500,"loss":float(total.detach()),"tau_mean":float(terms["tau"].mean().detach())}),flush=True)
    eligible=[row for row in validations if bool(row["prior_eligible"])]
    if not eligible: status("TRAINING",state="BLOCKED",reason="NO_PRIOR_ELIGIBLE_CHECKPOINT"); raise RuntimeError("no prior-eligible v2 checkpoint")
    best=min(eligible,key=lambda x:(x["selection_objective"],x["step"])); source=ARTIFACT/f"checkpoints/step{int(best['step']):05d}.ckpt"
    # Candidate checkpoint payloads are materialized from recovery checkpoints at each frozen candidate.
    if int(best["step"])==12500: shutil.copyfile(LAST,BEST)
    elif source.is_file(): shutil.copyfile(source,BEST)
    else: raise RuntimeError("selected candidate checkpoint missing")
    atomic_text(REPORT/"07_CHECKPOINT_AUDIT.md",f"""# V2 checkpoint audit

- selection rule frozen before training: {cfg['validation']['selection_rule']}
- initial v1-warm-start VAL prior: {initial['prior']:.9f}
- selected step: {int(best['step'])}
- selected VAL prior/post/move/selection objective: {best['prior']:.9f} / {best['post']:.9f} / {best['move']:.9f} / {best['selection_objective']:.9f}
- selected checkpoint SHA256: `{sha256(BEST)}`
- V3D/PoseBusters/xTB/ORCA used for selection: no
""")
    status("TRAINING",state="PASS",completed_steps=12500,selected_step=int(best["step"]),best_checkpoint=str(BEST),best_checkpoint_sha256=sha256(BEST))


def write_config_artifact() -> None:
    cfg=config(); payload={**cfg,"config_path":str(CONFIG_PATH),"config_sha256":sha256(CONFIG_PATH),"branch":subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),"head_before_experiment_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"v1_frozen_unchanged":True,"formal_test_records_read":0,"frozen_holdout_records_read":0}; atomic_json(REPORT/"03_V2_CONFIG.json",payload)


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("phase",choices=("audit","preflight","train","all","status")); args=parser.parse_args()
    if args.phase=="status": print(STATUS.read_text(encoding="utf-8") if STATUS.is_file() else json.dumps({"status":"NOT_STARTED"},indent=2)); return 0
    try:
        architecture_report(); write_config_artifact()
        if args.phase in ("audit","all") or not SOURCE_PAYLOAD.is_file(): audit_sources()
        if args.phase in ("preflight","all"): preflight()
        if args.phase in ("train","all"): train()
    except Exception as exc:
        status("FAILED",state="FAILED",error_type=type(exc).__name__,error=str(exc)); raise
    return 0


if __name__=="__main__": raise SystemExit(main())
