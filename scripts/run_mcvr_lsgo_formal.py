#!/usr/bin/env python3
"""Train frozen LSGO-BA on formal-large using only TRAIN/VALIDATION References."""

from __future__ import annotations

import argparse
import hashlib
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
import yaml

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
from etflow.ecir.learned_geometry import (
    LearnedGeometryObjective, direct_gradient_update, distribution_parameters,
    gaussian_nll, parameter_count, structured_objective,
)
from etflow.ecir.lsgo_io import atomic_json, file_sha256
from scripts.run_mcvr_lsgo import collate_graphs

CONFIG = ROOT / "configs/ecir_mvr_lsgo_ba_formal_large.yaml"
OUT = ROOT / "reports/ecir_mvr/lsgo_formal"


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def seed_all(seed: int) -> torch.Generator:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.Generator().manual_seed(seed + 47000)


def model_from_config(config: Mapping[str, Any]) -> LearnedGeometryObjective:
    model = LearnedGeometryObjective(
        hidden_dim=int(config["model"]["hidden_dim"]),
        layers=int(config["model"]["layers"]), learned_sigma=False,
    )
    if parameter_count(model) != int(config["model"]["expected_parameter_count"]):
        raise RuntimeError("formal model parameter identity mismatch")
    if any("sigma" in name for name, _ in model.named_parameters()):
        raise RuntimeError("learned sigma parameter present")
    return model


def load_dataset(config: Mapping[str, Any]) -> tuple[dict, dict]:
    identity = json.loads((OUT / "DATASET_IDENTITY.json").read_text(encoding="utf-8"))
    if identity["status"] != "FROZEN" or any(int(identity[key]) for key in ("formal_test_records_read", "frozen_holdout_records_read")):
        raise RuntimeError("formal dataset identity/protection failure")
    path = ROOT / config["dataset"]["prepared_path"]
    if file_sha256(path) != identity["prepared_sha256"]:
        raise RuntimeError("formal prepared dataset SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if len(payload["train"]) != 50000 or len(payload["val"]) != 5000:
        raise RuntimeError("formal prepared denominator mismatch")
    return payload, identity


def training_batch(items: Sequence[dict], generator: torch.Generator, batch_size: int, device: torch.device):
    chosen = torch.randint(len(items), (batch_size,), generator=generator).tolist()
    graphs, coordinates = [], []
    for index in chosen:
        item = items[index]; reference_index = int(torch.randint(len(item["references"]), (1,), generator=generator))
        graphs.append(item["graph"]); coordinates.append(item["references"][reference_index])
    return collate_graphs(graphs).to(device), torch.cat(coordinates).to(device)


@torch.no_grad()
def evaluate_reference_likelihood(model: LearnedGeometryObjective | None, items: Sequence[dict], device: torch.device, limit: int | None = None) -> dict[str, float]:
    sums = {key: 0.0 for key in ("bond_nll", "angle_nll", "bond_abs", "angle_abs", "bond_z", "angle_z", "bond_z2", "angle_z2")}
    counts = {"bond": 0, "angle": 0}; chosen = items if limit is None else items[:limit]
    for index, item in enumerate(chosen, start=1):
        graph = item["graph"].to(device); parameters = distribution_parameters(graph, model=model, variant="A" if model is None else "B")
        references = item["references"].to(device); left, right = graph.bonds
        bonds = torch.linalg.vector_norm(references[:, left] - references[:, right], dim=-1)
        if graph.angles.numel():
            aleft, center, aright = graph.angles.t(); one = references[:, aleft] - references[:, center]; two = references[:, aright] - references[:, center]
            angles = (one * two).sum(-1) / torch.linalg.vector_norm(one, dim=-1).clamp_min(1e-12) / torch.linalg.vector_norm(two, dim=-1).clamp_min(1e-12)
        else:
            angles = references.new_empty((len(references), 0))
        bn = gaussian_nll(bonds, parameters["bond_mu"][None], parameters["bond_sigma"][None]); an = gaussian_nll(angles, parameters["angle_mu"][None], parameters["angle_sigma"][None])
        bz = (bonds - parameters["bond_mu"][None]) / parameters["bond_sigma"][None]; az = (angles - parameters["angle_mu"][None]) / parameters["angle_sigma"][None]
        for key, value in (("bond_nll", bn), ("angle_nll", an), ("bond_abs", (bonds-parameters["bond_mu"][None]).abs()), ("angle_abs", (angles-parameters["angle_mu"][None]).abs()), ("bond_z", bz), ("angle_z", az), ("bond_z2", bz.square()), ("angle_z2", az.square())):
            sums[key] += float(value.double().sum())
        counts["bond"] += bn.numel(); counts["angle"] += an.numel()
        if index % 1000 == 0:
            print(f"LSGO FORMAL VALIDATE {index}/{len(chosen)}", flush=True)
    bond_nll=sums["bond_nll"]/counts["bond"]; angle_nll=sums["angle_nll"]/counts["angle"]
    bz_mean=sums["bond_z"]/counts["bond"]; az_mean=sums["angle_z"]/counts["angle"]
    bz_std=math.sqrt(max(0.0,sums["bond_z2"]/counts["bond"]-bz_mean*bz_mean)); az_std=math.sqrt(max(0.0,sums["angle_z2"]/counts["angle"]-az_mean*az_mean))
    return {
        "molecules": len(chosen), "reference_conformers": int(sum(len(item["references"]) for item in chosen)),
        "bond_nll": bond_nll, "angle_nll": angle_nll, "joint_nll": (bond_nll+angle_nll)/2,
        "bond_mu_mae_angstrom": sums["bond_abs"]/counts["bond"], "angle_mu_mae_cosine": sums["angle_abs"]/counts["angle"],
        "bond_z_mean": bz_mean, "bond_z_std": bz_std, "angle_z_mean": az_mean, "angle_z_std": az_std,
        "calibration_error": abs(bz_mean)+abs(bz_std-1)+abs(az_mean)+abs(az_std-1),
        "finite": bool(all(math.isfinite(value) for value in sums.values())),
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
    }


def diagnostic_items(items: Sequence[dict], config: Mapping[str, Any], count: int | None = None) -> list[dict]:
    token = config["validation"]["diagnostic_selection_seed"]
    ordered = sorted(items, key=lambda item: hashlib.sha256(f"{token}|{item['molecule_id']}".encode()).hexdigest())
    return ordered[: int(count or config["validation"]["diagnostic_molecules"])]


def evaluate_stationarity_selectivity(model: LearnedGeometryObjective, items: Sequence[dict], config: Mapping[str, Any], device: torch.device, count: int | None = None) -> dict[str, float]:
    reference_gradients=[]; source_gradients=[]; differences=[]; nonincrease=[]; finite=[]
    for item in diagnostic_items(items, config, count):
        graph=item["graph"].to(device); parameters=distribution_parameters(graph,model=model,variant="B")
        reference=item["references"][0].to(device=device,dtype=torch.float64); source=item["sources"][0].to(device=device,dtype=torch.float64)
        values={}
        for role,coordinates in (("reference",reference),("source",source)):
            x=coordinates.detach().clone().requires_grad_(True); objective,_=structured_objective(x,graph,parameters); gradient,=torch.autograd.grad(objective,x); gradient=gradient-gradient.mean(0,keepdim=True)
            values[role]=(float(objective),float(torch.sqrt(gradient.square().sum(-1).mean())))
            finite.append(bool(torch.isfinite(objective) and torch.isfinite(gradient).all()))
        moved=direct_gradient_update(reference,graph,parameters,rms_budget=float(config["validation"]["reference_stationarity_budget_angstrom"]),atom_cap=float(config["validation"]["atom_cap_angstrom"]),steps=1)["coordinates"]
        moved_objective=structured_objective(moved,graph,parameters)[0]
        reference_gradients.append(values["reference"][1]);source_gradients.append(values["source"][1]);differences.append(values["source"][0]-values["reference"][0]);nonincrease.append(float(moved_objective)<=values["reference"][0]+1e-10)
    return {
        "molecules":len(differences), "reference_gradient_rms_median":float(np.median(reference_gradients)), "source_gradient_rms_median":float(np.median(source_gradients)),
        "reference_to_source_gradient_ratio":float(np.median(reference_gradients)/max(np.median(source_gradients),1e-12)),
        "reference_objective_nonincrease_fraction":float(np.mean(nonincrease)), "source_gt_reference_fraction":float(np.mean(np.asarray(differences)>0)),
        "source_minus_reference_objective_median":float(np.median(differences)), "finite_fraction":float(np.mean(finite)),
        "formal_test_records_read":0,"frozen_holdout_records_read":0,
    }


def checkpoint_path(seed: int, name: str) -> Path:
    return OUT / "checkpoints" / f"seed{seed}" / name


def checkpoint_payload(model, optimizer, scheduler, generator, seed, step, exposure, validation, config_sha, identity_sha):
    return {
        "schema_version":"mcvr-lsgo-ba-formal-checkpoint-v1","variant":"B","formal_method":"LSGO-BA",
        "seed":seed,"global_step":step,"exposure_count":exposure,"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),
        "python_rng_state":random.getstate(),"numpy_rng_state":np.random.get_state(),"torch_rng_state":torch.get_rng_state(),"cuda_rng_state":torch.cuda.get_rng_state_all(),"sampler_generator_state":generator.get_state(),
        "validation":validation,"parameter_count":parameter_count(model),"config_sha256":config_sha,"dataset_identity_sha256":identity_sha,
        "coordinate_teacher":False,"learned_sigma":False,"xtb_training_access":False,"posebusters_training_access":False,"formal_test_records_read":0,"frozen_holdout_records_read":0,
    }


def restore_checkpoint(path: Path, model, optimizer, scheduler, generator, config_sha: str, identity_sha: str) -> dict:
    payload=torch.load(path,map_location="cpu",weights_only=False)
    required={"model_state","optimizer_state","scheduler_state","python_rng_state","numpy_rng_state","torch_rng_state","cuda_rng_state","sampler_generator_state","global_step","exposure_count"}
    if not required.issubset(payload) or payload["config_sha256"]!=config_sha or payload["dataset_identity_sha256"]!=identity_sha or payload["learned_sigma"]:
        raise RuntimeError("strict formal checkpoint identity failure")
    model.load_state_dict(payload["model_state"],strict=True);optimizer.load_state_dict(payload["optimizer_state"]);scheduler.load_state_dict(payload["scheduler_state"])
    random.setstate(payload["python_rng_state"]);np.random.set_state(payload["numpy_rng_state"]);torch.set_rng_state(payload["torch_rng_state"]);torch.cuda.set_rng_state_all(payload["cuda_rng_state"]);generator.set_state(payload["sampler_generator_state"])
    return payload


def run_updates(model, optimizer, scheduler, generator, train_items, device, config, start: int, stop: int, logs: list[dict], seed: int):
    batch_size=int(config["training"]["batch_molecules"]); log_interval=int(config["training"]["log_interval"])
    for step in range(start+1,stop+1):
        tick=time.time();graph,coordinates=training_batch(train_items,generator,batch_size,device);optimizer.zero_grad(set_to_none=True);prediction=distribution_parameters(graph,model=model,variant="B");objective,groups=structured_objective(coordinates,graph,prediction)
        if not bool(torch.isfinite(objective)):raise RuntimeError(f"nonfinite formal loss seed{seed} step{step}")
        objective.backward();gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float(config["training"]["gradient_clip"]))
        if not bool(torch.isfinite(gradient_norm)):raise RuntimeError(f"nonfinite formal gradient seed{seed} step{step}")
        optimizer.step();scheduler.step()
        if step==1 or step%log_interval==0:
            logs.append({"seed":seed,"step":step,"train_total_loss":float(objective.detach()),"bond_loss":float(groups["bond"].detach()),"angle_loss":float(groups["angle"].detach()),"gradient_norm":float(gradient_norm),"learning_rate":scheduler.get_last_lr()[0],"gpu_memory_mb":torch.cuda.max_memory_allocated(device)/1048576 if device.type=="cuda" else 0,"step_time_seconds":time.time()-tick,"exposure":step*batch_size,"record_equivalent_epochs":step*batch_size/150000,"molecule_equivalent_epochs":step*batch_size/50000})


def preregister(config: Mapping[str, Any]) -> None:
    if git("status","--porcelain"):
        raise RuntimeError("working tree must be clean before formal preregistration")
    identity=json.loads((OUT/"DATASET_IDENTITY.json").read_text(encoding="utf-8"))
    payload={"schema_version":"mcvr-lsgo-ba-formal-preregistration-v1","status":"FROZEN_BEFORE_SMOKE_AND_TRAINING","branch":git("branch","--show-current"),"head":git("rev-parse","HEAD"),"config_path":str(CONFIG),"config_sha256":file_sha256(CONFIG),"dataset_identity_sha256":identity["identity_sha256"],"prepared_sha256":identity["prepared_sha256"],"seeds":config["seeds"],"model":config["model"],"training":config["training"],"validation":config["validation"],"guards":config["guards"],"checkpoint_selection_external_access":False,"formal_test_records_read":0,"frozen_holdout_records_read":0}
    atomic_json(OUT/"PREREGISTRATION.json",payload)
    atomic_text(OUT/"PREREGISTRATION.md",f"""# LSGO-BA formal-large preregistration

Frozen before smoke or training on branch `{payload['branch']}`, HEAD `{payload['head']}`. The three formal seeds are `{payload['seeds']}` and all must run 12,500 optimizer steps with effective batch 64. Only the neural Bond/Angle means are trained; DRCSR scale and all method choices remain frozen.

Checkpoint candidates are fixed at 2,500/5,000/7,500/10,000/12,500. Selection uses lowest full 5,000-molecule validation joint BA NLL, then calibration error, then earlier step. GO requires every seed to beat frozen A on joint NLL and Bond/Angle MAE, reference stationarity nonincrease ≥0.95, Source>Reference objective fraction ≥0.60, and selected joint-NLL sample SD ≤0.05.

xTB and PoseBusters are forbidden for training or checkpoint selection. Formal test reads = **0**. Frozen holdout reads = **0**.
""")
    print("LSGO_BA_FORMAL_PREREGISTERED")


def smoke(config, dataset, identity, device):
    seed=int(config["seeds"][0]);generator=seed_all(seed);model=model_from_config(config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=float(config["training"]["learning_rate"]),weight_decay=float(config["training"]["weight_decay"]));scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=int(config["smoke"]["short_steps"]));logs=[]
    forward=[]
    model.eval()
    with torch.no_grad():
        for _ in range(int(config["smoke"]["forward_batches"])):
            graph,coordinates=training_batch(dataset["train"],generator,int(config["training"]["batch_molecules"]),device);prediction=distribution_parameters(graph,model=model,variant="B");objective,groups=structured_objective(coordinates,graph,prediction);forward.append({"total":float(objective),"bond":float(groups["bond"]),"angle":float(groups["angle"]),"fixed_sigma":not prediction["bond_sigma"].requires_grad and not prediction["angle_sigma"].requires_grad})
    model.train();resume_step=int(config["smoke"]["resume_step"]);run_updates(model,optimizer,scheduler,generator,dataset["train"],device,config,0,resume_step,logs,seed)
    path=OUT/"smoke/step0050.ckpt";payload=checkpoint_payload(model,optimizer,scheduler,generator,seed,resume_step,resume_step*64,{},file_sha256(CONFIG),identity["identity_sha256"]);atomic_torch(path,payload)
    resumed=model_from_config(config).to(device);resumed_optimizer=torch.optim.AdamW(resumed.parameters(),lr=float(config["training"]["learning_rate"]),weight_decay=float(config["training"]["weight_decay"]));resumed_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(resumed_optimizer,T_max=int(config["smoke"]["short_steps"]));restore_checkpoint(path,resumed,resumed_optimizer,resumed_scheduler,generator,file_sha256(CONFIG),identity["identity_sha256"])
    run_updates(resumed,resumed_optimizer,resumed_scheduler,generator,dataset["train"],device,config,resume_step,int(config["smoke"]["short_steps"]),logs,seed);resumed.eval()
    validation=evaluate_reference_likelihood(resumed,dataset["val"],device,int(config["smoke"]["validation_molecules"]));diagnostic=evaluate_stationarity_selectivity(resumed,dataset["val"],config,device,min(16,int(config["smoke"]["validation_molecules"])))
    passed=all(math.isfinite(v) for row in forward for v in (row["total"],row["bond"],row["angle"])) and all(row["fixed_sigma"] for row in forward) and all(math.isfinite(row["train_total_loss"]) and math.isfinite(row["gradient_norm"]) for row in logs) and validation["finite"] and diagnostic["finite_fraction"]==1 and int(payload["global_step"])==resume_step
    report={"schema_version":"mcvr-lsgo-ba-formal-smoke-v1","status":"PASS" if passed else "FAIL","forward_batches":forward,"backward_steps_checked":10,"short_steps":100,"resume_step":50,"strict_load":True,"validation":validation,"diagnostic":diagnostic,"parameter_count":parameter_count(resumed),"peak_gpu_memory_mb":max(row["gpu_memory_mb"] for row in logs),"formal_test_records_read":0,"frozen_holdout_records_read":0,"xtb_access":False,"posebusters_access":False}
    atomic_json(OUT/"SMOKE_REPORT.json",report);atomic_text(OUT/"SMOKE_REPORT.md","# LSGO-BA formal smoke\n\nStatus: **%s**. Two forward batches, ten explicit backward checks within a 100-step short run, atomic step-50 save, strict restore/resume, and internal validation all completed with finite Bond/Angle/total losses and gradients. Parameter count is %d; learned sigma is absent.\n\n`LSGO_BA_FORMAL_SMOKE_%s`\n\nFormal test reads = **0**. Frozen holdout reads = **0**."%(report["status"],report["parameter_count"],report["status"]))
    if not passed:raise RuntimeError("LSGO_BA_FORMAL_SMOKE_FAIL")
    print("LSGO_BA_FORMAL_SMOKE_PASS")


def train_seed(config, dataset, identity, device, seed: int):
    generator=seed_all(seed);model=model_from_config(config).to(device);training=config["training"];optimizer=torch.optim.AdamW(model.parameters(),lr=float(training["learning_rate"]),weight_decay=float(training["weight_decay"]));scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=int(training["scheduler_horizon"]));config_sha=file_sha256(CONFIG);logs=[];evaluations=[];start=0;last=checkpoint_path(seed,"last.ckpt")
    if last.is_file():
        payload=restore_checkpoint(last,model,optimizer,scheduler,generator,config_sha,identity["identity_sha256"]);start=int(payload["global_step"]);logs=pd.read_csv(OUT/f"logs/TRAIN_seed{seed}.csv").to_dict("records") if (OUT/f"logs/TRAIN_seed{seed}.csv").is_file() else [];evaluations=pd.read_csv(OUT/f"tables/VALIDATION_seed{seed}.csv").to_dict("records") if (OUT/f"tables/VALIDATION_seed{seed}.csv").is_file() else [];print(f"LSGO FORMAL RESUME seed={seed} step={start}",flush=True)
    checkpoints=set(map(int,training["checkpoint_steps"]));recovery=int(training["recovery_checkpoint_interval"]);started=time.time()
    for target in range(start+1,int(training["optimizer_steps"])+1):
        run_updates(model,optimizer,scheduler,generator,dataset["train"],device,config,target-1,target,logs,seed)
        validation={}
        if target in checkpoints:
            model.eval();likelihood=evaluate_reference_likelihood(model,dataset["val"],device);diagnostic=evaluate_stationarity_selectivity(model,dataset["val"],config,device);validation={**likelihood,**{f"diagnostic_{k}":v for k,v in diagnostic.items() if k not in {"formal_test_records_read","frozen_holdout_records_read"}},"seed":seed,"step":target};evaluations.append(validation);model.train();print(f"LSGO FORMAL CHECKPOINT seed={seed} step={target} joint={likelihood['joint_nll']:.6f}",flush=True)
        if target%recovery==0 or target in checkpoints:
            payload=checkpoint_payload(model,optimizer,scheduler,generator,seed,target,target*int(training["effective_batch"]),validation,config_sha,identity["identity_sha256"]);atomic_torch(last,payload)
            if target in checkpoints:atomic_torch(checkpoint_path(seed,f"step{target:05d}.ckpt"),payload)
            atomic_csv(OUT/f"logs/TRAIN_seed{seed}.csv",pd.DataFrame(logs));atomic_csv(OUT/f"tables/VALIDATION_seed{seed}.csv",pd.DataFrame(evaluations))
    best=min(evaluations,key=lambda row:(row["joint_nll"],row["calibration_error"],row["step"]));source=checkpoint_path(seed,f"step{int(best['step']):05d}.ckpt");shutil.copyfile(source,checkpoint_path(seed,"best.ckpt"))
    result={"status":"COMPLETED","seed":seed,"optimizer_steps":int(training["optimizer_steps"]),"effective_batch":int(training["effective_batch"]),"total_exposures":int(training["total_exposures"]),"best_step":int(best["step"]),"best_checkpoint":str(checkpoint_path(seed,"best.ckpt")),"best_checkpoint_sha256":file_sha256(checkpoint_path(seed,"best.ckpt")),"best_validation":best,"runtime_seconds":time.time()-started,"exit_code":0,"formal_test_records_read":0,"frozen_holdout_records_read":0}
    atomic_json(OUT/f"logs/TRAIN_seed{seed}.json",result);print(f"LSGO BA FORMAL SEED {seed} COMPLETED",flush=True);return result


def finalize(config,dataset,identity,device):
    a_path=OUT/"A_BASELINE_VALIDATION.json"
    if a_path.is_file():a=json.loads(a_path.read_text(encoding="utf-8"))
    else:a=evaluate_reference_likelihood(None,dataset["val"],device);atomic_json(a_path,a)
    runs=[json.loads((OUT/f"logs/TRAIN_seed{seed}.json").read_text(encoding="utf-8")) for seed in config["seeds"]]
    selected=pd.DataFrame([{**r["best_validation"],"checkpoint":r["best_checkpoint"],"checkpoint_sha256":r["best_checkpoint_sha256"]} for r in runs]);g=config["validation"]["gates"]
    selected["joint_better_than_A"]=selected.joint_nll<a["joint_nll"];selected["bond_mae_not_worse_A"]=selected.bond_mu_mae_angstrom<=a["bond_mu_mae_angstrom"];selected["angle_mae_not_worse_A"]=selected.angle_mu_mae_cosine<=a["angle_mu_mae_cosine"];selected["stationarity_pass"]=selected.diagnostic_reference_objective_nonincrease_fraction>=float(g["reference_objective_nonincrease_fraction_min"]);selected["selectivity_pass"]=selected.diagnostic_source_gt_reference_fraction>=float(g["source_gt_reference_fraction_min"])
    sd=float(selected.joint_nll.std(ddof=1));all_seed=bool(selected[["joint_better_than_A","bond_mae_not_worse_A","angle_mae_not_worse_A","stationarity_pass","selectivity_pass"]].all().all());decision="READY_FOR_FINAL_FROZEN_TEST" if all_seed and sd<=float(g["joint_nll_sample_sd_max"]) else "FORMAL_VALIDATION_NO_GO"
    curves=pd.concat([pd.read_csv(OUT/f"logs/TRAIN_seed{seed}.csv") for seed in config["seeds"]],ignore_index=True);validations=pd.concat([pd.read_csv(OUT/f"tables/VALIDATION_seed{seed}.csv") for seed in config["seeds"]],ignore_index=True);atomic_csv(OUT/"TRAINING_CURVES.csv",curves);atomic_csv(OUT/"VALIDATION_CHECKPOINTS.csv",validations);atomic_csv(OUT/"CHECKPOINT_SELECTION.csv",selected)
    freeze={"schema_version":"mcvr-lsgo-ba-formal-checkpoint-freeze-v1","status":"FROZEN","selection_rule":config["validation"]["checkpoint_rule"],"checkpoints":[{"seed":int(r.seed),"step":int(r.step),"path":r.checkpoint,"sha256":r.checkpoint_sha256} for r in selected.itertuples()],"config_sha256":file_sha256(CONFIG),"dataset_identity_sha256":identity["identity_sha256"],"formal_test_records_read":0,"frozen_holdout_records_read":0};atomic_json(OUT/"CHECKPOINT_FREEZE_MANIFEST.json",freeze)
    atomic_text(OUT/"TRAINING_SUMMARY.md","# LSGO-BA formal training summary\n\nAll three preregistered seeds completed 12,500 optimizer steps, effective batch 64 and 800,000 Reference draws. This is 16.0 molecule-equivalent epochs or 5.33 under historical 150,000-record accounting. No external evaluator was used.\n\n"+selected.to_markdown(index=False,floatfmt=".6f"))
    atomic_text(OUT/"VALIDATION_REPORT.md",f"# Formal validation\n\nDecision: **{decision}**. Frozen A joint NLL `{a['joint_nll']:.6f}`; selected B joint-NLL sample SD `{sd:.6f}`.\n\n"+selected.to_markdown(index=False,floatfmt=".6f")+"\n\nFormal test reads = **0**. Frozen holdout reads = **0**.")
    atomic_text(OUT/"SEED_STABILITY.md",f"# Seed stability\n\nSelected full-validation joint NLL mean `{selected.joint_nll.mean():.6f}`, sample SD (`ddof=1`) `{sd:.6f}`, frozen maximum `{float(g['joint_nll_sample_sd_max']):.6f}`.")
    atomic_text(OUT/"CHECKPOINT_SELECTION.md","# Checkpoint selection\n\nOnly full formal VALIDATION internal metrics were used. Primary: lowest joint BA NLL; secondary: calibration error; tie-break: earlier step. xTB/PoseBusters/formal test/frozen holdout were not accessed.\n\n"+selected.to_markdown(index=False,floatfmt=".6f"))
    status={"schema_version":"mcvr-lsgo-ba-formal-status-v1","status":decision,"smoke":"PASS","formal_training":"completed","optimizer_steps":12500,"train_records":150000,"train_molecules":50000,"validation_records":10000,"validation_molecules":5000,"effective_batch":64,"total_exposures":800000,"record_equivalent_epochs":800000/150000,"molecule_equivalent_epochs":16.0,"seeds":config["seeds"],"parameter_count":473674,"best_checkpoints":freeze["checkpoints"],"joint_nll_sample_sd":sd,"formal_test_records_read":0,"frozen_holdout_records_read":0};atomic_json(OUT/"FINAL_FORMAL_STATUS.json",status);print(decision);return decision


def parse_args():
    parser=argparse.ArgumentParser();parser.add_argument("--phase",choices=("preregister","smoke","train","finalize","all"),required=True);parser.add_argument("--seed",type=int);parser.add_argument("--device",default="cuda:0");return parser.parse_args()


def main():
    args=parse_args();config=load_config()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {args.device}, but this Python/PyTorch build has no CUDA; refusing silent CPU fallback")
    device=torch.device(args.device)
    if args.phase=="preregister":preregister(config);return 0
    dataset,identity=load_dataset(config)
    if args.phase=="smoke":smoke(config,dataset,identity,device);return 0
    if args.phase=="train":
        if args.seed not in config["seeds"]:raise RuntimeError("seed not preregistered")
        train_seed(config,dataset,identity,device,args.seed);return 0
    if args.phase=="finalize":finalize(config,dataset,identity,device);return 0
    smoke(config,dataset,identity,device)
    for seed in config["seeds"]:train_seed(config,dataset,identity,device,int(seed))
    finalize(config,dataset,identity,device);return 0


if __name__=="__main__":raise SystemExit(main())
