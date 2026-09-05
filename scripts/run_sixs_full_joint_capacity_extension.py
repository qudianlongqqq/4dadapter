#!/usr/bin/env python3
"""Exact step-17,500 -> 22,500 continuation and frozen DEV evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
import scripts.run_sixs_j1r1_full_joint_adaptive_ba_movement as original

RUN = "sixs_j1r1_full_joint_adaptive_ba_movement_seed307"
SOURCE_REPORT = ROOT / "reports/ecir_mvr" / RUN
SOURCE_ARTIFACT = ROOT / "artifacts/ecir_mvr" / RUN
OUT = ROOT / "reports/ecir_mvr/sixs_full_joint_final_capacity_audit"
EXT_REPORT = OUT / "extension_22500"
EXT_ARTIFACT = SOURCE_ARTIFACT / "capacity_extension_22500"
STATUS = OUT / "EXTENSION_STATUS.json"
CONFIG = ROOT / "configs/sixs_j1r1_full_joint_adaptive_ba_movement.json"
BASE_RECOVERY = SOURCE_ARTIFACT / "RECOVERY_CHECKPOINT.pt"
EXT_RECOVERY = EXT_ARTIFACT / "RECOVERY_CHECKPOINT.pt"
FINAL_22500 = EXT_REPORT / "FINAL_CHECKPOINT_STEP22500.pt"
TRAIN_LOG = OUT / "04_TRAINING_CURVE.csv"
EXT_LOG = EXT_REPORT / "EXTENDED_TRAIN_LOG.csv"
TARGET_STEP = 22500


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str))


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(tmp, index=False)
    else:
        frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, tmp)
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config() -> dict[str, Any]:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["training"]["optimizer_steps"] = TARGET_STEP
    value["training"]["checkpoint_policy"] = "NATURAL_SCHEDULER_HORIZON_STEP22500_ONLY_NO_DEV_SELECTION"
    return value


def spec_sha() -> str:
    payload = {"base_config_sha256":sha256(CONFIG),"start":17500,"target":TARGET_STEP,"scheduler_horizon":22500,"recipe_changes":"NONE"}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def guards() -> dict[str, Any]:
    return {"EXPERIMENT_ID":"SIXS_J1_R1_FULL_JOINT_ADAPTIVE_BA_MOVEMENT_CAPACITY_EXTENSION","SEED":307,"START_STEP":17500,"TARGET_STEP":22500,"FORMAL_READ":"NO","LARGE_HOLDOUT_READ":"NO","SEED331_STARTED":"NO","SEED353_STARTED":"NO","NEW_ARCHITECTURE_CREATED":"NO","HYPERPARAMETER_SWEEP":"NO","CHECKPOINT_SELECTED_BY_DEV_OUTCOME":"NO","RECIPE_CHANGED":"NO","XTB_NEW_COMPUTATION_STARTED":"NO"}


def status(stage: str, state: str = "RUNNING", **extra: Any) -> None:
    atomic_json(STATUS,{"schema_version":"sixs-full-joint-capacity-extension-status-v1","state":state,"stage":stage,"pid":os.getpid(),"updated_at":pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),**guards(),**extra})


def restore_rng(saved: dict[str, Any], generator: torch.Generator) -> None:
    generator.set_state(saved["generator_state"])
    random.setstate(saved["python_rng_state"])
    np.random.set_state(saved["numpy_rng_state"])
    torch.set_rng_state(saved["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved["cuda_rng_state"])


def checkpoint_payload(step: int, model, optimizer, scheduler, generator) -> dict[str, Any]:
    return {"schema_version":"sixs-full-joint-capacity-continuation-v1","step":step,"base_config_sha256":sha256(CONFIG),"continuation_spec_sha256":spec_sha(),"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),"generator_state":generator.get_state(),"python_rng_state":random.getstate(),"numpy_rng_state":np.random.get_state(),"torch_rng_state":torch.get_rng_state(),"cuda_rng_state":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],**guards()}


def train() -> None:
    if FINAL_22500.is_file():
        return
    prereg = json.loads((OUT / "SATURATION_PREREGISTRATION.json").read_text(encoding="utf-8"))
    gate = json.loads((OUT / "STATE_CONTINUITY_GATE.json").read_text(encoding="utf-8"))
    if prereg["training_saturation"] not in ("CLEARLY_NOT_SATURATED","LIKELY_NOT_SATURATED") or gate["MODEL_STATE"] != "PASS":
        raise RuntimeError("extension not preregistered or continuity gate failed")
    cfg = config()
    original.cfg = lambda: cfg
    device = original.execution_device()
    prepared, source_payload = original.frozen.load_inputs()
    sources = original.frozen.source_index(source_payload["train"])
    generator = original.seed_all(cfg["seed"])
    model = original.build_model(device)
    optimizer, scheduler = original.optimizer_for(model)
    groups = model.parameter_groups()
    saved_path = EXT_RECOVERY if EXT_RECOVERY.is_file() else BASE_RECOVERY
    saved = torch.load(saved_path,map_location="cpu",weights_only=False)
    if saved_path == BASE_RECOVERY:
        if saved.get("config_sha256") != sha256(CONFIG) or int(saved.get("step",-1)) != 17500:
            raise RuntimeError("base recovery identity mismatch")
    else:
        if saved.get("continuation_spec_sha256") != spec_sha():
            raise RuntimeError("extension recovery identity mismatch")
    model.load_state_dict(saved["model_state"],strict=True)
    optimizer.load_state_dict(saved["optimizer_state"])
    scheduler.load_state_dict(saved["scheduler_state"])
    restore_rng(saved,generator)
    start = int(saved["step"])
    if scheduler.state_dict()["T_max"] != 22500 or scheduler.state_dict()["last_epoch"] != start:
        raise RuntimeError("scheduler continuity mismatch")
    if start < 17500 or start > TARGET_STEP:
        raise RuntimeError("invalid continuation step")
    if EXT_LOG.is_file():
        logs = pd.read_csv(EXT_LOG).to_dict("records")
    else:
        base_columns = pd.read_csv(SOURCE_REPORT/"TRAIN_LOG.csv").columns
        logs = pd.read_csv(SOURCE_REPORT/"TRAIN_LOG.csv",usecols=base_columns).to_dict("records")
    status("TRAINING_CONTINUATION","RUNNING",current_step=start,total_steps=TARGET_STEP,device=str(device),source_checkpoint=str(saved_path),scheduler_last_epoch=scheduler.state_dict()["last_epoch"])
    started = time.time()
    for step in range(start+1,TARGET_STEP+1):
        tick=time.time(); model.train()
        graphs,bg,source,reference,_=original.frozen.sample_batch(prepared["train"],sources,generator,cfg["training"]["batch_molecules"],device)
        optimizer.zero_grad(set_to_none=True)
        prediction,action,losses,ref_bond,ref_angle=original.batch_losses(model,graphs,bg,source,reference)
        if not bool(torch.isfinite(losses["total"])) or not bool(torch.isfinite(action.proposal).all()):
            raise RuntimeError(f"nonfinite continuation forward at step {step}")
        losses["total"].backward()
        if step % cfg["training"]["log_interval"] == 0:
            logs.append(original.training_row(step,losses,prediction,action,ref_bond,ref_angle,groups,time.time()-tick))
        gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["training"]["gradient_clip"])
        if not bool(torch.isfinite(gradient)):
            raise RuntimeError(f"nonfinite continuation gradient at step {step}")
        optimizer.step(); scheduler.step()
        if step % cfg["training"]["log_interval"] == 0:
            row=logs[-1]
            status("TRAINING_CONTINUATION","RUNNING",current_step=step,total_steps=TARGET_STEP,total_loss=row["TOTAL_LOSS"],post_loss=row["POST_LOSS"],mu_bond_mae=row["MU_BOND_MAE"],mu_angle_mae=row["MU_ANGLE_MAE"],w_B_mean=row["W_B_MEAN"],tau_mean=row["TAU_MEAN"],sigma_mean=row["SIGMA_MEAN"],reliability_mean=row["R_MEAN"],elapsed_seconds=time.time()-started)
            print(json.dumps({"stage":"TRAINING_CONTINUATION",**row}),flush=True)
        if step % cfg["training"]["recovery_interval"] == 0:
            atomic_torch(EXT_RECOVERY,checkpoint_payload(step,model,optimizer,scheduler,generator))
            atomic_frame(EXT_LOG,pd.DataFrame(logs))
    final=checkpoint_payload(TARGET_STEP,model,optimizer,scheduler,generator)
    final["schema_version"]="sixs-full-joint-capacity-final-v1"
    atomic_torch(FINAL_22500,final)
    atomic_frame(EXT_LOG,pd.DataFrame(logs))
    status("TRAINING_COMPLETE","PASS",current_step=TARGET_STEP,total_steps=TARGET_STEP,checkpoint_sha256=sha256(FINAL_22500),scheduler_last_epoch=scheduler.state_dict()["last_epoch"],final_lrs=scheduler.state_dict()["_last_lr"])


def configure_original_evaluator() -> dict[str, Path]:
    cfg=config(); original.cfg=lambda:cfg
    original.EXPERIMENT_ID="SIXS_J1_R1_FULL_JOINT_ADAPTIVE_BA_MOVEMENT_STEP22500"
    original.REPORT=EXT_REPORT
    original.ARTIFACT=EXT_ARTIFACT
    original.STATUS=OUT/"EXTENSION_EVALUATION_STATUS.json"
    original.FINAL=FINAL_22500
    dev=EXT_ARTIFACT/"dev_evaluation"
    original.COORDINATES_READY=dev/"COORDINATES_READY.json"
    original.PAYLOAD=dev/"EVALUATION_PAYLOAD.pt"
    original.SDF=dev/"PROPOSAL.sdf"
    original.PER_RECORD=dev/"PER_RECORD.parquet"
    original.PB=dev/"POSEBUSTERS.parquet"
    original.V3D=dev/"VALIDITY3D.parquet"
    original.guards=lambda:{**guards(),"FULL_JOINT_TRAINING":"YES","REFERENCE_USED_AT_INFERENCE":"NO","SECOND_ORDER_REQUIRED":"NO"}
    base_status=original.status
    def corrected(stage,state="RUNNING",**extra):
        extra["CURRENT_STEP"]=TARGET_STEP; extra["TOTAL_STEPS"]=TARGET_STEP
        return base_status(stage,state,**extra)
    original.status=corrected
    return {"sdf":original.SDF,"per_record":original.PER_RECORD,"pb":original.PB,"v3d":original.V3D}


def summarize(records: pd.DataFrame) -> dict[str, Any]:
    return {"molecules":int(records.molecule_id.nunique()),"records":len(records),"proposal_V3D":float(records.V3D.mean()),"proposal_PB":float(records.PB.mean()),"internal_post":float(records.internal_post.mean()),"direction_improvement":float(records.direction_improvement.mean()),"bond_raw_mae":float(records.bond_raw_mae.mean()),"angle_raw_mae":float(records.angle_raw_mae.mean()),"source_rmsd":float(records.source_rmsd.mean()),"proposal_movement":float(records.proposal_movement.mean()),"tau_mean":float(records.tau.mean()),"tau_median":float(records.tau.median()),"tau_p05":float(records.tau.quantile(.05)),"tau_p95":float(records.tau.quantile(.95)),"tau_p99":float(records.tau.quantile(.99)),"tau_max":float(records.tau.max()),"atom_cap_fraction":float(records.atom_cap_active.mean()),"w_B_mean":float(records.w_B.mean()),"w_B_p05":float(records.w_B.quantile(.05)),"w_B_p95":float(records.w_B.quantile(.95)),"w_A_mean":float(records.w_A.mean())}


def evaluate() -> None:
    done=OUT/"EXTENSION_EVALUATION_COMPLETE.json"
    if done.is_file(): return
    if not FINAL_22500.is_file(): raise RuntimeError("step22500 checkpoint absent")
    paths=configure_original_evaluator()
    status("COORDINATE_GENERATION","RUNNING",current_step=TARGET_STEP,total_steps=TARGET_STEP)
    original.evaluate_coordinates()
    records=pd.read_parquet(paths["per_record"])
    ids=records.record_id.astype(str).tolist()
    if original.evaluator_artifact_status(paths["pb"],ids)!="COMPLETE" or original.evaluator_artifact_status(paths["v3d"],ids)!="COMPLETE":
        status("EXTERNAL_EVALUATION","RUNNING",evaluation_records=5000)
        original.frozen.run_external_evaluators(original.EXPERIMENT_ID,paths,ids)
    pb=pd.read_parquet(paths["pb"]); v3d=pd.read_parquet(paths["v3d"])
    records=records.drop(columns=["PB","V3D"],errors="ignore").merge(pb[["record_id","PB"]],on="record_id",validate="one_to_one").merge(v3d[["record_id","validity3d"]].rename(columns={"validity3d":"V3D"}),on="record_id",validate="one_to_one")
    atomic_frame(paths["per_record"],records)
    baseline=pd.read_parquet(SOURCE_ARTIFACT/"dev_evaluation/PER_RECORD.parquet")
    if baseline.record_id.astype(str).tolist()!=records.record_id.astype(str).tolist(): raise RuntimeError("17500/22500 record alignment failed")
    metrics=("V3D","PB","internal_post","direction_improvement","bond_raw_mae","angle_raw_mae","source_rmsd","proposal_movement")
    boots=[]
    seed=20260830
    for index,metric in enumerate(metrics):
        boots.append(original.cluster_bootstrap(baseline,records,metric,seed+index))
    boot=pd.DataFrame(boots); atomic_frame(OUT/"08_TRAINING_BUDGET_BOOTSTRAP.csv",boot)
    summary_17500=summarize(baseline); summary_22500=summarize(records)
    columns=list(summary_17500)
    comparison=pd.DataFrame([{"arm":"FULL_JOINT_17500",**summary_17500},{"arm":"FULL_JOINT_22500",**summary_22500},{"arm":"DELTA_22500_MINUS_17500",**{key:(summary_22500[key]-summary_17500[key] if isinstance(summary_17500[key],(int,float)) else None) for key in columns}}])
    atomic_frame(OUT/"07_17500_VS_22500.csv",comparison)
    base_pb=pd.read_parquet(SOURCE_ARTIFACT/"dev_evaluation/POSEBUSTERS.parquet"); base_v3d=pd.read_parquet(SOURCE_ARTIFACT/"dev_evaluation/VALIDITY3D.parquet")
    atomic_frame(OUT/"EXTENSION_V3D_COMPONENT_TRANSITIONS.csv",pd.DataFrame([original.transitions(base_v3d,v3d,c) for c in original.V3D_COMPONENTS]))
    atomic_frame(OUT/"EXTENSION_PB_COMPONENT_TRANSITIONS.csv",pd.DataFrame([original.transitions(base_pb,pb,c) for c in original.PB_COMPONENTS]))
    by={row.metric:row for row in boot.itertuples(index=False)}
    continuous_same=all(by[m].delta_candidate_minus_baseline<=0 for m in ("internal_post","bond_raw_mae","angle_raw_mae"))
    no_clear_reverse=all(by[m].ci95_low<=0 for m in ("internal_post","bond_raw_mae","angle_raw_mae"))
    v3d_positive=by["V3D"].delta_candidate_minus_baseline>0 and by["V3D"].ci95_low>0
    pb_ok=by["PB"].delta_candidate_minus_baseline>=-0.001
    limiting="YES" if v3d_positive and continuous_same and no_clear_reverse and pb_ok else "NO_OR_WEAK"
    result={"schema_version":"sixs-full-joint-capacity-extension-evaluation-v1","status":"PASS","summary_17500":summary_17500,"summary_22500":summary_22500,"bootstrap":{row.metric:row._asdict() for row in boot.itertuples(index=False)},"TRAINING_BUDGET_17500_WAS_LIMITING":limiting,"decision_components":{"V3D_CLEAR_POSITIVE":v3d_positive,"CONTINUOUS_GEOMETRY_SAME_DIRECTION":continuous_same,"NO_CLEAR_CONTINUOUS_REVERSE":no_clear_reverse,"PB_NO_MATERIAL_DROP":pb_ok},"record_alignment":"PASS",**guards()}
    atomic_json(done,result)
    status("EVALUATION_COMPLETE","PASS",proposal_v3d=summary_22500["proposal_V3D"],proposal_pb=summary_22500["proposal_PB"],delta_v3d=by["V3D"].delta_candidate_minus_baseline,v3d_ci95=[by["V3D"].ci95_low,by["V3D"].ci95_high],training_budget_17500_was_limiting=limiting)


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",choices=("train","evaluate","pipeline"),default="pipeline"); args=parser.parse_args()
    try:
        if args.stage in ("train","pipeline"): train()
        if args.stage in ("evaluate","pipeline"): evaluate()
        return 0
    except Exception as exc:
        status("FAILED","FAIL",error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        traceback.print_exc(); return 1


if __name__=="__main__":
    raise SystemExit(main())
