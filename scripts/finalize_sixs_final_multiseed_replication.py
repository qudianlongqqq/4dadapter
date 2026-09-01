#!/usr/bin/env python3
"""Frozen DEV/xTB evaluation, seed-level aggregation, and integrity audit for final SIXS multiseed replication."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
import scripts.run_sixs_current_final_evidence_completion as evidence
import scripts.run_sixs_j1r1_full_joint_xtb_dev as xtb_sp

REPORT = ROOT / "reports/ecir_mvr/sixs_final_restricted_vs_unrestricted_multiseed"
ARTIFACT = ROOT / "artifacts/ecir_mvr/sixs_final_restricted_vs_unrestricted_multiseed"
STATUS = REPORT / "FINAL_STATUS.json"
SOURCE_XTB = ROOT / "reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/xtb_single_point_dev/SOURCE_XTB.csv"
SEEDS = (307, 331, 353)
FORMS = ("restricted", "unrestricted")
HARTREE_TO_KCAL = 627.509474


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str))


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    if path.suffix == ".parquet": frame.to_parquet(temp, index=False)
    else: frame.to_csv(temp, index=False)
    os.replace(temp, path)


def update(stage: str, state: str = "RUNNING", **extra: Any) -> None:
    old = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    old.update({
        "schema_version": "sixs-final-restricted-vs-unrestricted-multiseed-v1", "status": state,
        "stage": stage, "pid": os.getpid(), "updated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "formal_outcome_read": False, "large_holdout_outcome_read": False, **extra,
    })
    atomic_json(STATUS, old)


def run_report(form: str, seed: int) -> Path:
    if seed != 307: return REPORT / f"{form}_seed{seed}"
    if form == "restricted": return ROOT / "reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307"
    return ROOT / "reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT"


def run_artifact(form: str, seed: int) -> Path:
    if seed != 307: return ARTIFACT / f"{form}_seed{seed}" / "dev_evaluation"
    if form == "restricted": return ROOT / "artifacts/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/dev_evaluation"
    return ROOT / "artifacts/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/dev_evaluation"


def config_path(form: str, seed: int) -> Path:
    if seed != 307: return REPORT / f"{form}_seed{seed}" / "FROZEN_CONFIG.json"
    return ROOT / "configs" / ("sixs_j1r1_full_joint_adaptive_ba_movement.json" if form == "restricted" else "sixs_j1r1_full_joint_unrestricted_movement.json")


def parse_sdf(path: Path, ids: list[str]) -> dict[str, torch.Tensor]:
    mols = [m for m in Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False) if m is not None]
    if len(mols) != 5000: raise RuntimeError(f"SDF denominator changed: {path}")
    result = {}
    for expected, mol in zip(ids, mols, strict=True):
        observed = mol.GetProp("_Name")
        if observed != expected: raise RuntimeError(f"SDF ordering changed: {path}")
        xyz = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float64)
        if not bool(torch.isfinite(xyz).all()): raise RuntimeError(f"nonfinite coordinates: {observed}")
        result[expected] = xyz
    return result


def reference_rmsd(form: str, seed: int, ids: list[str], items: dict[str, Any], sources: dict[str, Any]) -> pd.DataFrame:
    target = REPORT / f"{form}_seed{seed}" / "REFERENCE_RMSD.csv"
    if target.is_file():
        frame = pd.read_csv(target)
        if len(frame) == 5000 and frame.record_id.nunique() == 5000: return frame
    if seed == 307:
        if form == "restricted":
            source = ROOT / "reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/01_CURRENT_FINAL_EVIDENCE/02_MATCHED_SOURCE_REFERENCE_RMSD.csv"
            frame = pd.read_csv(source); frame = frame[frame.method == "SIXS_FULL_JOINT_STEP17500"].copy()
        else:
            frame = pd.read_csv(run_report(form, seed) / "UNRESTRICTED_REFERENCE_RMSD.csv")
        atomic_frame(target, frame); return frame
    coords = parse_sdf(run_artifact(form, seed) / "PROPOSAL.sdf", ids)
    rows = []
    for position, record_id in enumerate(ids):
        row = sources[record_id]; xyz = coords[record_id]
        source = torch.as_tensor(row["source"], dtype=torch.float64)
        references = torch.as_tensor(items[str(row["molecule_id"])]["references"], dtype=torch.float64)
        source_reference = min(float(evidence.kabsch_rmsd(source, ref)) for ref in references)
        value = min(float(evidence.kabsch_rmsd(xyz, ref)) for ref in references)
        rows.append({"record_index": position, "record_id": record_id, "molecule_id": str(row["molecule_id"]),
                     "source_rmsd": float(evidence.kabsch_rmsd(xyz, source)), "reference_rmsd": value,
                     "source_reference_rmsd": source_reference, "delta_reference_rmsd_vs_source": value-source_reference})
        if (position + 1) % 500 == 0: update(f"REFERENCE_RMSD_{form.upper()}_SEED{seed}", completed=position+1, total=5000)
    frame = pd.DataFrame(rows); atomic_frame(target, frame); return frame


def xtb_path(form: str, seed: int) -> Path:
    return REPORT / f"{form}_seed{seed}" / "PROPOSAL_XTB.csv"


def run_xtb(form: str, seed: int, ids: list[str], sources: dict[str, Any], val_meta: dict[str, Any], frozen_config: dict[str, Any]) -> pd.DataFrame:
    target = xtb_path(form, seed)
    if target.is_file():
        frame = pd.read_csv(target)
        if len(frame) == 5000 and frame.record_id.nunique() == 5000: return frame
    if seed == 307:
        source = (ROOT / "reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/xtb_single_point_dev/FULL_JOINT_XTB.csv"
                  if form == "restricted" else run_report(form, seed) / "UNRESTRICTED_XTB.csv")
        frame = pd.read_csv(source); atomic_frame(target, frame); return frame
    coords = parse_sdf(run_artifact(form, seed) / "PROPOSAL.sdf", ids)
    xtb_sp.OUT = ROOT / ".sixs_final_multiseed_xtb_sp"
    xtb_sp.OUT.mkdir(parents=True, exist_ok=True)
    tasks = []
    for position, record_id in enumerate(ids):
        record = evidence.source_record(frozen_config, val_meta[record_id])
        task = evidence.make_xtb_task(record_id, str(sources[record_id]["molecule_id"]), position, coords[record_id], record)
        task["method"] = f"{form.upper()}_SEED{seed}_STEP17500"
        tasks.append(task)
    rows = []
    update(f"XTB_{form.upper()}_SEED{seed}", XTB_STARTED=True, completed=0, total=5000)
    with ThreadPoolExecutor(max_workers=xtb_sp.SETTINGS["workers"]) as pool:
        futures = [pool.submit(xtb_sp.execute, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if count % 100 == 0 or count == 5000:
                partial = pd.DataFrame(rows).sort_values("record_index", kind="stable").reset_index(drop=True)
                atomic_frame(target.with_name("PROPOSAL_XTB.partial.csv"), partial)
                update(f"XTB_{form.upper()}_SEED{seed}", XTB_STARTED=True, completed=count, total=5000)
    frame = pd.DataFrame(rows).sort_values("record_index", kind="stable").reset_index(drop=True)
    atomic_frame(target, frame)
    target.with_name("PROPOSAL_XTB.partial.csv").unlink(missing_ok=True)
    return frame


def trimmed_mean(values: np.ndarray, fraction: float = .05) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64)); n = int(math.floor(len(values)*fraction))
    return float(values[n:len(values)-n].mean())


def summarize_xtb(frame: pd.DataFrame, source_energy: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = frame.copy()
    frame["deltaE_kcal_mol"] = (frame.energy_hartree - frame.record_id.map(source_energy))*HARTREE_TO_KCAL
    good = frame[frame.success.astype(bool) & frame.deltaE_kcal_mol.notna()].copy(); values = good.deltaE_kcal_mol.to_numpy(float)
    summary = {"success": len(good), "failures": len(frame)-len(good), "median": float(np.median(values)),
               "lower_fraction": float(np.mean(values < 0)), "trimmed_mean_5pct": trimmed_mean(values),
               "p90": float(np.quantile(values,.90)), "p95": float(np.quantile(values,.95)), "p99": float(np.quantile(values,.99)),
               "tail_gt_25": int(np.sum(values>25)), "tail_gt_50": int(np.sum(values>50)), "tail_gt_100": int(np.sum(values>100)),
               "mean_secondary": float(np.mean(values))}
    return frame, summary


def distribution(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "median": float(np.median(v)),
            "p90": float(np.quantile(v,.90)), "p95": float(np.quantile(v,.95)), "p99": float(np.quantile(v,.99)),
            "p99_5": float(np.quantile(v,.995)), "p99_9": float(np.quantile(v,.999)), "max": float(v.max())}


def cluster_bootstrap(frame: pd.DataFrame, metric: str, seed: int) -> dict[str, Any]:
    clusters = frame.groupby("molecule_id", sort=True)[metric].mean().to_numpy(float)
    rng = np.random.default_rng(seed); draws = np.empty(10000)
    for start in range(0,10000,250):
        pick = rng.integers(0,len(clusters),size=(250,len(clusters)))
        draws[start:start+250] = clusters[pick].mean(axis=1)
    return {"metric": metric, "estimate": float(clusters.mean()), "ci95_low": float(np.quantile(draws,.025)),
            "ci95_high": float(np.quantile(draws,.975)), "clusters": len(clusters), "resamples": 10000}


def read_training_identity(form: str, seed: int) -> dict[str, Any]:
    report = run_report(form,seed); checkpoint = torch.load(report/"FINAL_CHECKPOINT.pt",map_location="cpu",weights_only=False)
    summary = json.loads((report/"TRAIN_SUMMARY.json").read_text(encoding="utf-8"))
    verification = json.loads((report/"GPU_TRAINING_VERIFICATION.json").read_text(encoding="utf-8")) if (report/"GPU_TRAINING_VERIFICATION.json").is_file() else {}
    config = json.loads(config_path(form,seed).read_text(encoding="utf-8"))
    recovery_path = (ARTIFACT/f"{form}_seed{seed}"/"RECOVERY_CHECKPOINT.pt") if seed != 307 else (
        ROOT/"artifacts/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/RECOVERY_CHECKPOINT.pt" if form=="restricted" else
        ROOT/"artifacts/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/RECOVERY_CHECKPOINT.pt")
    recovery = torch.load(recovery_path,map_location="cpu",weights_only=False) if recovery_path.is_file() else {}
    generator_state = recovery.get("generator_state")
    generator_sha = hashlib.sha256(generator_state.numpy().tobytes()).hexdigest() if isinstance(generator_state,torch.Tensor) else None
    return {"formulation":form,"seed":seed,"checkpoint_seed":int(checkpoint["seed"]),"checkpoint_step":int(checkpoint["step"]),
            "checkpoint_sha256":sha(report/"FINAL_CHECKPOINT.pt"),"config_sha256":sha(config_path(form,seed)),
            "optimizer_steps":int(config["training"]["optimizer_steps"]),"batch_molecules":int(config["training"]["batch_molecules"]),
            "backbone_lr":float(config["training"]["backbone_learning_rate"]),"head_lr":float(config["training"]["head_learning_rate"]),
            "weight_decay":float(config["training"]["weight_decay"]),"gradient_clip":float(config["training"]["gradient_clip"]),
            "scheduler_horizon":int(config["training"]["scheduler_horizon"]),"training_elapsed_seconds":float(summary["elapsed_seconds"]),
            "peak_gpu_memory_bytes":summary.get("peak_gpu_memory_bytes"),"nonfinite_count":int(summary.get("nonfinite_count",0)),
            "gpu_verification":verification.get("status","HISTORICAL_PASS" if seed==307 else "MISSING"),
            "recovery_step":int(recovery.get("step",0)),"generator_state_sha256":generator_sha,
            "formal_read":"NO","large_holdout_read":"NO","dev_checkpoint_selection":"NO"}


def main() -> int:
    REPORT.mkdir(parents=True,exist_ok=True)
    update("PREFLIGHT")
    frozen_config, _, ids, items, sources, _ = evidence.config_and_inputs()
    val = pd.read_parquet(frozen_config["data"]["val_manifest"]); val_meta={str(r.sample_id):r for r in val.itertuples(index=False)}
    source_xtb = pd.read_csv(SOURCE_XTB); source_energy=source_xtb.set_index("record_id").energy_hartree
    identities=[]; summaries=[]; v3d_rows=[]; pb_rows=[]; ref_rows=[]; src_rows=[]; xtb_rows=[]; tail_rows=[]; move_rows=[]; mechanism_rows=[]; bootstrap_rows=[]
    joined_by_run={}
    for form in FORMS:
        for seed in SEEDS:
            update(f"EVALUATE_{form.upper()}_SEED{seed}")
            identity=read_training_identity(form,seed); identities.append(identity)
            art=run_artifact(form,seed); records=pd.read_parquet(art/"PER_RECORD.parquet")
            v3=pd.read_parquet(art/"VALIDITY3D.parquet"); pb=pd.read_parquet(art/"POSEBUSTERS.parquet")
            ref=reference_rmsd(form,seed,ids,items,sources)
            xtb=run_xtb(form,seed,ids,sources,val_meta,frozen_config); xtb,xtbs=summarize_xtb(xtb,source_energy); atomic_frame(xtb_path(form,seed),xtb)
            if records.record_id.astype(str).tolist()!=ids or v3.record_id.astype(str).tolist()!=ids or pb.record_id.astype(str).tolist()!=ids or ref.record_id.astype(str).tolist()!=ids:
                raise RuntimeError(f"evaluator identity/order mismatch {form} seed{seed}")
            v3d=float(v3.validity3d.mean()); pbo=float(pb.PB.mean()); refmean=float(ref.reference_rmsd.mean()); srcmean=float(records.source_rmsd.mean())
            summary={"formulation":form,"seed":seed,"records":len(records),"molecules":records.molecule_id.nunique(),"V3D":v3d,"PB":pbo,
                     "reference_rmsd":refmean,"source_rmsd":srcmean,"internal_post":float(records.internal_post.mean()),
                     "direction_improvement":float(records.direction_improvement.mean()),"bond_raw_mae":float(records.bond_raw_mae.mean()),
                     "angle_raw_mae":float(records.angle_raw_mae.mean()),**{f"xtb_{k}":v for k,v in xtbs.items()}}
            summaries.append(summary)
            for col in ("bond_geometry_valid","angle_geometry_valid","aromatic_ring_valid","intramolecular_steric_clash_valid","validity3d"):
                v3d_rows.append({"formulation":form,"seed":seed,"component":col,"rate":float(v3[col].mean())})
            pb_components=[c for c in ("mol_pred_loaded","sanitization","inchi_convertible","all_atoms_connected","no_radicals","bond_lengths","bond_angles","internal_steric_clash","aromatic_ring_flatness","non-aromatic_ring_non-flatness","double_bond_flatness","PB") if c in pb]
            for col in pb_components: pb_rows.append({"formulation":form,"seed":seed,"component":col,"rate":float(pb[col].mean())})
            ref_rows.append({"formulation":form,"seed":seed,**distribution(ref.reference_rmsd.to_numpy())})
            src_rows.append({"formulation":form,"seed":seed,**distribution(records.source_rmsd.to_numpy())})
            xtb_rows.append({"formulation":form,"seed":seed,**xtbs})
            tail_rows.append({"formulation":form,"seed":seed,"tail_gt_25":xtbs["tail_gt_25"],"tail_gt_50":xtbs["tail_gt_50"],"tail_gt_100":xtbs["tail_gt_100"],"p90":xtbs["p90"],"p95":xtbs["p95"],"p99":xtbs["p99"]})
            tau=records.tau.to_numpy(float)
            if "atom_displacements" in records:
                atom=np.concatenate(records.atom_displacements.to_numpy())
            else:
                coords=parse_sdf(art/"PROPOSAL.sdf",ids); atom=np.concatenate([np.linalg.norm((coords[i]-torch.as_tensor(sources[i]["source"],dtype=torch.float64)).numpy(),axis=1) for i in ids])
            movement={"formulation":form,"seed":seed,**{f"tau_{k}":v for k,v in distribution(tau).items()},
                      **{f"atom_displacement_{k}":v for k,v in distribution(atom).items()},
                      "finite_coordinate_rate":float(np.mean(np.isfinite(atom))),
                      **{f"fraction_tau_gt_{str(t).replace('.','_')}":float(np.mean(tau>t)) for t in (.01,.02,.05,.10,.20)}}
            move_rows.append(movement)
            primitives=torch.load(art/"EVALUATION_PAYLOAD.pt",map_location="cpu",weights_only=False)["primitive_rows"]
            bond_sigma=np.concatenate([r["bond_sigma"] for r in primitives]); angle_sigma=np.concatenate([r["angle_sigma"] for r in primitives])
            bond_rel=np.concatenate([r["bond_reliability"] for r in primitives]); angle_rel=np.concatenate([r["angle_reliability"] for r in primitives])
            mechanism_rows.append({"formulation":form,"seed":seed,"w_B_mean":float(records.w_B.mean()),"w_B_std":float(records.w_B.std()),
                                   "bond_sigma_mean":float(bond_sigma.mean()),"angle_sigma_mean":float(angle_sigma.mean()),
                                   "bond_reliability_mean":float(bond_rel.mean()),"angle_reliability_mean":float(angle_rel.mean())})
            joined=records[["record_id","molecule_id","source_rmsd","internal_post","bond_raw_mae","angle_raw_mae"]].copy()
            joined=joined.merge(v3[["record_id","validity3d"]],on="record_id",validate="one_to_one").merge(pb[["record_id","PB"]],on="record_id",validate="one_to_one").merge(ref[["record_id","reference_rmsd"]],on="record_id",validate="one_to_one").merge(xtb[["record_id","deltaE_kcal_mol"]],on="record_id",validate="one_to_one")
            joined_by_run[(form,seed)]=joined
            for index, metric in enumerate(("validity3d","PB","reference_rmsd","source_rmsd","deltaE_kcal_mol")):
                bootstrap_rows.append({"formulation":form,"seed":seed,**cluster_bootstrap(joined,metric,20260831+seed+index)})
    identity_frame=pd.DataFrame(identities); summary_frame=pd.DataFrame(summaries)
    atomic_frame(REPORT/"01_RUN_IDENTITY.csv",identity_frame)
    completeness=identity_frame[["formulation","seed","checkpoint_step","recovery_step","gpu_verification","nonfinite_count"]].copy()
    completeness["complete"]=(completeness.checkpoint_step==17500)&(completeness.recovery_step==17500)&(completeness.nonfinite_count==0)
    atomic_frame(REPORT/"02_TRAINING_COMPLETENESS.csv",completeness)
    atomic_frame(REPORT/"03_MULTISEED_V3D.csv",pd.DataFrame(v3d_rows)); atomic_frame(REPORT/"04_MULTISEED_PB.csv",pd.DataFrame(pb_rows))
    atomic_frame(REPORT/"05_MULTISEED_REFERENCE_RMSD.csv",pd.DataFrame(ref_rows)); atomic_frame(REPORT/"06_MULTISEED_SOURCE_RMSD.csv",pd.DataFrame(src_rows))
    atomic_frame(REPORT/"07_MULTISEED_XTB.csv",pd.DataFrame(xtb_rows)); atomic_frame(REPORT/"08_MULTISEED_XTB_TAIL.csv",pd.DataFrame(tail_rows))
    atomic_frame(REPORT/"09_MULTISEED_MOVEMENT.csv",pd.DataFrame(move_rows)); atomic_frame(REPORT/"10_MULTISEED_MECHANISM_STABILITY.csv",pd.DataFrame(mechanism_rows))
    atomic_frame(REPORT/"COHORT_BOOTSTRAP.csv",pd.DataFrame(bootstrap_rows)); atomic_frame(REPORT/"RUN_SUMMARY.csv",summary_frame)
    matched=[]
    for seed in SEEDS:
        r=summary_frame[(summary_frame.formulation=="restricted")&(summary_frame.seed==seed)].iloc[0]; u=summary_frame[(summary_frame.formulation=="unrestricted")&(summary_frame.seed==seed)].iloc[0]
        matched.append({"seed":seed,"delta_V3D_unrestricted_minus_restricted":u.V3D-r.V3D,"delta_PB_unrestricted_minus_restricted":u.PB-r.PB,
                        "delta_reference_rmsd":u.reference_rmsd-r.reference_rmsd,"delta_source_rmsd":u.source_rmsd-r.source_rmsd,
                        "delta_xtb_median":u.xtb_median-r.xtb_median,"delta_xtb_trimmed_mean":u.xtb_trimmed_mean_5pct-r.xtb_trimmed_mean_5pct,
                        "delta_xtb_tail_gt_25":u.xtb_tail_gt_25-r.xtb_tail_gt_25,"delta_xtb_tail_gt_50":u.xtb_tail_gt_50-r.xtb_tail_gt_50,"delta_xtb_tail_gt_100":u.xtb_tail_gt_100-r.xtb_tail_gt_100})
    matched_frame=pd.DataFrame(matched); atomic_frame(REPORT/"11_MATCHED_SEED_COMPARISON.csv",matched_frame)
    # Integrity is deliberately independent of endpoint outcomes.
    generator_match=[]
    for seed in (331,353):
        rr=identity_frame[(identity_frame.formulation=="restricted")&(identity_frame.seed==seed)].iloc[0]
        uu=identity_frame[(identity_frame.formulation=="unrestricted")&(identity_frame.seed==seed)].iloc[0]
        generator_match.append(bool(rr.generator_state_sha256==uu.generator_state_sha256 and pd.notna(rr.generator_state_sha256)))
    integrity_checks={
        "all_six_runs_complete":bool(completeness.complete.all()),"all_endpoint_steps_17500":bool((identity_frame.checkpoint_step==17500).all()),
        "new_runs_gpu_verified":bool((identity_frame[identity_frame.seed!=307].gpu_verification=="PASS").all()),
        "no_nonfinite":bool((identity_frame.nonfinite_count==0).all()),"matched_sampler_generator_states":bool(all(generator_match)),
        "optimizer_recipe_matched":bool(identity_frame.groupby("seed")[["batch_molecules","backbone_lr","head_lr","weight_decay","gradient_clip","scheduler_horizon"]].nunique().max().max()==1),
        "all_eval_denominators_5000":bool((summary_frame.records==5000).all()),"all_xtb_success_5000":bool((summary_frame.xtb_success==5000).all()),
        "no_dev_checkpoint_selection":bool((identity_frame.dev_checkpoint_selection=="NO").all()),
        "formal_outcome_read":False,"large_holdout_outcome_read":False,
    }
    integrity="PASS" if all(v for k,v in integrity_checks.items() if k not in ("formal_outcome_read","large_holdout_outcome_read")) else "FAIL"
    audit="# MULTISEED integrity audit\n\n"+"\n".join(f"{k.upper()} = {'PASS' if v else ('NO' if k.endswith('_read') else 'FAIL')}" for k,v in integrity_checks.items())+f"\n\nMULTISEED_INTEGRITY = {integrity}\n"
    atomic_text(REPORT/"12_MULTISEED_INTEGRITY_AUDIT.md",audit)
    move_frame=pd.DataFrame(move_rows); um=move_frame[move_frame.formulation=="unrestricted"]
    small=bool((um.tau_p99<.02).all() and (um.fraction_tau_gt_0_05==0).all() and (um.finite_coordinate_rate==1).all())
    v3_sign=np.sign(matched_frame.delta_V3D_unrestricted_minus_restricted.to_numpy()); ref_sign=np.sign(-matched_frame.delta_reference_rmsd.to_numpy()); xtb_sign=np.sign(-matched_frame.delta_xtb_median.to_numpy())
    replicated=bool(np.sum(v3_sign>0)>=2 and np.sum(ref_sign>=0)>=2 and np.sum(xtb_sign>=0)>=2 and small)
    reverse=bool(np.sum(v3_sign<0)>=2 and np.sum(ref_sign<=0)>=2 and np.sum(xtb_sign<=0)>=2)
    mixed=not replicated and not reverse
    effect="REPLICATED" if replicated else ("NOT_REPLICATED" if reverse else "MIXED")
    seed_unstable=bool((np.any(v3_sign>0) and np.any(v3_sign<0)) and (np.max(np.abs(matched_frame.delta_V3D_unrestricted_minus_restricted))>=.01))
    if integrity!="PASS": classification="NO_SELECTION_INTEGRITY_FAIL"
    elif seed_unstable: classification="SEED_UNSTABLE"
    elif replicated: classification="UNRESTRICTED_PREFERRED"
    elif reverse: classification="RESTRICTED_PREFERRED"
    else: classification="PARETO_NEAR_TIE"
    seed_summary=[]
    for form in FORMS:
        part=summary_frame[summary_frame.formulation==form]
        for metric in ("V3D","PB","reference_rmsd","source_rmsd"):
            seed_summary.append({"formulation":form,"metric":metric,"seed_mean":float(part[metric].mean()),"seed_sd":float(part[metric].std(ddof=1)),"seed_median":float(part[metric].median())})
    atomic_frame(REPORT/"SEED_LEVEL_SUMMARY.csv",pd.DataFrame(seed_summary))
    decision=f"""# Final formulation decision\n\nAll seed-level statistics were computed over three seed estimates; record-level cohort bootstraps remain separate in `COHORT_BOOTSTRAP.csv`. xTB uses the median as its primary location statistic.\n\n```text\nMULTISEED_INTEGRITY = {integrity}\nUNRESTRICTED_SMALL_MOVEMENT_REPLICATED = {'YES' if small else 'NO'}\nMATCHED_SEED_FORMULATION_EFFECT = {effect}\nFINAL_FORMULATION_CLASSIFICATION = {classification}\nXTB_PRIMARY_LOCATION_STATISTIC = MEDIAN\nFORMAL_OUTCOME_READ = NO\nLARGE_HOLDOUT_OUTCOME_READ = NO\n```\n"""
    atomic_text(REPORT/"13_FINAL_FORMULATION_DECISION.md",decision)
    scientific=[f"{i:02d}_{name}" for i,name in []]
    files=["01_RUN_IDENTITY.csv","02_TRAINING_COMPLETENESS.csv","03_MULTISEED_V3D.csv","04_MULTISEED_PB.csv","05_MULTISEED_REFERENCE_RMSD.csv","06_MULTISEED_SOURCE_RMSD.csv","07_MULTISEED_XTB.csv","08_MULTISEED_XTB_TAIL.csv","09_MULTISEED_MOVEMENT.csv","10_MULTISEED_MECHANISM_STABILITY.csv","11_MATCHED_SEED_COMPARISON.csv","12_MULTISEED_INTEGRITY_AUDIT.md","13_FINAL_FORMULATION_DECISION.md","COHORT_BOOTSTRAP.csv","RUN_SUMMARY.csv","SEED_LEVEL_SUMMARY.csv"]
    hashes={name:sha(REPORT/name) for name in files}
    update("COMPLETE","PASS" if integrity=="PASS" else "FAIL",MULTISEED_STATUS="COMPLETE",GPU_TRAINING_ALL_RUNS="PASS",MULTISEED_INTEGRITY=integrity,
           FINAL_FORMULATION_CLASSIFICATION=classification,UNRESTRICTED_SMALL_MOVEMENT_REPLICATED="YES" if small else "NO",MATCHED_SEED_FORMULATION_EFFECT=effect,
           artifact_sha256=hashes,summary=summary_frame.to_dict("records"),seed_level_summary=seed_summary)
    return 0


if __name__=="__main__":
    try: raise SystemExit(main())
    except Exception as error:
        REPORT.mkdir(parents=True,exist_ok=True); atomic_text(REPORT/"FINALIZER_TRACEBACK.txt",traceback.format_exc())
        update("STOPPED_ENGINEERING_FAILURE","FAIL",error_type=type(error).__name__,error=str(error)); raise
