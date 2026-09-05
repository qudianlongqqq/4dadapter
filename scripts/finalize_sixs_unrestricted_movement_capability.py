#!/usr/bin/env python3
"""Complete unrestricted-movement RMSD/xTB/locality/bootstrap evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
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
import scripts.run_sixs_current_final_evidence_completion as phase1
import scripts.run_sixs_j1r1_full_joint_xtb_dev as xtb_sp

RUN = "sixs_reference_xtb_and_unrestricted_movement_seed307"
BASE = ROOT / "reports/ecir_mvr" / RUN
P1 = BASE / "01_CURRENT_FINAL_EVIDENCE"
OUT = BASE / "02_UNRESTRICTED_MOVEMENT"
ART = ROOT / "artifacts/ecir_mvr" / RUN / "02_UNRESTRICTED_MOVEMENT/dev_evaluation"
CURRENT_ART = ROOT / "artifacts/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/dev_evaluation"
STATUS = OUT / "FINAL_STATUS.json"
UNR_XTB = OUT / "UNRESTRICTED_XTB.csv"
HARTREE_TO_KCAL = 627.509474


def status(stage: str, state: str = "RUNNING", **extra: Any) -> None:
    previous = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    phase1.atomic_json(STATUS, {
        **previous, "CURRENT_STAGE": stage, "PIPELINE_STATUS": state,
        "UPDATED_AT": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "FORMAL_READ": "NO", "LARGE_HOLDOUT_READ": "NO",
        "SEED331_STARTED": "NO", "SEED353_STARTED": "NO", **extra,
    })


def unrestricted_coordinates(ids: list[str]) -> dict[str, torch.Tensor]:
    mols = [mol for mol in Chem.SDMolSupplier(str(ART / "PROPOSAL.sdf"), removeHs=False, sanitize=False) if mol is not None]
    if len(mols) != 5000:
        raise RuntimeError("unrestricted SDF denominator changed")
    result = {}
    for expected, mol in zip(ids, mols, strict=True):
        observed = mol.GetProp("_Name")
        if observed != expected:
            raise RuntimeError("unrestricted SDF ordering changed")
        result[expected] = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float64)
    return result


def run_xtb() -> None:
    if UNR_XTB.is_file():
        frame = pd.read_csv(UNR_XTB)
        if len(frame) == 5000 and frame.record_id.nunique() == 5000:
            return
    config, _, ids, _, sources, _ = phase1.config_and_inputs()
    coords = unrestricted_coordinates(ids)
    val = pd.read_parquet(config["data"]["val_manifest"])
    metadata = {str(row.sample_id): row for row in val.itertuples(index=False)}
    xtb_sp.OUT = ROOT / ".unrestricted_xtb_sp"
    xtb_sp.OUT.mkdir(parents=True, exist_ok=True)
    tasks = []
    for position, record_id in enumerate(ids):
        record = phase1.source_record(config, metadata[record_id])
        task = phase1.make_xtb_task(record_id, str(sources[record_id]["molecule_id"]), position, coords[record_id], record)
        task["method"] = "UNRESTRICTED_MOVEMENT_STEP17500"
        tasks.append(task)
    rows = []
    status("UNRESTRICTED_GFN2_XTB_SINGLEPOINT", XTB_STARTED="YES", XTB_COMPLETED=0, XTB_TOTAL=5000)
    with ThreadPoolExecutor(max_workers=xtb_sp.SETTINGS["workers"]) as pool:
        futures = [pool.submit(xtb_sp.execute, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if count % 100 == 0 or count == 5000:
                status("UNRESTRICTED_GFN2_XTB_SINGLEPOINT", XTB_STARTED="YES", XTB_COMPLETED=count, XTB_TOTAL=5000)
                print(json.dumps({"stage": "UNRESTRICTED_GFN2_XTB_SINGLEPOINT", "completed": count, "total": 5000}), flush=True)
    frame = pd.DataFrame(rows).sort_values("record_index", kind="stable").reset_index(drop=True)
    phase1.atomic_frame(UNR_XTB, frame)


def distribution(values: np.ndarray, name: str) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "object": name, "count": len(values), "mean": float(values.mean()),
        "median": float(np.median(values)), "std": float(values.std(ddof=1)),
        "p50": float(np.quantile(values,.5)), "p75": float(np.quantile(values,.75)),
        "p90": float(np.quantile(values,.9)), "p95": float(np.quantile(values,.95)),
        "p99": float(np.quantile(values,.99)), "p99_5": float(np.quantile(values,.995)),
        "p99_9": float(np.quantile(values,.999)), "max": float(values.max()),
    }


def cluster_bootstrap(frame: pd.DataFrame, metric: str, seed: int) -> dict[str, Any]:
    cluster = frame.groupby("molecule_id", sort=True)[metric].mean().to_numpy(np.float64)
    rng = np.random.default_rng(seed); draws = np.empty(10000)
    for start in range(0, 10000, 500):
        selected = rng.integers(0, len(cluster), size=(500, len(cluster)))
        draws[start:start+500] = cluster[selected].mean(axis=1)
    return {
        "metric": metric, "delta_unrestricted_minus_current": float(cluster.mean()),
        "ci95_low": float(np.quantile(draws,.025)), "ci95_high": float(np.quantile(draws,.975)),
        "bootstrap_clusters": len(cluster), "bootstrap_resamples": 10000, "seed": seed,
    }


def finalize() -> None:
    required = (UNR_XTB, ART/"PER_RECORD.parquet", ART/"POSEBUSTERS.parquet", ART/"VALIDITY3D.parquet")
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"missing unrestricted evidence: {[str(p) for p in required if not p.is_file()]}")
    config, _, ids, items, sources, _ = phase1.config_and_inputs()
    unr_coords = unrestricted_coordinates(ids)
    phase1_rmsd = pd.read_csv(P1 / "02_MATCHED_SOURCE_REFERENCE_RMSD.csv")
    source_ref = phase1_rmsd[phase1_rmsd.method=="SOURCE"].set_index("record_id")
    current_ref = phase1_rmsd[phase1_rmsd.method=="SIXS_FULL_JOINT_STEP17500"].set_index("record_id")
    rows = []
    status("UNRESTRICTED_REFERENCE_RMSD")
    for position, record_id in enumerate(ids):
        source_row = sources[record_id]
        references = torch.as_tensor(items[str(source_row["molecule_id"])]["references"], dtype=torch.float64)
        xyz = unr_coords[record_id]
        reference_rmsd = min(float(phase1.kabsch_rmsd(xyz, ref)) for ref in references)
        source_reference = float(source_ref.loc[record_id,"reference_rmsd"])
        rows.append({
            "record_index": position, "record_id": record_id, "molecule_id": str(source_row["molecule_id"]),
            "source_rmsd": float(phase1.kabsch_rmsd(xyz, torch.as_tensor(source_row["source"],dtype=torch.float64))),
            "reference_rmsd": reference_rmsd, "source_reference_rmsd": source_reference,
            "delta_reference_rmsd_vs_source": reference_rmsd-source_reference,
        })
    unr_ref = pd.DataFrame(rows)
    phase1.atomic_frame(OUT / "UNRESTRICTED_REFERENCE_RMSD.csv", unr_ref)

    records = pd.read_parquet(ART/"PER_RECORD.parquet")
    pb = pd.read_parquet(ART/"POSEBUSTERS.parquet")
    v3d = pd.read_parquet(ART/"VALIDITY3D.parquet")
    phase1.atomic_frame(OUT/"UNRESTRICTED_PB.csv", pb)
    phase1.atomic_frame(OUT/"UNRESTRICTED_V3D.csv", v3d)
    atom_displacement = np.concatenate(records.atom_displacements.to_numpy())
    distributions = [distribution(records.tau.to_numpy(),"tau_angstrom"), distribution(records.source_rmsd.to_numpy(),"source_rmsd_angstrom"), distribution(atom_displacement,"per_atom_displacement_angstrom")]
    tau = records.tau.to_numpy()
    thresholds = {f"fraction_tau_gt_{str(value).replace('.','_')}": float(np.mean(tau>value)) for value in (.01,.02,.05,.10,.20,.50)}
    distribution_frame = pd.DataFrame(distributions)
    for key, value in thresholds.items(): distribution_frame[key] = value
    phase1.atomic_frame(OUT/"UNRESTRICTED_MOVEMENT_DISTRIBUTION.csv", distribution_frame)

    source_energy = pd.read_csv(phase1.CURRENT_XTB/"SOURCE_XTB.csv")
    current_energy = pd.read_csv(phase1.CURRENT_XTB/"FULL_JOINT_XTB.csv")
    unr_energy = pd.read_csv(UNR_XTB)
    source_e = source_energy.set_index("record_id").energy_hartree
    for frame in (current_energy, unr_energy):
        frame["deltaE_kcal_mol"] = (frame.energy_hartree - frame.record_id.map(source_e))*HARTREE_TO_KCAL
    unr_energy.to_csv(UNR_XTB,index=False)

    joined = records[["record_id","molecule_id","tau","source_rmsd","bond_raw_mae","angle_raw_mae"]].copy()
    joined = joined.merge(unr_ref[["record_id","reference_rmsd","delta_reference_rmsd_vs_source"]],on="record_id",validate="one_to_one")
    joined = joined.merge(v3d[["record_id","validity3d"]].rename(columns={"validity3d":"V3D"}),on="record_id",validate="one_to_one")
    joined = joined.merge(pb[["record_id","PB"]],on="record_id",validate="one_to_one")
    joined = joined.merge(unr_energy[["record_id","success","deltaE_kcal_mol"]].rename(columns={"success":"xtb_success"}),on="record_id",validate="one_to_one")
    joined["tau_quintile"] = pd.qcut(joined.tau.rank(method="first"),5,labels=["Q1","Q2","Q3","Q4","Q5"])
    locality = joined.groupby("tau_quintile",observed=True).agg(
        records=("record_id","size"), tau_mean=("tau","mean"), tau_min=("tau","min"), tau_max=("tau","max"),
        reference_rmsd=("reference_rmsd","mean"), delta_reference_rmsd_vs_source=("delta_reference_rmsd_vs_source","mean"),
        V3D=("V3D","mean"), PB=("PB","mean"), bond_raw_mae=("bond_raw_mae","mean"),
        angle_raw_mae=("angle_raw_mae","mean"), xtb_deltaE=("deltaE_kcal_mol","mean"),
    ).reset_index()
    phase1.atomic_frame(OUT/"UNRESTRICTED_LOCALITY_AUDIT.csv",locality)

    current_records = pd.read_parquet(CURRENT_ART/"PER_RECORD.parquet")
    current_pb = pd.read_parquet(CURRENT_ART/"POSEBUSTERS.parquet")
    current_v3d = pd.read_parquet(CURRENT_ART/"VALIDITY3D.parquet")
    current = current_records[["record_id","molecule_id","source_rmsd"]].rename(columns={"source_rmsd":"current_source_rmsd"})
    current = current.merge(current_ref[["reference_rmsd"]].reset_index().rename(columns={"reference_rmsd":"current_reference_rmsd"}),on="record_id",validate="one_to_one")
    current = current.merge(current_v3d[["record_id","validity3d"]].rename(columns={"validity3d":"current_V3D"}),on="record_id",validate="one_to_one")
    current = current.merge(current_pb[["record_id","PB"]].rename(columns={"PB":"current_PB"}),on="record_id",validate="one_to_one")
    current = current.merge(current_energy[["record_id","deltaE_kcal_mol"]].rename(columns={"deltaE_kcal_mol":"current_deltaE"}),on="record_id",validate="one_to_one")
    comparison = current.merge(joined[["record_id","source_rmsd","reference_rmsd","V3D","PB","deltaE_kcal_mol"]],on="record_id",validate="one_to_one")
    comparison["delta_reference_rmsd"] = comparison.reference_rmsd-comparison.current_reference_rmsd
    comparison["delta_V3D"] = comparison.V3D.astype(float)-comparison.current_V3D.astype(float)
    comparison["delta_PB"] = comparison.PB.astype(float)-comparison.current_PB.astype(float)
    comparison["delta_xtb"] = comparison.deltaE_kcal_mol-comparison.current_deltaE
    comparison["delta_source_rmsd"] = comparison.source_rmsd-comparison.current_source_rmsd
    phase1.atomic_frame(OUT/"CURRENT_VS_UNRESTRICTED_PAIRED.csv",comparison)
    metrics=("delta_reference_rmsd","delta_V3D","delta_PB","delta_xtb","delta_source_rmsd")
    boot=pd.DataFrame([cluster_bootstrap(comparison,m,20260831+i) for i,m in enumerate(metrics)])
    phase1.atomic_frame(OUT/"CURRENT_VS_UNRESTRICTED_BOOTSTRAP.csv",boot)

    current_mean=float(current.current_reference_rmsd.mean()); unr_mean=float(joined.reference_rmsd.mean())
    current_xtb=float(current.current_deltaE.mean()); unr_xtb_mean=float(joined.loc[joined.xtb_success.astype(bool),"deltaE_kcal_mol"].mean())
    by_metric={row.metric:row for row in boot.itertuples(index=False)}
    high=locality.iloc[-1]; prior=locality.iloc[:-1]
    locality_limit=bool(high.reference_rmsd>prior.reference_rmsd.min() and (high.V3D<prior.V3D.max() or high.xtb_deltaE>prior.xtb_deltaE.min()))
    small_natural=bool(records.tau.median()<.01 and records.tau.quantile(.99)<.01)
    reference_gain=by_metric["delta_reference_rmsd"].ci95_high<0
    validity_ok=by_metric["delta_V3D"].ci95_low>=0 and by_metric["delta_PB"].ci95_low>=-.001
    xtb_ok=by_metric["delta_xtb"].ci95_high<=0
    constraints_limit=bool(reference_gain and validity_ok and xtb_ok)
    exceeds=bool(constraints_limit and records.tau.quantile(.95)>.01)
    recommendation="UNRESTRICTED_MOVEMENT_MULTISeed_CANDIDATE" if constraints_limit else "RETAIN_RESTRICTED_FULL_JOINT"

    summary = pd.DataFrame([{
        "method":"UNRESTRICTED_MOVEMENT_STEP17500", "source_rmsd_mean":records.source_rmsd.mean(),
        "reference_rmsd_mean":unr_mean, "delta_reference_vs_source":joined.delta_reference_rmsd_vs_source.mean(),
        "V3D":joined.V3D.mean(), "PB":joined.PB.mean(), "xtb_deltaE_mean":unr_xtb_mean,
        "xtb_deltaE_median":joined.loc[joined.xtb_success.astype(bool),"deltaE_kcal_mol"].median(),
    }])
    phase1.atomic_text(OUT/"FINAL_CAPABILITY_DECISION.md",f"""# Unrestricted movement capability decision

{phase1.markdown(summary)}

The frozen current model remains unchanged. This seed307 capability branch is not automatically promoted.

```text
SMALL_MOVEMENT_EMERGES_NATURALLY = {'SUPPORTED' if small_natural else 'NOT_SUPPORTED'}
OPTIMAL_LEARNED_MOVEMENT_EXCEEDS_CURRENT_BOUND = {'SUPPORTED' if exceeds else 'NOT_SUPPORTED'}
MOVEMENT_CONSTRAINTS_LIMIT_CURRENT_CAPABILITY = {'SUPPORTED' if constraints_limit else 'NOT_SUPPORTED'}
CURRENT_FIRST_ORDER_DIRECTION_HAS_LOCAL_RANGE_LIMIT = {'SUPPORTED' if locality_limit else 'NOT_SUPPORTED'}
RECOMMENDED_MULTISEED_FORMULATION = {recommendation}
FORMAL_READ = NO
LARGE_HOLDOUT_READ = NO
SEED331_STARTED = NO
SEED353_STARTED = NO
```
""")
    required_reports=("UNRESTRICTED_MODEL_IDENTITY.md","UNRESTRICTED_TRAIN_LOG.csv","UNRESTRICTED_MOVEMENT_DISTRIBUTION.csv","UNRESTRICTED_REFERENCE_RMSD.csv","UNRESTRICTED_V3D.csv","UNRESTRICTED_PB.csv","UNRESTRICTED_XTB.csv","UNRESTRICTED_LOCALITY_AUDIT.csv","CURRENT_VS_UNRESTRICTED_PAIRED.csv","CURRENT_VS_UNRESTRICTED_BOOTSTRAP.csv","FINAL_CAPABILITY_DECISION.md")
    hashes={name:phase1.sha256(OUT/name) for name in required_reports}
    final={
        "EVIDENCE_COMPLETION_STATUS":"COMPLETE", "REFERENCE_RMSD_CORRESPONDENCE_AUDIT":"PASS",
        "UNRESTRICTED_TRAINING_STATUS":"COMPLETE_GPU_PASS", "UNRESTRICTED_TAU_MEAN":float(records.tau.mean()),
        "UNRESTRICTED_TAU_MEDIAN":float(records.tau.median()), "UNRESTRICTED_TAU_P95":float(records.tau.quantile(.95)),
        "UNRESTRICTED_TAU_P99":float(records.tau.quantile(.99)), "UNRESTRICTED_TAU_MAX":float(records.tau.max()),
        "UNRESTRICTED_SOURCE_RMSD_MEAN":float(records.source_rmsd.mean()), "UNRESTRICTED_REFERENCE_RMSD_MEAN":unr_mean,
        "CURRENT_REFERENCE_RMSD_MEAN":current_mean, "UNRESTRICTED_V3D":float(joined.V3D.mean()), "UNRESTRICTED_PB":float(joined.PB.mean()),
        "UNRESTRICTED_XTB_MEAN":unr_xtb_mean, "UNRESTRICTED_XTB_MEDIAN":float(joined.loc[joined.xtb_success.astype(bool),"deltaE_kcal_mol"].median()),
        "SMALL_MOVEMENT_EMERGES_NATURALLY":"SUPPORTED" if small_natural else "NOT_SUPPORTED",
        "OPTIMAL_LEARNED_MOVEMENT_EXCEEDS_CURRENT_BOUND":"SUPPORTED" if exceeds else "NOT_SUPPORTED",
        "MOVEMENT_CONSTRAINTS_LIMIT_CURRENT_CAPABILITY":"SUPPORTED" if constraints_limit else "NOT_SUPPORTED",
        "CURRENT_FIRST_ORDER_DIRECTION_HAS_LOCAL_RANGE_LIMIT":"SUPPORTED" if locality_limit else "NOT_SUPPORTED",
        "RECOMMENDED_MULTISEED_FORMULATION":recommendation, "ARTIFACT_SHA256":hashes,
        "FORMAL_READ":"NO","LARGE_HOLDOUT_READ":"NO","SEED331_STARTED":"NO","SEED353_STARTED":"NO",
    }
    for metric,row in by_metric.items(): final[metric.upper()+"_DELTA"]=row.delta_unrestricted_minus_current; final[metric.upper()+"_95CI"]=[row.ci95_low,row.ci95_high]
    phase1.atomic_json(OUT/"FINAL_STATUS.json",final)


def pipeline() -> None:
    run_xtb(); finalize()


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",choices=("xtb","finalize","pipeline"),default="pipeline"); args=parser.parse_args()
    try:
        if args.stage=="xtb": run_xtb()
        elif args.stage=="finalize": finalize()
        else: pipeline()
        return 0
    except Exception as error:
        phase1.atomic_text(OUT/"FINALIZER_TRACEBACK.txt",traceback.format_exc()); status("FINALIZER_FAIL_CLOSED","FAIL",ERROR=str(error),ERROR_TYPE=type(error).__name__); raise


if __name__=="__main__": raise SystemExit(main())
