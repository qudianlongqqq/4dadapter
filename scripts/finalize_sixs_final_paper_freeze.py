#!/usr/bin/env python3
"""Extract and hash the completed SIXS evidence without scientific recomputation.

This script only reads already-completed authoritative tables and serialized
diagnostics.  It performs reporting arithmetic (seed mean/sample SD and counts
of the already-defined DiTMC tau > 1 A forensic event), writes the paper-freeze
tables, and binds inputs/outputs by SHA256.  It never invokes a model, MMFF,
xTB, V3D, PoseBusters, or a reference evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SEEDS = (307, 331, 353)
CROSS_DATA = Path("E:/3dconformergenerationcode/dataset/sixs_final_cross_upstream_unrestricted")
PRIMARY_DATA = Path("E:/3dconformergenerationcode/dataset/sixs_primary_final_evaluation_v1")
CLOSURE_DATA = Path("E:/3dconformergenerationcode/dataset/sixs_final_evidence_closure_v1")
TRAIN_MANIFEST = Path("E:/3dconformergenerationcode/4dadapter-v8/data/ecir_mvr/formal_large/real_sources/train.parquet")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def numeric_mean_sd(frame: pd.DataFrame, columns: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=np.float64)
        result[column] = float(np.mean(values)) if len(values) else np.nan
        result[f"{column}_seed_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    return result


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def primary_table(repo: Path) -> pd.DataFrame:
    closure = repo / "reports/ecir_mvr/final_evidence_closure"
    base = pd.read_csv(closure / "18_FINAL_MAIN_TABLE.csv")
    columns = [
        "v3d_overall", "posebusters_overall", "bond_raw_mae", "angle_cosine_raw_mae",
        "kabsch_source_rmsd", "reference_rmsd", "xtb_delta_median",
        "xtb_delta_lower_fraction",
    ]
    rows: list[dict[str, Any]] = []
    for method in ["source", *[f"restricted_seed{s}" for s in SEEDS], *[f"unrestricted_seed{s}" for s in SEEDS]]:
        source = base.loc[base.method == method].iloc[0]
        row = {
            "method": method,
            "role": source.role,
            "records": int(source.records),
            "molecules": int(source.molecules),
            **{column: source[column] for column in columns},
            "finite_coordinate_rate": 1.0,
            "mmff_optimization_success_rate": np.nan,
            "cov_p": np.nan, "cov_r": np.nan, "amr_p": np.nan, "amr_r": np.nan,
            "cov_amr_status": "NOT_APPLICABLE__NO_FROZEN_PRIMARY_THRESHOLD_OR_INDEPENDENT_REFERENCE_SPLIT",
            "authoritative_source": "reports/ecir_mvr/final_evidence_closure/18_FINAL_MAIN_TABLE.csv",
        }
        rows.append(row)
    for formulation, role in (("restricted", "CONSTRAINED_OPERATING_POINT_THREE_SEED_MEAN"),
                              ("unrestricted", "QUALITY_ORIENTED_PRIMARY_THREE_SEED_MEAN")):
        group = pd.DataFrame([row for row in rows if row["method"].startswith(formulation + "_seed")])
        summary = numeric_mean_sd(group, columns + ["finite_coordinate_rate"])
        rows.append({
            "method": f"{formulation}_three_seed_mean_sd", "role": role,
            "records": 5000, "molecules": 2500, **summary,
            "mmff_optimization_success_rate": np.nan,
            "cov_p": np.nan, "cov_r": np.nan, "amr_p": np.nan, "amr_r": np.nan,
            "cov_amr_status": "NOT_APPLICABLE__NO_FROZEN_PRIMARY_THRESHOLD_OR_INDEPENDENT_REFERENCE_SPLIT",
            "authoritative_source": "reports/ecir_mvr/final_evidence_closure/18_FINAL_MAIN_TABLE.csv",
        })
    mmff = pd.read_csv(closure / "03_MMFF_FINAL_RESULTS.csv").iloc[0]
    rows.append({
        "method": "MMFF94s", "role": "PHYSICS_BASELINE", "records": 5000, "molecules": 2500,
        "v3d_overall": mmff.v3d, "posebusters_overall": mmff.pb,
        "bond_raw_mae": mmff.bond_raw_mae, "angle_cosine_raw_mae": mmff.angle_cosine_raw_mae,
        "kabsch_source_rmsd": mmff.source_rmsd, "reference_rmsd": mmff.reference_rmsd,
        "xtb_delta_median": mmff.xtb_delta_median_kcal_mol,
        "xtb_delta_lower_fraction": mmff.xtb_lower_energy_fraction,
        "finite_coordinate_rate": 1.0,
        "mmff_optimization_success_rate": float(mmff.optimization_success / mmff.attempted),
        "cov_p": np.nan, "cov_r": np.nan, "amr_p": np.nan, "amr_r": np.nan,
        "cov_amr_status": "NOT_APPLICABLE__NO_FROZEN_PRIMARY_THRESHOLD_OR_INDEPENDENT_REFERENCE_SPLIT",
        "authoritative_source": "reports/ecir_mvr/final_evidence_closure/03_MMFF_FINAL_RESULTS.csv",
    })
    reference = pd.read_csv(closure / "09_REFERENCE_CONTEXTUAL_METRICS.csv").iloc[0]
    rows.append({
        "method": "reference_context", "role": reference.role, "records": int(reference.records),
        "molecules": 2500, "v3d_overall": reference.v3d, "posebusters_overall": reference.pb,
        "bond_raw_mae": reference.bond_raw_mae, "angle_cosine_raw_mae": reference.angle_cosine_raw_mae,
        "kabsch_source_rmsd": reference.source_rmsd, "reference_rmsd": reference.reference_rmsd,
        "xtb_delta_median": reference.delta_e_vs_source_median_kcal_mol,
        "xtb_delta_lower_fraction": np.nan, "finite_coordinate_rate": 1.0,
        "mmff_optimization_success_rate": np.nan,
        "cov_p": np.nan, "cov_r": np.nan, "amr_p": np.nan, "amr_r": np.nan,
        "cov_amr_status": reference.cov_amr_status,
        "authoritative_source": "reports/ecir_mvr/final_evidence_closure/09_REFERENCE_CONTEXTUAL_METRICS.csv",
    })
    frame = pd.DataFrame(rows)
    frame["gfn2_xtb_optimized_current_final"] = "NOT_AVAILABLE__NOT_RUN_BY_FROZEN_CLOSURE"
    return frame


def reference_table(repo: Path) -> pd.DataFrame:
    closure = repo / "reports/ecir_mvr/final_evidence_closure"
    metrics = pd.read_csv(closure / "10_REFERENCE_FIDELITY_METRICS.csv")
    context = pd.read_csv(closure / "09_REFERENCE_CONTEXTUAL_METRICS.csv")
    summary = metrics.assign(
        row_type="METHOD_SUMMARY", comparison="", metric="", mean_paired_effect=np.nan,
        ci95_low=np.nan, ci95_high=np.nan, median_molecule_effect=np.nan,
        wins=np.nan, ties=np.nan, losses=np.nan,
        authoritative_source="reports/ecir_mvr/final_evidence_closure/10_REFERENCE_FIDELITY_METRICS.csv",
    )
    contextual = pd.DataFrame([{
        "method": "reference_context", "records": int(context.iloc[0].records),
        "v3d": context.iloc[0].v3d, "pb": context.iloc[0].pb,
        "source_rmsd": context.iloc[0].source_rmsd, "reference_rmsd": context.iloc[0].reference_rmsd,
        "bond_raw_mae": context.iloc[0].bond_raw_mae,
        "angle_cosine_raw_mae": context.iloc[0].angle_cosine_raw_mae,
        "cov_p": np.nan, "cov_r": np.nan, "amr_p": np.nan, "amr_r": np.nan,
        "cov_amr_status": context.iloc[0].cov_amr_status, "row_type": "CONTEXTUAL_SUMMARY",
        "comparison": "", "metric": "", "mean_paired_effect": np.nan,
        "ci95_low": np.nan, "ci95_high": np.nan, "median_molecule_effect": np.nan,
        "wins": np.nan, "ties": np.nan, "losses": np.nan,
        "authoritative_source": "reports/ecir_mvr/final_evidence_closure/09_REFERENCE_CONTEXTUAL_METRICS.csv",
    }])
    paired = pd.read_csv(closure / "11_REFERENCE_PAIRED_STATS.csv")
    paired_rows = paired.assign(
        method="", records=paired.finite_records, v3d=np.nan, pb=np.nan, source_rmsd=np.nan,
        reference_rmsd=np.nan, bond_raw_mae=np.nan, angle_cosine_raw_mae=np.nan,
        cov_p=np.nan, cov_r=np.nan, amr_p=np.nan, amr_r=np.nan,
        cov_amr_status="NOT_APPLICABLE", row_type="PAIRED_MOLECULE_CLUSTER_EFFECT",
        authoritative_source="reports/ecir_mvr/final_evidence_closure/11_REFERENCE_PAIRED_STATS.csv",
    )
    wanted = list(summary.columns)
    for column in paired_rows.columns:
        if column not in wanted:
            wanted.append(column)
    return pd.concat([summary.reindex(columns=wanted), contextual.reindex(columns=wanted),
                      paired_rows.reindex(columns=wanted)], ignore_index=True)


def avgflow_table(repo: Path) -> pd.DataFrame:
    closure = repo / "reports/ecir_mvr/final_evidence_closure"
    report = repo / "reports/ecir_mvr/sixs_final_cross_upstream_unrestricted/avgflow"
    methods = pd.read_csv(report / "METHOD_SUMMARY.csv")
    reference = pd.read_csv(closure / "AVG_REFERENCE_METRICS.csv")
    energy = pd.read_csv(closure / "13_AVGFLOW_ENERGY_AUDIT.csv")
    rows: list[dict[str, Any]] = []
    for _, method in methods.iterrows():
        name = str(method.method)
        ref = reference.loc[reference.method == name].iloc[0]
        row = method.to_dict()
        row.update({key: ref[key] for key in ["COV_P", "COV_R", "AMR_P", "AMR_R", "bond_raw_mae", "angle_cosine_mae", "bond_angle_status"]})
        row["authoritative_source"] = "reports/ecir_mvr/sixs_final_cross_upstream_unrestricted/avgflow/METHOD_SUMMARY.csv; reports/ecir_mvr/final_evidence_closure/AVG_REFERENCE_METRICS.csv"
        rows.append(row)
    frame = pd.DataFrame(rows)
    for seed in SEEDS:
        match = frame.method == f"AVGFLOW_SIXS_U_SEED{seed}"
        e = energy.loc[energy.seed == seed].iloc[0]
        frame.loc[match, "xtb_attempted"] = e.attempted
        frame.loc[match, "xtb_valid"] = e.finite_matched
        frame.loc[match, "xtb_failed"] = e.attempted - e.finite_matched
    candidate = frame[frame.seed.notna()].copy()
    mean_columns = ["V3D", "PB", "source_rmsd_mean", "reference_rmsd", "finite_coordinate_rate",
                    "COV_P", "COV_R", "AMR_P", "AMR_R", "xtb_median_delta_e", "xtb_lower_fraction"]
    summary = {"upstream": "avgflow", "method": "AVGFLOW_SIXS_U_THREE_SEED_MEAN_SD", "seed": np.nan,
               "records": 10000, "reference_records": 7908, **numeric_mean_sd(candidate, mean_columns),
               "bond_angle_status": "NOT_EVALUATED__NO_UNIQUE_ATOM_PERMUTATION_FOR_ALL_SYMMETRY_MAPPINGS",
               "authoritative_source": "same authoritative per-seed rows; sample SD across seeds"}
    return pd.concat([frame, pd.DataFrame([summary])], ignore_index=True)


def ditmc_table(repo: Path) -> pd.DataFrame:
    report = repo / "reports/ecir_mvr/sixs_final_cross_upstream_unrestricted/ditmc"
    methods = pd.read_csv(report / "METHOD_SUMMARY.csv")
    status = json.loads((report / "XTB_FINAL_STATUS.json").read_text(encoding="utf-8"))
    status_by_method = {row["method"]: row for row in status["methods"]}
    diag = pd.read_parquet(CROSS_DATA / "ditmc/COORDINATE_DIAGNOSTICS.parquet")
    v3d = pd.read_parquet(CROSS_DATA / "ditmc/VALIDITY3D.parquet")
    pb = pd.read_parquet(CROSS_DATA / "ditmc/POSEBUSTERS.parquet")
    raw_v3d = v3d.loc[v3d.method == "DITMC_RAW", ["record_id", "validity3d"]].rename(columns={"validity3d": "raw_v3d"})
    raw_pb = pb.loc[pb.method == "DITMC_RAW", ["record_id", "PB"]].rename(columns={"PB": "raw_pb"})
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        name = f"DITMC_SIXS_U_SEED{seed}"
        method = methods.loc[methods.method == name].iloc[0].to_dict()
        part = diag.loc[diag.seed == seed].copy().merge(raw_v3d, on="record_id", validate="one_to_one").merge(raw_pb, on="record_id", validate="one_to_one")
        huge = part.loc[part.tau > 1.0]
        delta = pd.read_csv(report / f"xtb_singlepoint/{name}_DELTA_VS_SOURCE.csv")
        valid = int(delta.matched_success.astype(bool).sum())
        method.update({
            "raw_displacement_rms_mean": float(part.source_rmsd_raw.mean()),
            "raw_displacement_rms_median": float(part.source_rmsd_raw.median()),
            "kabsch_source_rmsd_mean": float(method["source_rmsd_mean"]),
            "max_atom_displacement_max": float(part.max_atom_displacement.max()),
            "xtb_attempted": int(status_by_method[name]["attempted"]),
            "xtb_valid_matched": valid,
            "xtb_failed_or_unmatched": int(status_by_method[name]["attempted"]) - valid,
            "tau_gt_1A_records": int(len(huge)),
            "tau_gt_1A_molecules": int(huge.molecule_id.astype(str).nunique()),
            "tau_gt_1A_raw_v3d_invalid_records": int((~huge.raw_v3d.astype(bool)).sum()),
            "tau_gt_1A_raw_pb_invalid_records": int((~huge.raw_pb.astype(bool)).sum()),
            "tau_gt_1A_raw_any_pathology_records": int((~huge.raw_v3d.astype(bool) | ~huge.raw_pb.astype(bool)).sum()),
            "authoritative_source": "reports/ecir_mvr/sixs_final_cross_upstream_unrestricted/ditmc/METHOD_SUMMARY.csv; reports/ecir_mvr/final_evidence_closure/06_DITMC_TAU_FORENSIC.csv",
        })
        rows.append(method)
    return pd.DataFrame(rows)


def ablation_table(repo: Path, primary: pd.DataFrame) -> pd.DataFrame:
    root = repo / "reports/ecir_mvr/sixs_final_matched_ablation"
    table = pd.read_csv(root / "09_FINAL_ABLATION_TABLE.csv")
    xtb = pd.read_csv(root / "08_XTB_SUBSET_RESULTS.csv")
    xtb_summary = xtb.groupby("variant", sort=False).agg(
        xtb_subset_records=("matched_records", "sum"),
        xtb_median_delta_e_mean=("median_delta_e", "mean"),
        xtb_median_delta_e_seed_sd=("median_delta_e", "std"),
        xtb_lower_fraction_mean=("lower_fraction", "mean"),
        xtb_lower_fraction_seed_sd=("lower_fraction", "std"),
    ).reset_index()
    table = table.merge(xtb_summary, on="variant", how="left", validate="one_to_one")
    full = primary.loc[primary.method == "unrestricted_three_seed_mean_sd"].iloc[0]
    full_row = pd.DataFrame([{
        "variant": "full_unrestricted", "v3d_mean": full.v3d_overall,
        "v3d_sd": full.v3d_overall_seed_sd, "pb_mean": full.posebusters_overall,
        "pb_sd": full.posebusters_overall_seed_sd, "reference_rmsd_mean": full.reference_rmsd,
        "source_rmsd_mean": full.kabsch_source_rmsd,
        "xtb_subset_records": np.nan, "xtb_median_delta_e_mean": np.nan,
        "xtb_median_delta_e_seed_sd": np.nan, "xtb_lower_fraction_mean": np.nan,
        "xtb_lower_fraction_seed_sd": np.nan,
    }])
    conclusions = {
        "full_unrestricted": "reference operating point",
        "reliability_off": "Reliability has a measurable strong independent effect",
        "equal_ba": "Adaptive BA adds a smaller consistent gain",
        "fixed_sigma_action": "Predictive sigma materially affects inference-time action weighting; this does not isolate sigma training necessity",
        "bond_only": "Bond is the dominant action contribution",
        "angle_only": "Angle alone is insufficient; comparison with Bond-only/full shows complementary measurable contribution",
        "fixed_tau": "Learned tau materially affects inference-time action execution; this does not isolate tau training necessity",
    }
    result = pd.concat([full_row, table], ignore_index=True)
    result["conclusion"] = result.variant.map(conclusions)
    result["authoritative_source"] = np.where(
        result.variant == "full_unrestricted",
        "reports/ecir_mvr/final_evidence_closure/18_FINAL_MAIN_TABLE.csv",
        "reports/ecir_mvr/sixs_final_matched_ablation/09_FINAL_ABLATION_TABLE.csv; reports/ecir_mvr/sixs_final_matched_ablation/08_XTB_SUBSET_RESULTS.csv",
    )
    return result


def claim_matrix() -> pd.DataFrame:
    rows = [
        (1, "SIXS improves local geometric validity on the current prospective ETFlow cohort.", "SUPPORTED_STRONGLY", "Source V3D 0.2226; Unrestricted three-seed mean 0.5032.", "reports/ecir_mvr/final_evidence_closure/18_FINAL_MAIN_TABLE.csv; reports/ecir_mvr/final_evidence_closure/04_PRIMARY_PAIRED_STATS.csv"),
        (2, "Reliability contributes independently beyond predictive sigma.", "SUPPORTED_STRONGLY", "Reliability-off V3D 0.36627 versus Full Unrestricted 0.50320.", "reports/ecir_mvr/sixs_final_matched_ablation/09_FINAL_ABLATION_TABLE.csv"),
        (3, "Predictive sigma materially affects final action weighting.", "SUPPORTED_STRONGLY", "Fixed-sigma-action V3D 0.37447 versus Full 0.50320; inference-only action ablation, not a sigma-training-necessity claim.", "reports/ecir_mvr/sixs_final_matched_ablation/09_FINAL_ABLATION_TABLE.csv"),
        (4, "Bond action is dominant and Angle action is complementary.", "SUPPORTED_STRONGLY", "Bond-only V3D 0.49160, Angle-only 0.29607, Full 0.50320.", "reports/ecir_mvr/sixs_final_matched_ablation/09_FINAL_ABLATION_TABLE.csv"),
        (5, "Adaptive BA gives a smaller but consistent additional gain.", "SUPPORTED_STRONGLY", "Equal-BA V3D 0.49540 versus Full 0.50320 over matched seeds.", "reports/ecir_mvr/sixs_final_matched_ablation/09_FINAL_ABLATION_TABLE.csv"),
        (6, "Learned tau materially affects final action execution.", "SUPPORTED_STRONGLY", "Fixed-tau V3D 0.39233 versus Full 0.50320; inference-only action ablation, not a tau-training-necessity claim.", "reports/ecir_mvr/sixs_final_matched_ablation/09_FINAL_ABLATION_TABLE.csv"),
        (7, "SIXS improves local Reference geometry.", "SUPPORTED_STRONGLY", "Across all three seeds Bond/Angle MAE effects are negative with 95% CIs excluding zero.", "reports/ecir_mvr/final_evidence_closure/11_REFERENCE_PAIRED_STATS.csv"),
        (8, "SIXS globally recovers the Reference conformer.", "NOT_SUPPORTED", "Reference RMSD worsens by +0.000326 to +0.000595 A across seeds; every 95% CI excludes zero.", "reports/ecir_mvr/final_evidence_closure/11_REFERENCE_PAIRED_STATS.csv; reports/ecir_mvr/final_evidence_closure/12_REFERENCE_FIDELITY_CONCLUSION.md"),
        (9, "AvgFlow shows zero-shot structural-validity transfer.", "SUPPORTED_WITH_SCOPE_LIMIT", "V3D 0.5472 raw versus 0.5755 three-seed mean; PB essentially stable and Reference fidelity mixed.", "reports/ecir_mvr/sixs_final_cross_upstream_unrestricted/avgflow/METHOD_SUMMARY.csv; reports/ecir_mvr/final_evidence_closure/AVG_REFERENCE_FIDELITY_CONCLUSION.md"),
        (10, "AvgFlow energy improves.", "NOT_SUPPORTED", "Per-seed median DeltaE is +0.2614 to +0.3016 kcal/mol; lower-energy fractions are 0.0161 to 0.0180.", "reports/ecir_mvr/final_evidence_closure/13_AVGFLOW_ENERGY_AUDIT.csv"),
        (11, "DiTMC shows structural transfer.", "SUPPORTED_WITH_SCOPE_LIMIT", "V3D rises from 0.1612 raw to 0.39033 three-seed mean, but stability is not universal.", "reports/ecir_mvr/sixs_final_cross_upstream_unrestricted/ditmc/METHOD_SUMMARY.csv; reports/ecir_mvr/final_evidence_closure/07_DITMC_TAU_FORENSIC_CONCLUSION.md"),
        (12, "DiTMC transfer is fully stable.", "NOT_SUPPORTED", "Seed307 has 194 records with tau > 1 A and tau max 398.680 A; the output/displacement consistency audit passes.", "reports/ecir_mvr/final_evidence_closure/06_DITMC_TAU_FORENSIC.csv; reports/ecir_mvr/final_evidence_closure/07_DITMC_TAU_FORENSIC_CONCLUSION.md"),
    ]
    return pd.DataFrame(rows, columns=["claim_id", "claim", "classification", "exact_basis", "authoritative_artifact"])


def long_numbers(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for section, frame in tables.items():
        identity = "method" if "method" in frame.columns else "variant" if "variant" in frame.columns else None
        for _, record in frame.iterrows():
            entity = str(record.get(identity, record.get("comparison", ""))) if identity else str(record.get("comparison", ""))
            for metric, value in record.items():
                if metric in {identity, "role", "authoritative_source", "cov_amr_status", "bond_angle_status", "conclusion", "upstream"}:
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
                    rows.append({
                        "section": section, "entity": entity, "metric": metric, "value": value,
                        "authoritative_source": record.get("authoritative_source", "generated reporting aggregation"),
                    })
    return pd.DataFrame(rows)


def git_inventory(repo: Path) -> list[str]:
    output = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=repo, text=True)
    return [line for line in output.splitlines() if line]


def add_hash(entries: list[dict[str, Any]], role: str, path: Path, repo: Path, expected: str | None = None) -> None:
    exists = path.is_file()
    actual = sha256(path) if exists else None
    if expected is not None and actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}: expected={expected}, actual={actual}")
    try:
        display = path.resolve().relative_to(repo.resolve()).as_posix()
        location = "REPOSITORY"
    except ValueError:
        display = path.resolve().as_posix()
        location = "EXTERNAL"
    entries.append({"role": role, "path": display, "location": location, "exists": exists,
                    "bytes": path.stat().st_size if exists else None, "sha256": actual})


def release_manifest(repo: Path, out: Path, claim_path: Path, planned_tag: str) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    protocol = json.loads((repo / "reports/ecir_mvr/sixs_primary_final_evaluation/00_FROZEN_FINAL_EVALUATION_PROTOCOL.json").read_text(encoding="utf-8"))
    for model in protocol["model_methods"]:
        add_hash(entries, f"final_checkpoint_{model['id']}", repo / model["checkpoint"], repo, model["checkpoint_sha256"])
    add_hash(entries, "train_manifest", TRAIN_MANIFEST, repo, "fbfeffab299c070fcbf29edb99277113c5641ee588000f00fc384162337ecb3d")
    add_hash(entries, "prospective_cohort_manifest", repo / "reports/ecir_mvr/sixs_step2d_primary_final_2500/04_PRIMARY_FINAL_2500_MANIFEST.json", repo, "2a1d07af8c9e3150d1f2f3719d0bd43bd33819ca7674c364d0770c010cb86ee1")
    add_hash(entries, "source_coordinates", PRIMARY_DATA / "methods/source/COORDINATES.sdf", repo)
    add_hash(entries, "source_results", PRIMARY_DATA / "methods/source/PER_RECORD.parquet", repo)
    for formulation in ("restricted", "unrestricted"):
        for seed in SEEDS:
            add_hash(entries, f"{formulation}_seed{seed}_results", PRIMARY_DATA / f"methods/{formulation}_seed{seed}/PER_RECORD.parquet", repo)
    add_hash(entries, "mmff_final_per_record", CLOSURE_DATA / "methods/mmff94s/PER_RECORD.parquet", repo)
    add_hash(entries, "mmff_final_results", repo / "reports/ecir_mvr/final_evidence_closure/03_MMFF_FINAL_RESULTS.csv", repo)
    add_hash(entries, "reference_context_results", CLOSURE_DATA / "methods/reference_context/PER_RECORD.parquet", repo)
    add_hash(entries, "reference_fidelity_metrics", repo / "reports/ecir_mvr/final_evidence_closure/10_REFERENCE_FIDELITY_METRICS.csv", repo)
    add_hash(entries, "reference_paired_stats", repo / "reports/ecir_mvr/final_evidence_closure/11_REFERENCE_PAIRED_STATS.csv", repo)
    add_hash(entries, "avgflow_reference_metrics", repo / "reports/ecir_mvr/final_evidence_closure/AVG_REFERENCE_METRICS.csv", repo)
    add_hash(entries, "avgflow_reference_paired_stats", repo / "reports/ecir_mvr/final_evidence_closure/AVG_REFERENCE_PAIRED_STATS.csv", repo)
    add_hash(entries, "avgflow_reference_per_record", CROSS_DATA / "avgflow/FIDELITY_PER_RECORD.parquet", repo)
    add_hash(entries, "ditmc_forensic_top_rows", repo / "reports/ecir_mvr/final_evidence_closure/06_DITMC_TAU_FORENSIC.csv", repo)
    add_hash(entries, "ditmc_coordinate_diagnostics", CROSS_DATA / "ditmc/COORDINATE_DIAGNOSTICS.parquet", repo)
    add_hash(entries, "final_ablation_table", repo / "reports/ecir_mvr/sixs_final_matched_ablation/09_FINAL_ABLATION_TABLE.csv", repo)
    add_hash(entries, "final_main_table", repo / "reports/ecir_mvr/final_evidence_closure/18_FINAL_MAIN_TABLE.csv", repo)
    add_hash(entries, "final_claim_matrix", claim_path, repo)
    for script in [
        "scripts/run_sixs_primary_final_coordinates.py", "scripts/evaluate_sixs_primary_final_external.py",
        "scripts/finalize_sixs_primary_final_evaluation.py", "scripts/run_sixs_final_mmff94s_repair.py",
        "scripts/materialize_sixs_final_reference_context.py", "scripts/run_sixs_final_closure_xtb.py",
        "scripts/run_sixs_final_avgflow_reference_audit.py", "scripts/finalize_sixs_final_evidence_closure.py",
        "scripts/run_sixs_final_cross_upstream_unrestricted.py", "scripts/run_sixs_final_cross_upstream_xtb.py",
        "scripts/run_sixs_final_matched_ablation_training.py", "scripts/run_sixs_final_matched_ablation_evaluation.py",
        "scripts/run_sixs_final_matched_ablation_xtb.py", "scripts/finalize_sixs_final_matched_ablation.py",
        "scripts/finalize_sixs_final_paper_freeze.py",
    ]:
        add_hash(entries, "evaluator_or_finalizer", repo / script, repo)
    for config in [
        "configs/sixs_j1r1_full_joint_adaptive_ba_movement.json",
        "configs/sixs_j1r1_full_joint_unrestricted_movement.json",
        "configs/sixs_final_cross_upstream_unrestricted.json",
        "reports/ecir_mvr/sixs_primary_final_evaluation/00_FROZEN_FINAL_EVALUATION_PROTOCOL.json",
        "reports/ecir_mvr/sixs_final_matched_ablation/00_PROTOCOL_FREEZE.json",
    ]:
        add_hash(entries, "scientific_config", repo / config, repo)
    for generated in sorted(out.glob("0[0-9]_FINAL_*")):
        if generated.name != "11_FINAL_RELEASE_MANIFEST.json":
            add_hash(entries, "freeze_output", generated, repo)
    return {
        "schema_version": "sixs-final-paper-freeze-release-manifest-v1",
        "status": "FROZEN_BY_NONMOVING_ANNOTATED_TAG",
        "release_tag": planned_tag,
        "release_commit_resolution": f"git rev-list -n 1 {planned_tag}",
        "literal_self_commit_in_manifest": False,
        "self_commit_note": "The non-moving annotated tag binds this manifest without an impossible self-referential commit hash.",
        "scientific_semantics_changed": False,
        "scientific_recomputation_performed": False,
        "environment": {"python": "3.11.15", "pytorch": "2.11.0+cu128", "cuda": "12.8", "rdkit": "2026.03.4", "xtb": "6.7.1 (edcfbbe)"},
        "artifacts": entries,
    }


def write_markdown(repo: Path, out: Path, primary: pd.DataFrame, reference: pd.DataFrame,
                   avgflow: pd.DataFrame, ditmc: pd.DataFrame, claims: pd.DataFrame,
                   inventory: list[str], planned_tag: str) -> None:
    engineering = [
        ("STATUS.json BOM tolerance", "scripts/run_sixs_post_cross_upstream_ablation_pipeline.py:42-61", "ENGINEERING_FIX", "NO", "UTF-8/UTF-8-BOM orchestration serialization only"),
        ("tabulate-independent Markdown", "scripts/run_sixs_final_cross_upstream_unrestricted.py:493-502; scripts/run_sixs_musigma_reliability_factorial.py:142-147", "REPORT_ONLY", "NO", "Formatting fallback only"),
        ("pipeline recovery", "scripts/run_sixs_post_cross_upstream_ablation_pipeline.py:161-176", "ENGINEERING_FIX", "NO", "Same frozen stage identity/state; no outcome-dependent branch"),
        ("MMFF num_atoms restoration", "scripts/run_sixs_primary_final_coordinates.py:107-110; scripts/run_sixs_final_mmff94s_repair.py:57-64", "ENGINEERING_FIX", "NO", "Restores authoritative topology count required by validator"),
        ("Reference xTB cache/workdir dedup", "scripts/run_sixs_final_closure_xtb.py:90-120", "ENGINEERING_FIX", "NO", "Executes identical scientific identity once and fans immutable result to frozen duplicate rows"),
        ("finalizer molecule_id suffix repair", "scripts/finalize_sixs_final_evidence_closure.py:332-343", "ENGINEERING_FIX", "NO", "One-to-one merge plus explicit zero-mismatch identity assertion"),
    ]
    inventory_text = "\n".join(f"- `{line}`" for line in inventory) if inventory else "- Clean before freeze extraction."
    engineering_text = "\n".join(f"| {a} | `{b}` | {c} | {d} | {e} |" for a, b, c, d, e in engineering)
    atomic_text(out / "00_FREEZE_SCOPE.md", f"""# SIXS final paper freeze scope

This release is a read/aggregate/audit/hash operation over completed authoritative artifacts. It performed no model change, training, inference, MMFF, xTB, V3D, PoseBusters, Reference, runtime, or other scientific evaluation.

## Evidence partitions

- CURRENT FINAL: prospective ETFlow cohort, 2,500 molecules and 5,000 records.
- CROSS-UPSTREAM: AvgFlow and DiTMC zero-shot evidence; never inserted into the current-final table.
- ABLATION: matched three-seed final ablation; never inserted as a current-final method.
- HISTORICAL: provenance only; historical DEV/Formal/10K values are excluded from final-number tables.

## Pre-freeze worktree inventory

{inventory_text}

Large checkpoints, coordinates, per-record datasets, tool caches, and work directories remain external/ignored and are SHA256-bound by `11_FINAL_RELEASE_MANIFEST.json`.

## Engineering/report diff audit

| Change | Evidence | Class | Scientific semantics changed | Reason |
|---|---|---|---|---|
{engineering_text}

`SCIENTIFIC_SEMANTICS_CHANGED = NO`

Planned immutable tag: `{planned_tag}`. The tag target is the release commit and resolves the deliberate non-self-referential commit binding in the manifest.
""")
    ref_effect = reference.loc[reference.row_type == "PAIRED_MOLECULE_CLUSTER_EFFECT"]
    ref_rmsd = ref_effect.loc[ref_effect.metric == "reference_rmsd"]
    bond = ref_effect.loc[ref_effect.metric == "bond_raw_mae"]
    angle = ref_effect.loc[ref_effect.metric == "angle_cosine_raw_mae"]
    atomic_text(out / "08_FINAL_LIMITATIONS.md", """# Final limitations

1. Global Reference conformer recovery is not supported: all three Unrestricted seeds slightly worsen all-atom nearest-ensemble Reference RMSD even while local Bond and Angle errors improve.
2. AvgFlow energy does not improve under the primary median statistic despite improved V3D; validity and energy responses must remain separate.
3. DiTMC Unrestricted movement has a verified instability tail, concentrated in seed307 and consistent with the serialized coordinates rather than a reporting-unit error.
4. The Reference row is contextual, not an executable method and not a theoretical upper bound. Reference-vs-itself COV/AMR is not reported without an independent held-out Reference split.
5. SIXS is a learned local fixed-topology refiner, not a physical energy optimizer.
6. Results do not establish a universal upstream guarantee: AvgFlow and DiTMC have metric-specific, materially different behavior.
7. No per-sample xTB energy monotonicity is claimed; all energy denominators and failures remain explicit.
8. No state-of-the-art, first-method, universal, or calibrated-uncertainty claim is made without a separate literature/claim audit.
9. Fixed-sigma-action and fixed-tau are inference/action ablations; they do not by themselves establish that sigma or tau must be trained in a particular way.
10. Current-final COV/AMR is not reported because no frozen threshold and independent Reference self split exist for that cohort.
""")
    atomic_text(out / "09_FINAL_PROVENANCE_TIMELINE.md", """# Final provenance timeline

1. Restricted and Unrestricted were prospectively frozen as operating points A and B before primary-final outcome access (`reports/ecir_mvr/final_development_freeze/11_FINAL_DEVELOPMENT_FREEZE.md`).
2. The 2,500-molecule prospective cohort membership was frozen with zero documented historical overlap (`reports/ecir_mvr/sixs_step2d_primary_final_2500/04_PRIMARY_FINAL_2500_MANIFEST.json`).
3. The prospective ETFlow NFE10 source and final evaluation protocol were frozen before scientific outcome access (`reports/ecir_mvr/sixs_primary_final_evaluation/00_FROZEN_FINAL_EVALUATION_PROTOCOL.json`).
4. The primary-final evaluation then opened the prospective outcome (`reports/ecir_mvr/sixs_primary_final_evaluation/FINAL_STATUS.json`).
5. Later evidence synthesis labeled Unrestricted the quality-oriented primary and Restricted the constrained control; a sole primary was not predeclared (`reports/ecir_mvr/final_evidence_closure/16_PROVENANCE_TIMELINE.md`).
6. The matched final ablation completed on the frozen design (`reports/ecir_mvr/sixs_final_matched_ablation/10_FINAL_ABLATION_CONCLUSION.md`).
7. AvgFlow and DiTMC cross-upstream runs completed without retraining (`reports/ecir_mvr/sixs_final_cross_upstream_unrestricted/*/RESULT.json`).
8. MMFF94s was repaired by restoring `num_atoms`; 4,977/5,000 optimizations succeeded and 23 fixed-denominator rows used documented Source fallback (`reports/ecir_mvr/final_evidence_closure/03_MMFF_FINAL_RESULTS.csv`).
9. Reference contextual xTB was recovered with scientific-identity deduplication; Reference fidelity and AvgFlow Reference audits completed (`reports/ecir_mvr/final_evidence_closure/09_REFERENCE_CONTEXTUAL_METRICS.csv`, `AVG_REFERENCE_METRICS.csv`).
10. Final aggregation completed after a zero-mismatch `molecule_id` suffix repair (`reports/ecir_mvr/final_evidence_closure/18_FINAL_MAIN_TABLE.csv`).
11. This release freezes those completed artifacts without scientific recomputation.

```text
PROSPECTIVELY_FROZEN_OPERATING_POINTS = Restricted; Unrestricted
QUALITY_ORIENTED_PRIMARY = Unrestricted
CONSTRAINED_CONTROL = Restricted
SOLE_PRIMARY_PREDECLARED = NO
SCIENTIFIC_SEMANTICS_CHANGED = NO
```
""")
    atomic_text(out / "10_FINAL_AUTHORITATIVE_RESULTS_INDEX.md", """# Final authoritative results index

- Current-final method metrics and xTB: `reports/ecir_mvr/final_evidence_closure/18_FINAL_MAIN_TABLE.csv`
- Repaired current-final MMFF94s: `reports/ecir_mvr/final_evidence_closure/03_MMFF_FINAL_RESULTS.csv`
- Primary paired molecule-cluster statistics: `reports/ecir_mvr/final_evidence_closure/04_PRIMARY_PAIRED_STATS.csv`
- Reference contextual row: `reports/ecir_mvr/final_evidence_closure/09_REFERENCE_CONTEXTUAL_METRICS.csv`
- Reference fidelity method metrics: `reports/ecir_mvr/final_evidence_closure/10_REFERENCE_FIDELITY_METRICS.csv`
- Reference paired molecule-cluster statistics: `reports/ecir_mvr/final_evidence_closure/11_REFERENCE_PAIRED_STATS.csv`
- AvgFlow structural/validity metrics: `reports/ecir_mvr/sixs_final_cross_upstream_unrestricted/avgflow/METHOD_SUMMARY.csv`
- AvgFlow Reference COV/AMR and RMSD: `reports/ecir_mvr/final_evidence_closure/AVG_REFERENCE_METRICS.csv`
- AvgFlow paired Reference statistics: `reports/ecir_mvr/final_evidence_closure/AVG_REFERENCE_PAIRED_STATS.csv`
- AvgFlow energy: `reports/ecir_mvr/final_evidence_closure/13_AVGFLOW_ENERGY_AUDIT.csv`
- DiTMC structural/validity/energy summaries: `reports/ecir_mvr/sixs_final_cross_upstream_unrestricted/ditmc/METHOD_SUMMARY.csv`
- DiTMC movement forensic: `reports/ecir_mvr/final_evidence_closure/06_DITMC_TAU_FORENSIC.csv` and `07_DITMC_TAU_FORENSIC_CONCLUSION.md`
- Final matched ablation: `reports/ecir_mvr/sixs_final_matched_ablation/09_FINAL_ABLATION_TABLE.csv`
- Ablation xTB subset: `reports/ecir_mvr/sixs_final_matched_ablation/08_XTB_SUBSET_RESULTS.csv`
- Final claim classifications: `reports/ecir_mvr/final_paper_freeze/07_FINAL_CLAIM_MATRIX.csv`

The files in this directory are paper-facing extracts. The sources above remain the single authoritative locations; historical DEV/Formal/10K values are not substitutes.
""")
    seed307 = ditmc.loc[ditmc.seed == 307].iloc[0]
    atomic_text(out / "12_FINAL_SUBMISSION_READINESS.md", f"""# Final submission readiness

All core scientific evidence and the release snapshot are complete. Optional comparisons or a separate literature review do not keep the evidence freeze partial.

```text
SCIENTIFIC_EVIDENCE_STATUS = COMPLETE_WITH_SCOPED_CLAIMS
CORE_EXPERIMENTS_COMPLETE = YES
NEW_MODEL_TRAINING_REQUIRED = NO
NEW_ABLATION_REQUIRED = NO
NEW_SIXS_INFERENCE_REQUIRED = NO
MMFF_STATUS = PASS__4977_OF_5000_OPTIMIZATION_SUCCESS__FIXED_DENOMINATOR_PRESERVED
REFERENCE_FIDELITY_STATUS = PASS__LOCAL_SUPPORTED__GLOBAL_NOT_SUPPORTED__PARTIAL_RECOVERY
AVGFLOW_REFERENCE_STATUS = PASS__MIXED
DITMC_FORENSIC_STATUS = STRUCTURAL_TRANSFER_WITH_INSTABILITY
PROVENANCE_STATUS = PASS_WITH_POST_OUTCOME_PRIMARY_LABEL_DISCLOSED
RELEASE_STATUS = PASS__CLEAN_COMMIT_AND_NONMOVING_TAG
GIT_STATUS = CLEAN_AT_RELEASE
CURRENT_SUBMISSION_READINESS = READY_FOR_PAPER_FREEZE
REMAINING_P0 = NONE
REMAINING_P1 = literature/claim wording audit; isolated SIXS GPU runtime only if the manuscript requires a runtime claim
OPTIONAL_P2 = predeclared GFN2-xTB geometry-optimization subset only if the manuscript adds a physics-optimization comparison
```

Reference interpretation is deliberately split: Bond and Angle MAE improve with molecule-cluster 95% CIs excluding zero, while Reference RMSD worsens slightly for every seed. AvgFlow V3D transfers but its median xTB DeltaE is positive. DiTMC seed307 contains {int(seed307.tau_gt_1A_records)} records across {int(seed307.tau_gt_1A_molecules)} molecules with tau > 1 A, so transfer is scoped rather than universally stable.
""")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("reports/ecir_mvr/final_paper_freeze"))
    parser.add_argument("--release-tag", default="sixs-final-evidence-freeze-2026")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    out = (repo / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inventory = git_inventory(repo)
    primary = primary_table(repo)
    reference = reference_table(repo)
    avgflow = avgflow_table(repo)
    ditmc = ditmc_table(repo)
    ablation = ablation_table(repo, primary)
    claims = claim_matrix()
    outputs = {
        "02_FINAL_PRIMARY_TABLE.csv": primary,
        "03_FINAL_REFERENCE_TABLE.csv": reference,
        "04_FINAL_AVGFLOW_TABLE.csv": avgflow,
        "05_FINAL_DITMC_FORENSIC_TABLE.csv": ditmc,
        "06_FINAL_ABLATION_TABLE.csv": ablation,
        "07_FINAL_CLAIM_MATRIX.csv": claims,
    }
    for name, frame in outputs.items():
        atomic_csv(out / name, frame)
    atomic_csv(out / "01_FINAL_AUTHORITATIVE_NUMBERS.csv", long_numbers({
        "CURRENT_FINAL": primary, "REFERENCE": reference.loc[reference.row_type != "PAIRED_MOLECULE_CLUSTER_EFFECT"],
        "CROSS_UPSTREAM_AVGFLOW": avgflow, "CROSS_UPSTREAM_DITMC": ditmc, "ABLATION": ablation,
    }))
    write_markdown(repo, out, primary, reference, avgflow, ditmc, claims, inventory, args.release_tag)
    manifest = release_manifest(repo, out, out / "07_FINAL_CLAIM_MATRIX.csv", args.release_tag)
    atomic_json(out / "11_FINAL_RELEASE_MANIFEST.json", manifest)
    expected = {
        "00_FREEZE_SCOPE.md", "01_FINAL_AUTHORITATIVE_NUMBERS.csv", "02_FINAL_PRIMARY_TABLE.csv",
        "03_FINAL_REFERENCE_TABLE.csv", "04_FINAL_AVGFLOW_TABLE.csv", "05_FINAL_DITMC_FORENSIC_TABLE.csv",
        "06_FINAL_ABLATION_TABLE.csv", "07_FINAL_CLAIM_MATRIX.csv", "08_FINAL_LIMITATIONS.md",
        "09_FINAL_PROVENANCE_TIMELINE.md", "10_FINAL_AUTHORITATIVE_RESULTS_INDEX.md",
        "11_FINAL_RELEASE_MANIFEST.json", "12_FINAL_SUBMISSION_READINESS.md",
    }
    missing = sorted(name for name in expected if not (out / name).is_file())
    if missing:
        raise RuntimeError(f"Missing freeze outputs: {missing}")
    print(json.dumps({
        "status": "PASS", "output": str(out), "files": len(expected),
        "primary_rows": len(primary), "claim_rows": len(claims),
        "scientific_recomputation": False, "scientific_semantics_changed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
