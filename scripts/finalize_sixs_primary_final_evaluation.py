#!/usr/bin/env python
"""Frozen molecule-cluster summary for the SIXS primary-final evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260901
PB_COMPONENTS = (
    "mol_pred_loaded", "sanitization", "inchi_convertible", "all_atoms_connected",
    "no_radicals", "bond_lengths", "bond_angles", "internal_steric_clash",
    "aromatic_ring_flatness", "non-aromatic_ring_non-flatness", "double_bond_flatness",
)
V3D_COMPONENTS = (
    "bond_geometry_valid", "angle_geometry_valid", "aromatic_ring_valid",
    "intramolecular_steric_clash_valid",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def trimmed_mean(values: np.ndarray, fraction: float = 0.05) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    trim = int(np.floor(fraction * len(ordered)))
    return float(ordered[trim : len(ordered) - trim].mean()) if len(ordered) > 2 * trim else float("nan")


def load_method(root: Path, xtb_root: Path, method: str) -> pd.DataFrame:
    target = root / "methods" / method
    records = pd.read_parquet(target / "PER_RECORD.parquet")
    pb = pd.read_parquet(target / "POSEBUSTERS.parquet")
    v3d = pd.read_parquet(target / "VALIDITY3D.parquet")
    frame = records.merge(pb[["record_id", "PB", *PB_COMPONENTS]], on="record_id", validate="one_to_one")
    frame = frame.merge(v3d[["record_id", "validity3d", *V3D_COMPONENTS]], on="record_id", validate="one_to_one")
    energy = pd.read_csv(xtb_root / f"{method.upper()}_XTB.csv")
    frame = frame.merge(energy[["record_id", "success", "energy_hartree", "failure_reason"]].rename(columns={"success": "xtb_success", "failure_reason": "xtb_failure_reason"}), on="record_id", validate="one_to_one")
    if len(frame) != 5000 or frame.molecule_id.nunique() != 2500 or not bool((frame.groupby("molecule_id").size() == 2).all()):
        raise RuntimeError(f"method denominator/alignment failure: {method}")
    return frame


def mean_difference_ci(candidate: pd.DataFrame, baseline: pd.DataFrame, column: str, seed_offset: int) -> tuple[float, float, float]:
    left = candidate.groupby("molecule_id", sort=True)[column].mean()
    right = baseline.groupby("molecule_id", sort=True)[column].mean()
    if not left.index.equals(right.index):
        raise RuntimeError("molecule alignment mismatch")
    delta = left.to_numpy(dtype=np.float64) - right.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws = rng.integers(0, len(delta), size=(RESAMPLES, len(delta)), endpoint=False)
    sampled = delta[draws].mean(axis=1)
    return float(delta.mean()), float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def median_delta_ci(candidate: pd.DataFrame, source: pd.DataFrame, seed_offset: int) -> tuple[float, float, float]:
    merged = candidate[["record_id", "molecule_id", "energy_hartree", "xtb_success"]].merge(
        source[["record_id", "energy_hartree", "xtb_success"]].rename(columns={"energy_hartree": "source_energy", "xtb_success": "source_success"}),
        on="record_id", validate="one_to_one",
    )
    merged["delta"] = (merged.energy_hartree - merged.source_energy) * 627.509474
    merged = merged[merged.xtb_success.astype(bool) & merged.source_success.astype(bool) & np.isfinite(merged.delta)]
    molecule_values = [group.delta.to_numpy(dtype=np.float64) for _, group in merged.groupby("molecule_id", sort=True)]
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    sampled = np.empty(RESAMPLES, dtype=np.float64)
    for index in range(RESAMPLES):
        chosen = rng.integers(0, len(molecule_values), size=len(molecule_values), endpoint=False)
        sampled[index] = np.median(np.concatenate([molecule_values[value] for value in chosen]))
    values = merged.delta.to_numpy(dtype=np.float64)
    return float(np.median(values)), float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def main(args: argparse.Namespace) -> int:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    methods = ["source"] + [method["id"] for method in protocol["model_methods"]] + ["mmff94s"]
    frames = {method: load_method(args.output_dir, args.output_dir / "xtb_single_point", method) for method in methods}
    source = frames["source"]
    source_energy = source[["record_id", "energy_hartree", "xtb_success"]].rename(columns={"energy_hartree": "source_energy", "xtb_success": "source_success"})
    summaries = []
    components = []
    bootstraps = []
    for method_index, method in enumerate(methods):
        frame = frames[method]
        merged_energy = frame.merge(source_energy, on="record_id", validate="one_to_one")
        merged_energy["delta_e"] = (merged_energy.energy_hartree - merged_energy.source_energy) * 627.509474
        finite = merged_energy.xtb_success.astype(bool) & merged_energy.source_success.astype(bool) & np.isfinite(merged_energy.delta_e)
        delta = merged_energy.loc[finite, "delta_e"].to_numpy(dtype=np.float64)
        summary = {
            "method": method,
            "records": len(frame),
            "molecules": frame.molecule_id.nunique(),
            "v3d_overall": float(frame.validity3d.astype(bool).mean()),
            "posebusters_overall": float(frame.PB.astype(bool).mean()),
            "bond_raw_mae": float(frame.bond_raw_mae.mean()),
            "angle_cosine_raw_mae": float(frame.angle_cosine_raw_mae.mean()),
            "raw_source_displacement_rms": float(frame.raw_source_displacement_rms.mean()),
            "kabsch_source_rmsd": float(frame.kabsch_source_rmsd.mean()),
            "reference_rmsd": float(frame.reference_rmsd.mean()),
            "xtb_success": int(finite.sum()),
            "xtb_failures": int((~finite).sum()),
            "xtb_delta_median": 0.0 if method == "source" else float(np.median(delta)),
            "xtb_delta_lower_fraction": 0.0 if method == "source" else float(np.mean(delta < 0)),
            "xtb_delta_5pct_trimmed_mean": 0.0 if method == "source" else trimmed_mean(delta),
            "xtb_delta_mean": 0.0 if method == "source" else float(np.mean(delta)),
            "xtb_delta_p90": 0.0 if method == "source" else float(np.quantile(delta, 0.90)),
            "xtb_delta_p95": 0.0 if method == "source" else float(np.quantile(delta, 0.95)),
            "xtb_delta_p99": 0.0 if method == "source" else float(np.quantile(delta, 0.99)),
            "xtb_delta_gt25": 0 if method == "source" else int(np.sum(delta > 25)),
            "xtb_delta_gt50": 0 if method == "source" else int(np.sum(delta > 50)),
            "xtb_delta_gt100": 0 if method == "source" else int(np.sum(delta > 100)),
        }
        if "tau" in frame:
            summary.update({
                "tau_median": float(frame.tau.median()),
                "tau_p99": float(frame.tau.quantile(0.99)),
                "tau_max": float(frame.tau.max()),
            })
        if "mmff_success" in frame:
            summary["mmff_success_rate"] = float(frame.mmff_success.astype(bool).mean())
        summaries.append(summary)
        for component in (*V3D_COMPONENTS, *PB_COMPONENTS):
            components.append({"method": method, "component": component, "pass_rate": float(frame[component].astype(bool).mean())})
        if method != "source":
            for column in ("validity3d", "PB", "bond_raw_mae", "angle_cosine_raw_mae", "reference_rmsd", "raw_source_displacement_rms"):
                point, low, high = mean_difference_ci(frame, source, column, method_index * 20 + len(bootstraps))
                bootstraps.append({"method": method, "baseline": "source", "metric": column, "effect_candidate_minus_baseline": point, "ci95_low": low, "ci95_high": high, "resampling_unit": "molecule"})
            point, low, high = median_delta_ci(frame, source, method_index)
            bootstraps.append({"method": method, "baseline": "source", "metric": "xtb_median_delta_e", "effect_candidate_minus_baseline": point, "ci95_low": low, "ci95_high": high, "resampling_unit": "molecule"})
    summary_frame = pd.DataFrame(summaries)
    seed_rows = []
    for formulation, prefix in (("Restricted", "restricted_seed"), ("Unrestricted", "unrestricted_seed")):
        group = summary_frame[summary_frame.method.str.startswith(prefix)]
        for metric in ("v3d_overall", "posebusters_overall", "reference_rmsd", "raw_source_displacement_rms", "xtb_delta_median", "xtb_delta_lower_fraction"):
            values = group[metric].to_numpy(dtype=np.float64)
            seed_rows.append({"formulation": formulation, "metric": metric, "mean_across_seeds": float(values.mean()), "seed_sd_ddof1": float(values.std(ddof=1)), "seed_values": ";".join(f"{value:.12g}" for value in values)})
    atomic_csv(args.report_dir / "03_FINAL_METHOD_SUMMARY.csv", summary_frame)
    atomic_csv(args.report_dir / "04_FINAL_COMPONENT_SUMMARY.csv", pd.DataFrame(components))
    atomic_csv(args.report_dir / "05_MOLECULE_CLUSTER_BOOTSTRAP.csv", pd.DataFrame(bootstraps))
    atomic_csv(args.report_dir / "06_SEED_LEVEL_SUMMARY.csv", pd.DataFrame(seed_rows))
    artifacts = {}
    for name in ("03_FINAL_METHOD_SUMMARY.csv", "04_FINAL_COMPONENT_SUMMARY.csv", "05_MOLECULE_CLUSTER_BOOTSTRAP.csv", "06_SEED_LEVEL_SUMMARY.csv"):
        artifacts[name] = sha256_file(args.report_dir / name)
    final = {
        "schema_version": "sixs-primary-final-evaluation-status-v1",
        "status": "PASS",
        "primary_final_outcome_opened": True,
        "methods": methods,
        "molecules": 2500,
        "records": 5000,
        "molecule_cluster_bootstrap_resamples": RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "final_formulation_selected_from_outcome": False,
        "operating_points_remain_predeclared": ["Restricted", "Unrestricted"],
        "model_training_performed": False,
        "outcome_dependent_exclusion": False,
        "protocol_sha256": sha256_file(args.protocol),
        "artifact_sha256": artifacts,
    }
    atomic_json(args.report_dir / "FINAL_STATUS.json", final)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("protocol", "output_dir", "report_dir"):
        setattr(args, name, getattr(args, name).resolve())
    return args


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
