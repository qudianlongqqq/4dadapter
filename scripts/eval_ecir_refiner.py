#!/usr/bin/env python
"""Molecule-aggregated ECIR evaluation with paired bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import numpy as np
import pandas as pd
import torch

from etflow.commons.kabsch_utils import kabsch_rmsd
from etflow.ecir.dataset import ECIRMixedDataset
from etflow.ecir.geometry import geometry_error_vector, severe_clash
from etflow.ecir.model import ECIRFlowSystem
from etflow.ecir.target_building import restrained_force_field_relaxation


INTERNAL_METRICS = (
    "bond_violation",
    "angle_violation",
    "torsion_circular_error",
    "ring_invalidity",
    "clash_score",
    "chirality_error",
)


def _load_baseline(specification: str) -> tuple[str, dict[str, torch.Tensor]]:
    name, raw_path = specification.split("=", 1)
    path = Path(raw_path)
    payload = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.suffix.lower() == ".json"
        else torch.load(path, map_location="cpu", weights_only=False)
    )
    records = payload.get("records", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(records, list):
        raise ValueError(f"Baseline {name} has no record list")
    coordinates = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        value = next(
            (
                record[key]
                for key in ("coordinates", "x_refined", "x_final", "pos", "x_cart", "x_init")
                if key in record
            ),
            None,
        )
        if value is not None:
            sample_id = str(record.get("sample_id", record.get("mol_id")))
            coordinates[sample_id] = torch.as_tensor(value, dtype=torch.float32)
    return name, coordinates


def _nearest_rmsd(coordinates: torch.Tensor, references: torch.Tensor) -> float:
    if references.ndim == 2:
        references = references.unsqueeze(0)
    return float(torch.stack([kabsch_rmsd(coordinates, ref) for ref in references]).min())


def _subsets(record: Mapping[str, Any]) -> list[str]:
    rotatable = int(record.get("num_rotatable_bonds", torch.as_tensor(record["rotatable_bond_index"]).size(1)))
    ring = bool(torch.as_tensor(record.get("bond_is_in_ring", [])).any())
    return [
        "all",
        "rotatable_le_2" if rotatable <= 2 else ("rotatable_3_5" if rotatable <= 5 else "rotatable_ge_6"),
        "ring" if ring else "non_ring",
    ]


def _metric_row(
    method: str,
    coordinates: torch.Tensor,
    target: torch.Tensor,
    references: torch.Tensor,
    record: Mapping[str, Any],
    elapsed_ms: float,
    ff_metadata: Mapping[str, Any] | None = None,
    source_type: str = "unknown",
) -> dict[str, Any]:
    error = geometry_error_vector(coordinates, target, record)
    ff = dict(ff_metadata or {})
    return {
        "molecule_id": str(record.get("source_mol_id", record.get("mol_id"))),
        "sample_id": str(record.get("sample_id", record.get("mol_id"))),
        "method": method,
        "source_type": source_type,
        "rotatable_bond_count": int(record.get("num_rotatable_bonds", 0)),
        "has_ring": bool(torch.as_tensor(record.get("bond_is_in_ring", [])).any()),
        "bond_violation": float(error[0]),
        "angle_violation": float(error[1]),
        "torsion_circular_error": float(error[2]),
        "ring_invalidity": float(error[3]),
        "clash_score": float(error[4]),
        "severe_clash": bool(severe_clash(coordinates, record["edge_index"])),
        "chirality_error": float(error[5]),
        "chirality_preserved": float(error[5]) == 0.0,
        "aligned_RMSD": _nearest_rmsd(coordinates, references),
        "MMFF_coverage": ff.get("method") == "MMFF94s" and bool(ff.get("supported")),
        "force_field_method": ff.get("method", "not_run"),
        "relaxation_energy_drop": ff.get("energy_drop"),
        "relaxation_RMSD": ff.get("relaxation_rmsd"),
        "optimization_success": ff.get("optimization_success"),
        "ms_per_conformer": elapsed_ms,
    }


def _cov_mat(records: list[dict[str, Any]], coordinate_map, reference_map, threshold: float):
    per_molecule = {}
    for molecule in sorted({row["molecule_id"] for row in records}):
        generated = coordinate_map[molecule]
        references = reference_map[molecule]
        matrix = torch.stack(
            [torch.stack([kabsch_rmsd(gen, ref) for ref in references]) for gen in generated]
        )
        per_molecule[molecule] = {
            "COV_P": float((matrix.min(1).values < threshold).float().mean()),
            "COV_R": float((matrix.min(0).values < threshold).float().mean()),
            "MAT_P": float(matrix.min(1).values.mean()),
            "MAT_R": float(matrix.min(0).values.mean()),
            "diversity": float(
                torch.stack(
                    [kabsch_rmsd(generated[i], generated[j]) for i in range(len(generated)) for j in range(i + 1, len(generated))]
                ).mean()
            ) if len(generated) > 1 else 0.0,
        }
    return per_molecule


def _paired_bootstrap(
    molecule_frame: pd.DataFrame,
    metric: str,
    baseline: str,
    candidate: str,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    pivot = molecule_frame.pivot(index="molecule_id", columns="method", values=metric).dropna()
    delta = pivot[candidate].to_numpy() - pivot[baseline].to_numpy()
    if delta.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(delta, size=delta.size, replace=True).mean() for _ in range(draws)])
    return float(delta.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--target_cache_dir", type=Path, required=True)
    parser.add_argument("--atlas_path", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_records", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--coverage_threshold", type=float, default=1.25)
    parser.add_argument("--bootstrap_draws", type=int, default=1000)
    parser.add_argument("--cov_noninferiority_margin", type=float, default=0.02)
    parser.add_argument("--rmsd_noninferiority_margin", type=float, default=0.02)
    parser.add_argument("--mat_noninferiority_margin", type=float, default=0.02)
    parser.add_argument("--unseen_source_result", type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("diagnostics/ecir/eval"))
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="Optional name=sample_payload for Cartesian/Strict/Serial comparison",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ECIRFlowSystem(**dict(payload["config"].get("model") or {})).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    dataset = ECIRMixedDataset(
        args.cache_dir,
        args.split,
        atlas_path=args.atlas_path,
        target_cache_dir=args.target_cache_dir,
        real_error_ratio=1.0,
        synthetic_error_ratio=0.0,
        clean_identity_ratio=0.0,
        max_records=args.max_records,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    baselines = dict(_load_baseline(specification) for specification in args.baseline)
    baseline_hits = {name: 0 for name in baselines}
    rows: list[dict[str, Any]] = []
    coordinates_by_method: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    references_by_molecule: dict[str, torch.Tensor] = {}
    total_started = time.perf_counter()
    for index, source_path in enumerate(dataset.files):
        record = torch.load(source_path, map_location="cpu", weights_only=False)
        entry = dataset.entries[index]
        data = dataset.get(index)
        target = data.x_target.cpu()
        references = torch.as_tensor(record.get("x_ref_candidates", record["x_ref_aligned"]), dtype=torch.float32)
        if references.ndim == 2:
            references = references.unsqueeze(0)
        molecule = str(record.get("source_mol_id", record.get("mol_id")))
        references_by_molecule.setdefault(molecule, references)
        upstream = torch.as_tensor(
            record[entry.get("coordinate_key", "x_init")], dtype=torch.float32
        )
        started = time.perf_counter()
        refined, diagnostics = model.refine(
            data.to(device), coordinates=upstream.to(device), steps=args.steps
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ecir_ms = (time.perf_counter() - started) * 1000.0
        methods = {"upstream": (upstream, 0.0), "ECIR_4step_teacher": (refined.cpu(), ecir_ms)}
        method_force_field: dict[str, Mapping[str, Any]] = {}
        sample_id = str(record.get("sample_id", record.get("mol_id")))
        for name, baseline in baselines.items():
            if sample_id in baseline:
                methods[name] = (baseline[sample_id], 0.0)
                baseline_hits[name] += 1
        for mmff_steps in (10, 50):
            started = time.perf_counter()
            ff_result = restrained_force_field_relaxation(record, upstream, max_steps=mmff_steps)
            elapsed = (time.perf_counter() - started) * 1000.0
            if ff_result.accepted and ff_result.coordinates is not None:
                method = f"MMFF_{mmff_steps}step" if ff_result.method == "MMFF94s" else f"UFF_{mmff_steps}step"
                methods[method] = (ff_result.coordinates, elapsed)
                method_force_field[method] = ff_result.metadata()
        for method, (coordinates, elapsed) in methods.items():
            ff = method_force_field.get(method)
            if method in {"upstream", "ECIR_4step_teacher"}:
                ff = restrained_force_field_relaxation(record, coordinates, max_steps=10).metadata()
            source_type = str(entry.get("source_type", record.get("generator_name", "unknown")))
            row = _metric_row(
                method, coordinates, target, references, record, elapsed, ff,
                source_type=source_type,
            )
            row["subsets"] = ";".join(
                _subsets(record)
                + [f"source_{source_type}"]
                + (["MMFF_supported"] if ff and ff.get("method") == "MMFF94s" else ["MMFF_unsupported"])
            )
            rows.append(row)
            coordinates_by_method[method][molecule].append(coordinates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conformer_frame = pd.DataFrame(rows)
    conformer_frame.to_csv(args.output_dir / "per_conformer.csv", index=False)
    numeric = list(INTERNAL_METRICS) + ["aligned_RMSD", "ms_per_conformer"]
    molecule_frame = conformer_frame.groupby(["molecule_id", "method"], as_index=False)[numeric].mean()
    for method, molecules in coordinates_by_method.items():
        cov = _cov_mat(
            [row for row in rows if row["method"] == method],
            molecules,
            references_by_molecule,
            args.coverage_threshold,
        )
        for molecule, values in cov.items():
            mask = (molecule_frame["molecule_id"] == molecule) & (molecule_frame["method"] == method)
            for key, value in values.items():
                molecule_frame.loc[mask, key] = value
    molecule_frame.to_csv(args.output_dir / "per_molecule.csv", index=False)
    summary_rows = []
    for subset in sorted({part for value in conformer_frame["subsets"] for part in value.split(";")}):
        subset_ids = conformer_frame.loc[conformer_frame["subsets"].str.split(";").apply(lambda values: subset in values), "molecule_id"].unique()
        selected_conformers = conformer_frame[
            conformer_frame["subsets"].str.split(";").apply(lambda values: subset in values)
        ]
        selected = selected_conformers.groupby(
            ["molecule_id", "method"], as_index=False
        )[numeric].mean()
        if subset == "all":
            selected = selected.merge(
                molecule_frame[["molecule_id", "method", "COV_P", "COV_R", "MAT_P", "MAT_R", "diversity"]],
                on=["molecule_id", "method"],
                how="left",
            )
        for method, frame in selected.groupby("method"):
            summary_rows.append({"subset": subset, "method": method, "molecules": frame["molecule_id"].nunique(), **{column: frame[column].mean() for column in frame.select_dtypes(include=[np.number]).columns if column != "molecule_id"}})
    summary = pd.DataFrame(summary_rows)
    bootstrap = {}
    for metric in (*INTERNAL_METRICS, "aligned_RMSD", "COV_P", "COV_R", "MAT_P", "MAT_R"):
        bootstrap[metric] = _paired_bootstrap(
            molecule_frame,
            metric,
            "upstream",
            "ECIR_4step_teacher",
            draws=args.bootstrap_draws,
            seed=42,
        )
    internal_passes = [metric for metric in INTERNAL_METRICS[:5] if bootstrap[metric][2] < 0.0]
    all_summary = summary[(summary["subset"] == "all")].set_index("method")
    rmsd_ok = bootstrap["aligned_RMSD"][2] <= args.rmsd_noninferiority_margin
    mat_p_ok = bootstrap["MAT_P"][2] <= args.mat_noninferiority_margin
    mat_r_ok = bootstrap["MAT_R"][2] <= args.mat_noninferiority_margin
    cov_ok = all_summary.loc["ECIR_4step_teacher", "COV_R"] >= all_summary.loc["upstream", "COV_R"] - args.cov_noninferiority_margin
    metric_gate_pass = (
        len(internal_passes) >= 2
        and rmsd_ok
        and mat_p_ok
        and mat_r_ok
        and cov_ok
    )
    require_unseen = bool(
        payload["config"].get("go_no_go", {}).get("require_unseen_source_pass", True)
    )
    unseen_source_pass = not require_unseen
    unseen_source_evidence = None
    if args.unseen_source_result is not None:
        unseen_source_evidence = json.loads(
            args.unseen_source_result.read_text(encoding="utf-8")
        )
        unseen_source_pass = (
            unseen_source_evidence.get("status") in {"PASS", "GO"}
            and not bool(unseen_source_evidence.get("sample_leakage", True))
        )
    go = metric_gate_pass and unseen_source_pass
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    result = {
        "status": "GO" if go else "NO_GO",
        "internal_metrics_with_directional_95ci": internal_passes,
        "rmsd_noninferiority_pass": bool(rmsd_ok),
        "mat_p_noninferiority_pass": bool(mat_p_ok),
        "mat_r_noninferiority_pass": bool(mat_r_ok),
        "cov_noninferiority_pass": bool(cov_ok),
        "metric_gate_pass": bool(metric_gate_pass),
        "unseen_source_required": require_unseen,
        "unseen_source_pass": bool(unseen_source_pass),
        "unseen_source_evidence": unseen_source_evidence,
        "bootstrap_delta_candidate_minus_upstream": {key: {"mean": value[0], "ci95_low": value[1], "ci95_high": value[2]} for key, value in bootstrap.items()},
        "NFE": args.steps,
        "total_pipeline_seconds": time.perf_counter() - total_started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "optional_baseline_records": baseline_hits,
        "test_used_for_selection": False,
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = [
        "# ECIR evaluation",
        "",
        f"Decision: **{result['status']}**",
        "",
        "Statistics are aggregated per molecule. Confidence intervals use paired molecule bootstrap resampling.",
        "",
        f"Internal metrics with a directional 95% CI: {', '.join(internal_passes) or 'none'}.",
        f"RMSD noninferiority: {rmsd_ok}; COV-R noninferiority: {cov_ok}.",
        f"MAT-P noninferiority: {mat_p_ok}; MAT-R noninferiority: {mat_r_ok}.",
        f"Unseen checkpoint/NFE/seed evidence required: {require_unseen}; pass: {unseen_source_pass}.",
        f"NFE: {args.steps}; total time: {result['total_pipeline_seconds']:.3f}s.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
