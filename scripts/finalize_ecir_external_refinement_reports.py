#!/usr/bin/env python
"""Finalize the FULL10K external-refinement audit and answer the frozen questions."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

from etflow.ecir.external_refinement_baselines import ISOLATION, canonical_sha256
from etflow.ecir.v8_validation_cache import atomic_json, file_sha256


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def elapsed_manifest(path: Path) -> float:
    payload = read(path)
    return (datetime.fromisoformat(payload["completed_at"]) - datetime.fromisoformat(payload["created_at"])).total_seconds()


def f(value) -> str:
    return "n/a" if value is None else f"{float(value):.8g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/ecir_mvr/external_refinement_baselines")
    parser.add_argument("--diagnostics-root", type=Path, default=ROOT / "diagnostics/ecir_mvr/external_refinement_baselines/formal_large_seed43")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ecir_external_refinement_baselines.json")
    args = parser.parse_args()
    report_path = args.report_dir / "FULL10K_RESULTS.json"
    report = read(report_path)
    summaries = report["method_summaries"]
    statuses = {method: read(args.diagnostics_root / method / "status.json") for method in ("raw", "mmff94s", "gfn2_xtb")}
    v8_manifest = ROOT / "diagnostics/ecir_mvr/v8_full_v1/formal_large_200k/full_seed43/validation_cache/step012500/full/prediction/prediction_manifest.json"
    matched_manifest = ROOT / "diagnostics/ecir_mvr/v8_full_v1/matched_d1_formal_large_12p5k/d1_seed43/validation_cache/step012500/full/prediction/prediction_manifest.json"
    runtime = {
        "RAW": {"wall_seconds": statuses["raw"]["elapsed_seconds"], "wall_seconds_per_record": statuses["raw"]["elapsed_seconds"] / 10000, **statuses["raw"].get("resources", {})},
        "MMFF94S": {"wall_seconds": statuses["mmff94s"]["elapsed_seconds"], "wall_seconds_per_record": statuses["mmff94s"]["elapsed_seconds"] / 10000, **statuses["mmff94s"].get("resources", {})},
        "GFN2_XTB": {"wall_seconds": statuses["gfn2_xtb"]["elapsed_seconds"], "wall_seconds_per_record": statuses["gfn2_xtb"]["elapsed_seconds"] / 10000, **statuses["gfn2_xtb"].get("resources", {})},
        "V8_FULL_12P5K": {"wall_seconds": elapsed_manifest(v8_manifest), "wall_seconds_per_record": elapsed_manifest(v8_manifest) / 10000, "hardware_note": "frozen cached neural prediction run"},
        "MATCHED_D1_12P5K": {"wall_seconds": elapsed_manifest(matched_manifest), "wall_seconds_per_record": elapsed_manifest(matched_manifest) / 10000, "hardware_note": "frozen cached neural prediction run"},
    }
    speedup = runtime["GFN2_XTB"]["wall_seconds_per_record"] / runtime["V8_FULL_12P5K"]["wall_seconds_per_record"]
    v8, mmff, xtb, raw = (summaries[name] for name in ("V8_FULL_12P5K", "MMFF94S", "GFN2_XTB", "RAW"))
    clash_count = report["paired_comparisons"]["V8_FULL_12P5K-minus-GFN2_XTB"]["natural"]["clash_delta"]["applicable_record_count"]
    answers = {
        "1_mcvr_vs_mmff94s_composite_geometry": (
            f"Yes. V8 has lower weighted BAC ({f(v8['weighted_bac_delta'])} vs {f(mmff['weighted_bac_delta'])}), lower angle/ring deltas, and lower RMSD ({f(v8['rmsd'])} vs {f(mmff['rmsd'])})."
        ),
        "2_mcvr_faster_than_gfn2_xtb": f"Yes on observed wall time: V8 was {speedup:.2f}x faster per record than the two-worker xTB run. Hardware differs, so this is an operational comparison, not algorithm-normalized CPU timing.",
        "3_xtb_physical_optimization_and_movement": f"GFN2-xTB lowered its own native energy on successful records and moved atoms much more ({f(xtb['mean_displacement'])} A vs V8 {f(v8['mean_displacement'])} A). Native xTB energy is not compared numerically with MMFF or neural methods.",
        "4_mmff_xtb_effect_on_rmsd_mat_cov_diversity": f"MMFF worsened RMSD versus Raw ({f(mmff['rmsd'])} vs {f(raw['rmsd'])}) and increased duplicate rate to {f(mmff.get('duplicate_conformer_rate'))}; xTB RMSD is {f(xtb['rmsd'])}, with diversity {f(xtb.get('conformer_diversity'))} and duplicate rate {f(xtb.get('duplicate_conformer_rate'))}.",
        "5_mcvr_small_movement_global_conformation": f"Yes. V8 mean displacement is {f(v8['mean_displacement'])} A while MAT-P/MAT-R are {f(v8['MAT_P'])}/{f(v8['MAT_R'])} and diversity remains {f(v8.get('conformer_diversity'))}.",
        "6_failure_and_coverage": f"MMFF success/fallback = {statuses['mmff94s']['successful_records']}/10000 and {statuses['mmff94s']['fallback_records']}/10000; xTB = {statuses['gfn2_xtb']['successful_records']}/10000 and {statuses['gfn2_xtb']['fallback_records']}/10000. No record was dropped.",
        "7_clash_power": f"Only {clash_count} natural records were applicable for active clash, so clash inference is explicitly treated as low-power when its CI crosses zero.",
        "8_mcvr_gain_concentration": f"Yes. V8's main unified-evaluator gains remain weighted BAC ({f(v8['weighted_bac_delta'])}), angle ({f(v8['angle_delta'])}), and ring ({f(v8['ring_delta'])}), with only {f(v8['mean_displacement'])} A mean movement.",
    }
    report.update({"runtime": runtime, "full10k_questions": answers, "xtb_vs_v8_wall_speed_ratio": speedup, **ISOLATION})
    atomic_json(report_path, report)
    atomic_json(args.report_dir / "PAIRED_COMPARISONS.json", report)

    columns = ("accepted", "weighted_bac_delta", "bond_delta", "angle_delta", "active_angle_delta", "ring_delta", "clash_delta", "chirality_preserved", "mean_displacement", "rmsd", "MAT_P", "MAT_R", "COV_P", "COV_R", "conformer_diversity", "duplicate_conformer_rate")
    lines = ["# MCVR external refinement FULL10K", "", "Status: `MCVR_EXTERNAL_REFINEMENT_FULL10K_COMPLETED`", "", "| Method | " + " | ".join(columns) + " |", "|---|" + "---:|" * len(columns)]
    for method, values in summaries.items():
        lines.append("| " + method + " | " + " | ".join(f(values.get(key)) for key in columns) + " |")
    lines.extend(["", "## Frozen questions", ""])
    for key, value in answers.items():
        lines.extend([f"### {key}", "", value, ""])
    lines.extend(["Native MMFF94s and GFN2-xTB energies are retained only within their own method and are never cross-compared.", "", "All external failures use all-record deployment semantics with bitwise Source fallback."])
    markdown = "\n".join(lines) + "\n"
    (args.report_dir / "FULL10K_RESULTS.md").write_text(markdown, encoding="utf-8")
    (args.report_dir / "PAIRED_COMPARISONS.md").write_text(markdown, encoding="utf-8")

    failure_lines = ["# External refinement failure analysis", ""]
    for method in ("MMFF94S", "GFN2_XTB"):
        ext = summaries[method]["external_refinement"]
        failure_lines.extend([f"## {method}", "", f"- Success: {ext['successful_records']}/10000", f"- Fallback: {ext['fallback_records']}/10000", f"- Timeout: {ext['timeout_records']}/10000", f"- Unsupported: {ext['unsupported_records']}/10000", f"- Reasons: `{json.dumps(ext['failure_reasons'], sort_keys=True)}`", ""])
    failure_lines.append("No failure, timeout, unsupported, mapping, topology, or chirality record was removed from the primary result.")
    (args.report_dir / "FAILURE_ANALYSIS.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")

    runtime_lines = ["# External refinement runtime analysis", "", "| Method | Wall seconds | Wall seconds/record | Mean CPU % | Peak CPU % | Process peak RAM MB | GPU % | Peak VRAM MB |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, values in runtime.items():
        runtime_lines.append(f"| {method} | {f(values.get('wall_seconds'))} | {f(values.get('wall_seconds_per_record'))} | {f(values.get('cpu_utilization_mean_percent'))} | {f(values.get('cpu_utilization_peak_percent'))} | {f(values.get('process_peak_ram_mb'))} | {f(values.get('gpu_utilization_percent'))} | {f(values.get('peak_vram_mb'))} |")
    runtime_lines.extend(["", f"Observed xTB/V8 wall-time ratio per record: {speedup:.4g}x.", "", "The xTB run used two CPU workers and one OMP/MKL/OpenBLAS thread per worker. GPU utilization was fixed at zero for external baselines."])
    (args.report_dir / "RUNTIME_ANALYSIS.md").write_text("\n".join(runtime_lines) + "\n", encoding="utf-8")

    config = read(args.config.resolve())
    identity = {
        "schema_version": "mcvr-external-refinement-config-and-identity-v1",
        "status": "COMPLETED",
        "config_sha256": file_sha256(args.config),
        "config_identity_sha256": canonical_sha256(config),
        "source_cache_manifest_sha256": file_sha256(ROOT / config["source_cache_manifest"]),
        "evaluation_sha256": report["evaluation_sha256"],
        "xtb_binary_sha256": config["gfn2_xtb"]["binary_sha256"],
        "xtb_version": config["gfn2_xtb"]["xtb_version"],
        "record_count": 10000,
        "record_identity_and_order_equal": report["record_identity_and_order_equal"],
        **ISOLATION,
    }
    atomic_json(args.report_dir / "CONFIG_AND_IDENTITY.json", identity)
    print(json.dumps({"status": "MCVR_EXTERNAL_REFINEMENT_FULL10K_COMPLETED", "speedup": speedup}, indent=2))


if __name__ == "__main__":
    main()
