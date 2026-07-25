#!/usr/bin/env python3
"""Frozen BA versus BA+C internal mechanism gate (no external evaluators)."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
from etflow.ecir.bat_refinement import combined_gradient_update, frozen_ba_update, prepare_bat_graph, steric_metrics
from etflow.ecir.learned_geometry import LearnedGeometryObjective, distribution_parameters, prepare_graph, structured_objective
from etflow.ecir.lsgo_io import atomic_json, file_sha256


OUT = ROOT / "reports/ecir_mvr/bat_refinement"
CONFIG_PATH = ROOT / "configs/ecir_mvr_bat_refinement.yaml"
ISOLATION = {"formal_test_records_read": 0, "frozen_holdout_records_read": 0, "posebusters_access": False, "xtb_access": False}


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def deterministic(items, partition: str, count: int):
    rows = [item for item in items if item["partition"] == partition]
    rows.sort(key=lambda item: hashlib.sha256(str(item["molecule_id"]).encode()).hexdigest())
    return rows[:count]


def load_model(row, device):
    path = Path(row["path"])
    if file_sha256(path) != row["sha256"]:
        raise RuntimeError("frozen BA checkpoint SHA mismatch")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = LearnedGeometryObjective(hidden_dim=128, layers=3, learned_sigma=False).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def objective(coordinates, graph, parameters):
    return float(structured_objective(coordinates, graph, parameters)[0].detach())


def evaluate_partition(items, calibration, model, seed, partition, device, config):
    records = []
    steric = config["steric"]
    for molecule_index, item in enumerate(deterministic(items, partition, 48)):
        base = prepare_graph(item["record"], calibration).to(device)
        bat = prepare_bat_graph(
            base, item["record"], safe_factor_nonbonded=float(steric["safe_factor_nonbonded"]),
            safe_factor_1_4=float(steric["safe_factor_1_4"]),
            catastrophic_factor=float(steric["catastrophic_factor"]), include_hydrogens=False,
        ).to(device)
        with torch.no_grad():
            parameters = distribution_parameters(base, model=model, variant="B")
        for source_index, source_cpu in enumerate(item["sources"]):
            source = torch.as_tensor(source_cpu, dtype=torch.float64, device=device)
            ba = frozen_ba_update(source, bat, parameters, rms_budget=.003, atom_cap=.03)
            bac = combined_gradient_update(
                source, bat, parameters, rms_budget=.003, atom_cap=.03,
                tau=float(steric["tau_angstrom"]), fallback_coordinates=ba["coordinates"],
                backtracking_fractions=steric["backtracking_fractions"],
            )
            source_steric = steric_metrics(source, bat)
            ba_steric = steric_metrics(ba["coordinates"], bat)
            bac_steric = steric_metrics(bac["coordinates"], bat)
            delta = bac["coordinates"] - source
            records.append({
                "partition": partition, "seed": seed, "molecule_id": item["molecule_id"], "source_index": source_index,
                "source_ba_objective": objective(source, base, parameters),
                "ba_ba_objective": objective(ba["coordinates"], base, parameters),
                "bac_ba_objective": objective(bac["coordinates"], base, parameters),
                "source_violation_count": source_steric["violation_count"],
                "ba_violation_count": ba_steric["violation_count"],
                "bac_violation_count": bac_steric["violation_count"],
                "source_penetration_sum": source_steric["penetration_sum"],
                "ba_penetration_sum": ba_steric["penetration_sum"],
                "bac_penetration_sum": bac_steric["penetration_sum"],
                "source_catastrophic": source_steric["catastrophic_count"],
                "ba_catastrophic": ba_steric["catastrophic_count"],
                "bac_catastrophic": bac_steric["catastrophic_count"],
                "active_steric_count": bac["active_steric_count"], "fallback": int(bac["fallback"]),
                "rejected": int(bac["rejected"]), "trust_clipped": int(bac["trust_clipped"]),
                "backtracking_fraction": bac["backtracking_fraction"],
                "rms": float(torch.sqrt(delta.square().sum(-1).mean())),
                "atom_max": float(torch.linalg.vector_norm(delta, dim=-1).max()),
                "chirality_preserved": int(bac["safety"]["chirality_preserved"]),
                "ring_nonregression": int(bac["safety"]["ring_nonregression"]),
            })
        if (molecule_index + 1) % 12 == 0:
            print(f"STERIC {partition} seed{seed} {molecule_index + 1}/48", flush=True)
    return records


def reference_stationarity(items, calibration, model, seed, device, config):
    movements, safety = [], []
    steric = config["steric"]
    for item in deterministic(items, "dev_a", 24) + deterministic(items, "dev_b", 24):
        base = prepare_graph(item["record"], calibration).to(device)
        bat = prepare_bat_graph(base, item["record"]).to(device)
        with torch.no_grad():
            parameters = distribution_parameters(base, model=model, variant="B")
        reference = torch.as_tensor(item["references"][0], dtype=torch.float64, device=device)
        ba = frozen_ba_update(reference, bat, parameters, rms_budget=.001, atom_cap=.03)
        result = combined_gradient_update(
            reference, bat, parameters, rms_budget=.001, atom_cap=.03, tau=float(steric["tau_angstrom"]),
            fallback_coordinates=ba["coordinates"], backtracking_fractions=steric["backtracking_fractions"],
        )
        delta = result["coordinates"] - reference
        movements.append(float(torch.sqrt(delta.square().sum(-1).mean())))
        safety.append(result["safety"])
    return {
        "seed": seed, "records": len(movements), "rms_median": float(np.median(movements)),
        "rms_p95": float(np.quantile(movements, .95)),
        "chirality_fraction": float(np.mean([row["chirality_preserved"] for row in safety])),
        "ring_nonregression_fraction": float(np.mean([row["ring_nonregression"] for row in safety])),
    }


def summarize(frame, partition):
    rows = frame[frame.partition == partition]
    ba_pen = float(rows.ba_penetration_sum.sum())
    bac_pen = float(rows.bac_penetration_sum.sum())
    return {
        "partition": partition, "records": len(rows),
        "active_fraction": float((rows.active_steric_count > 0).mean()),
        "ba_violation_total": int(rows.ba_violation_count.sum()), "bac_violation_total": int(rows.bac_violation_count.sum()),
        "violation_reduction_fraction": float((rows.ba_violation_count.sum() - rows.bac_violation_count.sum()) / max(rows.ba_violation_count.sum(), 1)),
        "ba_penetration_sum": ba_pen, "bac_penetration_sum": bac_pen,
        "penetration_reduction_fraction": float((ba_pen - bac_pen) / max(ba_pen, 1e-12)),
        "new_catastrophic": int(((rows.bac_catastrophic > rows.ba_catastrophic)).sum()),
        "ba_objective_delta_vs_source_mean": float((rows.ba_ba_objective - rows.source_ba_objective).mean()),
        "bac_objective_delta_vs_source_mean": float((rows.bac_ba_objective - rows.source_ba_objective).mean()),
        "bac_minus_ba_objective_mean": float((rows.bac_ba_objective - rows.ba_ba_objective).mean()),
        "rms_mean": float(rows.rms.mean()), "rms_p95": float(rows.rms.quantile(.95)),
        "fallback_fraction": float(rows.fallback.mean()), "reject_fraction": float(rows.rejected.mean()),
        "chirality_fraction": float(rows.chirality_preserved.mean()), "ring_nonregression_fraction": float(rows.ring_nonregression.mean()),
    }


def main() -> int:
    started = time.time()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    identity = json.loads((OUT / "DATASET_IDENTITY.json").read_text(encoding="utf-8"))
    if identity["formal_test_records_read"] or identity["frozen_holdout_records_read"]:
        raise RuntimeError("protected split access")
    dataset_path = Path(config["dataset"]["training_compact"])
    if file_sha256(dataset_path) != config["dataset"]["training_compact_sha256"]:
        raise RuntimeError("training dataset SHA mismatch")
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    calibration = json.loads(Path(config["dataset"]["drcsr_calibration"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(config["ba_anchor"]["manifest"]).read_text(encoding="utf-8"))
    lookup = {(row["variant"], int(row["seed"])): row for row in manifest["checkpoints"]}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    records, stations = [], []
    for seed in config["ba_seeds"]:
        model = load_model(lookup[("B", int(seed))], device)
        for partition in ("dev_a", "dev_b"):
            records.extend(evaluate_partition(payload["items"], calibration, model, int(seed), partition, device, config))
        stations.append(reference_stationarity(payload["items"], calibration, model, int(seed), device, config))
    frame = pd.DataFrame(records)
    path = OUT / "tables/BA_C_INTERNAL.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp"); frame.to_csv(temporary, index=False); os.replace(temporary, path)
    summaries = [summarize(frame, partition) for partition in ("dev_a", "dev_b")]
    gate = config["internal_gates"]
    checks = {
        "dev_a_penetration_improves": summaries[0]["penetration_reduction_fraction"] >= gate["steric_violation_reduction_min"],
        "dev_b_penetration_improves": summaries[1]["penetration_reduction_fraction"] >= gate["steric_violation_reduction_min"],
        "no_new_catastrophic": all(row["new_catastrophic"] <= gate["newly_created_catastrophic_max"] for row in summaries),
        "ba_objective_improves_from_source": all(row["bac_objective_delta_vs_source_mean"] <= gate["ba_objective_mean_regression_max"] for row in summaries),
        "reference_stationary": all(row["rms_median"] <= gate["reference_movement_median_max_angstrom"] + 1e-12 for row in stations),
        "topology_chirality_safe": all(row["chirality_fraction"] == 1.0 and row["ring_nonregression_fraction"] == 1.0 for row in summaries + stations),
        "rms_trust": all(row["rms_p95"] <= config["ba_anchor"]["budget_angstrom"] + 1e-12 for row in summaries),
        "fallback_reasonable": all(row["fallback_fraction"] <= gate["fallback_max"] for row in summaries),
    }
    decision = "STERIC_INTERNAL_GO" if all(checks.values()) else "STERIC_NO_GO"
    result = {
        "schema_version": "mcvr-bat-steric-internal-v1", "decision": decision, "checks": checks,
        "summaries": summaries, "reference_stationarity": stations, "runtime_seconds": time.time() - started,
        "config_sha256": file_sha256(CONFIG_PATH), "ba_manifest_sha256": file_sha256(Path(config["ba_anchor"]["manifest"])),
        "records_sha256": file_sha256(path), **ISOLATION,
    }
    atomic_json(OUT / "manifests/STERIC_INTERNAL.json", result)
    table = "\n".join(
        f"| {row['partition']} | {row['records']} | {row['active_fraction']:.4f} | {row['ba_violation_total']} | {row['bac_violation_total']} | {row['penetration_reduction_fraction']:.4%} | {row['new_catastrophic']} | {row['bac_objective_delta_vs_source_mean']:.6g} | {row['rms_mean']:.6g} | {row['fallback_fraction']:.3%} |"
        for row in summaries
    )
    atomic_text(OUT / "STERIC_INTERNAL.md", f"""# BA+C internal steric gate

Decision: **{decision}**

| partition | records | active | BA violations | BA+C violations | penetration reduction | new catastrophic | BA+C objective Δ vs Source | RMS | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

The gate is based on penetration magnitude because a 0.003 Å micro-step can reduce overlap continuously without necessarily crossing the binary boundary in one step. Binary violation totals remain fully reported.

Checks: `{json.dumps(checks, sort_keys=True)}`

No PB, xTB, MVT, formal test, or frozen holdout was accessed.
""")
    atomic_text(OUT / "BA_REPLICATION_INTERNAL.md", f"# Frozen BA replication internal\n\nThree frozen B checkpoints were SHA-verified and evaluated without retraining. Exact BA inheritance is covered by the regression suite. BA objective deltas versus Source are `{[row['ba_objective_delta_vs_source_mean'] for row in summaries]}` for DEV_A/DEV_B.\n")
    atomic_text(OUT / "REFERENCE_STATIONARITY.md", "# Reference stationarity\n\n" + "\n".join(f"- seed {row['seed']}: median RMS `{row['rms_median']:.6g}` Å; p95 `{row['rms_p95']:.6g}` Å; chirality/ring `{row['chirality_fraction']:.3f}`/`{row['ring_nonregression_fraction']:.3f}`" for row in stations))
    print(decision)
    return 0 if decision == "STERIC_INTERNAL_GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
