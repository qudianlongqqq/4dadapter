#!/usr/bin/env python3
"""Train and gate lightweight torsion heads on internal Reference labels only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
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
from etflow.ecir.bat_refinement import (
    TorsionHead, canonical_rotatable_torsions, dihedral_angles, mixture_responsibilities,
    prepare_bat_graph, torsion_nll, torsion_pathology,
)
from etflow.ecir.learned_geometry import LearnedGeometryObjective, prepare_graph
from etflow.ecir.lsgo_io import atomic_json, atomic_torch_save as atomic_torch, file_sha256

OUT = ROOT / "reports/ecir_mvr/bat_refinement"
CONFIG_PATH = ROOT / "configs/ecir_mvr_bat_refinement.yaml"
ISOLATION = {"formal_test_records_read": 0, "frozen_holdout_records_read": 0, "posebusters_access": False, "xtb_access": False, "mvt_used": False}


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def seed_all(seed: int) -> torch.Generator:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.Generator().manual_seed(seed + 91000)


def deterministic(items, partition: str, count: int | None = None):
    rows = [item for item in items if item["partition"] == partition]
    rows.sort(key=lambda item: hashlib.sha256(str(item["molecule_id"]).encode()).hexdigest())
    return rows if count is None else rows[:count]


def load_ba(row, device):
    path = Path(row["path"])
    if file_sha256(path) != row["sha256"]:
        raise RuntimeError("frozen BA checkpoint SHA mismatch")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = LearnedGeometryObjective(hidden_dim=128, layers=3, learned_sigma=False).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True); model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def prepare_rows(items, calibration, ba_model, device, *, smoke: bool):
    limits = {"train": 48 if smoke else 400, "dev_a": 24 if smoke else None, "dev_b": 24 if smoke else None}
    rows = []
    for partition in ("train", "dev_a", "dev_b"):
        chosen = deterministic(items, partition, limits[partition])
        for index, item in enumerate(chosen):
            base = prepare_graph(item["record"], calibration).to(device)
            torsions, _, metadata = canonical_rotatable_torsions(item["record"])
            if not torsions.numel():
                continue
            embedding = ba_model(base)["node_embedding"].detach().cpu()
            reference_angles = torch.stack([
                dihedral_angles(torch.as_tensor(reference, dtype=torch.float64), torsions)
                for reference in item["references"]
            ]).float()
            source_angles = torch.stack([
                dihedral_angles(torch.as_tensor(source, dtype=torch.float64), torsions)
                for source in item["sources"]
            ]).float()
            rows.append({
                "molecule_id": item["molecule_id"], "partition": partition,
                "embedding": embedding, "torsions": torsions, "references": reference_angles,
                "sources": source_angles, "metadata": metadata,
            })
        print(f"TORSION PREP {partition} {len(chosen)}", flush=True)
    return rows


def checkpoint_path(kind: str, seed: int, label: str) -> Path:
    return OUT / "checkpoints" / f"torsion_{kind}_seed{seed}" / f"{label}.ckpt"


def train_head(rows, *, seed: int, components: int, learned_kappa: bool, fixed_kappa: float, steps: int, config, device, label: str):
    generator = seed_all(seed)
    head = TorsionHead(hidden_dim=128, components=components, learned_kappa=learned_kappa).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(config["torsion"]["learning_rate"]), weight_decay=float(config["torsion"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    train_rows = [row for row in rows if row["partition"] == "train"]
    logs = []
    started = time.time()
    for step in range(1, steps + 1):
        indices = torch.randint(len(train_rows), (int(config["torsion"]["batch_molecules"]),), generator=generator).tolist()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for index in indices:
            row = train_rows[index]
            reference_index = int(torch.randint(row["references"].size(0), (1,), generator=generator))
            prediction = head(row["embedding"].to(device), row["torsions"].to(device), fixed_kappa=fixed_kappa)
            values = row["references"][reference_index].to(device)
            losses.append(torsion_nll(values, prediction).mean())
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite torsion loss {label}/{seed}/{step}")
        loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        optimizer.step(); scheduler.step()
        if step == 1 or step % 20 == 0 or step == steps:
            logs.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(gradient), "learning_rate": scheduler.get_last_lr()[0]})
        if step % 100 == 0 or step == steps:
            print(f"TORSION TRAIN {label} seed{seed} {step}/{steps}", flush=True)
    payload = {
        "schema_version": "mcvr-bat-torsion-checkpoint-v1", "kind": label, "seed": seed,
        "components": components, "learned_kappa": learned_kappa, "fixed_kappa": fixed_kappa,
        "steps": steps, "model_state": head.state_dict(), "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(), "parameter_count": sum(p.numel() for p in head.parameters()),
        "runtime_seconds": time.time() - started, "config_sha256": file_sha256(CONFIG_PATH), **ISOLATION,
    }
    path = checkpoint_path(label, seed, f"step{steps:04d}")
    atomic_torch(path, payload)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(logs).to_csv(OUT / "logs" / f"TORSION_{label}_seed{seed}.csv", index=False)
    return head.eval(), payload, path


@torch.no_grad()
def evaluate(head, rows, partition, fixed_kappa, device):
    primitive_nll, per_molecule, source_greater, source_active_values = [], [], [], []
    predictions_for_pathology, responsibility_values = [], []
    reference_by_primitive = []
    for row in (value for value in rows if value["partition"] == partition):
        prediction = head(row["embedding"].to(device), row["torsions"].to(device), fixed_kappa=fixed_kappa)
        local_reference = []
        for values_cpu in row["references"]:
            values = values_cpu.to(device)
            nll = torsion_nll(values, prediction)
            primitive_nll.extend(nll.cpu().tolist()); local_reference.append(float(nll.mean()))
            responsibility_values.append(mixture_responsibilities(values, prediction["logits"], prediction["means"], prediction["kappa"]).cpu())
            reference_by_primitive.extend(nll.cpu().tolist())
        local_source = []
        for values_cpu in row["sources"]:
            nll = torsion_nll(values_cpu.to(device), prediction)
            local_source.append(float(nll.mean())); source_active_values.extend(nll.cpu().tolist())
        reference_mean, source_mean = float(np.mean(local_reference)), float(np.mean(local_source))
        source_greater.append(source_mean > reference_mean)
        per_molecule.append({"molecule_id": row["molecule_id"], "reference_nll": reference_mean, "source_nll": source_mean, "rotors": row["torsions"].size(0)})
        predictions_for_pathology.append({key: value.detach().cpu() for key, value in prediction.items()})
    all_prediction = {key: torch.cat([row[key] for row in predictions_for_pathology]) for key in ("weights", "means", "kappa")}
    pathology = torsion_pathology(all_prediction)
    responsibilities = torch.cat(responsibility_values)
    occupancy = responsibilities.mean(0)
    array = np.asarray(primitive_nll)
    return {
        "partition": partition, "molecules": len(per_molecule), "primitive_count": len(primitive_nll),
        "nll_mean": float(array.mean()), "nll_median": float(np.median(array)),
        "nll_p95": float(np.quantile(array, .95)), "uniform_margin": float(math.log(2 * math.pi) - array.mean()),
        "source_gt_reference_fraction": float(np.mean(source_greater)),
        "source_nll_values": source_active_values, "reference_nll_values": reference_by_primitive,
        "posterior_occupancy": occupancy.tolist(), "posterior_occupancy_min": float(occupancy.min()),
        "pathology": pathology, "per_molecule": per_molecule,
    }


def estimate_kappa(head, rows, device):
    cosines = []
    with torch.no_grad():
        for row in (value for value in rows if value["partition"] == "train"):
            prediction = head(row["embedding"].to(device), row["torsions"].to(device), fixed_kappa=8.)
            for values_cpu in row["references"]:
                values = values_cpu.to(device)
                responsibilities = mixture_responsibilities(values, prediction["logits"], prediction["means"], prediction["kappa"])
                mode = responsibilities.argmax(-1)
                selected = prediction["means"].gather(1, mode[:, None]).squeeze(1)
                cosines.extend(torch.cos(values - selected).cpu().tolist())
    resultant = float(np.mean(cosines))
    if resultant < .53:
        kappa = 2 * resultant + resultant ** 3 + 5 * resultant ** 5 / 6
    elif resultant < .85:
        kappa = -.4 + 1.39 * resultant + .43 / (1 - resultant)
    else:
        kappa = 1 / max(resultant ** 3 - 4 * resultant ** 2 + 3 * resultant, 1e-9)
    return resultant, float(np.clip(kappa, .25, 32.))


@torch.no_grad()
def nll_at_kappa(head, rows, partition, kappa, device):
    values = []
    for row in (value for value in rows if value["partition"] == partition):
        prediction = head(row["embedding"].to(device), row["torsions"].to(device), fixed_kappa=float(kappa))
        prediction["kappa"] = torch.full_like(prediction["kappa"], float(kappa))
        for reference in row["references"]:
            values.extend(torsion_nll(reference.to(device), prediction).cpu().tolist())
    return float(np.mean(values))


def compact_evaluations(evaluations):
    if evaluations is None:
        return None
    return {
        partition: {
            key: value for key, value in metrics.items()
            if key not in {"source_nll_values", "reference_nll_values", "per_molecule"}
        }
        for partition, metrics in evaluations.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    canonical = json.loads((OUT / "manifests/TORSION_CANONICALIZATION_AUDIT.json").read_text(encoding="utf-8"))
    if canonical["decision"] != "TORSION_LABEL_GO":
        raise RuntimeError("canonicalization gate is not GO")
    identity = json.loads((OUT / "DATASET_IDENTITY.json").read_text(encoding="utf-8"))
    if identity["formal_test_records_read"] or identity["frozen_holdout_records_read"]:
        raise RuntimeError("protected split access")
    dataset_path = Path(config["dataset"]["training_compact"])
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    calibration = json.loads(Path(config["dataset"]["drcsr_calibration"]).read_text(encoding="utf-8"))
    ba_manifest = json.loads(Path(config["ba_anchor"]["manifest"]).read_text(encoding="utf-8"))
    lookup = {(row["variant"], int(row["seed"])): row for row in ba_manifest["checkpoints"]}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    steps = 20 if args.smoke else int(config["torsion"]["pilot_steps"])
    results, kappa_rows, checkpoints = [], [], []
    seed_pairs = list(zip(config["ba_seeds"], config["torsion_seeds"], strict=True))
    if args.smoke:
        seed_pairs = seed_pairs[:1]
    for ba_seed, torsion_seed in seed_pairs:
        ba_model = load_ba(lookup[("B", int(ba_seed))], device)
        rows = prepare_rows(payload["items"], calibration, ba_model, device, smoke=args.smoke)
        single, single_payload, single_path = train_head(rows, seed=int(torsion_seed), components=1, learned_kappa=False, fixed_kappa=8., steps=steps, config=config, device=device, label="single_smoke" if args.smoke else "single_pilot")
        initial, initial_payload, initial_path = train_head(rows, seed=int(torsion_seed), components=3, learned_kappa=False, fixed_kappa=8., steps=steps, config=config, device=device, label="k3_initial_smoke" if args.smoke else "k3_initial_pilot")
        resultant, estimate = estimate_kappa(initial, rows, device)
        candidates = sorted(set([.5, 1., 2., 4., 8., 16., 32., round(estimate, 6)]))
        grid = [{"kappa": value, "dev_a_nll": nll_at_kappa(initial, rows, "dev_a", value, device), "dev_b_nll": nll_at_kappa(initial, rows, "dev_b", value, device)} for value in candidates]
        selected = min(grid, key=lambda row: (row["dev_a_nll"], row["kappa"]))["kappa"]
        calibrated, calibrated_payload, calibrated_path = train_head(rows, seed=int(torsion_seed), components=3, learned_kappa=False, fixed_kappa=selected, steps=steps, config=config, device=device, label="k3_fixed_smoke" if args.smoke else "k3_fixed_pilot")
        learned_metrics = None
        if int(torsion_seed) == int(seed_pairs[0][1]):
            learned, learned_payload, learned_path = train_head(rows, seed=int(torsion_seed) + 1000, components=3, learned_kappa=True, fixed_kappa=selected, steps=steps, config=config, device=device, label="k3_learned_diagnostic_smoke" if args.smoke else "k3_learned_diagnostic_pilot")
            learned_metrics = {partition: evaluate(learned, rows, partition, selected, device) for partition in ("dev_a", "dev_b")}
            checkpoints.append((learned_path, learned_payload))
        single_metrics = {partition: evaluate(single, rows, partition, 8., device) for partition in ("dev_a", "dev_b")}
        fixed_metrics = {partition: evaluate(calibrated, rows, partition, selected, device) for partition in ("train", "dev_a", "dev_b")}
        train_threshold = float(np.quantile(fixed_metrics["train"]["reference_nll_values"], config["torsion"]["active_reference_quantile"]))
        for partition in ("dev_a", "dev_b"):
            fixed_metrics[partition]["active_threshold"] = train_threshold
            fixed_metrics[partition]["source_active_fraction"] = float(np.mean(np.asarray(fixed_metrics[partition]["source_nll_values"]) > train_threshold))
        results.append({
            "ba_seed": int(ba_seed), "torsion_seed": int(torsion_seed), "selected_kappa": selected,
            "train_resultant_length": resultant, "train_kappa_estimate": estimate, "kappa_grid": grid,
            "single": compact_evaluations(single_metrics), "fixed_k3": compact_evaluations(fixed_metrics),
            "learned_k3_diagnostic": compact_evaluations(learned_metrics),
        })
        kappa_rows.extend({"ba_seed": ba_seed, "torsion_seed": torsion_seed, **row, "selected": row["kappa"] == selected} for row in grid)
        checkpoints.extend(((single_path, single_payload), (initial_path, initial_payload), (calibrated_path, calibrated_payload)))
    if args.smoke:
        smoke = {"status": "PASS", "steps": steps, "runs": len(checkpoints), "results": results, **ISOLATION}
        atomic_json(OUT / "manifests/TORSION_SMOKE.json", smoke)
        print("TORSION_20_STEP_SMOKE_PASS")
        return 0
    gate = config["internal_gates"]
    checks = {
        "better_than_uniform": all(run["fixed_k3"][part]["uniform_margin"] >= gate["uniform_nll_margin_min"] for run in results for part in ("dev_a", "dev_b")),
        "k3_not_worse_than_single": all(run["fixed_k3"][part]["nll_mean"] <= run["single"][part]["nll_mean"] - gate["mixture_vs_single_margin_min"] for run in results for part in ("dev_a", "dev_b")),
        "no_component_collapse": all(run["fixed_k3"][part]["posterior_occupancy_min"] >= gate["component_occupancy_min"] for run in results for part in ("dev_a", "dev_b")),
        "no_mode_duplication": all(run["fixed_k3"][part]["pathology"]["mode_duplication_fraction"] <= gate["duplicated_mode_fraction_max"] for run in results for part in ("dev_a", "dev_b")),
        "source_selectivity": all(run["fixed_k3"][part]["source_gt_reference_fraction"] >= gate["source_surprise_gt_reference_fraction_min"] for run in results for part in ("dev_a", "dev_b")),
        "active_fraction_bounded": all(gate["torsion_active_fraction_min"] <= run["fixed_k3"][part]["source_active_fraction"] <= gate["torsion_active_fraction_max"] for run in results for part in ("dev_a", "dev_b")),
    }
    decision = "TORSION_INTERNAL_GO" if all(checks.values()) else "TORSION_NO_GO"
    checkpoint_manifest = [{"path": str(path), "sha256": file_sha256(path), "kind": data["kind"], "seed": data["seed"], "steps": data["steps"], "parameter_count": data["parameter_count"]} for path, data in checkpoints]
    result = {"schema_version": "mcvr-bat-torsion-internal-v1", "decision": decision, "checks": checks, "results": results, "checkpoints": checkpoint_manifest, **ISOLATION}
    atomic_json(OUT / "manifests/TORSION_INTERNAL.json", result)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(kappa_rows).to_csv(OUT / "tables/TORSION_KAPPA_GRID.csv", index=False)
    summary_rows = []
    for run in results:
        for partition in ("dev_a", "dev_b"):
            fixed, single = run["fixed_k3"][partition], run["single"][partition]
            summary_rows.append({"seed": run["torsion_seed"], "partition": partition, "kappa": run["selected_kappa"], "uniform_nll": math.log(2 * math.pi), "single_nll": single["nll_mean"], "k3_nll": fixed["nll_mean"], "source_gt_reference": fixed["source_gt_reference_fraction"], "source_active": fixed["source_active_fraction"], "occupancy_min": fixed["posterior_occupancy_min"], "duplication": fixed["pathology"]["mode_duplication_fraction"]})
    pd.DataFrame(summary_rows).to_csv(OUT / "BAT_INTERNAL_ABLATION.csv", index=False)
    table = "\n".join(f"| {row['seed']} | {row['partition']} | {row['kappa']:.4g} | {row['uniform_nll']:.4f} | {row['single_nll']:.4f} | {row['k3_nll']:.4f} | {row['source_gt_reference']:.3f} | {row['source_active']:.3f} | {row['occupancy_min']:.3f} | {row['duplication']:.3f} |" for row in summary_rows)
    atomic_text(OUT / "TORSION_CALIBRATION.md", f"""# Torsion internal calibration

Decision: **{decision}**

| seed | partition | fixed κ | uniform NLL | single VM | K3 VM | Source>Reference | active | min occupancy | duplicated |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

Checks: `{json.dumps(checks, sort_keys=True)}`

All κ estimation and selection used TRAIN residuals plus DEV_A/DEV_B likelihood only. External metrics were locked.
""")
    atomic_text(OUT / "TORSION_KAPPA_AUDIT.md", "# Torsion κ audit\n\n" + "\n".join(f"- seed {run['torsion_seed']}: TRAIN resultant `{run['train_resultant_length']:.5f}`, residual estimate `{run['train_kappa_estimate']:.5f}`, DEV_A-selected fixed κ `{run['selected_kappa']}`." for run in results) + "\n\nκ=8 was used only to learn initial centers for the post-hoc residual audit. The selected fixed κ was then used in a fresh head training run. Learned κ remained diagnostic only.")
    atomic_text(OUT / "TORSION_MODEL_AUDIT.md", f"# Torsion model audit\n\nThe primary model is a 3-component von Mises mixture head over frozen BA node embeddings. It learns π and circular μ only; κ is fixed after internal calibration. Maximum primary parameter count: `{max(row['parameter_count'] for row in checkpoint_manifest if 'k3_fixed' in row['kind'])}` (<0.5M). BA is never fine-tuned. Decision: **{decision}**.\n")
    atomic_text(OUT / "TORSION_SOURCE_SELECTIVITY.md", "# Torsion Source selectivity\n\n" + "\n".join(f"- seed {row['seed']} {row['partition']}: Source>Reference molecule fraction `{row['source_gt_reference']:.3f}`; tail-active primitive fraction `{row['source_active']:.3f}`." for row in summary_rows))
    print(decision)
    return 0 if decision == "TORSION_INTERNAL_GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
