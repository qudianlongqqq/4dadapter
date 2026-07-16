#!/usr/bin/env python
"""Train the one authorized conservative-torsion MCVR Run B pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.loader import DataLoader

from etflow.commons.global_coupled_4d_sampling import atomic_json_save, atomic_torch_save
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.mvr_loss import MCVRLoss
from etflow.ecir.mvr_model import MCVRModel
from etflow.ecir.run_a_evaluation import build_clean_control_items, build_items
from etflow.ecir.run_b_evaluation import evaluate_three_way
from scripts.train_ecir_mvr_run_a import _assert_identity, _dataset, _git, _seed, _sha


LOSS_NAMES = (
    "flow_loss", "validity_mode_loss", "identity_loss", "anchor_loss", "sparse_loss",
    "torsion_mode_loss", "torsion_anchor_loss", "torsion_gate_sparsity_loss",
    "high_flex_torsion_trust_loss", "error_loss", "uncertainty_loss",
    "trust_loss", "total_loss",
)
DIAGNOSTIC_NAMES = (
    "rigid_gate_mean", "global_safety_gate_mean", "uncertainty_mean",
    "velocity_norm_mean", "molecule_displacement_mean", "max_atom_displacement_mean",
    "identity_subset_displacement", "torsion_gate_mean",
    "torsion_gate_active_fraction", "torsion_velocity_norm",
    "torsion_velocity_fraction", "mean_torsion_change", "p95_torsion_change",
    "high_flex_mean_torsion_change", "high_flex_p95_torsion_change",
)
METRIC_FIELDS = ("step", "split", *LOSS_NAMES, *DIAGNOSTIC_NAMES, "records_per_second")


def _loss_value(values, name):
    return values["loss" if name == "total_loss" else name]


def _assert_run_a(config):
    checkpoint = Path(config["run_a_checkpoint"])
    if _sha(checkpoint) != config["run_a_checkpoint_sha256"]:
        raise RuntimeError("Run A selected checkpoint identity changed")
    result = json.loads(Path(
        "diagnostics/ecir_mvr/stage2b/run_a/result.json"
    ).read_text(encoding="utf-8"))
    if result["decision"] != "RUN_A_PASS" or result["test_records_read"] != 0:
        raise RuntimeError("Run A is no longer a test-free PASS")
    if result["20k_permitted"] or result["100k_permitted"]:
        raise RuntimeError("Run A long-run permissions unexpectedly changed")


@torch.inference_mode()
def _diagnostics(model, batch, step_size=0.25):
    model.eval()
    graphs = int(batch.num_graphs)
    output = model(batch, batch.x_input, batch.x_input.new_full((graphs,), 0.5))
    atom_batch = batch.batch
    displacement = float(step_size) * output["v_final"]
    norms = torch.linalg.vector_norm(displacement, dim=-1)
    energy = displacement.new_zeros(graphs)
    energy.index_add_(0, atom_batch, displacement.square().sum(-1))
    counts = torch.bincount(atom_batch, minlength=graphs).clamp_min(1).to(displacement.dtype)
    graph_rms = torch.sqrt(energy / counts + 1e-12)
    active = batch.active_mode_mask.reshape(graphs, 6)
    clean = active[:, 5] > 0
    identity = norms[clean[atom_batch]].mean() if bool(clean.any()) else norms.new_zeros(())
    torsion_norms = torch.linalg.vector_norm(output["v_torsion_contribution"], dim=-1)
    rigid_norms = torch.linalg.vector_norm(output["v_rigid_contribution"], dim=-1)
    fraction = torsion_norms / (torsion_norms + rigid_norms).clamp_min(1e-12)
    high = batch.num_rotatable_bonds.reshape(graphs) >= 6
    high_atoms = high[atom_batch]
    torsion_change_proxy = float(step_size) * torsion_norms
    model.train()
    return {
        "rigid_gate_mean": float(output["rigid_gate"].mean()),
        "global_safety_gate_mean": float(output["global_safety_gate"].mean()),
        "uncertainty_mean": float(output["uncertainty"].mean()),
        "velocity_norm_mean": float(torch.linalg.vector_norm(output["v_final"], dim=-1).mean()),
        "molecule_displacement_mean": float(graph_rms.mean()),
        "max_atom_displacement_mean": float(norms.max()),
        "identity_subset_displacement": float(identity),
        "torsion_gate_mean": float(output["torsion_gate"].mean()),
        "torsion_gate_active_fraction": float(output["torsion_gate_active"].mean()),
        "torsion_velocity_norm": float(torsion_norms.mean()),
        "torsion_velocity_fraction": float(fraction.mean()),
        "mean_torsion_change": float(torsion_change_proxy.mean()),
        "p95_torsion_change": float(torch.quantile(torsion_change_proxy, 0.95)),
        "high_flex_mean_torsion_change": float(torsion_change_proxy[high_atoms].mean()) if bool(high_atoms.any()) else 0.0,
        "high_flex_p95_torsion_change": float(torch.quantile(torsion_change_proxy[high_atoms], 0.95)) if bool(high_atoms.any()) else 0.0,
    }


@torch.inference_mode()
def _validate_losses(model, loss_fn, loader, device):
    model.eval(); losses, diagnostics = [], []
    for batch in loader:
        batch = batch.to(device)
        values = loss_fn(model, batch)
        losses.append({name: float(_loss_value(values, name)) for name in LOSS_NAMES})
        diagnostics.append(_diagnostics(model, batch))
    model.train()
    result = {name: float(np.mean([row[name] for row in losses])) for name in LOSS_NAMES}
    result.update({name: float(np.mean([row[name] for row in diagnostics])) for name in DIAGNOSTIC_NAMES})
    return result


def _payload(model, optimizer, step, resolved, validation=None):
    return {
        "schema_version": "ecir-mvr-run-b-v1", "model_type": "MCVRModel",
        "run_mode": "rigid_plus_conservative_torsion", "step": int(step),
        "config": resolved, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "validation": validation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_audit", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["experiment_name"] != "ecir_mvr_stage2b_run_b_conservative_torsion_seed42_5k":
        raise RuntimeError("only frozen Run B is authorized")
    if int(config["training"]["optimizer_steps"]) != 5000:
        raise RuntimeError("Run B must use exactly 5000 optimizer steps")
    mode = config["run_b_mode"]
    if not mode["enable_torsion_repair"] or mode["torsion_gate_fixed_zero"]:
        raise RuntimeError("Run B conservative torsion is not enabled")
    audit = _assert_identity(config, args.data_audit)
    _assert_run_a(config)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Run B requires CUDA")
    _seed(int(config["seed"]))

    output = Path(config["output_dir"]); checkpoints = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True); checkpoints.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = Path(config["diagnostics_dir"]); diagnostic_dir.mkdir(parents=True, exist_ok=True)
    config_sha = _sha(args.config); git_commit = _git("rev-parse", "HEAD")
    resolved = {**config, "resolved": {
        "config_sha256": config_sha, "git_commit": git_commit, "device": str(device),
        "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
    }}
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    metadata = {
        "status": "RUNNING", "experiment_name": config["experiment_name"],
        "seed": 42, "optimizer_steps": 5000, "git_commit": git_commit,
        "config_sha256": config_sha, "data_audit_identity": audit["identity_sha256"],
        "run_a_checkpoint_sha256": config["run_a_checkpoint_sha256"],
        "frozen_identities": config["frozen_identities"],
        "gpu": torch.cuda.get_device_name(0), "cuda": str(torch.version.cuda),
        "torch": str(torch.__version__), "python": platform.python_version(),
        "test_records_read": 0, "run_c_started": False,
        "20k_started": False, "100k_started": False, "started_at_unix": time.time(),
    }
    atomic_json_save(metadata, output / "run_metadata.json")
    log_handle = (output / "training.log").open("w", encoding="utf-8", buffering=1)
    def log(message):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True); log_handle.write(line + "\n")
    log(json.dumps({"audit": audit["identity_sha256"], "config_sha256": config_sha,
                    "run_a_checkpoint": config["run_a_checkpoint_sha256"],
                    "gpu": metadata["gpu"], "seed": 42}, sort_keys=True))

    validity = ChemicalValidity(config["data"]["validity_statistics"])
    train_data = _dataset(config, "train", validity); val_data = _dataset(config, "val", validity)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    val_items = build_items(config["data"]["val_sources"], config["data"]["val_targets"], validity)
    clean_items = build_clean_control_items(val_items, validity, limit=20)
    if len(clean_items) != 20:
        raise RuntimeError("Run B requires 20 clean validation controls")

    run_a_payload = torch.load(config["run_a_checkpoint"], map_location=device, weights_only=False)
    run_a_model = MCVRModel(**run_a_payload["config"]["model"]).to(device)
    run_a_model.load_state_dict(run_a_payload["model_state_dict"], strict=True); run_a_model.eval()
    model = MCVRModel(**config["model"]).to(device)
    loss_fn = MCVRLoss(config["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.000001)

    metrics_path = output / "metrics.csv"; metrics_handle = metrics_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(metrics_handle, fieldnames=METRIC_FIELDS); writer.writeheader()
    iterator = iter(train_loader); epoch = 0; seen = 0; started = time.perf_counter()
    window = []; diagnostics_history = []; validation_history = []; comparison = []
    stop_reason = None; best = None
    for step in range(1, 5001):
        if step % 50 == 1:
            _assert_identity(config, args.data_audit); _assert_run_a(config)
        try: batch = next(iterator)
        except StopIteration:
            epoch += 1; train_data.set_epoch(epoch); iterator = iter(train_loader); batch = next(iterator)
        batch = batch.to(device); optimizer.zero_grad(set_to_none=True)
        losses = loss_fn(model, batch)
        if not all(bool(torch.isfinite(value)) for value in losses.values()):
            stop_reason = "nan_or_inf"; break
        losses["loss"].backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(grad)) or float(grad) > 100.0:
            stop_reason = "gradient_explosion"; break
        optimizer.step(); seen += int(batch.num_graphs)
        window.append({name: float(_loss_value(losses, name).detach()) for name in LOSS_NAMES})
        if step == 1 or step % 50 == 0:
            diag = _diagnostics(model, batch)
            row = {"step": step, "split": "train",
                   **{name: float(np.mean([x[name] for x in window])) for name in LOSS_NAMES},
                   **diag, "records_per_second": seen / max(time.perf_counter() - started, 1e-9)}
            writer.writerow(row); metrics_handle.flush(); window.clear(); diagnostics_history.append(row)
            log(f"step={step} loss={row['total_loss']:.6f} torsion_gate={row['torsion_gate_mean']:.5f} "
                f"active={row['torsion_gate_active_fraction']:.4f} torsion_fraction={row['torsion_velocity_fraction']:.4f}")
            if row["torsion_velocity_fraction"] > 0.5:
                stop_reason = "torsion_velocity_exceeds_rigid"
            if row["torsion_gate_mean"] > 0.95:
                stop_reason = "torsion_gate_near_all_one"
            if stop_reason: break

        if step % 1000 == 0:
            _assert_identity(config, args.data_audit); _assert_run_a(config)
            val_losses = _validate_losses(model, loss_fn, val_loader, device)
            writer.writerow({"step": step, "split": "val", **val_losses, "records_per_second": ""}); metrics_handle.flush()
            full = evaluate_three_way(
                model, run_a_model, val_items, validity, device=device,
                inference=config["inference"], upstream_margins=config["noninferiority"],
                incremental_margins=config["incremental_noninferiority"],
                bootstrap_draws=500, clean_control_items=clean_items,
            )
            full["summary"].to_csv(output / f"validation_step{step:06d}.csv", index=False)
            validation = {
                "step": step, "losses": val_losses,
                "upstream_accuracy_gate": full["upstream_accuracy_gate"],
                "incremental_accuracy_gate": full["incremental_accuracy_gate"],
                "upstream_accuracy_noninferior": full["upstream_accuracy_noninferior"],
                "incremental_accuracy_noninferior": full["incremental_accuracy_noninferior"],
                "total_validity_delta_vs_run_a": full["total_validity_delta_vs_run_a"],
                "torsion_delta_vs_run_a": full["torsion_delta_vs_run_a"],
                "high_flex_total_delta_vs_run_a": full["high_flex_total_delta_vs_run_a"],
                "clean_identity_fraction": full["clean_identity_fraction"],
                "acceptance_fraction": full["acceptance_fraction"],
                "mean_displacement": full["mean_displacement"],
                "torsion_gate_mean": full["torsion_gate_mean"],
                "torsion_gate_active_fraction": full["torsion_gate_active_fraction"],
                "torsion_velocity_fraction": full["torsion_velocity_fraction"],
                "high_flex_mean_torsion_change": full["high_flex_mean_torsion_change"],
                "high_flex_p95_torsion_change": full["high_flex_p95_torsion_change"],
                "bootstrap_vs_run_a": full["bootstrap_vs_run_a"],
                "bootstrap_high_flex_vs_run_a": full["bootstrap_high_flex_vs_run_a"],
            }
            validation_history.append(validation)
            payload = _payload(model, optimizer, step, resolved, validation)
            path = checkpoints / f"step{step:06d}.ckpt"; atomic_torch_save(payload, path)
            both = full["upstream_accuracy_noninferior"] and full["incremental_accuracy_noninferior"]
            score = (
                full["torsion_delta_vs_run_a"] + full["high_flex_total_delta_vs_run_a"],
                full["total_validity_delta_vs_run_a"], full["mean_displacement"],
            )
            if both and (best is None or score < best["score"]):
                best = {"step": step, "score": score}; atomic_torch_save(payload, checkpoints / "best_noninferior_incremental.ckpt")
            comparison.append({
                "step": step, "checkpoint": str(path.resolve()),
                "upstream_accuracy_noninferior": full["upstream_accuracy_noninferior"],
                "incremental_accuracy_noninferior": full["incremental_accuracy_noninferior"],
                "total_validity_delta_vs_run_a": full["total_validity_delta_vs_run_a"],
                "torsion_delta_vs_run_a": full["torsion_delta_vs_run_a"],
                "high_flex_total_delta_vs_run_a": full["high_flex_total_delta_vs_run_a"],
                "clean_identity_fraction": full["clean_identity_fraction"],
                "acceptance_fraction": full["acceptance_fraction"],
                "torsion_gate_active_fraction": full["torsion_gate_active_fraction"],
                "high_flex_mean_torsion_change": full["high_flex_mean_torsion_change"],
                "high_flex_p95_torsion_change": full["high_flex_p95_torsion_change"],
            })
            pd.DataFrame(comparison).to_csv(diagnostic_dir / "checkpoint_comparison.csv", index=False)
            log(f"validation step={step} upstream_accuracy={full['upstream_accuracy_noninferior']} "
                f"incremental_accuracy={full['incremental_accuracy_noninferior']} "
                f"validity_vs_a={full['total_validity_delta_vs_run_a']:.6f} "
                f"torsion_vs_a={full['torsion_delta_vs_run_a']:.6f}")
            if full["clean_identity_fraction"] < 1.0:
                stop_reason = "clean_identity_broken"
            if full["acceptance_fraction"] < 0.10:
                stop_reason = "acceptance_collapsed"
            if full["high_flex_mean_torsion_change"] > 0.010 or full["high_flex_p95_torsion_change"] > 0.030:
                stop_reason = "high_flex_torsion_limit_exceeded"
            if len(validation_history) >= 2:
                recent = validation_history[-2:]
                if all(not value["incremental_accuracy_noninferior"] for value in recent):
                    stop_reason = "two_incremental_accuracy_failures"
                if all(value["torsion_gate_active_fraction"] == 0.0 for value in recent):
                    stop_reason = "torsion_gate_all_zero_two_validations"
                if (recent[-1]["total_validity_delta_vs_run_a"] >= 0 and
                    recent[-1]["mean_displacement"] > recent[-2]["mean_displacement"]):
                    stop_reason = "no_incremental_validity_with_displacement_growth"
            if stop_reason: log(f"EARLY_STOP {stop_reason} step={step}"); break

    final_step = step
    atomic_torch_save(_payload(model, optimizer, final_step, resolved,
                              validation_history[-1] if validation_history else None), checkpoints / "last.ckpt")
    metrics_handle.close(); frame = pd.read_csv(metrics_path)
    loss_summary = frame.groupby("split", as_index=False)[list(LOSS_NAMES)].agg(["first", "last", "min"])
    loss_summary.columns = ["_".join(part for part in column if part) for column in loss_summary.columns]
    loss_summary.to_csv(diagnostic_dir / "loss_summary.csv", index=False)
    elapsed = time.perf_counter() - started
    metadata.update({
        "status": "COMPLETED" if stop_reason is None else "EARLY_STOPPED",
        "completed_steps": final_step, "elapsed_seconds": elapsed,
        "stop_reason": stop_reason, "best_noninferior_incremental_step": best["step"] if best else None,
        "completed_at_unix": time.time(),
    })
    atomic_json_save(metadata, output / "run_metadata.json")
    log(f"finished steps={final_step} elapsed={elapsed:.3f} stop_reason={stop_reason}")
    log_handle.close(); print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
