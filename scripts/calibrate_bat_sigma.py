#!/usr/bin/env python3
"""Audit frozen BA scales against neural-mean Reference residuals."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import yaml

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
from etflow.ecir.learned_geometry import LearnedGeometryObjective, geometry_values, prepare_graph
from etflow.ecir.lsgo_io import atomic_json, file_sha256

OUT = ROOT / "reports/ecir_mvr/bat_refinement"
CONFIG_PATH = ROOT / "configs/ecir_mvr_bat_refinement.yaml"


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8"); os.replace(temporary, path)


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    absolute = np.abs(values)
    return {
        "count": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
        "p50_abs": float(np.quantile(absolute, .50)), "p90_abs": float(np.quantile(absolute, .90)),
        "p95_abs": float(np.quantile(absolute, .95)), "p99_abs": float(np.quantile(absolute, .99)),
        "coverage_abs_lt_1": float((absolute < 1).mean()), "coverage_abs_lt_2": float((absolute < 2).mean()),
        "robust_scale": float(1.4826 * np.median(np.abs(values - np.median(values)))),
    }


@torch.no_grad()
def collect(model, items, calibration, partition: str, device: torch.device) -> dict[str, np.ndarray]:
    bond_z, angle_z = [], []
    for index, item in enumerate(item for item in items if item["partition"] == partition):
        graph = prepare_graph(item["record"], calibration).to(device)
        prediction = model(graph)
        bond_mu, angle_mu = prediction["bond_mu"].double(), prediction["angle_mu"].double()
        bond_sigma, angle_sigma = graph.bond_fixed[:, 1], graph.angle_fixed[:, 1]
        for reference in torch.as_tensor(item["references"], dtype=torch.float64, device=device):
            bonds, angles = geometry_values(reference, graph)
            bond_z.append(((bonds - bond_mu) / bond_sigma).cpu())
            angle_z.append(((angles - angle_mu) / angle_sigma).cpu())
        if (index + 1) % 250 == 0:
            print(f"SIGMA {partition} {index + 1}", flush=True)
    return {"bond": torch.cat(bond_z).numpy(), "angle": torch.cat(angle_z).numpy()}


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    identity = json.loads((OUT / "DATASET_IDENTITY.json").read_text(encoding="utf-8"))
    if int(identity["formal_test_records_read"]) or int(identity["frozen_holdout_records_read"]):
        raise RuntimeError("protected split access")
    training_path = Path(config["dataset"]["training_compact"])
    training = torch.load(training_path, map_location="cpu", weights_only=False)
    calibration = json.loads(Path(config["dataset"]["drcsr_calibration"]).read_text(encoding="utf-8"))
    manifest_path = Path(config["ba_anchor"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lookup = {(row["variant"], int(row["seed"])): row for row in manifest["checkpoints"]}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = []
    for seed in config["ba_seeds"]:
        row = lookup[("B", int(seed))]; checkpoint_path = Path(row["path"])
        if file_sha256(checkpoint_path) != row["sha256"]:
            raise RuntimeError(f"BA checkpoint SHA changed: {seed}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = LearnedGeometryObjective(hidden_dim=128, layers=3, learned_sigma=False).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True); model.eval()
        partition_values = {partition: collect(model, training["items"], calibration, partition, device) for partition in ("train", "dev_a", "dev_b")}
        train_gamma = {group: max(stats(partition_values["train"][group])["robust_scale"], 1e-6) for group in ("bond", "angle")}
        for partition in ("train", "dev_a", "dev_b"):
            for group in ("bond", "angle"):
                raw = stats(partition_values[partition][group])
                calibrated = stats(partition_values[partition][group] / train_gamma[group])
                results.append({
                    "seed": int(seed), "partition": partition, "group": group,
                    "frozen_drcsr_scale_z": raw, "train_residual_gamma": train_gamma[group],
                    "posthoc_train_gamma_z": calibrated,
                })
    payload = {
        "schema_version": "mcvr-bat-sigma-posthoc-v1", "status": "COMPLETED",
        "primary_coordinate_scale": "exact frozen DRCSR/reference scale inherited from LSGO-B",
        "posthoc_residual_gamma_role": "calibration audit only; does not alter BA/BA+C/BAT+C coordinate objective",
        "results": results, "ba_checkpoint_manifest_sha256": file_sha256(manifest_path),
        "dataset_identity_sha256": identity["identity_sha256"],
        "learned_sigma": False, "sigma_inflation_possible": False,
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0, "full10k_used_for_tuning": False,
    }
    atomic_json(OUT / "manifests/SIGMA_POSTHOC_CALIBRATION.json", payload)
    lines = [
        "# Post-hoc BA scale calibration", "",
        "The exact historical BA anchor uses the frozen typed DRCSR/reference scales. Changing them would invalidate the incremental BA→BA+C comparison. This audit measures neural-mean residual calibration but does not modify coordinate inference.", "",
        "The neural network does not output sigma. Consequently sigma inflation cannot occur. `train_residual_gamma` is the robust MAD scale of `(q_ref-mu_neural)/sigma_frozen`; values different from one quantify calibration mismatch rather than a trainable uncertainty.", "",
        "| seed | partition | group | gamma | raw z std | calibrated z std | raw |z|<1 | calibrated |z|<1 |", "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['seed']} | {row['partition']} | {row['group']} | {row['train_residual_gamma']:.4f} | "
            f"{row['frozen_drcsr_scale_z']['std']:.4f} | {row['posthoc_train_gamma_z']['std']:.4f} | "
            f"{row['frozen_drcsr_scale_z']['coverage_abs_lt_1']:.3f} | {row['posthoc_train_gamma_z']['coverage_abs_lt_1']:.3f} |"
        )
    lines += ["", "No MVT coordinate, Cartesian delta, PB, xTB, formal test or frozen holdout was accessed."]
    atomic_text(OUT / "SIGMA_POSTHOC_CALIBRATION.md", "\n".join(lines))
    print("BAT_SIGMA_POSTHOC_CALIBRATION_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
