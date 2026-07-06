#!/usr/bin/env python
"""Summarize resolved training settings, data exposure, and checkpoints."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch
import yaml


COLUMNS = (
    "run_dir",
    "mode",
    "t_min",
    "t_max",
    "max_steps",
    "learning_rate",
    "batch_size",
    "accumulate_grad_batches",
    "effective_batch_size",
    "correction_scale",
    "q_loss_weight",
    "corr_reg_weight",
    "ridge_eps",
    "max_q_norm",
    "max_condition",
    "train_records",
    "val_records",
    "num_epochs_estimated",
    "best_val_final_loss_checkpoint",
    "last_checkpoint",
    "rollout_eval_best_checkpoint",
)


def _cache_root(value: str, run_dir: Path) -> Path:
    raw = Path(value).expanduser()
    candidates = (
        raw,
        Path(__file__).resolve().parents[1] / raw,
        run_dir / raw,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return raw.resolve()


def _selected_count(root: Path, split: str, max_molecules: int | None) -> int:
    directory = root / split if (root / split).is_dir() else root
    files = sorted(directory.glob("*.pt"))
    if max_molecules is None:
        return len(files)
    selected, count = set(), 0
    for path in files:
        record = torch.load(path, map_location="cpu", weights_only=False)
        mol_id = str(record.get("source_mol_id", record.get("mol_id", "")))
        if mol_id in selected:
            count += 1
        elif len(selected) < int(max_molecules):
            selected.add(mol_id)
            count += 1
    return count


def _recursive_best(value) -> str | None:
    if isinstance(value, dict):
        candidate = value.get("best_model_path")
        if candidate:
            return str(candidate)
        for nested in value.values():
            found = _recursive_best(nested)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found = _recursive_best(nested)
            if found:
                return found
    return None


def _checkpoint_step(path: Path) -> int:
    matches = re.findall(r"(?:step[=_-]?)?(\d+)", path.stem)
    return int(matches[-1]) if matches else -1


def _best_from_metrics(run_dir: Path, checkpoints: list[Path]) -> Path | None:
    final_measurements = []
    fallback_measurements = []
    for metrics in run_dir.rglob("metrics.csv"):
        with metrics.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                step = int(float(row.get("step") or -1))
                final_value = row.get("val/final_loss")
                fallback_value = row.get("val/loss")
                if final_value not in (None, ""):
                    final_measurements.append((float(final_value), step))
                if fallback_value not in (None, ""):
                    fallback_measurements.append((float(fallback_value), step))
    measurements = final_measurements or fallback_measurements
    if not measurements or not checkpoints:
        return None
    _, best_step = min(measurements)
    return min(checkpoints, key=lambda path: abs(_checkpoint_step(path) - best_step))


def _checkpoints(run_dir: Path) -> tuple[str, str]:
    checkpoint_dir = run_dir / "checkpoints"
    last = checkpoint_dir / "last.ckpt"
    all_checkpoints = sorted(checkpoint_dir.glob("*.ckpt"))
    non_last = [path for path in all_checkpoints if path.name != "last.ckpt"]
    best = None
    if last.is_file():
        payload = torch.load(last, map_location="cpu", weights_only=False)
        best_path = _recursive_best(payload.get("callbacks", payload))
        if best_path:
            best = Path(best_path)
            if not best.is_absolute():
                best = run_dir / best
    if best is None:
        best = _best_from_metrics(run_dir, non_last)
    if best is None and non_last:
        best = non_last[-1]
    return str(best or ""), str(last if last.is_file() else "")


def _rollout_best(
    run_dir: Path, mode: str, rollout_summaries: list[Path]
) -> str:
    expected_method = (
        "cartesian_adapter" if mode == "cartesian_optimizer" else "flexbond4d_adapter"
    )
    candidates = []
    checkpoint_names = {path.name for path in (run_dir / "checkpoints").glob("*.ckpt")}
    paths = list(run_dir.rglob("sweep_summary.csv")) + rollout_summaries
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("method") != expected_method or row.get("subset") != "all":
                    continue
                checkpoint_path = row.get("checkpoint_path")
                if checkpoint_path:
                    try:
                        if Path(checkpoint_path).expanduser().resolve().parent != (
                            run_dir / "checkpoints"
                        ).resolve():
                            continue
                    except OSError:
                        continue
                elif checkpoint_names and row.get("checkpoint_name") not in checkpoint_names:
                    continue
                candidates.append(row)
    if not candidates:
        return ""
    best = min(
        candidates,
        key=lambda row: (float(row["rmsd_mean"]), float(row["failure_rate"])),
    )
    return (
        f"{best.get('checkpoint_name', '')};step={best.get('step', '')};"
        f"alpha={best.get('update_scale', '')};rmsd_mean={best.get('rmsd_mean', '')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--rollout_summaries", nargs="*", type=Path, default=())
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for run_dir in args.run_dirs:
        run_dir = run_dir.expanduser().resolve()
        config_path = run_dir / "config.resolved.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing resolved config: {config_path}")
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        model = config.get("model", {})
        data = config.get("data", {})
        trainer = config.get("trainer", {})
        time_sampling = config.get("time_sampling", {})
        batch_size = int(data.get("batch_size", 0))
        accumulate = int(trainer.get("accumulate_grad_batches", 1))
        effective_batch = batch_size * accumulate
        cache_root = _cache_root(str(data.get("cache_dir", "")), run_dir)
        maximum = data.get("max_molecules")
        train_records = _selected_count(cache_root, "train", maximum)
        val_records = _selected_count(cache_root, "val", maximum)
        max_steps = int(trainer.get("max_steps", 0))
        best, last = _checkpoints(run_dir)
        rows.append(
            {
                "run_dir": str(run_dir),
                "mode": model.get("mode"),
                "t_min": time_sampling.get("t_min", model.get("t_min", 0.0)),
                "t_max": time_sampling.get("t_max", model.get("t_max", 1.0)),
                "max_steps": max_steps,
                "learning_rate": model.get("lr"),
                "batch_size": batch_size,
                "accumulate_grad_batches": accumulate,
                "effective_batch_size": effective_batch,
                "correction_scale": model.get("correction_scale"),
                "q_loss_weight": model.get("q_loss_weight"),
                "corr_reg_weight": model.get("corr_reg_weight"),
                "ridge_eps": model.get("ridge_eps"),
                "max_q_norm": model.get("max_q_norm"),
                "max_condition": model.get("max_condition"),
                "train_records": train_records,
                "val_records": val_records,
                "num_epochs_estimated": (
                    max_steps * effective_batch / train_records
                    if train_records
                    else float("nan")
                ),
                "best_val_final_loss_checkpoint": best,
                "last_checkpoint": last,
                "rollout_eval_best_checkpoint": _rollout_best(
                    run_dir,
                    str(model.get("mode", "")),
                    [path.expanduser().resolve() for path in args.rollout_summaries],
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "run_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} run summaries to {output}")


if __name__ == "__main__":
    main()
