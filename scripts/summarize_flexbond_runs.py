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
    "best_checkpoint",
    "last_checkpoint",
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
    measurements = []
    for metrics in run_dir.rglob("metrics.csv"):
        with metrics.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                value = row.get("val/loss") or row.get("val/final_loss")
                if value not in (None, ""):
                    measurements.append((float(value), int(float(row.get("step") or -1))))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="+", required=True, type=Path)
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
                "best_checkpoint": best,
                "last_checkpoint": last,
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
