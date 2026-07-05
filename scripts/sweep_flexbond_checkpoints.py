#!/usr/bin/env python
"""Sweep adapter checkpoints and inference update scales on one fair cohort."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import torch


def _alpha_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _checkpoint_step(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and checkpoint.get("global_step") is not None:
        return int(checkpoint["global_step"])
    matches = re.findall(r"(?:step[=_-]?)?(\d+)", path.stem)
    return int(matches[-1]) if matches else -1


def _discover_config(checkpoint: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Config does not exist: {explicit}")
        return explicit.resolve()
    candidates = (
        checkpoint.parent.parent / "config.resolved.yaml",
        checkpoint.parent / "config.resolved.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not discover config.resolved.yaml for checkpoint {checkpoint}"
    )


def _run(command: list[str], dry_run: bool) -> None:
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--inference_cache", required=True)
    parser.add_argument("--reference_cache", required=True)
    parser.add_argument("--cartesian_checkpoints", nargs="*", type=Path, default=())
    parser.add_argument("--flexbond_checkpoints", nargs="*", type=Path, default=())
    parser.add_argument("--cartesian_config", type=Path)
    parser.add_argument("--flexbond_config", type=Path)
    parser.add_argument(
        "--update_scales", nargs="+", type=float, default=(0.1, 0.2, 0.5, 1.0)
    )
    parser.add_argument("--max_displacement", type=float)
    parser.add_argument("--refinement_steps", type=int, default=10)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if not args.cartesian_checkpoints and not args.flexbond_checkpoints:
        raise ValueError("At least one checkpoint list must be non-empty.")
    if any(scale < 0 for scale in args.update_scales):
        raise ValueError("update_scales must be non-negative.")

    root = Path(__file__).resolve().parents[1]
    sample_script = root / "scripts/sample_flexbond_optimizer.py"
    eval_script = root / "scripts/eval_flexbond_optimizer.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    jobs = (
        [
            ("cartesian_adapter", path, args.cartesian_config)
            for path in args.cartesian_checkpoints
        ]
        + [
            ("flexbond4d_adapter", path, args.flexbond_config)
            for path in args.flexbond_checkpoints
        ]
    )

    for method, checkpoint, explicit_config in jobs:
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        config = _discover_config(checkpoint, explicit_config)
        step = _checkpoint_step(checkpoint)
        for update_scale in args.update_scales:
            tag = _alpha_tag(update_scale)
            run_name = f"{method}_step{step}_alpha{tag}"
            run_dir = args.output_dir / run_name
            sample_path = run_dir / f"samples_alpha{tag}.pt"
            evaluation_dir = run_dir / "evaluation"
            if not (args.skip_existing and sample_path.is_file()):
                sample_command = [
                    sys.executable,
                    str(sample_script),
                    "--checkpoint",
                    str(checkpoint),
                    "--config",
                    str(config),
                    "--cache_dir",
                    args.inference_cache,
                    "--manifest",
                    str(args.manifest),
                    "--split",
                    args.split,
                    "--refinement_steps",
                    str(args.refinement_steps),
                    "--update_scale",
                    str(update_scale),
                    "--device",
                    args.device,
                    "--output",
                    str(sample_path),
                ]
                if args.max_displacement is not None:
                    sample_command.extend(
                        ["--max_displacement", str(args.max_displacement)]
                    )
                _run(sample_command, args.dry_run)

            eval_command = [
                sys.executable,
                str(eval_script),
                "--manifest",
                str(args.manifest),
                "--inference_cache",
                args.inference_cache,
                "--reference_cache",
                args.reference_cache,
                "--split",
                args.split,
                "--output_dir",
                str(evaluation_dir),
            ]
            sample_argument = (
                "--cartesian_samples"
                if method == "cartesian_adapter"
                else "--flexbond_samples"
            )
            eval_command.extend([sample_argument, str(sample_path)])
            _run(eval_command, args.dry_run)
            if args.dry_run:
                continue

            with (evaluation_dir / "summary.csv").open(encoding="utf-8") as handle:
                for metric in csv.DictReader(handle):
                    if metric["method"] != method:
                        continue
                    rows.append(
                        {
                            "checkpoint": str(checkpoint),
                            "step": step,
                            "update_scale": update_scale,
                            "max_displacement": args.max_displacement,
                            "method": method,
                            "subset": metric["subset"],
                            "rmsd_mean": float(metric["rmsd_mean"]),
                            "COV-R": float(metric["COV-R"]),
                            "COV-P": float(metric["COV-P"]),
                            "MAT-R": float(metric["MAT-R"]),
                            "MAT-P": float(metric["MAT-P"]),
                            "failure_rate": float(metric["failure_rate"]),
                        }
                    )

    if args.dry_run:
        print("Dry run complete; no samples or evaluations were written.")
        return
    if not rows:
        raise RuntimeError("Sweep produced no metric rows.")
    rows.sort(key=lambda row: (row["method"], row["subset"], row["step"], row["update_scale"]))
    columns = list(rows[0])
    with (args.output_dir / "sweep_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    best = []
    for method in sorted({row["method"] for row in rows}):
        for subset in sorted({row["subset"] for row in rows if row["method"] == method}):
            candidates = [
                row
                for row in rows
                if row["method"] == method and row["subset"] == subset
            ]
            best.append(min(candidates, key=lambda row: (row["rmsd_mean"], row["failure_rate"])))
    with (args.output_dir / "sweep_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"metrics": rows, "best": best}, handle, indent=2)
    print(f"Wrote {len(rows)} sweep rows and {len(best)} best settings to {args.output_dir}")


if __name__ == "__main__":
    main()
