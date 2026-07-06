#!/usr/bin/env python
"""Run a fair checkpoint/update-scale sample, evaluation, and diagnosis sweep."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import torch


COLUMNS = (
    "method",
    "checkpoint_name",
    "checkpoint_path",
    "step",
    "update_scale",
    "subset",
    "rmsd_mean",
    "rmsd_median",
    "COV-R",
    "COV-P",
    "MAT-R",
    "MAT-P",
    "failure_rate",
    "fraction_improved",
    "fraction_worsened",
    "mean_delta_rmsd",
    "median_delta_rmsd",
    "mean_update_norm",
    "max_update_norm",
)


def _tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _step(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and checkpoint.get("global_step") is not None:
        return int(checkpoint["global_step"])
    matches = re.findall(r"(?:step[=_-]?)?(\d+)", path.stem)
    return int(matches[-1]) if matches else -1


def _run(command: list[str], dry_run: bool) -> None:
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--inference_cache", required=True)
    parser.add_argument("--reference_cache", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--cartesian_checkpoints", nargs="*", type=Path, default=())
    parser.add_argument("--flexbond_checkpoints", nargs="*", type=Path, default=())
    parser.add_argument("--cartesian_config", type=Path)
    parser.add_argument("--flexbond_config", type=Path)
    parser.add_argument(
        "--update_scales", nargs="+", type=float, default=(0.1, 0.2, 0.5, 1.0)
    )
    parser.add_argument("--max_displacement", type=float)
    parser.add_argument("--adaptive_alpha_by_update_norm", action="store_true")
    parser.add_argument("--target_update_norm", type=float)
    parser.add_argument("--refinement_steps", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if not args.cartesian_checkpoints and not args.flexbond_checkpoints:
        raise ValueError("At least one checkpoint list must be non-empty")
    if args.cartesian_checkpoints and args.cartesian_config is None:
        raise ValueError("--cartesian_config is required for Cartesian checkpoints")
    if args.flexbond_checkpoints and args.flexbond_config is None:
        raise ValueError("--flexbond_config is required for FlexBond checkpoints")
    if any(scale < 0 for scale in args.update_scales):
        raise ValueError("update_scales must be non-negative")

    root = Path(__file__).resolve().parents[1]
    sample_script = root / "scripts/sample_flexbond_optimizer.py"
    eval_script = root / "scripts/eval_flexbond_optimizer.py"
    diagnose_script = root / "scripts/diagnose_flexbond_samples.py"
    diversity_script = root / "scripts/diagnose_flexbond_diversity.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("cartesian_adapter", path, args.cartesian_config)
        for path in args.cartesian_checkpoints
    ] + [
        ("flexbond4d_adapter", path, args.flexbond_config)
        for path in args.flexbond_checkpoints
    ]
    output_rows = []
    for method, checkpoint, config in jobs:
        checkpoint = checkpoint.expanduser().resolve()
        config = config.expanduser().resolve()
        if not checkpoint.is_file() or not config.is_file():
            raise FileNotFoundError(f"Missing checkpoint/config: {checkpoint}, {config}")
        step = _step(checkpoint)
        sample_flag = (
            "--cartesian_samples"
            if method == "cartesian_adapter"
            else "--flexbond_samples"
        )
        for scale in args.update_scales:
            name = f"{method}_step{step}_alpha{_tag(scale)}"
            run_dir = args.output_dir / name
            sample_path = run_dir / f"samples_alpha{_tag(scale)}.pt"
            eval_dir = run_dir / "evaluation"
            diagnostic_dir = run_dir / "diagnostics"
            diversity_dir = run_dir / "diversity"
            sample_command = [
                sys.executable,
                str(sample_script),
                "--checkpoint", str(checkpoint),
                "--config", str(config),
                "--cache_dir", args.inference_cache,
                "--manifest", str(args.manifest),
                "--split", args.split,
                "--output", str(sample_path),
                "--refinement_steps", str(args.refinement_steps),
                "--update_scale", str(scale),
                "--device", args.device,
            ]
            if args.max_displacement is not None:
                sample_command.extend(["--max_displacement", str(args.max_displacement)])
            if args.adaptive_alpha_by_update_norm:
                sample_command.append("--adaptive_alpha_by_update_norm")
                if args.target_update_norm is None:
                    raise ValueError(
                        "--target_update_norm is required with adaptive alpha"
                    )
                sample_command.extend(
                    ["--target_update_norm", str(args.target_update_norm)]
                )
            if not (args.skip_existing and sample_path.is_file()):
                _run(sample_command, args.dry_run)

            common = [
                "--manifest", str(args.manifest),
                "--inference_cache", args.inference_cache,
                "--reference_cache", args.reference_cache,
                "--split", args.split,
                sample_flag, str(sample_path),
            ]
            _run(
                [sys.executable, str(eval_script), *common,
                 "--threshold", str(args.threshold), "--output_dir", str(eval_dir)],
                args.dry_run,
            )
            _run(
                [sys.executable, str(diagnose_script), *common,
                 "--upstream_only", "--output_dir", str(diagnostic_dir)],
                args.dry_run,
            )
            _run(
                [sys.executable, str(diversity_script), *common,
                 "--threshold", str(args.threshold), "--output_dir", str(diversity_dir)],
                args.dry_run,
            )
            if args.dry_run:
                continue

            diagnostic_rows = _read(
                diagnostic_dir / "diagnostics_by_rotatable.csv"
            )
            diagnostic_by_subset = {
                row["group"]: row
                for row in diagnostic_rows
                if row["method"] == method
            }
            for metric in _read(eval_dir / "summary.csv"):
                if metric["method"] != method:
                    continue
                diagnostic = diagnostic_by_subset[metric["subset"]]
                output_rows.append(
                    {
                        "method": method,
                        "checkpoint_name": checkpoint.name,
                        "checkpoint_path": str(checkpoint),
                        "step": step,
                        "update_scale": scale,
                        "subset": metric["subset"],
                        "rmsd_mean": metric["rmsd_mean"],
                        "rmsd_median": metric["rmsd_median"],
                        "COV-R": metric["COV-R"],
                        "COV-P": metric["COV-P"],
                        "MAT-R": metric["MAT-R"],
                        "MAT-P": metric["MAT-P"],
                        "failure_rate": metric["failure_rate"],
                        "fraction_improved": diagnostic["fraction_improved"],
                        "fraction_worsened": diagnostic["fraction_worsened"],
                        "mean_delta_rmsd": diagnostic["mean_delta_rmsd"],
                        "median_delta_rmsd": diagnostic["median_delta_rmsd"],
                        "mean_update_norm": diagnostic["mean_update_norm"],
                        "max_update_norm": diagnostic["max_update_norm"],
                    }
                )

    if args.dry_run:
        print("Dry run complete; no sampling or evaluation was executed.")
        return
    output_rows.sort(
        key=lambda row: (row["method"], int(row["step"]), float(row["update_scale"]), row["subset"])
    )
    with (args.output_dir / "sweep_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)
    best = []
    for method in sorted({row["method"] for row in output_rows}):
        for subset in sorted(
            {row["subset"] for row in output_rows if row["method"] == method}
        ):
            candidates = [
                row
                for row in output_rows
                if row["method"] == method and row["subset"] == subset
            ]
            best.append(
                min(
                    candidates,
                    key=lambda row: (
                        float(row["rmsd_mean"]),
                        float(row["failure_rate"]),
                    ),
                )
            )
    with (args.output_dir / "sweep_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({"metrics": output_rows, "best": best}, handle, indent=2)
    print(f"Wrote {len(output_rows)} sweep rows to {args.output_dir / 'sweep_summary.csv'}")


if __name__ == "__main__":
    main()
