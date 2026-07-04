"""Run (or print) the canonical one-step 4D Jacobian smoke test."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/drugs-so3-jacobian-4d-bs4.yaml"
    )
    parser.add_argument(
        "--output_dir", default="logs_smoke/jacobian_4d_1step"
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--correction_scale", type=float, default=0.03)
    parser.add_argument("--q_loss_weight", type=float, default=0.001)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="print the resolved training command without starting it",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> List[str]:
    project_root = Path(__file__).resolve().parents[1]
    return [
        sys.executable,
        str(project_root / "scripts" / "train_jacobian_4d.py"),
        "--config",
        args.config,
        "--output_dir",
        args.output_dir,
        "--max_steps",
        "1",
        "--batch_size",
        str(args.batch_size),
        "--val_check_interval",
        "1",
        "--limit_val_batches",
        "1",
        "--log_every_n_steps",
        "1",
        "--use_jacobian_4d_correction",
        "true",
        "--jacobian_4d_correction_scale",
        str(args.correction_scale),
        "--jacobian_4d_q_loss_weight",
        str(args.q_loss_weight),
    ]


def main() -> int:
    args = parse_args()
    command = build_command(args)
    print("command:", " ".join(shlex.quote(part) for part in command))
    if args.dry_run:
        return 0
    project_root = Path(__file__).resolve().parents[1]
    return subprocess.run(command, cwd=project_root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
