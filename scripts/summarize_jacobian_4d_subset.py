"""Summarize baseline and 4D subset-sampling diagnostic outputs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from etflow.models.utils import find_rigid_alignment


FIELDS = (
    "model",
    "num_successes",
    "num_failures",
    "mean_num_atoms",
    "mean_sampling_time",
    "mean_jacobian_4d_head_calls",
    "generated_shapes_valid",
    "mean_first_conformer_kabsch_rmsd",
    "input_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_output", required=True)
    parser.add_argument("--jacobian_output", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def _resolve_output(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "subset_output.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Subset output does not exist: {path}")
    return path


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def _kabsch_rmsd(generated: torch.Tensor, reference: torch.Tensor) -> float:
    generated = generated.detach().to(dtype=torch.float64, device="cpu")
    reference = reference.detach().to(dtype=torch.float64, device="cpu")
    rotation, translation = find_rigid_alignment(generated, reference)
    aligned = (rotation @ generated.transpose(0, 1)).transpose(0, 1) + translation
    return float(
        torch.sqrt((aligned - reference).square().sum(dim=-1).mean()).item()
    )


def _summarize(label: str, path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    molecules: List[Dict[str, Any]] = list(payload.get("molecules", []))
    atom_counts = []
    sampling_times = []
    head_calls = []
    rmsds = []
    shape_validity = []

    for row in molecules:
        generated = torch.as_tensor(row["generated_pos"])
        reference = torch.as_tensor(row["reference_pos"])
        num_atoms = int(row["num_atoms"])
        valid_shape = (
            generated.ndim == 3
            and reference.ndim == 3
            and generated.size(1) == num_atoms
            and reference.size(1) == num_atoms
            and generated.size(2) == 3
            and reference.size(2) == 3
            and generated.size(0) > 0
            and reference.size(0) > 0
        )
        shape_validity.append(valid_shape)
        atom_counts.append(num_atoms)
        sampling_times.append(float(row.get("sampling_time_mean", float("nan"))))
        calls = row.get("jacobian_4d_head_calls")
        if calls is not None:
            head_calls.append(float(calls))
        if valid_shape:
            rmsds.append(_kabsch_rmsd(generated[0], reference[0]))

    return {
        "model": label,
        "num_successes": int(payload.get("num_successes", len(molecules))),
        "num_failures": int(payload.get("num_failures", 0)),
        "mean_num_atoms": _mean(atom_counts),
        "mean_sampling_time": _mean(sampling_times),
        "mean_jacobian_4d_head_calls": _mean(head_calls) if head_calls else 0.0,
        "generated_shapes_valid": bool(shape_validity) and all(shape_validity),
        "mean_first_conformer_kabsch_rmsd": _mean(rmsds),
        "input_path": str(path),
    }


def _display(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if not math.isfinite(value) else f"{value:.6g}"
    return str(value)


def main() -> int:
    args = parse_args()
    base_path = _resolve_output(args.base_output)
    jacobian_path = _resolve_output(args.jacobian_output)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _summarize("base", base_path),
        _summarize("jacobian_4d", jacobian_path),
    ]
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Jacobian 4D subset sampling summary\n\n")
        handle.write("| " + " | ".join(FIELDS) + " |\n")
        handle.write("| " + " | ".join("---" for _ in FIELDS) + " |\n")
        for row in rows:
            handle.write(
                "| "
                + " | ".join(_display(row[field]) for field in FIELDS)
                + " |\n"
            )
        handle.write(
            "\nThe RMSD column compares only the first generated conformer to "
            "the first reference after Kabsch alignment; it is a sanity check, "
            "not COV/MAT/AMR.\n"
        )

    for row in rows:
        print(" ".join(f"{field}={_display(row[field])}" for field in FIELDS))
    print(f"summary.csv: {csv_path}")
    print(f"summary.md: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
