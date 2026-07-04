"""Compare baseline and Jacobian 4D coverage metrics at key thresholds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


KEY_THRESHOLDS = (0.8, 1.0, 1.25, 1.5, 2.0)
REQUIRED_COLUMNS = ("Threshold", "COV-R_mean", "COV-P_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--jacobian_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="scale001_q0001")
    return parser.parse_args()


def _load_metrics(directory: Path) -> List[Dict[str, str]]:
    path = directory / "eval_cov_mat_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Coverage metrics file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return list(reader)


def _find_threshold(rows: List[Dict[str, str]], threshold: float) -> Dict[str, str]:
    for row in rows:
        if abs(float(row["Threshold"]) - threshold) <= 1e-8:
            return row
    raise ValueError(f"Threshold {threshold} is missing from coverage metrics")


def main() -> None:
    args = parse_args()
    base_rows = _load_metrics(Path(args.base_dir))
    jacobian_rows = _load_metrics(Path(args.jacobian_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for threshold in KEY_THRESHOLDS:
        base = _find_threshold(base_rows, threshold)
        jacobian = _find_threshold(jacobian_rows, threshold)
        base_cov_r = float(base["COV-R_mean"])
        jacobian_cov_r = float(jacobian["COV-R_mean"])
        base_cov_p = float(base["COV-P_mean"])
        jacobian_cov_p = float(jacobian["COV-P_mean"])
        summary_rows.append(
            {
                "Threshold": threshold,
                "model_name": args.model_name,
                "base_COV-R_mean": base_cov_r,
                "model_COV-R_mean": jacobian_cov_r,
                "delta_COV-R_mean": jacobian_cov_r - base_cov_r,
                "base_COV-P_mean": base_cov_p,
                "model_COV-P_mean": jacobian_cov_p,
                "delta_COV-P_mean": jacobian_cov_p - base_cov_p,
            }
        )

    csv_path = output_dir / "cov_mat_pair_summary.csv"
    md_path = output_dir / "cov_mat_pair_summary.md"
    columns = list(summary_rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary_rows)

    cov_r_wins = sum(row["delta_COV-R_mean"] > 0 for row in summary_rows)
    cov_p_wins = sum(row["delta_COV-P_mean"] > 0 for row in summary_rows)
    strongest = max(summary_rows, key=lambda row: row["delta_COV-P_mean"])
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Base vs 4D COV/MAT subset summary\n\n")
        handle.write(f"4D model: `{args.model_name}`\n\n")
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in summary_rows:
            handle.write("| " + " | ".join(str(row[key]) for key in columns) + " |\n")
        handle.write("\n## Short interpretation\n\n")
        handle.write(
            f"- 4D has higher COV-R at {cov_r_wins}/{len(KEY_THRESHOLDS)} key thresholds.\n"
        )
        handle.write(
            f"- 4D has higher COV-P at {cov_p_wins}/{len(KEY_THRESHOLDS)} key thresholds.\n"
        )
        handle.write(
            "- Strongest COV-P gain is "
            f"{strongest['delta_COV-P_mean']:.6g} at threshold "
            f"{strongest['Threshold']}.\n"
        )
        handle.write(
            "\nThis is a subset comparison only; no statistical significance is claimed.\n"
        )

    print(f"COV/MAT pair summary saved:\n- {csv_path}\n- {md_path}")


if __name__ == "__main__":
    main()
