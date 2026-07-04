#!/usr/bin/env python
"""Combine one or more FlexBond evaluation summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        csv_path = path / "summary.csv" if path.is_dir() else path
        with csv_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({"run": csv_path.parent.name, **row})
    if not rows:
        raise SystemExit("No summary rows found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# FlexBond optimizer run summary\n\n")
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row[c]) for c in columns) + " |\n")
    print(f"Wrote combined summary to {args.output_dir}")


if __name__ == "__main__":
    main()
