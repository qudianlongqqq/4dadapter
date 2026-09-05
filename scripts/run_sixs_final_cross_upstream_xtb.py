#!/usr/bin/env python
"""Adapt the frozen final xTB worker to 10K cross-upstream SIXS-U arms."""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE = Path(__file__).with_name("run_sixs_primary_final_xtb.py")


def main() -> None:
    joined = " ".join(sys.argv).lower()
    if "avgflow" in joined:
        prefix = "AVGFLOW"
    elif "ditmc" in joined:
        prefix = "DITMC"
    else:
        raise RuntimeError("cannot determine cross-upstream branch from arguments")
    text = SOURCE.read_text(encoding="utf-8")
    text = text.replace("EXPECTED_RECORDS = 5000", "EXPECTED_RECORDS = 10000")
    text = text.replace(
        'method_ids = ["source"] + [method["id"] for method in protocol["model_methods"]] + ["mmff94s"]',
        f'method_ids = ["source"] + {[f"{prefix}_SIXS_U_SEED{seed}" for seed in (307, 331, 353)]!r}',
    )
    text = text.replace(
        'source_records = pd.read_parquet(args.coordinate_dir / "methods/source/PER_RECORD.parquet")',
        'source_records = pd.read_parquet(args.coordinate_dir / "SOURCE_RECORDS.parquet")',
    )
    text = text.replace(
        'source_molecules = load_sdf(args.coordinate_dir / "methods/source/COORDINATES.sdf")',
        f'source_molecules = load_sdf(args.coordinate_dir / "sdf/{prefix}_RAW.sdf")',
    )
    text = text.replace(
        'molecules = load_sdf(args.coordinate_dir / f"methods/{method}/COORDINATES.sdf")',
        f'molecules = load_sdf(args.coordinate_dir / "sdf" / ("{prefix}_RAW.sdf" if method == "source" else f"{{method}}.sdf"))',
    )
    code = compile(text, str(SOURCE), "exec")
    namespace = {"__name__": "__main__", "__file__": str(SOURCE)}
    exec(code, namespace)


if __name__ == "__main__":
    main()
