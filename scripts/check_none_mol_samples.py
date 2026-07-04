"""Scan processed ETFlow samples for missing, None, or unrecoverable molecules."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch
from loguru import logger as log
from tqdm import tqdm

from etflow.commons.featurization import (
    MOL_BLOCK_FIELDS,
    MOL_OBJECT_FIELDS,
    SMILES_FIELDS,
    get_sample_field,
    get_sample_field_names,
    recover_mol_from_sample,
)
from etflow.commons.io import get_base_data_dir


COLUMNS = [
    "partition",
    "split",
    "sample_idx",
    "file_path",
    "mol_status",
    "available_recovery_fields",
    "recoverable",
    "recovery_source",
    "error",
]


def _mol_status(sample) -> str:
    field_names = set(get_sample_field_names(sample))
    if "mol" not in field_names:
        return "missing"
    return "none" if get_sample_field(sample, "mol") is None else "present"


def _available_recovery_fields(sample) -> List[str]:
    fields = MOL_OBJECT_FIELDS + MOL_BLOCK_FIELDS + SMILES_FIELDS
    return [name for name in fields if get_sample_field(sample, name) is not None]


def scan(args) -> int:
    data_dir = (
        Path(args.data_dir)
        if args.data_dir is not None
        else Path(get_base_data_dir()) / "processed"
    )
    rows: List[Dict[str, object]] = []
    counts = {"present": 0, "none": 0, "missing": 0, "unrecoverable": 0}
    total = 0

    for split in args.splits:
        split_dir = data_dir / args.partition.lower() / split
        files = sorted(split_dir.glob("*.pt"))
        if not files:
            log.warning(f"No .pt files found in {split_dir}")
            continue

        for sample_idx, file_path in enumerate(
            tqdm(files, desc=f"Scanning {args.partition}/{split}")
        ):
            total += 1
            sample = None
            status = "load_failed"
            recoverable = False
            recovery_source = ""
            error = ""
            available_fields: List[str] = []
            try:
                sample = torch.load(file_path, map_location="cpu", weights_only=False)
                status = _mol_status(sample)
                counts[status] += 1
                available_fields = _available_recovery_fields(sample)
                recovery = recover_mol_from_sample(
                    sample,
                    expected_atomic_numbers=get_sample_field(
                        sample,
                        "atomic_numbers",
                    ),
                )
                recoverable = True
                recovery_source = recovery.source
            except Exception as exc:
                counts["unrecoverable"] += 1
                error = str(exc)
                if sample is not None:
                    available_fields = _available_recovery_fields(sample)

            should_report = (
                status == "none"
                or not recoverable
                or (args.include_missing and status == "missing")
            )
            if should_report:
                rows.append(
                    {
                        "partition": args.partition,
                        "split": split,
                        "sample_idx": sample_idx,
                        "file_path": str(file_path),
                        "mol_status": status,
                        "available_recovery_fields": ";".join(available_fields),
                        "recoverable": recoverable,
                        "recovery_source": recovery_source,
                        "error": error,
                    }
                )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    log.info(
        f"Scanned {total} samples: present={counts['present']}, "
        f"None={counts['none']}, missing={counts['missing']}, "
        f"unrecoverable={counts['unrecoverable']}"
    )
    log.info(f"Wrote {len(rows)} diagnostic rows to {output_path}")
    return 1 if args.fail_on_unrecoverable and counts["unrecoverable"] else 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--partition", choices=["drugs", "qm9"], default="drugs")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
    )
    parser.add_argument("--output_csv", type=str, default="none_mol_samples.csv")
    parser.add_argument("--include_missing", action="store_true")
    parser.add_argument("--fail_on_unrecoverable", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(scan(parse_args()))
