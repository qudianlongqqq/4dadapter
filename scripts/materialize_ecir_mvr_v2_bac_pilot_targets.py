#!/usr/bin/env python3
"""Materialize a deterministic train-only BAC target subset for pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etflow.ecir.bac_target import BACMinimalTargetBuilder  # noqa: E402
from etflow.ecir.chemical_validity import ChemicalValidity  # noqa: E402
from etflow.ecir.mvr_dataset import _load_record_and_coordinates  # noqa: E402


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument(
        "--validity-statistics",
        type=Path,
        default=Path("data/ecir_mvr/validity_reference_stats.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v2_bac_overnight/pilot_targets"),
    )
    parser.add_argument("--records", type=int, default=4096)
    parser.add_argument("--selection-seed", type=int, default=42019)
    parser.add_argument("--maximum-fallback-fraction", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formal_root = args.formal_root.expanduser().resolve()
    source_root = args.source_cache_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    target_dir = output / "targets" / "train"
    target_dir.mkdir(parents=True, exist_ok=True)
    source_metadata = json.loads(
        (formal_root / "real_sources" / "metadata.json").read_text(encoding="utf-8")
    )
    if int(source_metadata.get("test_records_read", -1)) != 0:
        raise RuntimeError("source identity does not certify zero test reads")
    sources = pd.read_parquet(formal_root / "real_sources" / "train.parquet")
    targets = pd.read_parquet(
        formal_root / "minimal_targets" / "train.parquet"
    ).set_index("sample_id")
    if set(map(str, sources.split.unique())) != {"train"}:
        raise RuntimeError("pilot source manifest contains a non-train split")
    sources = sources.assign(
        _rank=[
            hashlib.sha256(
                f"{args.selection_seed}|{sample_id}".encode("utf-8")
            ).hexdigest()
            for sample_id in sources.sample_id
        ]
    ).sort_values("_rank")
    selected = sources.head(int(args.records)).drop(columns=["_rank"]).copy()
    validity = ChemicalValidity(args.validity_statistics)
    builder = BACMinimalTargetBuilder(
        validity,
        source_identity_sha256=source_metadata["formal_source_identity_sha256"],
    )
    rows = []
    statuses: Counter[str] = Counter()
    stops: Counter[str] = Counter()
    started = time.perf_counter()
    for dataset_index, (frame_index, original) in enumerate(selected.iterrows()):
        row = original.copy()
        row.source_path = str(source_root / "train" / Path(row.source_path).name)
        old_target = targets.loc[row.sample_id]
        old_target_path = (
            formal_root
            / "minimal_targets"
            / "train"
            / Path(old_target.target_cache_path).name
        )
        record, coordinates = _load_record_and_coordinates(
            row,
            dataset_index=dataset_index,
            target_path=old_target_path,
            formal_adapter_cache=None,
        )
        result = builder.build(coordinates, record)
        metadata = result["target_metadata"]
        statuses[str(metadata["target_status"])] += 1
        stops[str(metadata["stop_reason"])] += 1
        filename = hashlib.sha256(str(row.sample_id).encode("utf-8")).hexdigest()
        target_path = target_dir / f"{filename}.pt"
        payload = {
            "x_target": result["x_target"].detach().cpu(),
            "target_metadata": metadata,
            "source_atomic_numbers": torch.as_tensor(
                record["atomic_numbers"], dtype=torch.long
            ),
            "sample_id": str(row.sample_id),
            "molecule_id": str(row.molecule_id),
            "split": "train",
        }
        temporary = target_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(target_path)
        rows.append(
            {
                "split": "train",
                "sample_id": str(row.sample_id),
                "molecule_id": str(row.molecule_id),
                "target_cache_path": str(target_path),
                "target_file_sha256": _sha(target_path),
                "target_status": str(metadata["target_status"]),
                "target_identity_sha256": metadata["target_identity_sha256"],
            }
        )
        if (dataset_index + 1) % 250 == 0:
            print(
                f"targets={dataset_index + 1}/{len(selected)} "
                f"seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )
    target_manifest = output / "targets_train.parquet"
    source_manifest = output / "sources_train.parquet"
    pd.DataFrame(rows).to_parquet(target_manifest, index=False)
    selected.to_parquet(source_manifest, index=False)
    fallback = statuses["identity_fallback"] / max(len(selected), 1)
    summary = {
        "schema_version": "mcvr-v2-bac-pilot-target-assets-v1",
        "records": len(selected),
        "selection_seed": int(args.selection_seed),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha(source_manifest),
        "target_manifest": str(target_manifest),
        "target_manifest_sha256": _sha(target_manifest),
        "status_counts": dict(statuses),
        "stop_counts": dict(stops),
        "fallback_fraction": fallback,
        "maximum_fallback_fraction": float(args.maximum_fallback_fraction),
        "formal_source_identity_sha256": source_metadata[
            "formal_source_identity_sha256"
        ],
        "validity_statistics_identity_sha256": validity.statistics[
            "identity_sha256"
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "test_records_read": 0,
        "test_assets_opened": False,
        "validation_only": True,
    }
    _write_json(output / "summary.json", summary)
    if fallback > float(args.maximum_fallback_fraction):
        raise RuntimeError(
            f"BAC target fallback fraction {fallback:.6f} exceeds limit"
        )
    print(json.dumps({"status": "BAC_PILOT_TARGETS_READY", **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
