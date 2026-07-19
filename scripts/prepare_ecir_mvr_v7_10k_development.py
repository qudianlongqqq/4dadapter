#!/usr/bin/env python3
"""Freeze the train-derived unseen 10K-molecule V7 development cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SEED = 43018
MOLECULES = 10_000
FORMAL_SOURCE_IDENTITY = (
    "3d86eec9ebd82ae96860330ded0fad35938be74111929ed29b9487f8b7e39a0a"
)
D1_CHECKPOINT_SHA256 = (
    "9348744817ef7eec6d9d682dd95a35f0be86f0565b6dd060e8d5fe54e609e426"
)
SOURCE_MANIFEST_SHA256 = (
    "fbfeffab299c070fcbf29edb99277113c5641ee588000f00fc384162337ecb3d"
)
TARGET_MANIFEST_SHA256 = (
    "7e97c5d92529608cfcace8cd279cbd25f20e08b28e1739a191483ba3b574c242"
)
D1_TRAIN_SOURCE_SHA256 = (
    "767eb0db025d85df7421c7418dd4460463c5e332cf21673870833bed34a85c14"
)
EXPECTED_SELECTION = {
    "ordered_molecule_ids_sha256": (
        "17f19269598d7985b16bd0beb82f8e00f0401b2a44ba91c42b631bdc8489bf78"
    ),
    "ordered_sample_ids_sha256": (
        "880c68ced3e8f3e74b9aa44a207ea1abbc0715776e542298d06514840695c0a3"
    ),
    "source_coordinate_hashes_sha256": (
        "40561efd8201bc663b24dda58769cf5f09cf1b28fffa4ffd772c92a5684ba340"
    ),
    "target_hashes_sha256": (
        "f16b9483df647e20860e324fc55db6f64e89cc2548038d2f9242e4f36eef3e04"
    ),
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _rank_molecule_ids(molecule_ids: set[str]) -> list[str]:
    def rank_key(molecule_id: str) -> tuple[str, str]:
        digest = hashlib.sha256(
            (
                f"{SEED}|{FORMAL_SOURCE_IDENTITY}|{D1_CHECKPOINT_SHA256}|"
                f"{molecule_id}"
            ).encode("utf-8")
        ).hexdigest()
        return digest, molecule_id

    return sorted(map(str, molecule_ids), key=rank_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument(
        "--d1-training-source-manifest",
        type=Path,
        default=Path(
            "diagnostics/ecir_mvr/v2_bac_overnight/pilot_targets/"
            "sources_train.parquet"
        ),
    )
    parser.add_argument(
        "--validation-cohorts",
        type=Path,
        default=Path(
            "diagnostics/ecir_mvr/v2_bac_overnight/validation_cohorts.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v7_10k/manifests"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in (
        "formal_root",
        "source_cache_root",
        "d1_training_source_manifest",
        "validation_cohorts",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    source_path = args.formal_root / "real_sources" / "train.parquet"
    target_path = args.formal_root / "minimal_targets" / "train.parquet"
    expected_files = {
        source_path: SOURCE_MANIFEST_SHA256,
        target_path: TARGET_MANIFEST_SHA256,
        args.d1_training_source_manifest: D1_TRAIN_SOURCE_SHA256,
    }
    for path, expected in expected_files.items():
        actual = _sha(path)
        if actual != expected:
            raise RuntimeError(f"frozen input SHA mismatch: {path}: {actual}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite V7 10K manifest dir: {args.output_dir}")

    sources = pd.read_parquet(source_path)
    targets = pd.read_parquet(target_path)
    d1_training = pd.read_parquet(args.d1_training_source_manifest)
    if set(map(str, sources.split.unique())) != {"train"}:
        raise RuntimeError("V7 10K source pool is not exclusively train split")
    if bool(sources.test_record.astype(bool).any()):
        raise RuntimeError("V7 10K source pool contains a test record")
    if int(targets.test_records_read.max()) != 0:
        raise RuntimeError("V7 10K target manifest reports test reads")

    seen = set(d1_training.molecule_id.astype(str))
    candidates = set(sources.molecule_id.astype(str)) - seen
    ranked = _rank_molecule_ids(candidates)
    selected_ids = set(ranked[:MOLECULES])
    selected_sources = sources[
        sources.molecule_id.astype(str).isin(selected_ids)
    ].sort_values(["molecule_id", "sample_id"])
    selected_targets = targets[
        targets.sample_id.isin(selected_sources.sample_id)
    ].sort_values("sample_id")

    cohorts = json.loads(args.validation_cohorts.read_text(encoding="utf-8"))
    tune_ids = set(map(str, cohorts["validation_tune"]["molecule_ids"]))
    holdout_ids = set(map(str, cohorts["validation_holdout"]["molecule_ids"]))
    overlaps = {
        "d1_training_molecules": len(selected_ids & seen),
        "validation_tune_molecules": len(selected_ids & tune_ids),
        "frozen_holdout_molecules": len(selected_ids & holdout_ids),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"V7 10K cohort overlap detected: {overlaps}")
    if int(selected_sources.molecule_id.nunique()) != MOLECULES:
        raise RuntimeError("V7 10K molecule count changed")
    if len(selected_sources) != 30_000:
        raise RuntimeError("V7 10K record count changed")
    if set(selected_sources.groupby("molecule_id").size()) != {3}:
        raise RuntimeError("V7 10K records-per-molecule changed")
    if set(selected_sources.sample_id) != set(selected_targets.sample_id):
        raise RuntimeError("V7 10K source/target sample identity differs")

    source_paths = [
        args.source_cache_root / str(split) / Path(str(value)).name
        for split, value in zip(
            selected_sources.split, selected_sources.source_path, strict=True
        )
    ]
    target_paths = [
        args.formal_root / "minimal_targets" / str(split) / Path(str(value)).name
        for split, value in zip(
            selected_targets.split, selected_targets.target_cache_path, strict=True
        )
    ]
    missing_source = [str(path) for path in source_paths if not path.is_file()]
    missing_target = [str(path) for path in target_paths if not path.is_file()]
    if missing_source or missing_target:
        raise RuntimeError(
            f"V7 10K missing PT files: source={len(missing_source)} "
            f"target={len(missing_target)}"
        )

    selection = {
        "ordered_molecule_ids_sha256": _canonical_sha(sorted(selected_ids)),
        "ordered_sample_ids_sha256": _canonical_sha(
            selected_sources.sample_id.astype(str).tolist()
        ),
        "source_coordinate_hashes_sha256": _canonical_sha(
            selected_sources.coordinate_sha256.astype(str).tolist()
        ),
        "target_hashes_sha256": _canonical_sha(
            selected_targets.target_sha256.astype(str).tolist()
        ),
    }
    if selection != EXPECTED_SELECTION:
        raise RuntimeError(f"V7 10K frozen selection changed: {selection}")

    args.output_dir.mkdir(parents=True)
    derived_source = args.output_dir / "development_sources.parquet"
    derived_target = args.output_dir / "development_targets.parquet"
    selected_sources.to_parquet(derived_source, index=False)
    selected_targets.to_parquet(derived_target, index=False)
    stable_identity = {
        "schema_version": "mcvr-v7-10k-development-manifest-v1",
        "seed": SEED,
        "cohort_policy": "train-derived-unseen-development",
        "formal_source_identity_sha256": FORMAL_SOURCE_IDENTITY,
        "d1_checkpoint_sha256": D1_CHECKPOINT_SHA256,
        "parent_source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "parent_target_manifest_sha256": TARGET_MANIFEST_SHA256,
        "d1_training_source_manifest_sha256": D1_TRAIN_SOURCE_SHA256,
        "molecules": MOLECULES,
        "records": len(selected_sources),
        "records_per_molecule": {"3": MOLECULES},
        "source_manifest_sha256": _sha(derived_source),
        "target_manifest_sha256": _sha(derived_target),
        **selection,
        "overlaps": overlaps,
        "missing_source_pt_count": 0,
        "missing_target_pt_count": 0,
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "formal_large_run": False,
        "training_performed": False,
        "target_rematerialization": False,
        "validation_only": True,
    }
    manifest = {
        **stable_identity,
        "identity_sha256": _canonical_sha(stable_identity),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(derived_source),
        "target_manifest": str(derived_target),
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "V7_10K_DEVELOPMENT_MANIFEST_FROZEN",
                "identity_sha256": manifest["identity_sha256"],
                "molecules": MOLECULES,
                "records": len(selected_sources),
                "test_records_read": 0,
                "test_assets_opened": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
