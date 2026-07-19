#!/usr/bin/env python3
"""Bind one completed formal D1-B prior checkpoint to frozen V7."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from etflow.ecir.mvr_v7_formal import (  # noqa: E402
    build_v7_formal_model,
    canonical_sha256,
    file_sha256,
    load_v7_formal_config,
)


EXPECTED_MANIFEST_SHA = {
    "train_sources": "fbfeffab299c070fcbf29edb99277113c5641ee588000f00fc384162337ecb3d",
    "val_sources": "e7d29f971124f51bd385ec987372ab85181b152250ec0789407a867ff81e3c1a",
    "train_targets": "7e97c5d92529608cfcace8cd279cbd25f20e08b28e1739a191483ba3b574c242",
    "val_targets": "4b4ef42c9905c3bbe2dbe911c57827ce594583c66a52f94d7c4d9b5ca70de4c7",
}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(42, 43), required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--v7-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-mode", choices=("trained", "resumed"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("training_config", "v7_config", "checkpoint", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite V7 formal binding: {args.output_dir}")
    training = yaml.safe_load(args.training_config.read_text(encoding="utf-8"))
    wrapper = load_v7_formal_config(args.v7_config)
    if int(training.get("seed", -1)) != args.seed:
        raise RuntimeError("V7 formal seed differs from training config")
    data = training["data"]
    if any("test" in str(key).lower() for key in data):
        raise RuntimeError("V7 formal training config names test data")
    if int(data["train_molecules"]) != 50_000 or int(data["val_molecules"]) != 5_000:
        raise RuntimeError("V7 formal train/validation molecule counts changed")
    frozen = training.get("frozen_identities", {})
    for key, expected in wrapper["formal_identities"].items():
        if frozen.get(key) != expected:
            raise RuntimeError(f"V7 formal frozen identity changed: {key}")
    checkpoint_sha = file_sha256(args.checkpoint)
    if (
        args.expected_checkpoint_sha256
        and checkpoint_sha != args.expected_checkpoint_sha256
    ):
        raise RuntimeError("V7 formal prior checkpoint SHA mismatch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint["config"].get("seed", -1)) != args.seed:
        raise RuntimeError("V7 formal checkpoint seed changed")
    model = build_v7_formal_model(checkpoint, wrapper, device="cpu")
    del model

    args.output_dir.mkdir(parents=True)
    resolved = {
        **wrapper,
        "seed": args.seed,
        "prior_checkpoint": str(args.checkpoint),
        "prior_checkpoint_sha256": checkpoint_sha,
        "prior_training_config": str(args.training_config),
        "prior_training_config_sha256": file_sha256(args.training_config),
        "v7_wrapper_config_sha256": file_sha256(args.v7_config),
    }
    resolved_path = args.output_dir / "config.resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    checkpoint_identity = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_schema_version": checkpoint["schema_version"],
        "strict_load": True,
        "v7_trainable_parameter_count": 0,
    }
    _write_json(args.output_dir / "checkpoint_identity.json", checkpoint_identity)
    metadata = {
        "schema_version": "mcvr-v7-formal-seed-binding-v1",
        "status": "V7_FORMAL_PRIOR_BOUND",
        "seed": args.seed,
        "prior_mode": args.prior_mode,
        "formal_large_prior_training": True,
        "v7_training_performed": False,
        "train_molecules": 50_000,
        "validation_molecules": 5_000,
        "checkpoint_sha256": checkpoint_sha,
        "wrapper_config_sha256": file_sha256(args.v7_config),
        "test_records_read": 0,
        "test_assets_opened": False,
        "formal_test_run": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(args.output_dir / "run_metadata.json", metadata)
    provenance = {
        "schema_version": "mcvr-v7-formal-provenance-v1",
        "seed": args.seed,
        "git_commit": _git_commit(),
        "host": platform.node(),
        "platform": platform.platform(),
        "checkpoint_sha256": checkpoint_sha,
        "training_config_sha256": file_sha256(args.training_config),
        "wrapper_config_sha256": file_sha256(args.v7_config),
        "resolved_config_identity_sha256": canonical_sha256(resolved),
        "dataset_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "formal_identities": wrapper["formal_identities"],
        "test_records_read": 0,
        "test_assets_opened": False,
    }
    _write_json(args.output_dir / "PROVENANCE.json", provenance)
    files = (
        "checkpoint_identity.json",
        "config.resolved.yaml",
        "PROVENANCE.json",
        "run_metadata.json",
    )
    sums = "".join(
        f"{file_sha256(args.output_dir / name)}  {name}\n" for name in files
    )
    (args.output_dir / "SHA256SUMS.txt").write_text(sums, encoding="ascii")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
