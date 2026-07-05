#!/usr/bin/env python
"""Check that Cartesian and FlexBond samples use one identical frozen cohort."""

import argparse
from pathlib import Path

import torch

from etflow.data.flexbond_cache_schema import x_init_sha256
from etflow.data.flexbond_eval_manifest import load_eval_manifest


def _check(path: Path, method: str, manifest: dict) -> tuple[list[str], list[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("manifest", {}).get("records") != manifest["records"]:
        raise ValueError(f"{path} embeds a different evaluation manifest")
    expected = {str(row["sample_id"]): row for row in manifest["records"]}
    actual = {}
    failed = []
    for record in payload.get("records", []):
        sample_id = str(record["sample_id"])
        if sample_id in actual:
            raise ValueError(f"Duplicate sample id {sample_id!r} in {path}")
        if record.get("method_name") != method:
            raise ValueError(f"Unexpected method name in {path}: {record.get('method_name')}")
        digest = x_init_sha256(record["x_init"], record["atomic_numbers"])
        if sample_id not in expected or digest != str(expected[sample_id]["x_init_hash"]):
            raise ValueError(f"x_init cohort mismatch for {sample_id!r} in {path}")
        actual[sample_id] = record
        if record.get("status") != "success":
            failed.append(sample_id)
    return sorted(set(expected).difference(actual)), sorted(failed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cartesian_samples", required=True, type=Path)
    parser.add_argument("--flexbond_samples", required=True, type=Path)
    args = parser.parse_args()
    manifest = load_eval_manifest(args.manifest)
    cart_missing, cart_failed = _check(
        args.cartesian_samples, "cartesian_adapter", manifest
    )
    flex_missing, flex_failed = _check(
        args.flexbond_samples, "flexbond4d_adapter", manifest
    )
    print(f"cartesian: missing={cart_missing} failed={cart_failed}")
    print(f"flexbond: missing={flex_missing} failed={flex_failed}")
    if cart_missing or flex_missing:
        raise SystemExit("FAIL: sample payload is missing frozen cohort ids")
    print(f"PASS: both methods use identical x_init for {len(manifest['records'])} samples")


if __name__ == "__main__":
    main()
