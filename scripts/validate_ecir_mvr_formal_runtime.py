#!/usr/bin/env python
"""CPU-only exhaustive runtime readiness validation for formal D1-B assets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from etflow.ecir import formal_target_assets as target_assets  # noqa: E402
from etflow.ecir.chemical_validity import ChemicalValidity  # noqa: E402
from etflow.ecir.formal_rdkit_adapter import adapt_formal_cache_record  # noqa: E402
from etflow.ecir.formal_runtime_readiness import (  # noqa: E402
    RUNTIME_READY,
    RUNTIME_REPORT,
    canonical_sha256,
    file_sha256,
    formal_asset_identities,
    git_commit,
    runtime_code_identity,
)
from scripts.train_ecir_mvr_run_a import _dataset  # noqa: E402


REPORT_MD = ROOT / "reports/ecir_mvr/D1B_FORMAL_RUNTIME_VALIDATION.md"
PROGRESS_PATH = ROOT / "reports/ecir_mvr/D1B_FORMAL_RUNTIME_VALIDATION.progress.json"
FAILURE_CLASSIFICATIONS = (
    "disconnected_explicit_hydrogen",
    "disconnected_non_hydrogen_atom",
    "multi_component_ionic_molecule",
    "atom_count_mismatch",
    "atomic_number_mismatch",
    "formal_charge_mismatch",
    "atom_map_missing",
    "atom_map_duplicate",
    "atom_mapping_not_unique",
    "hydrogen_parent_not_unique",
    "topology_signature_mismatch",
    "source_target_identity_mismatch",
    "coordinate_shape_mismatch",
    "other",
)


def classify_failure(error: BaseException) -> str:
    message = str(error).lower()
    rules = (
        ("disconnected explicit h", "disconnected_explicit_hydrogen"),
        ("disconnected non-hydrogen", "disconnected_non_hydrogen_atom"),
        ("formal charge", "formal_charge_mismatch"),
        ("atom map missing", "atom_map_missing"),
        ("atom-map identity", "source_target_identity_mismatch"),
        ("atom-map ids are incomplete", "atom_map_missing"),
        ("atom_map_ids", "atom_map_duplicate"),
        ("mapping is not unique", "atom_mapping_not_unique"),
        ("not uniquely proven", "atom_mapping_not_unique"),
        ("hydrogen counts differ", "hydrogen_parent_not_unique"),
        ("topology signature", "topology_signature_mismatch"),
        ("atomic-number", "atomic_number_mismatch"),
        ("atomic number", "atomic_number_mismatch"),
        ("atom counts differ", "atom_count_mismatch"),
        ("num_atoms", "atom_count_mismatch"),
        ("coordinate shape", "coordinate_shape_mismatch"),
        ("does not match source", "source_target_identity_mismatch"),
        ("source sha", "source_target_identity_mismatch"),
        ("sample_id", "source_target_identity_mismatch"),
    )
    return next((label for text, label in rules if text in message), "other")


def validate_manifests(
    sources: Mapping[str, pd.DataFrame],
    targets: Mapping[str, pd.DataFrame],
    expected: Mapping[str, int],
) -> None:
    for split, count in expected.items():
        source = sources[split]
        target = targets[split]
        if len(source) != count or len(target) != count:
            raise RuntimeError(
                f"formal runtime {split} manifest counts differ from {count}"
            )
        for name, frame in (("source", source), ("target", target)):
            if "sample_id" not in frame or frame["sample_id"].duplicated().any():
                raise RuntimeError(
                    f"formal runtime {split} {name} sample identities are not unique"
                )
            if "split" not in frame or set(frame["split"].astype(str)) != {split}:
                raise RuntimeError(
                    f"formal runtime {split} {name} split identity changed"
                )
        if set(source["sample_id"].astype(str)) != set(
            target["sample_id"].astype(str)
        ):
            raise RuntimeError(
                f"formal runtime {split} source/target pairing identities differ"
            )
        if "test_record" in source and source["test_record"].astype(bool).any():
            raise RuntimeError(f"formal runtime {split} source manifest contains test rows")
        if "test_records_read" in target and (
            target["test_records_read"].fillna(-1).astype(int) != 0
        ).any():
            raise RuntimeError(f"formal runtime {split} target manifest reports test reads")


def _target_validation_identities(config: Mapping[str, Any]) -> dict[str, Any]:
    metadata = json.loads(
        Path(config["data"]["target_metadata"]).read_text(encoding="utf-8")
    )
    identities = {
        "builder_code_sha256": metadata["builder_code_sha256"],
        "builder_config_sha256": metadata["builder_config_sha256"],
        "target_builder_config": metadata["target_builder_config"],
        "formal_rdkit_adapter_sha256": metadata[
            "formal_rdkit_adapter_sha256"
        ],
    }
    if metadata.get("config_file_sha256") is not None:
        identities["config_file_sha256"] = metadata["config_file_sha256"]
    return identities


def validate_pair(
    source_row: Mapping[str, Any],
    target_row: Mapping[str, Any],
    target_identities: Mapping[str, Any],
) -> dict[str, Any]:
    source_path = Path(str(source_row["source_path"]))
    target_path = Path(str(target_row["target_cache_path"]))
    if file_sha256(source_path) != str(source_row["source_file_sha256"]):
        raise ValueError("source SHA does not match source manifest")
    if file_sha256(target_path) != str(target_row["target_file_sha256"]):
        raise ValueError("target SHA does not match target manifest")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    target = torch.load(target_path, map_location="cpu", weights_only=False)
    target_assets.validate_target_payload(
        target, dict(source_row), target_identities
    )
    source_atoms = torch.as_tensor(source["atomic_numbers"], dtype=torch.long).view(-1)
    target_atoms = torch.as_tensor(
        target["source_atomic_numbers"], dtype=torch.long
    ).view(-1)
    if source_atoms.numel() != target_atoms.numel():
        raise ValueError("source and target atom counts differ")
    if not torch.equal(source_atoms, target_atoms):
        raise ValueError("source and target atomic-number sequences differ")
    for name in ("x_input", "x_target"):
        coordinates = torch.as_tensor(target[name])
        if tuple(coordinates.shape) != (source_atoms.numel(), 3):
            raise ValueError(f"target {name} coordinate shape differs from source")
    adapted = adapt_formal_cache_record(source)
    disconnected = tuple(adapted.get("_formal_disconnected_cache_atoms", ()))
    disconnected_h = sum(int(source_atoms[index]) == 1 for index in disconnected)
    disconnected_non_h = len(disconnected) - disconnected_h
    component_count = int(adapted.get("_formal_component_count", 1))
    charges = [
        atom.GetFormalCharge() for atom in adapted["_formal_rdkit_mol"].GetAtoms()
    ]
    ionic = component_count > 1 and any(charge != 0 for charge in charges)
    charge_keys = ("formal_charges", "atom_formal_charges", "atomic_formal_charges")
    isotope_keys = ("isotopes", "atom_isotopes", "atomic_isotopes")
    return {
        "disconnected_explicit_hydrogen": disconnected_h,
        "disconnected_non_hydrogen_atom": disconnected_non_h,
        "multi_component_ionic_molecule": int(ionic),
        "_runtime_observation": (
            {
                "source_target_atom_count": int(source_atoms.numel()),
                "source_target_atomic_numbers_match": True,
                "source_target_coordinate_shapes": {
                    name: list(torch.as_tensor(target[name]).shape)
                    for name in ("x_input", "x_target")
                },
                "disconnected_cache_atoms": list(disconnected),
                "disconnected_atomic_numbers": [
                    int(source_atoms[index]) for index in disconnected
                ],
                "disconnected_formal_charges": [
                    int(charges[index]) for index in disconnected
                ],
                "disconnected_cache_atom_map_ids": [
                    list(value)
                    for value in adapted.get("_formal_disconnected_atom_map_ids", ())
                ],
                "cache_identity_kind": adapted.get("_formal_cache_identity_kind"),
                "formal_charge_metadata_available": any(
                    source.get(key) is not None for key in charge_keys
                ),
                "isotope_metadata_available": any(
                    source.get(key) is not None for key in isotope_keys
                ),
                "cache_component_count": int(
                    adapted.get("_formal_cache_component_count", component_count)
                ),
                "rdkit_component_count": component_count,
                "multi_component_ionic_molecule": bool(ionic),
            }
            if disconnected or ionic
            else None
        ),
    }


def _real_error_plan(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "row_index": index,
            "sample_type": "real_error",
            "corruption_type": "real",
            "source": str(row.generator_name),
            "severity": str(row.source_severity),
        }
        for index, row in frame.iterrows()
    ]


def scan_split(
    dataset,
    source_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    split: str,
    start_index: int = 0,
    pair_validator: Callable[..., Mapping[str, int]],
    target_identities: Mapping[str, Any],
    progress: Callable[[int, list[dict[str, Any]], Counter], None] | None = None,
    observation_records: list[dict[str, Any]] | None = None,
) -> tuple[int, list[dict[str, Any]], Counter]:
    dataset.plan = _real_error_plan(source_frame)
    targets = target_frame.set_index("sample_id")
    if not targets.index.is_unique:
        raise ValueError(f"{split} target sample identities are not unique")
    failures: list[dict[str, Any]] = []
    observations: Counter = Counter()
    checked = int(start_index)
    for index in range(int(start_index), len(source_frame)):
        source = source_frame.iloc[index]
        target = None
        try:
            target = targets.loc[source.sample_id]
            pair_observations = dict(
                pair_validator(source.to_dict(), target.to_dict(), target_identities)
            )
            detail = pair_observations.pop("_runtime_observation", None)
            observations.update(pair_observations)
            if detail is not None and observation_records is not None:
                observation_records.append(
                    {
                        "split": split,
                        "dataset_index": index,
                        "sample_id": str(source.sample_id),
                        "source_path": str(source.source_path),
                        "target_path": str(target.target_cache_path),
                        **detail,
                    }
                )
            item = dataset[index]
            if int(item.num_nodes) != int(source.num_atoms):
                raise ValueError("PyG atom count does not match source num_atoms")
        except Exception as error:
            failures.append(
                {
                    "failure_classification": classify_failure(error),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "split": split,
                    "dataset_index": index,
                    "sample_id": str(source.sample_id),
                    "source_path": str(source.source_path),
                    "target_path": (
                        str(target.target_cache_path) if target is not None else None
                    ),
                }
            )
        checked = index + 1
        if progress is not None and (checked % 1000 == 0 or checked == len(source_frame)):
            progress(checked, failures, observations)
    return checked, failures, observations


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# D1-B Formal Runtime Validation",
        "",
        f"Decision: `{report['decision']}`",
        "",
        f"- Train checked: {report['train_checked']}",
        f"- Val checked: {report['val_checked']}",
        f"- Test read: {report['test_records_read']}",
        f"- Passed: {report['passed_count']}",
        f"- Failed: {report['failed_count']}",
        "",
        "## Failure Classifications",
        "",
    ]
    lines.extend(
        f"- `{name}`: {count}"
        for name, count in report["failure_classifications"].items()
    )
    lines.extend(["", "## Failures", ""])
    if report["failures"]:
        lines.extend(
            f"- `{row['sample_id']}` ({row['failure_classification']}): {row['error']}"
            for row in report["failures"]
        )
    else:
        lines.append("None.")
    lines.extend(["", "## Runtime Observations", ""])
    lines.extend(
        f"- `{name}`: {count}"
        for name, count in report["observations"].items()
    )
    lines.extend(["", "## Disconnected Or Ionic Records", ""])
    if report["observation_records"]:
        lines.extend(
            "- "
            f"`{row['sample_id']}`: disconnected={row['disconnected_cache_atoms']}, "
            f"charges={row['disconnected_formal_charges']}, "
            f"components={row['rdkit_component_count']}"
            for row in report["observation_records"]
        )
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/ecir_mvr_formal_large_d1b_base.yaml",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if any("test" in str(key).lower() for key in config["data"]):
        raise RuntimeError("formal runtime validation configuration names test data")
    expected = {"train": 150_000, "val": 10_000}
    frames = {
        split: pd.read_parquet(config["data"][f"{split}_sources"])
        .reset_index(drop=True)
        for split in ("train", "val")
    }
    targets = {
        split: pd.read_parquet(config["data"][f"{split}_targets"])
        .reset_index(drop=True)
        for split in ("train", "val")
    }
    validate_manifests(frames, targets, expected)
    target_identities = _target_validation_identities(config)
    validity = ChemicalValidity(config["data"]["validity_statistics"])
    datasets = {split: _dataset(config, split, validity) for split in ("train", "val")}
    code = runtime_code_identity()
    state_identity = {
        "base_config_sha256": file_sha256(args.config),
        "runtime_code_identity_sha256": code["identity_sha256"],
        "git_commit": git_commit(),
        "formal_asset_identities": formal_asset_identities(config),
    }
    state = {
        **state_identity,
        "split": "train",
        "next_index": 0,
        "failures": [],
        "observations": {},
        "elapsed_seconds": 0.0,
    }
    if args.resume and PROGRESS_PATH.is_file():
        state = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        if any(state.get(key) != value for key, value in state_identity.items()):
            raise RuntimeError("runtime validation resume identity changed")
    prior_elapsed = float(state.get("elapsed_seconds", 0.0))
    started = time.time()
    all_failures = list(state.get("failures", []))
    observations = Counter(state.get("observations", {}))
    observation_records = list(state.get("observation_records", []))
    checked_by_split = {"train": 0, "val": 0}
    for split in ("train", "val"):
        if split == "train" and state.get("split") == "val":
            checked_by_split["train"] = expected["train"]
            continue
        start_index = int(state.get("next_index", 0)) if state.get("split") == split else 0
        split_observation_records: list[dict[str, Any]] = []

        def write_progress(checked, failures, current_observations):
            payload = {
                **state_identity,
                "split": split,
                "next_index": checked,
                "failures": all_failures + failures,
                "observations": dict(observations + current_observations),
                "observation_records": (
                    observation_records + split_observation_records
                ),
                "elapsed_seconds": prior_elapsed + time.time() - started,
                "test_records_read": 0,
            }
            _atomic_json(PROGRESS_PATH, payload)
            print(
                f"split={split} checked={checked}/{expected[split]} "
                f"failed={len(payload['failures'])}",
                flush=True,
            )

        checked, failures, split_observations = scan_split(
            datasets[split],
            frames[split],
            targets[split],
            split=split,
            start_index=start_index,
            pair_validator=validate_pair,
            target_identities=target_identities,
            progress=write_progress,
            observation_records=split_observation_records,
        )
        checked_by_split[split] = checked
        all_failures.extend(failures)
        observations.update(split_observations)
        observation_records.extend(split_observation_records)
        state = {**state_identity, "split": "val", "next_index": 0}
    checked_total = sum(checked_by_split.values())
    classifications = {name: 0 for name in FAILURE_CLASSIFICATIONS}
    classifications.update(Counter(row["failure_classification"] for row in all_failures))
    report = {
        "schema_version": "ecir-mvr-formal-runtime-validation-v1",
        "decision": (
            RUNTIME_READY
            if checked_by_split == expected and not all_failures
            else "D1B_FORMAL_RUNTIME_NOT_READY"
        ),
        "train_checked": checked_by_split["train"],
        "val_checked": checked_by_split["val"],
        "test_records_read": 0,
        "passed_count": checked_total - len(all_failures),
        "failed_count": len(all_failures),
        "failure_classifications": classifications,
        "failures": all_failures,
        "observations": dict(observations),
        "observation_records": observation_records,
        "elapsed_seconds": prior_elapsed + time.time() - started,
        "base_config_path": str(args.config.resolve()),
        "base_config_sha256": file_sha256(args.config),
        "runtime_adapter_sha256": file_sha256(
            ROOT / "etflow/ecir/formal_rdkit_adapter.py"
        ),
        "runtime_code_identity_sha256": code["identity_sha256"],
        "runtime_code_files": code["files"],
        "formal_asset_identities": formal_asset_identities(config),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "formal_target_modified": False,
        "checkpoint_created": False,
    }
    report["runtime_validation_identity_sha256"] = canonical_sha256(report)
    _atomic_json(RUNTIME_REPORT, report)
    _atomic_text(REPORT_MD, _markdown(report))
    _atomic_json(
        PROGRESS_PATH,
        {
            **state_identity,
            "status": "COMPLETED",
            "decision": report["decision"],
            "next_index": 0,
            "test_records_read": 0,
        },
    )
    print(report["decision"])
    if report["decision"] != RUNTIME_READY:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
