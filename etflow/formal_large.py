"""Pure contracts shared by the formal-large experiment entry points."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from etflow.commons.record_identity import source_record_identity


SEED = 42
ALPHAS = (0.2, 0.5)
CHECKPOINT_STEPS = (50_000, 100_000, 150_000, 200_000)
TRAIN_MOLECULES = 50_000
VAL_MOLECULES = 5_000
TEST_MOLECULES = 100
SCREEN_MAX_RECORDS = 200
CONFIRM_MAX_RECORDS = 600
TRAIN_PAIRS_PER_MOLECULE = 3
VAL_PAIRS_PER_MOLECULE = 2
REFINEMENT_STEPS = 10
TRAINING_BUDGET = {
    "max_steps": 200_000,
    "batch_size": 4,
    "accumulate_grad_batches": 2,
    "effective_batch_size": 8,
    "learning_rate": 0.0002,
    "t_min": 0.0,
    "t_max": 0.25,
    "seed": SEED,
    "precision": "32-true",
    "val_check_interval": 5_000,
    "limit_val_batches": 100,
}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_molecule_order(molecule_ids: Iterable[str], seed: int = SEED) -> list[str]:
    unique = set(map(str, molecule_ids))
    return sorted(
        unique,
        key=lambda value: (
            hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
            value,
        ),
    )


def select_pair_records(
    records: Iterable[Mapping[str, Any]],
    *,
    molecule_limit: int,
    pairs_per_molecule: int | None,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        molecule_id = source_record_identity(record)
        grouped[molecule_id].append(dict(record))
    order = deterministic_molecule_order(grouped, seed)[:molecule_limit]
    selected = []
    for molecule_id in order:
        rows = sorted(grouped[molecule_id], key=lambda row: str(row["sample_id"]))
        selected.extend(rows if pairs_per_molecule is None else rows[:pairs_per_molecule])
    return selected


def molecule_ids(records: Iterable[Mapping[str, Any]]) -> set[str]:
    return {source_record_identity(record) for record in records}


def assert_disjoint_splits(split_records: Mapping[str, Iterable[Mapping[str, Any]]]) -> None:
    ids = {name: molecule_ids(records) for name, records in split_records.items()}
    names = sorted(ids)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = ids[left].intersection(ids[right])
            if overlap:
                raise ValueError(
                    f"Molecule leakage between {left} and {right}: {sorted(overlap)[:20]}"
                )


def flexibility_tier(num_rotatable_bonds: int) -> str:
    value = int(num_rotatable_bonds)
    if value <= 2:
        return "low"
    if value <= 5:
        return "medium"
    return "high"


def select_stratified_manifest(
    manifest: Mapping[str, Any],
    counts: Mapping[str, int],
    *,
    seed: int = SEED,
    max_records: int | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest["records"]:
        grouped[str(row["mol_id"])].append(dict(row))
    tiers: dict[str, list[str]] = defaultdict(list)
    for molecule_id, rows in grouped.items():
        tiers[flexibility_tier(int(rows[0]["num_rotatable_bonds"]))].append(molecule_id)
    chosen = set()
    for tier, count in counts.items():
        ordered = deterministic_molecule_order(tiers.get(tier, []), seed)
        if len(ordered) < int(count):
            raise ValueError(
                f"Need {count} {tier}-flexibility molecules, found {len(ordered)}"
            )
        chosen.update(ordered[: int(count)])
    candidate_rows = [
        dict(row) for row in manifest["records"] if str(row["mol_id"]) in chosen
    ]
    if max_records is not None:
        if int(max_records) < len(chosen):
            raise ValueError(
                "max_records must permit at least one record per selected molecule"
            )
        rows = []
        unseen = set(chosen)
        for row in candidate_rows:
            molecule_id = str(row["mol_id"])
            if molecule_id in unseen:
                rows.append(row)
                unseen.remove(molecule_id)
            elif len(rows) < int(max_records) - len(unseen):
                rows.append(row)
            if len(rows) >= int(max_records) and not unseen:
                break
        if unseen:
            raise ValueError(f"Selected molecules have no records: {sorted(unseen)}")
    else:
        rows = candidate_rows
    original_counts = Counter(str(row["mol_id"]) for row in candidate_rows)
    kept_counts = Counter(str(row["mol_id"]) for row in rows)
    truncated = [
        {
            "mol_id": molecule_id,
            "original_records": original_counts[molecule_id],
            "kept_records": kept_counts[molecule_id],
        }
        for molecule_id in sorted(chosen)
        if kept_counts[molecule_id] < original_counts[molecule_id]
    ]
    selection_report = {
        "selected_molecule_count": len(chosen),
        "selected_record_count": len(rows),
        "max_records": max_records,
        "records_per_molecule": dict(sorted(kept_counts.items())),
        "truncated_molecules": truncated,
        "record_order": "original_manifest_order",
    }
    return {
        **dict(manifest),
        "selection_seed": seed,
        "selection_counts": dict(counts),
        "selection_report": selection_report,
        "records": rows,
    }


def pair_count_distribution(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(
        source_record_identity(record) for record in records
    )
    return {str(key): value for key, value in sorted(Counter(counts.values()).items())}


def training_budget_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    trainer = config["trainer"]
    data = config["data"]
    optimizer = config["optimizer"]
    time_sampling = config["time_sampling"]
    return {
        "max_steps": int(trainer["max_steps"]),
        "batch_size": int(data["batch_size"]),
        "accumulate_grad_batches": int(trainer["accumulate_grad_batches"]),
        "effective_batch_size": int(data["batch_size"])
        * int(trainer["accumulate_grad_batches"]),
        "learning_rate": float(optimizer.get("lr", config.get("model", {}).get("lr"))),
        "t_min": float(time_sampling["t_min"]),
        "t_max": float(time_sampling["t_max"]),
        "seed": int(config["seed"]),
        "precision": str(trainer["precision"]),
        "val_check_interval": int(trainer["val_check_interval"]),
        "limit_val_batches": int(trainer["limit_val_batches"]),
    }


def assert_matched_training_budgets(configs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    signatures = {name: training_budget_signature(config) for name, config in configs.items()}
    first_name = next(iter(signatures))
    first = signatures[first_name]
    for name, signature in signatures.items():
        if signature != first:
            raise ValueError(f"Training budget mismatch: {first_name}={first}, {name}={signature}")
    if first != TRAINING_BUDGET:
        raise ValueError(f"Formal-large budget differs from the frozen contract: {first}")
    return first


def ranking_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    def metric(*names: str, default: float = 0.0) -> float:
        for name in names:
            if row.get(name) not in (None, ""):
                return float(row[name])
        return default

    return (
        metric("failure_rate", default=1.0),
        metric("rmsd_mean", "MAT-P", default=float("inf")),
        metric("MAT-R", default=float("inf")),
        -metric("COV-R"),
        -metric("COV-P"),
        metric("high_flex_rmsd", default=float("inf")),
    )


def top_candidates(rows: Iterable[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    return [dict(row) for row in sorted(rows, key=ranking_key)[:count]]


def assert_same_evaluation_cohort(manifests: Mapping[str, Mapping[str, Any]]) -> None:
    expected = None
    for method, manifest in manifests.items():
        identity = [
            (str(row["sample_id"]), str(row["mol_id"]), str(row["x_init_hash"]))
            for row in manifest["records"]
        ]
        if expected is None:
            expected = identity
        elif identity != expected:
            raise ValueError(f"Evaluation cohort differs for method {method}")


def verify_frozen_config(
    config: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    resolved_config_path: str | Path,
    manifest: Mapping[str, Any],
) -> None:
    if file_sha256(checkpoint_path) != str(config["checkpoint_file_sha256"]):
        raise ValueError("Frozen checkpoint hash mismatch")
    if file_sha256(resolved_config_path) != str(config["config_file_sha256"]):
        raise ValueError("Frozen config hash mismatch")
    if canonical_sha256(manifest) != str(config["validation_manifest_sha256"]):
        raise ValueError("Frozen validation manifest hash mismatch")
