"""Frozen evaluation-cohort manifests for fair adapter comparisons."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .flexbond_cache_schema import x_init_sha256


EVAL_MANIFEST_VERSION = "1.0"
SAMPLE_PAYLOAD_PROVENANCE_VERSION = "1.0"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_content_sha256(manifest: Mapping[str, Any]) -> str:
    """Return a path-independent identity for a parsed evaluation manifest."""

    return _canonical_sha256(dict(manifest))


def data_manifest_row(data: Any) -> dict[str, Any]:
    return {
        "mol_id": str(data.source_mol_id),
        "sample_id": str(data.sample_id),
        "x_init_hash": str(data.x_init_hash),
        "num_rotatable_bonds": int(data.num_rotatable_bonds.item()),
    }


def build_eval_manifest(dataset: Iterable[Any]) -> dict[str, Any]:
    rows = [data_manifest_row(data) for data in dataset]
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Evaluation manifest contains duplicate sample_id values.")
    per_molecule = {}
    for row in rows:
        previous = per_molecule.setdefault(row["mol_id"], row["num_rotatable_bonds"])
        if previous != row["num_rotatable_bonds"]:
            raise ValueError("num_rotatable_bonds differs within one molecule cohort.")
    return {
        "manifest_version": EVAL_MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records": rows,
    }


def limit_manifest_molecules(manifest: dict[str, Any], limit: int) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("Molecule limit must be positive.")
    selected = []
    molecule_ids: set[str] = set()
    for row in manifest["records"]:
        mol_id = str(row["mol_id"])
        if mol_id in molecule_ids:
            selected.append(row)
        elif len(molecule_ids) < limit:
            molecule_ids.add(mol_id)
            selected.append(row)
    return {**manifest, "records": selected}


def load_eval_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if str(manifest.get("manifest_version")) != EVAL_MANIFEST_VERSION:
        raise ValueError("Unsupported or missing evaluation manifest version.")
    rows = manifest.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Evaluation manifest has no records.")
    required = {"mol_id", "sample_id", "x_init_hash", "num_rotatable_bonds"}
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Manifest row is missing fields: {sorted(missing)}.")
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation manifest contains duplicate sample_id values.")
    return manifest


def validate_dataset_against_manifest(dataset: Iterable[Any], manifest: dict) -> dict[str, Any]:
    by_id = {str(data.sample_id): data for data in dataset}
    expected_ids = {str(row["sample_id"]) for row in manifest["records"]}
    missing = sorted(expected_ids.difference(by_id))
    if missing:
        raise ValueError(f"Inference cache is missing manifest sample ids: {missing[:20]}.")
    for row in manifest["records"]:
        data = by_id[str(row["sample_id"])]
        actual = data_manifest_row(data)
        if actual != {
            "mol_id": str(row["mol_id"]),
            "sample_id": str(row["sample_id"]),
            "x_init_hash": str(row["x_init_hash"]),
            "num_rotatable_bonds": int(row["num_rotatable_bonds"]),
        }:
            raise ValueError(f"Manifest mismatch for sample {row['sample_id']!r}.")
    return by_id


def _ordered_manifest_rows(
    manifest: Mapping[str, Any], ordered_sample_ids: list[str]
) -> list[dict[str, Any]]:
    by_id = {str(row["sample_id"]): row for row in manifest["records"]}
    manifest_order = [str(row["sample_id"]) for row in manifest["records"]]
    if len(ordered_sample_ids) != len(set(ordered_sample_ids)):
        raise ValueError("Sample payload contains duplicate ordered sample IDs.")
    missing = [sample_id for sample_id in ordered_sample_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"Sample payload contains IDs outside the manifest: {missing[:20]}.")
    positions = [manifest_order.index(sample_id) for sample_id in ordered_sample_ids]
    if positions != sorted(positions):
        raise ValueError("Sample payload IDs do not preserve manifest order.")
    return [by_id[sample_id] for sample_id in ordered_sample_ids]


def _inference_cohort_sha256(
    ordered_sample_ids: list[str], inference_by_id: Mapping[str, Any]
) -> str:
    rows = []
    for sample_id in ordered_sample_ids:
        if sample_id not in inference_by_id:
            raise ValueError(f"Inference cache is missing sample {sample_id!r}.")
        rows.append(data_manifest_row(inference_by_id[sample_id]))
    return _canonical_sha256(rows)


def _validate_record_x_init_hash(record: Mapping[str, Any], expected_hash: str) -> None:
    if str(record.get("x_init_hash")) != str(expected_hash):
        raise ValueError(f"x_init_hash mismatch for sample {record.get('sample_id')!r}.")
    if record.get("x_init") is not None and record.get("atomic_numbers") is not None:
        actual_hash = x_init_sha256(record["x_init"], record["atomic_numbers"])
        if actual_hash != str(expected_hash):
            raise ValueError(
                f"x_init tensor hash mismatch for sample {record.get('sample_id')!r}."
            )


def build_sample_payload_provenance(
    *,
    manifest: Mapping[str, Any],
    manifest_path: str | Path,
    selected_manifest: Mapping[str, Any],
    split: str,
    inference_cache_path: str | Path,
    inference_by_id: Mapping[str, Any],
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the shared sample/evaluator manifest provenance contract.

    ``manifest`` is always the complete source manifest. ``selected_manifest``
    records the ordered sampling subset (for example ``--max_molecules``).
    Paths are retained for auditability, while hashes are the identities used
    for validation.
    """

    ordered_ids = [str(row["sample_id"]) for row in selected_manifest["records"]]
    record_ids = [str(record.get("sample_id")) for record in records]
    if record_ids != ordered_ids:
        raise ValueError("Sample records do not exactly preserve selected manifest order.")
    selected_rows = _ordered_manifest_rows(manifest, ordered_ids)
    molecule_ids = [str(row["mol_id"]) for row in selected_rows]
    x_init_hashes = [str(row["x_init_hash"]) for row in selected_rows]
    for record, row in zip(records, selected_rows):
        record_molecule = record.get("source_mol_id", record.get("mol_id"))
        if str(record_molecule) != str(row["mol_id"]):
            raise ValueError(f"mol_id mismatch for sample {row['sample_id']!r}.")
        _validate_record_x_init_hash(record, str(row["x_init_hash"]))
    manifest_hash = manifest_content_sha256(manifest)
    return {
        "schema_version": SAMPLE_PAYLOAD_PROVENANCE_VERSION,
        "manifest": {
            "identity": f"sha256:{manifest_hash}",
            "sha256": manifest_hash,
            "path": str(Path(manifest_path).expanduser().resolve()),
        },
        "split": str(split),
        "ordered_sample_ids": ordered_ids,
        "molecule_ids": molecule_ids,
        "x_init_hashes": x_init_hashes,
        "sample_count": len(records),
        "inference_cache": {
            "path": str(Path(inference_cache_path).expanduser().resolve()),
            "cohort_sha256": _inference_cohort_sha256(ordered_ids, inference_by_id),
            "sample_count": len(records),
        },
    }


def build_manifest_aware_sample_payload(
    *,
    records: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    manifest_path: str | Path,
    selected_manifest: Mapping[str, Any],
    split: str,
    inference_cache_path: str | Path,
    inference_by_id: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a sample payload using the one shared provenance implementation."""

    reserved = {"records", "manifest", "manifest_provenance"}
    overlap = reserved.intersection(extra or {})
    if overlap:
        raise ValueError(f"Sample payload extras use reserved keys: {sorted(overlap)}.")
    payload = dict(extra or {})
    payload.update(
        {
            "records": records,
            "manifest": dict(manifest),
            "manifest_provenance": build_sample_payload_provenance(
                manifest=manifest,
                manifest_path=manifest_path,
                selected_manifest=selected_manifest,
                split=split,
                inference_cache_path=inference_cache_path,
                inference_by_id=inference_by_id,
                records=records,
            ),
        }
    )
    return payload


def validate_sample_payload_provenance(
    payload: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_path: str | Path,
    split: str,
    inference_cache_path: str | Path,
    inference_by_id: Mapping[str, Any],
) -> None:
    """Validate every field in the shared manifest provenance contract."""

    if payload.get("partial") is True:
        raise ValueError("Partial sample payloads cannot be evaluated as final results.")
    provenance = payload.get("manifest_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Sample payload is missing manifest provenance.")
    if str(provenance.get("schema_version")) != SAMPLE_PAYLOAD_PROVENANCE_VERSION:
        raise ValueError("Sample payload has an unsupported provenance schema version.")
    requested_hash = manifest_content_sha256(manifest)
    embedded = payload.get("manifest")
    if not isinstance(embedded, Mapping) or manifest_content_sha256(embedded) != requested_hash:
        raise ValueError("Sample payload was not produced from the requested manifest.")
    manifest_identity = provenance.get("manifest")
    if not isinstance(manifest_identity, Mapping):
        raise ValueError("Sample payload is missing manifest identity metadata.")
    if (
        str(manifest_identity.get("sha256")) != requested_hash
        or str(manifest_identity.get("identity")) != f"sha256:{requested_hash}"
    ):
        raise ValueError("Sample payload was not produced from the requested manifest.")
    # manifest_path is deliberately not compared: the content hash is the
    # identity, and absolute/relative aliases must not create false mismatch.
    if not manifest_identity.get("path"):
        raise ValueError("Sample payload is missing its manifest path.")
    if str(provenance.get("split")) != str(split):
        raise ValueError("Sample payload split does not match the evaluator split.")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Sample payload records must be a list.")
    ordered_ids = provenance.get("ordered_sample_ids")
    molecule_ids = provenance.get("molecule_ids")
    x_init_hashes = provenance.get("x_init_hashes")
    if not all(isinstance(value, list) for value in (ordered_ids, molecule_ids, x_init_hashes)):
        raise ValueError("Sample payload is missing ordered cohort provenance.")
    ordered_ids = [str(value) for value in ordered_ids]
    selected_rows = _ordered_manifest_rows(manifest, ordered_ids)
    expected_molecules = [str(row["mol_id"]) for row in selected_rows]
    expected_hashes = [str(row["x_init_hash"]) for row in selected_rows]
    if [str(record.get("sample_id")) for record in records] != ordered_ids:
        raise ValueError("Sample record order differs from ordered sample IDs.")
    if [str(value) for value in molecule_ids] != expected_molecules:
        raise ValueError("Sample payload molecule IDs differ from the manifest.")
    if [str(value) for value in x_init_hashes] != expected_hashes:
        raise ValueError("Sample payload x_init hashes differ from the manifest.")
    if int(provenance.get("sample_count", -1)) != len(records):
        raise ValueError("Sample payload sample count is incorrect.")
    for record, row in zip(records, selected_rows):
        record_molecule = record.get("source_mol_id", record.get("mol_id"))
        if str(record_molecule) != str(row["mol_id"]):
            raise ValueError(f"mol_id mismatch for sample {row['sample_id']!r}.")
        _validate_record_x_init_hash(record, str(row["x_init_hash"]))
    cache = provenance.get("inference_cache")
    if not isinstance(cache, Mapping) or not cache.get("path"):
        raise ValueError("Sample payload is missing inference cache provenance.")
    expected_cache_hash = _inference_cohort_sha256(ordered_ids, inference_by_id)
    if str(cache.get("cohort_sha256")) != expected_cache_hash:
        raise ValueError("Sample payload inference cache provenance does not match.")
    if int(cache.get("sample_count", -1)) != len(records):
        raise ValueError("Sample payload inference cache sample count is incorrect.")
    # As with manifests, the requested paths are audit context only. The
    # content hashes above are the identities used for acceptance.


def write_eval_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
