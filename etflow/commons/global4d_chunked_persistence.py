"""Crash-safe, append-only persistence for Global Coupled 4D sampling."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from etflow.commons.global_coupled_4d_sampling import (
    atomic_torch_save,
    file_sha256,
)


CHUNK_FORMAT_VERSION = "global4d-chunk-v1"
STATE_FORMAT_VERSION = "global4d-sampling-state-v2"
CHUNK_PATTERN = re.compile(r"^chunk_(\d{6})\.pt$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ordered_sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    return canonical_sha256([str(value) for value in sample_ids])


def run_identity_sha256(run_identity: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(run_identity))


def _hash_value(digest: "hashlib._Hash", value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: str(item)):
            _hash_value(digest, str(key))
            _hash_value(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for item in value:
            _hash_value(digest, item)
        return
    if value is None or isinstance(value, (str, int, float, bool)):
        digest.update(
            json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
        )
        digest.update(b"\0")
        return
    raise TypeError(f"Unsupported chunk value for hashing: {type(value).__name__}")


def records_content_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, list(records))
    return digest.hexdigest()


def _ensure_safe_root(root: Path, *, create: bool) -> Path:
    root = Path(root)
    if root.exists():
        if root.is_symlink():
            raise ValueError(f"Chunk directory must not be a symlink: {root}")
        if not root.is_dir():
            raise ValueError(f"Chunk path is not a directory: {root}")
    elif create:
        root.mkdir(parents=True, exist_ok=False)
    return root


def chunk_path(root: Path, index: int) -> Path:
    if int(index) < 0:
        raise ValueError("Chunk index must be non-negative")
    root = _ensure_safe_root(root, create=True)
    path = root / f"chunk_{int(index):06d}.pt"
    if path.parent != root or path.name != f"chunk_{int(index):06d}.pt":
        raise ValueError("Unsafe chunk path")
    if path.is_symlink():
        raise ValueError(f"Chunk file must not be a symlink: {path}")
    return path


def _expected_ids(rows: Sequence[Mapping[str, Any]], start: int, end: int) -> list[str]:
    if start < 0 or end < start or end > len(rows):
        raise ValueError(f"Invalid chunk bounds [{start}, {end})")
    return [str(row["sample_id"]) for row in rows[start:end]]


def build_chunk_payload(
    *,
    records: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    chunk_index: int,
    start: int,
    run_identity: Mapping[str, Any],
    previous_chunk_sha256: str | None,
    auxiliary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    end = start + len(records)
    ids = [str(record.get("sample_id")) for record in records]
    expected = _expected_ids(selected_rows, start, end)
    if not records:
        raise ValueError("Refusing to create an empty chunk")
    if ids != expected:
        raise ValueError("Chunk records are not the exact contiguous manifest interval")
    if len(ids) != len(set(ids)):
        raise ValueError("Chunk contains duplicate sample IDs")
    identity = dict(run_identity)
    return {
        "format_version": CHUNK_FORMAT_VERSION,
        "chunk_index": int(chunk_index),
        "start": int(start),
        "end": int(end),
        "sample_count": len(records),
        "ordered_sample_ids": ids,
        "ordered_sample_ids_sha256": ordered_sample_ids_sha256(ids),
        "records_content_sha256": records_content_sha256(records),
        "previous_chunk_sha256": previous_chunk_sha256,
        "run_identity": identity,
        "run_identity_sha256": run_identity_sha256(identity),
        "records": list(records),
        "auxiliary": dict(auxiliary or {}),
    }


def validate_chunk_payload(
    payload: Mapping[str, Any],
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    expected_index: int,
    expected_start: int,
    run_identity: Mapping[str, Any],
    previous_chunk_sha256: str | None,
) -> None:
    if str(payload.get("format_version")) != CHUNK_FORMAT_VERSION:
        raise ValueError("Unsupported or missing Global4D chunk format")
    if int(payload.get("chunk_index", -1)) != expected_index:
        raise ValueError("Chunk index is not contiguous")
    records = payload.get("records")
    ids = payload.get("ordered_sample_ids")
    if not isinstance(records, list) or not isinstance(ids, list):
        raise ValueError("Chunk records or ordered IDs are missing")
    end = expected_start + len(records)
    if int(payload.get("start", -1)) != expected_start or int(
        payload.get("end", -1)
    ) != end:
        raise ValueError("Chunk bounds are not contiguous")
    if int(payload.get("sample_count", -1)) != len(records):
        raise ValueError("Chunk sample count is incorrect")
    expected_ids = _expected_ids(selected_rows, expected_start, end)
    record_ids = [str(record.get("sample_id")) for record in records]
    ids = [str(value) for value in ids]
    if ids != expected_ids or record_ids != expected_ids:
        raise ValueError("Chunk IDs do not match the exact manifest interval")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Chunk contains duplicate records")
    if str(payload.get("ordered_sample_ids_sha256")) != ordered_sample_ids_sha256(ids):
        raise ValueError("Chunk ordered sample ID hash mismatch")
    if str(payload.get("records_content_sha256")) != records_content_sha256(records):
        raise ValueError("Chunk record content hash mismatch")
    if payload.get("previous_chunk_sha256") != previous_chunk_sha256:
        raise ValueError("Chunk hash chain mismatch")
    if dict(payload.get("run_identity") or {}) != dict(run_identity):
        raise ValueError("Chunk belongs to a different sampling command")
    if str(payload.get("run_identity_sha256")) != run_identity_sha256(run_identity):
        raise ValueError("Chunk run identity hash mismatch")


@dataclass
class ChunkScan:
    records: list[dict[str, Any]]
    trajectory: list[dict[str, Any]]
    profile_rows: list[dict[str, Any]]
    chunk_hashes: list[str]
    chunk_paths: list[Path]
    total_bytes: int
    scan_seconds: float

    @property
    def completed_count(self) -> int:
        return len(self.records)

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_paths)

    @property
    def latest_chunk_sha256(self) -> str | None:
        return self.chunk_hashes[-1] if self.chunk_hashes else None


def scan_chunks(
    root: Path,
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    run_identity: Mapping[str, Any],
) -> ChunkScan:
    started = time.perf_counter()
    root = _ensure_safe_root(root, create=True)
    paths = []
    for path in root.iterdir():
        if path.is_symlink():
            raise ValueError(f"Chunk directory contains a symlink: {path}")
        match = CHUNK_PATTERN.fullmatch(path.name)
        if match:
            if not path.is_file():
                raise ValueError(f"Chunk entry is not a regular file: {path}")
            paths.append((int(match.group(1)), path))
    paths.sort(key=lambda item: item[0])
    records: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    chunk_hashes: list[str] = []
    chunk_paths: list[Path] = []
    previous_hash = None
    for expected_index, (actual_index, path) in enumerate(paths):
        if actual_index != expected_index:
            raise ValueError("Chunk numbering is missing, duplicated, or out of order")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_chunk_payload(
            payload,
            selected_rows=selected_rows,
            expected_index=expected_index,
            expected_start=len(records),
            run_identity=run_identity,
            previous_chunk_sha256=previous_hash,
        )
        chunk_records = list(payload["records"])
        records.extend(chunk_records)
        auxiliary = payload.get("auxiliary") or {}
        trajectory.extend(list(auxiliary.get("trajectory", [])))
        profile_rows.extend(list(auxiliary.get("profile_rows", [])))
        previous_hash = file_sha256(path)
        chunk_hashes.append(previous_hash)
        chunk_paths.append(path)
    ids = [str(record.get("sample_id")) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Chunk sequence contains duplicate sample IDs")
    return ChunkScan(
        records=records,
        trajectory=trajectory,
        profile_rows=profile_rows,
        chunk_hashes=chunk_hashes,
        chunk_paths=chunk_paths,
        total_bytes=sum(path.stat().st_size for path in chunk_paths),
        scan_seconds=time.perf_counter() - started,
    )


def write_chunk(
    root: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    chunk_index: int,
    start: int,
    run_identity: Mapping[str, Any],
    previous_chunk_sha256: str | None,
    auxiliary: Mapping[str, Any] | None = None,
) -> tuple[Path, str, bool]:
    path = chunk_path(root, chunk_index)
    payload = build_chunk_payload(
        records=records,
        selected_rows=selected_rows,
        chunk_index=chunk_index,
        start=start,
        run_identity=run_identity,
        previous_chunk_sha256=previous_chunk_sha256,
        auxiliary=auxiliary,
    )
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=False)
        validate_chunk_payload(
            existing,
            selected_rows=selected_rows,
            expected_index=chunk_index,
            expected_start=start,
            run_identity=run_identity,
            previous_chunk_sha256=previous_chunk_sha256,
        )
        if str(existing["records_content_sha256"]) != str(
            payload["records_content_sha256"]
        ):
            raise ValueError("Refusing to overwrite a completed chunk")
        return path, file_sha256(path), False
    atomic_torch_save(payload, path)
    return path, file_sha256(path), True


def convert_records_to_chunks(
    records: Sequence[Mapping[str, Any]],
    root: Path,
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    run_identity: Mapping[str, Any],
    save_every_records: int,
) -> ChunkScan:
    """Idempotently append a validated record prefix as fixed-size chunks."""

    if save_every_records < 1:
        raise ValueError("save_every_records must be positive")
    scan = scan_chunks(
        root, selected_rows=selected_rows, run_identity=run_identity
    )
    existing_ids = [str(row.get("sample_id")) for row in scan.records]
    record_ids = [str(row.get("sample_id")) for row in records]
    if existing_ids != record_ids[: len(existing_ids)]:
        raise ValueError("Existing chunks are not a prefix of conversion records")
    offset = scan.completed_count
    chunk_index = scan.chunk_count
    previous_hash = scan.latest_chunk_sha256
    while offset < len(records):
        end = min(offset + save_every_records, len(records))
        _, previous_hash, _ = write_chunk(
            root,
            records=records[offset:end],
            selected_rows=selected_rows,
            chunk_index=chunk_index,
            start=offset,
            run_identity=run_identity,
            previous_chunk_sha256=previous_hash,
        )
        offset = end
        chunk_index += 1
    result = scan_chunks(
        root, selected_rows=selected_rows, run_identity=run_identity
    )
    if [str(row.get("sample_id")) for row in result.records] != record_ids:
        raise ValueError("Converted chunks do not exactly reproduce input records")
    return result


def compact_sampling_state(
    *,
    status: str,
    completed_count: int,
    total_count: int,
    completed_chunk_count: int,
    current_chunk_size: int,
    save_every_records: int,
    run_identity: Mapping[str, Any],
    ordered_sample_ids_hash: str,
    output: Path,
    device: str,
    started_at: str,
    latest_chunk_sha256: str | None,
    average_seconds_per_record: float = 0.0,
    eta_seconds: float | None = None,
    latest_error: Mapping[str, Any] | None = None,
    total_seconds: float | None = None,
    legacy_source_sha256: str | None = None,
) -> dict[str, Any]:
    identity = dict(run_identity)
    state = {
        "format_version": STATE_FORMAT_VERSION,
        "partial_format": "chunked",
        "status": str(status).upper(),
        "completed_count": int(completed_count),
        "total_count": int(total_count),
        "completed_chunk_count": int(completed_chunk_count),
        "current_chunk_size": int(current_chunk_size),
        "next_chunk_index": int(completed_chunk_count),
        "save_every_records": int(save_every_records),
        "checkpoint_inference_sha256": identity.get("checkpoint_inference_sha256"),
        "config_sha256": identity.get("config_sha256"),
        "manifest_sha256": identity.get("manifest_sha256"),
        "ordered_sample_ids_sha256": str(ordered_sample_ids_hash),
        "run_identity_sha256": run_identity_sha256(identity),
        "latest_chunk_sha256": latest_chunk_sha256,
        "output": str(Path(output).resolve()),
        "device": str(device),
        "started_at": str(started_at),
        "updated_at": utc_now(),
        "latest_error": dict(latest_error) if latest_error else None,
        "average_seconds_per_record": float(average_seconds_per_record),
        "eta_seconds": eta_seconds,
    }
    if total_seconds is not None:
        state["total_seconds"] = float(total_seconds)
    if legacy_source_sha256 is not None:
        state["legacy_source_sha256"] = str(legacy_source_sha256)
    return state


def validate_compact_state(
    state: Mapping[str, Any],
    *,
    scan: ChunkScan,
    run_identity: Mapping[str, Any],
    ordered_sample_ids_hash: str,
    save_every_records: int,
) -> None:
    if str(state.get("format_version")) != STATE_FORMAT_VERSION:
        raise ValueError("Chunked sampling state has an unsupported format")
    if str(state.get("partial_format")) != "chunked":
        raise ValueError("Sampling state is not chunked")
    if str(state.get("run_identity_sha256")) != run_identity_sha256(run_identity):
        raise ValueError("Sampling state belongs to a different command")
    if str(state.get("ordered_sample_ids_sha256")) != ordered_sample_ids_hash:
        raise ValueError("Sampling state manifest order hash mismatch")
    if int(state.get("save_every_records", -1)) != int(save_every_records):
        raise ValueError("Cannot change chunk size while resuming existing chunks")
    state_chunks = int(state.get("completed_chunk_count", -1))
    state_count = int(state.get("completed_count", -1))
    if state_chunks > scan.chunk_count or state_count > scan.completed_count:
        raise ValueError("Sampling state is ahead of durable chunks")
    if state_chunks:
        if state.get("latest_chunk_sha256") != scan.chunk_hashes[state_chunks - 1]:
            raise ValueError("Sampling state latest chunk hash mismatch")
    elif state.get("latest_chunk_sha256") not in (None, ""):
        raise ValueError("Sampling state has a hash without a completed chunk")
    if state_chunks == scan.chunk_count and state_count != scan.completed_count:
        raise ValueError("Sampling state count disagrees with durable chunks")


def cleanup_chunks(root: Path) -> None:
    root = _ensure_safe_root(root, create=False)
    for path in root.iterdir():
        if path.is_symlink():
            raise ValueError(f"Refusing to clean symlink: {path}")
        if path.is_file() and (
            CHUNK_PATTERN.fullmatch(path.name) or ".tmp." in path.name
        ):
            path.unlink()
        elif path.is_dir():
            raise ValueError(f"Unexpected directory inside chunk root: {path}")
    if root.exists() and not any(root.iterdir()):
        root.rmdir()
