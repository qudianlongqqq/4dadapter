"""Identity-bound, resumable prediction caches for MCVR V8 validation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch import Tensor


ISOLATION = {
    "formal_test_records_read": 0,
    "formal_test_assets_opened": False,
    "minimal_validity_target_test_used": False,
    "frozen_holdout_records_read": 0,
    "parameter_selection_from_formal_test": False,
}

# Frozen before the 10K parity run from two independent legacy-compatible
# batch=1/no-grad repeats (observed max 1.0180473327636719e-4 Angstrom).
CONTINUOUS_PARITY_ATOL = 1.1e-4
CONTINUOUS_PARITY_RTOL = 1.0e-3
METRIC_PARITY_ATOL = 1.0e-6
METRIC_PARITY_RTOL = 3.0e-4
PHASES = {
    "TRAINING",
    "FAST_VALIDATING",
    "FULL_PREDICTING",
    "FULL_EVALUATING",
    "FINALIZING",
    "COMPLETED",
    "FAILED_CLOSED",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(value: Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    descriptor = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode()
    return hashlib.sha256(descriptor + tensor.numpy().tobytes()).hexdigest()


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)


def atomic_torch_save(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(target)


def jsonable(value: Any) -> Any:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class ValidationStatus:
    path: Path
    status: str
    phase: str
    training_step: int
    validation_mode: str
    started_at: str

    @classmethod
    def start(
        cls,
        path: str | Path,
        *,
        phase: str,
        training_step: int,
        validation_mode: str,
    ) -> "ValidationStatus":
        if phase not in PHASES:
            raise ValueError(f"unknown validation phase: {phase}")
        tracker = cls(
            Path(path), phase, phase, int(training_step), str(validation_mode), utc_now()
        )
        tracker.update()
        return tracker

    def update(self, **updates: Any) -> None:
        phase = str(updates.pop("phase", self.phase))
        status = str(updates.pop("status", phase))
        if phase not in PHASES or status not in PHASES:
            raise ValueError(f"invalid validation status transition: {status}/{phase}")
        self.phase, self.status = phase, status
        payload = {
            "schema_version": "mcvr-v8-validation-live-status-v1",
            "status": status,
            "phase": phase,
            "training_step": self.training_step,
            "validation_mode": self.validation_mode,
            "current_validation_record": 0,
            "prediction_chunks_completed": 0,
            "evaluation_chunks_completed": 0,
            "records_per_second": 0.0,
            "elapsed_seconds": 0.0,
            "estimated_remaining_seconds": None,
            "last_update_time": utc_now(),
            "started_at": self.started_at,
            "error": None,
            **ISOLATION,
            **updates,
        }
        atomic_json(self.path, payload)

    def fail(self, error: BaseException | str, **updates: Any) -> None:
        self.update(
            status="FAILED_CLOSED",
            phase="FAILED_CLOSED",
            error=str(error),
            **updates,
        )


def prediction_identity(
    *,
    checkpoint_sha256: str,
    config_sha256: str,
    validation_sources_sha256: str,
    validation_targets_sha256: str,
    record_ids: Sequence[str],
    evaluator_semantics_sha256: str,
    safety_semantics_sha256: str,
    method: str = "V8 Full v1",
) -> dict[str, Any]:
    identity = {
        "schema_version": "mcvr-v8-prediction-cache-identity-v1",
        "method": method,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "validation_sources_sha256": validation_sources_sha256,
        "validation_targets_sha256": validation_targets_sha256,
        "record_identity_sha256": canonical_sha256(list(record_ids)),
        "record_count": len(record_ids),
        "evaluator_semantics_sha256": evaluator_semantics_sha256,
        "safety_semantics_sha256": safety_semantics_sha256,
        **ISOLATION,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def new_prediction_manifest(
    identity: Mapping[str, Any], *, chunk_size: int, output_dir: str | Path
) -> dict[str, Any]:
    return {
        "schema_version": "mcvr-v8-prediction-cache-manifest-v1",
        "status": "RUNNING",
        "identity": dict(identity),
        "chunk_size": int(chunk_size),
        "output_dir": str(Path(output_dir).resolve()),
        "chunks": [],
        "records_written": 0,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        **ISOLATION,
    }


def validate_manifest_identity(
    manifest: Mapping[str, Any], expected_identity: Mapping[str, Any]
) -> None:
    if manifest.get("schema_version") != "mcvr-v8-prediction-cache-manifest-v1":
        raise RuntimeError("prediction cache manifest schema changed")
    if manifest.get("identity") != dict(expected_identity):
        raise RuntimeError("prediction cache identity changed")
    for key, expected in ISOLATION.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"prediction cache isolation changed: {key}")


def completed_chunk_ranges(manifest: Mapping[str, Any]) -> set[tuple[int, int]]:
    return {
        (int(chunk["record_start"]), int(chunk["record_end"]))
        for chunk in manifest.get("chunks", [])
    }


def append_prediction_chunk(
    manifest_path: str | Path,
    manifest: dict[str, Any],
    *,
    record_start: int,
    record_end: int,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output = Path(manifest_path).parent
    name = f"chunk_{record_start:06d}_{record_end:06d}.pt"
    path = output / "chunks" / name
    payload = {
        "schema_version": "mcvr-v8-prediction-cache-chunk-v1",
        "identity_sha256": manifest["identity"]["identity_sha256"],
        "record_start": int(record_start),
        "record_end": int(record_end),
        "record_count": len(records),
        "records": list(records),
        **ISOLATION,
    }
    atomic_torch_save(path, payload)
    entry = {
        "path": str(path.relative_to(output)).replace("\\", "/"),
        "record_start": int(record_start),
        "record_end": int(record_end),
        "record_count": len(records),
        "sha256": file_sha256(path),
    }
    manifest["chunks"].append(entry)
    manifest["chunks"] = sorted(manifest["chunks"], key=lambda row: row["record_start"])
    manifest["records_written"] = sum(int(row["record_count"]) for row in manifest["chunks"])
    manifest["updated_at"] = utc_now()
    atomic_json(manifest_path, manifest)
    return entry


def iter_prediction_records(
    manifest_path: str | Path, *, require_completed: bool = True
) -> Iterator[Mapping[str, Any]]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if require_completed and manifest.get("status") != "COMPLETED":
        raise RuntimeError("prediction cache is incomplete")
    expected = 0
    for chunk in sorted(manifest.get("chunks", []), key=lambda row: row["record_start"]):
        chunk_path = path.parent / chunk["path"]
        if file_sha256(chunk_path) != chunk["sha256"]:
            raise RuntimeError(f"prediction chunk SHA changed: {chunk_path}")
        payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
        if payload.get("identity_sha256") != manifest["identity"]["identity_sha256"]:
            raise RuntimeError("prediction chunk identity changed")
        if int(payload["record_start"]) != expected:
            raise RuntimeError("prediction cache record range is not contiguous")
        for record in payload["records"]:
            yield record
        expected = int(payload["record_end"])
    if expected != int(manifest.get("records_written", -1)):
        raise RuntimeError("prediction cache record count changed")


def finish_prediction_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    expected = int(manifest["identity"]["record_count"])
    if int(manifest.get("records_written", -1)) != expected:
        raise RuntimeError("prediction cache cannot complete with missing records")
    manifest["status"] = "COMPLETED"
    manifest["completed_at"] = utc_now()
    manifest["updated_at"] = manifest["completed_at"]
    stable = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = canonical_sha256(stable)
    atomic_json(path, manifest)


def compare_prediction_records(
    left: Iterable[Mapping[str, Any]],
    right: Iterable[Mapping[str, Any]],
    *,
    atol: float = CONTINUOUS_PARITY_ATOL,
    rtol: float = CONTINUOUS_PARITY_RTOL,
) -> dict[str, Any]:
    max_absolute = 0.0
    max_relative = 0.0
    count = 0
    for count, (a, b) in enumerate(zip(left, right, strict=True), start=1):
        for key in ("sample_id", "molecule_id", "source_coordinate_sha256", "accepted"):
            if a[key] != b[key]:
                raise RuntimeError(f"prediction parity mismatch for {key} at record {count}")
        for key in ("raw_coordinates", "safe_coordinates"):
            first = torch.as_tensor(a[key])
            second = torch.as_tensor(b[key])
            difference = (first - second).abs()
            absolute = float(difference.max()) if difference.numel() else 0.0
            scale = second.abs().clamp_min(atol)
            relative = float((difference / scale).max()) if difference.numel() else 0.0
            max_absolute = max(max_absolute, absolute)
            max_relative = max(max_relative, relative)
            if not torch.allclose(first, second, atol=atol, rtol=rtol):
                raise RuntimeError(f"prediction coordinate parity failed: {key} record {count}")
        for key in ("rollback", "backtracking_decision"):
            if a.get(key) != b.get(key):
                raise RuntimeError(f"prediction parity mismatch for {key} at record {count}")
    return {
        "status": "PARITY_OK",
        "records": count,
        "continuous_atol": atol,
        "continuous_rtol": rtol,
        "max_absolute_difference": max_absolute,
        "max_relative_difference": max_relative,
        "discrete_bitwise_equal": True,
    }


def compare_metric_reports(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    atol: float = METRIC_PARITY_ATOL,
    rtol: float = METRIC_PARITY_RTOL,
) -> dict[str, Any]:
    if left.get("records") != right.get("records"):
        raise RuntimeError("metric parity record count changed")
    if left.get("rejection_reasons") != right.get("rejection_reasons"):
        raise RuntimeError("metric parity rejection reasons changed")
    differences = {}
    for key, expected in left["metrics"].items():
        if key not in right["metrics"]:
            raise RuntimeError(f"metric parity field missing: {key}")
        actual = float(right["metrics"][key])
        expected = float(expected)
        absolute = abs(actual - expected)
        relative = absolute / max(abs(expected), atol)
        differences[key] = {"absolute": absolute, "relative": relative}
        if absolute > atol + rtol * abs(expected):
            raise RuntimeError(f"metric parity tolerance exceeded: {key}")
    return {
        "status": "PARITY_OK",
        "metric_atol": atol,
        "metric_rtol": rtol,
        "max_absolute_difference": max(
            (row["absolute"] for row in differences.values()), default=0.0
        ),
        "max_relative_difference": max(
            (row["relative"] for row in differences.values()), default=0.0
        ),
        "differences": differences,
        "discrete_bitwise_equal": True,
    }
