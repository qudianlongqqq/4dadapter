"""Small audited I/O and identity helpers used only by the LSGO pilot."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor


FORBIDDEN_PATH_TOKENS = (
    "formal_test", "formal-test", "/test/", "frozen_holdout",
    "frozen-holdout", "/holdout/", "/real_sources/val.parquet",
)


def guarded_train_path(path: str | Path) -> Path:
    result = Path(path)
    normalized = str(result.resolve()).replace("\\", "/").lower()
    if any(token in normalized for token in FORBIDDEN_PATH_TOKENS):
        raise RuntimeError(f"forbidden non-train path: {result}")
    return result


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with guarded_train_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def atomic_torch_save(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def center_coordinates(values: Tensor) -> Tensor:
    values = torch.as_tensor(values)
    if values.ndim != 2 or values.size(-1) != 3 or values.size(0) < 1:
        raise ValueError("coordinates must be [N,3] and non-empty")
    if not torch.isfinite(values).all():
        raise ValueError("coordinates contain NaN/Inf")
    return values - values.mean(dim=0, keepdim=True)


def validate_record_identity(record: Mapping[str, Any]) -> None:
    atomic = torch.as_tensor(record["atomic_numbers"])
    if not torch.equal(atomic, torch.as_tensor(record["x_init_atomic_numbers"])):
        raise RuntimeError("Source atom identity changed")
    if not torch.equal(atomic, torch.as_tensor(record["x_ref_atomic_numbers"])):
        raise RuntimeError("Reference atom identity changed")
    if not (
        record["topology_signature"]
        == record["x_init_topology_signature"]
        == record["x_ref_topology_signature"]
    ):
        raise RuntimeError("topology identity changed")


def nearest_reference_metrics(coordinates: Tensor, references: Tensor) -> tuple[int, float]:
    coordinates = torch.as_tensor(coordinates)
    references = torch.as_tensor(references, device=coordinates.device, dtype=coordinates.dtype)
    if references.ndim != 3 or references.size(0) < 1:
        raise ValueError("references must be [M,N,3]")
    if coordinates.shape != references.shape[1:]:
        raise ValueError("coordinate/reference atom shapes differ")
    source = coordinates - coordinates.mean(dim=0, keepdim=True)
    targets = references - references.mean(dim=1, keepdim=True)
    covariance = torch.einsum("ni,mnj->mij", source, targets)
    left, _, right_t = torch.linalg.svd(covariance)
    determinant = torch.linalg.det(left @ right_t)
    corrected_left = left.clone()
    corrected_left[:, :, -1] *= determinant[:, None]
    rotations = corrected_left @ right_t
    aligned = torch.einsum("ni,mij->mnj", source, rotations)
    values = torch.sqrt((aligned - targets).square().sum(-1).mean(-1))
    return int(torch.argmin(values)), float(values.min())
