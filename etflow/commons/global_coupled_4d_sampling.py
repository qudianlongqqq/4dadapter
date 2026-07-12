"""Crash-safe sampling state and checkpoint identities for Global Coupled 4D."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: repr(item),
    ).encode("utf-8")


def checkpoint_inference_identity(path: str | Path) -> dict[str, Any]:
    """Hash all inference-relevant state, independently of checkpoint filename."""

    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Checkpoint has no state_dict mapping: {checkpoint_path}")
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = torch.as_tensor(state_dict[name]).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_stable_json(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes())
    global_step = int(payload.get("global_step", 0))
    hyper_parameters = payload.get("hyper_parameters", {})
    digest.update(_stable_json({
        "global_step": global_step,
        "hyper_parameters": hyper_parameters,
    }))
    return {
        "path": str(checkpoint_path.expanduser().resolve()),
        "file_sha256": file_sha256(checkpoint_path),
        "inference_sha256": digest.hexdigest(),
        "global_step": global_step,
    }


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def atomic_json_save(payload: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def configure_cpu_threads(threads: int) -> dict[str, Any]:
    if threads < 1:
        raise ValueError("CPU thread count must be positive")
    torch.set_num_threads(threads)
    interop_requested = min(threads, 4)
    try:
        torch.set_num_interop_threads(interop_requested)
    except RuntimeError:
        pass
    return {
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return value
