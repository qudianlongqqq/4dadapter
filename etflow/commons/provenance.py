"""Reproducible run metadata without mutating the Git working tree."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch


def _git(args: list[str], cwd: Path) -> str:
    command = ["git", "-c", f"safe.directory={cwd.as_posix()}", *args]
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def collect_run_provenance(
    *,
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    cache_path: Optional[str] = None,
    repo_root: Optional[str | Path] = None,
) -> dict[str, Any]:
    root = Path(repo_root or Path.cwd()).resolve()
    gpu_info = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            gpu_info.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"], root),
        "git_status": _git(["status", "--short"], root),
        "config_path": str(Path(config_path).resolve()) if config_path else None,
        "checkpoint_path": (
            str(Path(checkpoint_path).resolve()) if checkpoint_path else None
        ),
        "cache_path": str(Path(cache_path).resolve()) if cache_path else None,
        "cuda_available": torch.cuda.is_available(),
        "gpu_info": gpu_info,
        "torch_version": torch.__version__,
    }


def write_run_provenance(path: str | Path, **kwargs) -> dict[str, Any]:
    provenance = collect_run_provenance(**kwargs)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
    return provenance
