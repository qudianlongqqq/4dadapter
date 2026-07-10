"""Crash-safe state markers for long-running experiment stages."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def atomic_write_json(path: str | Path, payload: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def update_run_state(directory: str | Path, status: str, **details) -> dict:
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat(), **details}
    atomic_write_json(directory / "run_state.json", payload)
    for old_marker in ("STARTED", "COMPLETED", "FAILED"):
        old_path = directory / old_marker
        if old_path.exists():
            old_path.unlink()
    marker = {"started":"STARTED", "completed":"COMPLETED", "failed":"FAILED"}.get(status.lower())
    if marker:
        (directory / marker).touch(exist_ok=True)
    return payload
