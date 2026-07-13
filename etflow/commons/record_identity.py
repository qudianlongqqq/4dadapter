"""Dependency-free helpers for persisted upstream record identities."""

from __future__ import annotations

from typing import Any, Mapping


def _record_field(record: Any, key: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(key)
    getter = getattr(record, "get", None)
    if callable(getter):
        return getter(key, None)
    return getattr(record, key, None)


def source_record_identity(record: Any) -> str:
    """Return the persisted upstream-record identity, never a cache filename."""

    for key in (
        "source_record_id",
        "source_mol_id",
        "molecule_id",
        "id",
        "mol_id",
    ):
        value = _record_field(record, key)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError("Record has no persisted source-record identity.")
