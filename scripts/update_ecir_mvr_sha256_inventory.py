#!/usr/bin/env python
"""Refresh the existing MCVR inventory and add files from a completed stage."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    inventory = args.inventory.resolve()
    existing: list[str] = []
    if inventory.is_file():
        for line in inventory.read_text(encoding="utf-8").splitlines():
            if "  " in line:
                existing.append(line.split("  ", 1)[1])
    additions: list[Path] = []
    for configured in args.paths:
        additions.extend(configured.rglob("*") if configured.is_dir() else [configured])
    names = set(existing)
    for path in additions:
        if path.is_file() and path.resolve() != inventory:
            try:
                names.add(path.resolve().relative_to(ROOT).as_posix())
            except ValueError:
                names.add(str(path.resolve()))
    rows = []
    for name in sorted(names):
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file() and path.resolve() != inventory:
            rows.append(f"{_sha(path)}  {name}")
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"refreshed {len(rows)} entries in {args.inventory}")


if __name__ == "__main__":
    main()
