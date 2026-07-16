#!/usr/bin/env python
"""Write deterministic SHA256 inventory for one completed MCVR stage."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

from etflow.ecir.audit import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    files = []
    for configured in args.paths:
        if configured.is_dir():
            files.extend(path for path in configured.rglob("*") if path.is_file())
        elif configured.is_file():
            files.append(configured)
    files = sorted({path.resolve() for path in files if path.resolve() != output})
    lines = []
    for path in files:
        try:
            relative = path.relative_to(ROOT).as_posix()
        except ValueError:
            relative = str(path)
        lines.append(f"{file_sha256(path)}  {relative}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} entries to {args.output}")


if __name__ == "__main__":
    main()
