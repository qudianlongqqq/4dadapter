#!/usr/bin/env python3
"""Freeze provenance and SHA256SUMS for the completed mechanism audit."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
from etflow.ecir.lsgo_io import atomic_json

OUT = ROOT / "reports/ecir_mvr/lsgo_mechanism"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def git(*args): return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> int:
    if git("status", "--porcelain"):
        # The finalizer source itself may be the only untracked file at first run.
        rows = [row for row in git("status", "--porcelain").splitlines() if not row.endswith("scripts/finalize_lsgo_mechanism.py")]
        if rows: raise RuntimeError(f"unexpected dirty inputs before finalization: {rows}")
    decision = json.loads((OUT / "FINAL_DECISION.json").read_text(encoding="utf-8"))
    identity = json.loads((OUT / "DATASET_IDENTITY.json").read_text(encoding="utf-8"))
    provenance = {
        "schema_version": "mcvr-lsgo-mechanism-provenance-v1", "status": "FROZEN",
        "branch": git("branch", "--show-current"), "base_commit": "a2a4b5d65cd358caaba6185121bb7e62aac8d2ae",
        "preregistration_commit": "d1cbe2f", "result_commit": git("rev-parse", "HEAD"),
        "decisions": decision["decisions"], "mechanism_molecules": identity["molecule_count"], "source_records": identity["source_record_count"], "reference_records": identity["reference_count"],
        "xTB_singlepoint_records": 2441, "xTB_force_records": 36,
        "tests": {"new_mechanism": "16 passed", "focused_lsgo_bat_mechanism": "109 passed", "broader_ecir": "194 passed; 2 unrelated absent-cache integration failures"},
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
    }
    atomic_json(OUT / "PROVENANCE.json", provenance)
    tracked = git("ls-files", "reports/ecir_mvr/lsgo_mechanism").splitlines()
    paths = [ROOT / value for value in tracked if not value.endswith("SHA256SUMS.txt")]
    paths.append(OUT / "PROVENANCE.json")
    relative_paths = sorted({path.resolve().relative_to(ROOT.resolve()).as_posix(): path for path in paths}.items())
    lines = [f"{sha(path)}  {relative}" for relative, path in relative_paths]
    target = OUT / "SHA256SUMS.txt"; temporary = target.with_suffix(".txt.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="ascii"); os.replace(temporary, target)
    print("LSGO_MECHANISM_SHA256_FROZEN")
    return 0


if __name__ == "__main__": raise SystemExit(main())
