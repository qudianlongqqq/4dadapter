#!/usr/bin/env python3
"""Freeze provenance and checksums for LSGO Bond minimality confirmation."""

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

OUT = ROOT / "reports/ecir_mvr/lsgo_bond_confirm"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> int:
    dirty = [
        row for row in git("status", "--porcelain").splitlines()
        if not row.endswith("scripts/finalize_lsgo_bond_confirm.py")
    ]
    if dirty:
        raise RuntimeError(f"unexpected dirty inputs before finalization: {dirty}")
    decision = json.loads((OUT / "FINAL_DECISION.json").read_text(encoding="utf-8"))
    identity = json.loads((OUT / "DATASET_IDENTITY.json").read_text(encoding="utf-8"))
    provenance = {
        "schema_version": "mcvr-lsgo-bond-confirm-provenance-v1",
        "status": "FROZEN",
        "branch": git("branch", "--show-current"),
        "base_commit": "02d140036257d6e2c15fe18aa18aef7ed24ca33f",
        "preregistration_commit": "87f7fdc47324923aedc6e75ba61abf6138a5d6a0",
        "comparison_implementation_commit": "e35269b",
        "coordinate_freeze_commit": "530c652",
        "external_evaluation_commit": "318e98d",
        "decision_report_commit": git("rev-parse", "HEAD"),
        "decision": decision["decision"],
        "molecules": identity["molecule_count"],
        "source_records": identity["source_record_count"],
        "reference_records": identity["reference_count"],
        "posebusters_evaluations": 6000,
        "xtb_single_point_evaluations": 6000,
        "new_model_training": False,
        "tests": {
            "bond_confirm_lsgo_bat": "103 passed; 1 pre-finalization SHA test skipped",
            "historical_mechanism_logic": "15 passed",
            "historical_mechanism_checksum": "1 checkout-byte SHA mismatch at FINAL_SUMMARY.md; frozen logic and BA equivalence passed; historical artifact not modified",
        },
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
    }
    atomic_json(OUT / "PROVENANCE.json", provenance)
    tracked = git("ls-files", "reports/ecir_mvr/lsgo_bond_confirm").splitlines()
    paths = [ROOT / value for value in tracked if not value.endswith("SHA256SUMS.txt")]
    paths.append(OUT / "PROVENANCE.json")
    unique = sorted({path.resolve().relative_to(ROOT.resolve()).as_posix(): path for path in paths}.items())
    lines = [f"{sha(path)}  {relative}" for relative, path in unique]
    target = OUT / "SHA256SUMS.txt"
    temporary = target.with_suffix(".txt.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="ascii")
    os.replace(temporary, target)
    print("LSGO_BOND_CONFIRM_SHA256_FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
