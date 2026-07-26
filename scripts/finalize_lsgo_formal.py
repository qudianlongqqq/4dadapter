#!/usr/bin/env python3
"""Freeze provenance and checksums for completed LSGO-BA formal training."""

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

OUT = ROOT / "reports/ecir_mvr/lsgo_formal"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> int:
    generated = {
        "reports/ecir_mvr/lsgo_formal/PROVENANCE.json",
        "reports/ecir_mvr/lsgo_formal/SHA256SUMS.txt",
    }
    dirty = [
        row
        for row in git("status", "--porcelain").splitlines()
        if row[3:].replace("\\", "/") not in generated
        and not row.endswith("scripts/finalize_lsgo_formal.py")
    ]
    if dirty:
        raise RuntimeError(f"unexpected dirty inputs before finalization: {dirty}")
    status = json.loads((OUT / "FINAL_FORMAL_STATUS.json").read_text(encoding="utf-8"))
    freeze = json.loads((OUT / "CHECKPOINT_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    identity = json.loads((OUT / "DATASET_IDENTITY.json").read_text(encoding="utf-8"))
    if status["status"] != "READY_FOR_FINAL_FROZEN_TEST" or freeze["status"] != "FROZEN":
        raise RuntimeError("formal result is not ready to freeze")
    provenance = {
        "schema_version": "mcvr-lsgo-ba-formal-provenance-v1", "status": "FROZEN",
        "branch": git("branch", "--show-current"), "base_commit": "e24138bfa7280ead0e7003d8fa96eefbdf9717bb",
        "preregistration_head": json.loads((OUT / "PREREGISTRATION.json").read_text(encoding="utf-8"))["head"],
        "result_commit": git("rev-parse", "HEAD"), "formal_status": status["status"],
        "dataset_identity_sha256": identity["identity_sha256"], "config_sha256": freeze["config_sha256"],
        "checkpoint_sha256": {str(row["seed"]): row["sha256"] for row in freeze["checkpoints"]},
        "optimizer_steps_per_seed": 12500, "effective_batch": 64, "exposures_per_seed": 800000,
        "parameter_count": 473674, "tests": "103 passed (formal + LSGO + BAT)",
        "external_checkpoint_selection": False, "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
    }
    atomic_json(OUT / "PROVENANCE.json", provenance)
    tracked = git("ls-files", "reports/ecir_mvr/lsgo_formal").splitlines()
    paths = [ROOT / value for value in tracked if not value.endswith("SHA256SUMS.txt")]
    paths.extend([OUT / "PROVENANCE.json", ROOT / "configs/ecir_mvr_lsgo_ba_formal_large.yaml"])
    paths.extend(Path(row["path"]) for row in freeze["checkpoints"])
    unique = sorted({path.resolve().relative_to(ROOT.resolve()).as_posix(): path for path in paths}.items())
    lines = [f"{sha(path)}  {relative}" for relative, path in unique]
    target = OUT / "SHA256SUMS.txt"; temporary = target.with_suffix(".txt.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="ascii"); os.replace(temporary, target)
    print("LSGO_BA_FORMAL_SHA256_FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
