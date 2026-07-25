#!/usr/bin/env python3
"""Freeze the minimum eligible BAT mechanism before prospective coordinates."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
from etflow.ecir.lsgo_io import atomic_json, file_sha256

OUT = ROOT / "reports/ecir_mvr/bat_refinement"
CONFIG_PATH = ROOT / "configs/ecir_mvr_bat_refinement.yaml"


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    steric = json.loads((OUT / "manifests/STERIC_INTERNAL.json").read_text(encoding="utf-8"))
    torsion = json.loads((OUT / "manifests/TORSION_INTERNAL.json").read_text(encoding="utf-8"))
    identity = json.loads((OUT / "DATASET_IDENTITY.json").read_text(encoding="utf-8"))
    if steric["decision"] != "STERIC_INTERNAL_GO" or torsion["decision"] != "TORSION_NO_GO":
        raise RuntimeError("unexpected internal decision chain")
    if identity["formal_test_records_read"] or identity["frozen_holdout_records_read"]:
        raise RuntimeError("protected split access")
    ba_manifest_path = Path(config["ba_anchor"]["manifest"])
    ba_manifest = json.loads(ba_manifest_path.read_text(encoding="utf-8"))
    rows = []
    for seed in config["ba_seeds"]:
        source = next(row for row in ba_manifest["checkpoints"] if row["variant"] == "B" and int(row["seed"]) == int(seed))
        if file_sha256(Path(source["path"])) != source["sha256"]:
            raise RuntimeError("frozen BA checkpoint SHA mismatch")
        rows.append(source)
    prereg = {
        "schema_version": "mcvr-bat-formal-freeze-v1", "status": "FROZEN",
        "branch": git("branch", "--show-current"), "head_before_freeze_commit": git("rev-parse", "HEAD"),
        "config_path": str(CONFIG_PATH), "config_sha256": file_sha256(CONFIG_PATH),
        "dataset_identity_sha256": file_sha256(OUT / "DATASET_IDENTITY.json"),
        "internal_steric_sha256": file_sha256(OUT / "manifests/STERIC_INTERNAL.json"),
        "internal_torsion_sha256": file_sha256(OUT / "manifests/TORSION_INTERNAL.json"),
        "eligible_variant": "BA+C", "selection_rule": "minimum validated mechanism",
        "ineligible_variants": {"BA+T": "TORSION_NO_GO", "BAT+C": "TORSION_NO_GO"},
        "trust_rms_angstrom": .003, "atom_cap_angstrom": .03, "steps": 1,
        "steric": config["steric"], "ba_seeds": config["ba_seeds"],
        "external_conditions": ["Source"] + [name for seed in config["ba_seeds"] for name in (f"BA_seed{seed}", f"BA+C_seed{seed}")],
        "no_external_selection": True, "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
    }
    atomic_json(OUT / "PREREGISTRATION.json", prereg)
    atomic_text(OUT / "PREREGISTRATION.md", f"""# BAT formal preregistration

Status: **FROZEN**

- Eligible prospective candidate: **BA+C** (minimum validated mechanism).
- BA remains the exact three-seed frozen LSGO-B anchor; C is analytic and has no trainable parameters.
- BA+T and BAT+C are stopped because `TORSION_NO_GO` failed Source selectivity.
- One direct-gradient step; graph RMS ≤0.003 Å; atom cap ≤0.03 Å.
- Fresh external conditions are frozen as Source plus paired BA and BA+C for seeds 173/181/193.
- Coordinates will be frozen once before the first PB/xTB access. No post-external selection or retuning is allowed.

Formal test reads=0; frozen holdout reads=0.
""")
    checkpoint = {
        "schema_version": "mcvr-bat-checkpoint-freeze-v1", "status": "FROZEN",
        "formal_variant": "BA+C", "new_trainable_checkpoint": None,
        "reason": "C is analytic; Torsion stopped at internal gate", "ba_checkpoints": rows,
        "ba_manifest_path": str(ba_manifest_path), "ba_manifest_sha256": file_sha256(ba_manifest_path),
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
    }
    atomic_json(OUT / "CHECKPOINT_FREEZE_MANIFEST.json", checkpoint)
    atomic_text(OUT / "FORMAL_TRAINING_SUMMARY.md", """# Formal training summary

Final formal variant: **BA+C**.

No new formal training is required: BA uses the already frozen 2,500-step LSGO-B checkpoints for seeds 173/181/193 (473,674 parameters each), and C is an analytic steric barrier with zero trainable parameters. The 83,337-parameter torsion pilot heads are retained as failed internal artifacts and are not formal checkpoints.

Formal added steps: 0. Formal added parameters: 0. The original frozen BA training identity and checkpoint SHA values are recorded in `CHECKPOINT_FREEZE_MANIFEST.json`.
""")
    print("BAT_FORMAL_BA_C_FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
