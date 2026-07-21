#!/usr/bin/env python
"""Freeze a deterministic risk-covering Smoke100 manifest from validation only."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import numpy as np
import torch
from rdkit.Chem import AllChem

from etflow.ecir.external_refinement_baselines import ISOLATION, canonical_sha256, derive_total_charge, derive_unpaired_electrons
from etflow.ecir.v8_validation_cache import atomic_json, file_sha256, iter_prediction_records
from scripts.evaluate_ecir_mvr_v8_prediction_cache import _memberships


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    rows = list(iter_prediction_records(args.source_cache_manifest.resolve()))
    risks = {}
    required = []
    for row in rows:
        index = int(row["record_index"])
        record, item = row["record"], row["item"]
        mol = record["_formal_rdkit_mol"]
        membership = _memberships(item)
        risk = {
            **membership,
            "mmff_unsupported": not bool(AllChem.MMFFHasAllMoleculeParams(mol)),
            "charged": derive_total_charge(mol) != 0,
            "radical_open_shell": derive_unpaired_electrons(mol) != 0,
        }
        risks[index] = risk
        if risk["mmff_unsupported"] or risk["radical_open_shell"]:
            required.append(index)
    rng = np.random.default_rng(args.seed)
    selected = list(dict.fromkeys(required))
    cohort_targets = {
        "active_clash": 3,
        "charged": 12,
        "active_angle": 20,
        "ring_risk": 15,
        "high_flexibility": 20,
        "low_error_minimal_movement": 15,
    }
    for cohort, target in cohort_targets.items():
        candidates = [index for index in range(len(rows)) if risks[index][cohort] and index not in selected]
        rng.shuffle(candidates)
        selected.extend(candidates[:target])
    remaining = [index for index in range(len(rows)) if index not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: 100 - len(selected)])
    selected = selected[:100]
    payload = {
        "schema_version": "mcvr-external-refinement-smoke100-manifest-v1",
        "status": "FROZEN_BEFORE_FAST1000",
        "selection_seed": args.seed,
        "selection_rule": "risk_covering_then_seed43_uniform_validation_only",
        "record_count": len(selected),
        "record_indices": selected,
        "record_ids": [str(rows[index]["sample_id"]) for index in selected],
        "cohort_counts": {name: sum(int(risks[index].get(name, False)) for index in selected) for name in sorted(next(iter(risks.values())))},
        "source_cache_manifest_sha256": file_sha256(args.source_cache_manifest),
        **ISOLATION,
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
