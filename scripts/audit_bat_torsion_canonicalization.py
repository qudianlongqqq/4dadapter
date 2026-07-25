#!/usr/bin/env python3
"""Exhaustive TRAIN/DEV torsion-label canonicalization audit."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
import yaml
from rdkit import Chem

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
from etflow.ecir.audit import field
from etflow.ecir.bat_refinement import canonical_rotatable_torsions, circular_difference, dihedral_angles
from etflow.ecir.lsgo_io import atomic_json, file_sha256
from etflow.ecir.target_building import _record_to_rdkit_mapping

OUT = ROOT / "reports/ecir_mvr/bat_refinement"
CONFIG_PATH = ROOT / "configs/ecir_mvr_bat_refinement.yaml"


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def is_amide_like(molecule, left, right):
    first, second = molecule.GetAtomWithIdx(left), molecule.GetAtomWithIdx(right)
    for carbon, nitrogen in ((first, second), (second, first)):
        if carbon.GetAtomicNum() != 6 or nitrogen.GetAtomicNum() != 7:
            continue
        if any(
            bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetOtherAtom(carbon).GetAtomicNum() in (8, 16)
            for bond in carbon.GetBonds() if bond.GetOtherAtomIdx(carbon.GetIdx()) != nitrogen.GetIdx()
        ):
            return True
    return False


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    identity = json.loads((OUT / "DATASET_IDENTITY.json").read_text(encoding="utf-8"))
    if identity["formal_test_records_read"] or identity["frozen_holdout_records_read"]:
        raise RuntimeError("protected split access")
    path = Path(config["dataset"]["training_compact"])
    if file_sha256(path) != config["dataset"]["training_compact_sha256"]:
        raise RuntimeError("training compact SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    totals = {
        "molecules": 0, "raw_rotors": 0, "canonical_rotors": 0, "zero_rotor_molecules": 0,
        "symmetric_terminal_rotors": 0, "duplicate_central_bonds": 0, "invalid_indices": 0,
        "ring_bonds": 0, "non_single_bonds": 0, "raw_amide_like": 0, "amide_like_included": 0,
        "nonfinite_angles": 0, "range_failures": 0, "reversal_failures": 0,
        "determinism_failures": 0, "source_angles": 0, "reference_angles": 0,
    }
    by_partition = {}
    maximum_reversal_error = 0.0
    for index, item in enumerate(payload["items"]):
        partition = str(item["partition"])
        local = by_partition.setdefault(partition, {"molecules": 0, "rotors": 0, "references": 0})
        local["molecules"] += 1; local["references"] += len(item["references"])
        record = item["record"]
        first = canonical_rotatable_torsions(record)
        second = canonical_rotatable_torsions(record)
        torsions, _, metadata = first
        raw = torch.as_tensor(field(record, "rotatable_bond_index", torch.empty((2, 0))), dtype=torch.long).reshape(2, -1)
        totals["molecules"] += 1; totals["raw_rotors"] += raw.size(1); totals["canonical_rotors"] += torsions.size(0)
        local["rotors"] += torsions.size(0)
        if not torsions.numel():
            totals["zero_rotor_molecules"] += 1
        if not torch.equal(first[0], second[0]) or first[2] != second[2]:
            totals["determinism_failures"] += 1
        atom_count = int(torch.as_tensor(record["atomic_numbers"]).numel())
        if torsions.numel() and (int(torsions.min()) < 0 or int(torsions.max()) >= atom_count):
            totals["invalid_indices"] += 1
        centers = [tuple(sorted(q[1:3])) for q in torsions.tolist()]
        totals["duplicate_central_bonds"] += len(centers) - len(set(centers))
        molecule, mapping = _record_to_rdkit_mapping(record)
        for left, right in raw.t().tolist():
            totals["raw_amide_like"] += int(is_amide_like(molecule, mapping[left], mapping[right]))
        for row in metadata:
            left, right = (mapping[value] for value in row["central_bond"])
            bond = molecule.GetBondBetweenAtoms(left, right)
            totals["ring_bonds"] += int(bond.IsInRing())
            totals["non_single_bonds"] += int(bond.GetBondType() != Chem.BondType.SINGLE)
            totals["amide_like_included"] += int(is_amide_like(molecule, left, right))
            totals["symmetric_terminal_rotors"] += int(row["symmetric_terminal_environment"])
        coordinate_sets = [("source", value) for value in item["sources"]]
        coordinate_sets += [("reference", value) for value in item["references"]]
        for kind, coordinates in coordinate_sets:
            values = dihedral_angles(torch.as_tensor(coordinates, dtype=torch.float64), torsions)
            reversed_values = dihedral_angles(torch.as_tensor(coordinates, dtype=torch.float64), torch.flip(torsions, dims=[1]))
            failures = int((circular_difference(values, reversed_values).abs() > 1e-10).sum())
            if values.numel():
                maximum_reversal_error = max(maximum_reversal_error, float(circular_difference(values, reversed_values).abs().max()))
            totals["reversal_failures"] += failures
            totals["nonfinite_angles"] += int((~torch.isfinite(values)).sum())
            totals["range_failures"] += int(((values < -math.pi) | (values >= math.pi)).sum())
            totals[f"{kind}_angles"] += values.numel()
        if (index + 1) % 250 == 0:
            print(f"TORSION CANONICAL {index + 1}/{len(payload['items'])}", flush=True)
    checks = {
        "deterministic": totals["determinism_failures"] == 0,
        "unique": totals["duplicate_central_bonds"] == 0,
        "valid_indices": totals["invalid_indices"] == 0,
        "nonring_single": totals["ring_bonds"] == totals["non_single_bonds"] == 0,
        "amide_restricted_excluded": totals["amide_like_included"] == 0,
        "finite_periodic_labels": totals["nonfinite_angles"] == totals["range_failures"] == 0,
        "forward_reverse_equivalent": totals["reversal_failures"] == 0,
        "nonempty_signal": totals["canonical_rotors"] > 0 and totals["reference_angles"] > 0,
    }
    decision = "TORSION_LABEL_GO" if all(checks.values()) else "TORSION_LABEL_NO_GO"
    result = {
        "schema_version": "mcvr-bat-torsion-canonical-audit-v1", "decision": decision,
        "checks": checks, "totals": totals, "by_partition": by_partition,
        "maximum_reversal_circular_error": maximum_reversal_error,
        "canonical_rule": "canonical central orientation plus heavy/canonical-ranked terminal; fixed atom order",
        "label_source": "TRAIN/DEV Reference internal coordinates only", "mvt_used": False,
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
    }
    atomic_json(OUT / "manifests/TORSION_CANONICALIZATION_AUDIT.json", result)
    atomic_text(OUT / "TORSION_CANONICALIZATION_AUDIT.md", f"""# Torsion canonicalization audit

Decision: **{decision}**

- Molecules: `{totals['molecules']}`; raw/canonical rotors: `{totals['raw_rotors']}` / `{totals['canonical_rotors']}`.
- Source/Reference angles audited: `{totals['source_angles']}` / `{totals['reference_angles']}`.
- Forward/reverse failures: `{totals['reversal_failures']}`; maximum circular error: `{maximum_reversal_error:.3g}`.
- Duplicate central bonds / invalid indices: `{totals['duplicate_central_bonds']}` / `{totals['invalid_indices']}`.
- Ring/non-single/amide-like restricted inclusions: `{totals['ring_bonds']}` / `{totals['non_single_bonds']}` / `{totals['amide_like_included']}`. Raw project rotors contained `{totals['raw_amide_like']}` amide-like definitions, all explicitly excluded before label construction.
- Symmetric terminal environments audited: `{totals['symmetric_terminal_rotors']}`. Canonical ranks resolve a deterministic representative; circular labels remain tied to the frozen atom order.
- Degenerate definitions are returned as finite zero by the dihedral primitive; none creates a non-finite or out-of-range label.

The labels are periodic angles directly computed from internal TRAIN/DEV Reference coordinates. No MVT, Cartesian Source→Reference delta, PB, xTB, formal test, or frozen holdout was read.
""")
    print(decision)
    return 0 if decision == "TORSION_LABEL_GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
