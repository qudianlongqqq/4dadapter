#!/usr/bin/env python
"""Failure-accounting PoseBusters and Validity3D worker for final SIXS."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import yaml
from rdkit import Chem


OFFICIAL_ADAPTER = Path(r"E:\3dconformergenerationcode\4dadapter-lsgo-v2\scripts\run_lsgo_standard_genbench3d.py")
GENBENCH_REPO = Path(r"E:\miniconda\envs\external-validity\src\genbench3d")
LIGBOUNDCONF = Path(r"E:\3dconformergenerationcode\external_data\genbench3d_official\ligboundconf_minimized\S2_LigBoundConf_minimized.sdf")
REFERENCE_ROOT = Path(r"E:\3dconformergenerationcode\4dadapter-lsgo-v2\reports\ecir_mvr\lsgo_standard_eval\genbench3d_reference_cache")
GENBENCH_COMMIT = "0926bc6614509aa10ccf6f69da0405d4be6af6b3"
GENBENCH_ADAPTER_SHA = "6ebc450b4f841a9f6a3b463b7838e50bc2951e92570146cd0a473abbfd970450"
REFERENCE_SDF_SHA = "15e8e4635525f3d9452292e86995d28f6c24eb50baad661eb4c2665274d00fe2"
REFERENCE_VALUE_SHA = "63659acddd04017a4b8fc5f2df767540e48cd36a7849e578ddb6caf6130deadc"
REFERENCE_KERNEL_SHA = "6c098fa5b10c85f12db49df3a35efa33963fc222754e5d2a9d0b64e61c604a19"
PB_COMPONENTS = (
    "mol_pred_loaded", "sanitization", "inchi_convertible", "all_atoms_connected",
    "no_radicals", "bond_lengths", "bond_angles", "internal_steric_clash",
    "aromatic_ring_flatness", "non-aromatic_ring_non-flatness", "double_bond_flatness",
)
V3D_COMPONENTS = (
    "bond_geometry_valid", "angle_geometry_valid", "aromatic_ring_valid",
    "intramolecular_steric_clash_valid",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--sdf", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--pb", type=Path, required=True)
    parser.add_argument("--v3d", type=Path, required=True)
    args = parser.parse_args()
    records = pd.read_parquet(args.records)
    ids = records.record_id.astype(str).tolist()
    if len(ids) != 5000 or len(set(ids)) != 5000:
        raise RuntimeError("external worker record identity changed")

    import posebusters
    from posebusters import PoseBusters

    pb_root = Path(posebusters.__file__).parent
    installed = yaml.safe_load((pb_root / "config/mol_fast.yml").read_text(encoding="utf-8"))
    runtime = copy.deepcopy(installed)
    for module in runtime.get("modules", []):
        if module.get("function") == "energy_ratio":
            module.setdefault("parameters", {})["num_threads"] = 1
    raw = PoseBusters(config=runtime, max_workers=1, chunk_size=50).bust(str(args.sdf), None, None, full_report=True).reset_index()
    if "molecule" not in raw.columns or raw.molecule.astype(str).duplicated().any():
        raise RuntimeError("PoseBusters did not return unique molecule identities")
    raw["record_id"] = raw.molecule.astype(str)
    base = pd.DataFrame({"record_id": ids, "_order": range(len(ids))})
    frame = base.merge(raw, on="record_id", how="left", validate="one_to_one").sort_values("_order", kind="stable")
    for component in PB_COMPONENTS:
        if component not in frame:
            frame[component] = False
        frame[component] = frame[component].fillna(False).astype(bool)
    frame["PB"] = frame[list(PB_COMPONENTS)].all(axis=1)
    frame["evaluator_record_present"] = frame.molecule.notna() if "molecule" in frame else False
    frame["arm"] = args.arm
    atomic_frame(args.pb, frame.drop(columns=["_order"]))
    print(f"POSEBUSTERS_COMPLETE {args.arm} {len(frame)} failures={int((~frame.evaluator_record_present).sum())}", flush=True)

    if sha256(OFFICIAL_ADAPTER) != GENBENCH_ADAPTER_SHA:
        raise RuntimeError("GenBench3D adapter identity changed")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=GENBENCH_REPO, text=True).strip() != GENBENCH_COMMIT:
        raise RuntimeError("GenBench3D commit changed")
    for path, expected in (
        (LIGBOUNDCONF, REFERENCE_SDF_SHA),
        (REFERENCE_ROOT / "LigBoundConf_geometry_values.p", REFERENCE_VALUE_SHA),
        (REFERENCE_ROOT / "LigBoundConf_geometry_kernel_densities.p", REFERENCE_KERNEL_SHA),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"GenBench3D reference identity changed: {path}")
    sys.modules.setdefault("gemmi", types.ModuleType("gemmi"))
    spec = importlib.util.spec_from_file_location("primary_final_genbench_external", OFFICIAL_ADAPTER)
    official = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(official)
    official.METHODS = (args.arm,)
    SDFSource, ReferenceGeometry, Validity3D = official.import_official_evaluator()
    reference = ReferenceGeometry(
        source=SDFSource(str(LIGBOUNDCONF), name="LigBoundConf", removeHs=False),
        root=str(REFERENCE_ROOT), minimum_pattern_values=50, use_generalized_patterns=False,
    )
    validity = Validity3D(
        reference_geometry=reference, q_value_threshold=0.001,
        steric_clash_safety_ratio=0.75, maximum_ring_plane_distance=0.1,
        include_torsions=False, consider_hydrogens=False,
    )
    supplier = Chem.SDMolSupplier(str(args.sdf), removeHs=False, sanitize=True)
    rows = []
    for index, sample_id in enumerate(ids):
        try:
            molecule = supplier[index]
            if molecule is None or molecule.GetProp("_Name") != sample_id:
                raise ValueError("SDF parse or identity failure")
            output = official.evaluate_record_group(validity, [molecule], index)
            if len(output) != 1:
                raise ValueError("Validity3D denominator mismatch")
            row = dict(output[0])
            for component in V3D_COMPONENTS:
                row[component] = bool(row.get(component, False))
            row["validity3d"] = all(row[component] for component in V3D_COMPONENTS)
            row["evaluator_success"] = True
            row["failure_reason"] = None
        except BaseException as exc:
            row = {component: False for component in V3D_COMPONENTS}
            row.update({"validity3d": False, "evaluator_success": False, "failure_reason": f"{type(exc).__name__}:{exc}"})
        row["arm"] = args.arm
        row["record_id"] = sample_id
        rows.append(row)
        if (index + 1) % 100 == 0:
            print(f"VALIDITY3D_PROGRESS {args.arm} {index + 1}/5000", flush=True)
    v3d = pd.DataFrame(rows)
    if len(v3d) != 5000 or v3d.record_id.astype(str).tolist() != ids:
        raise RuntimeError("Validity3D denominator changed")
    atomic_frame(args.v3d, v3d)
    print(f"VALIDITY3D_COMPLETE {args.arm} {len(v3d)} failures={int((~v3d.evaluator_success).sum())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
