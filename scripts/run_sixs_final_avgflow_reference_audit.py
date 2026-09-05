#!/usr/bin/env python
"""Current-final AvgFlow Reference audit on frozen coordinates/mappings only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdMolAlign

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()
from etflow.ecir.formal_rdkit_adapter import adapt_formal_cache_record


SEEDS = (307, 331, 353)
METHODS = ("AVGFLOW_RAW", *(f"AVGFLOW_SIXS_U_SEED{s}" for s in SEEDS))
VALID_IDENTITIES = {"IDENTITY_EXACT", "IDENTITY_SYMMETRY_AMBIGUOUS"}
EXPECTED_MAPPING_SHA = "525f0ca1edfe9436370f2db104c537205a4469cce265a6cd7930019541552a14"
EXPECTED_COVMAT_SHA = "74c32f953726d98aeab0a3e11295883ceb2cacf4a63b7f4a0deba759ee5f678c"
THRESHOLD = 0.75


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def graph_signature(molecule: Chem.Mol) -> tuple[Any, Any]:
    atoms = tuple(
        (int(a.GetAtomicNum()), int(a.GetFormalCharge()), int(a.GetIsotope()), bool(a.GetIsAromatic()), str(a.GetChiralTag()))
        for a in molecule.GetAtoms()
    )
    bonds = tuple(sorted(
        (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
         str(b.GetBondType()), bool(b.GetIsAromatic()), bool(b.IsInRing()))
        for b in molecule.GetBonds()
    ))
    return atoms, bonds


def graph_sha(molecule: Chem.Mol) -> str:
    value = json.dumps(graph_signature(Chem.RemoveHs(Chem.Mol(molecule))), sort_keys=True, default=str).encode()
    return hashlib.sha256(value).hexdigest()


def atom_order_signature(molecule: Chem.Mol) -> tuple[tuple[int, int, int], ...]:
    return tuple((int(a.GetAtomicNum()), int(a.GetFormalCharge()), int(a.GetIsotope())) for a in molecule.GetAtoms())


def normalized_graph_signature(molecule: Chem.Mol) -> tuple[Any, Any]:
    value = Chem.Mol(molecule)
    Chem.SanitizeMol(value)
    atoms = tuple(
        (int(a.GetAtomicNum()), int(a.GetFormalCharge()), int(a.GetIsotope()), bool(a.GetIsAromatic()))
        for a in value.GetAtoms()
    )
    bonds = tuple(sorted(
        (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
         str(b.GetBondType()), bool(b.GetIsAromatic()), bool(b.IsInRing()))
        for b in value.GetBonds()
    ))
    return atoms, bonds


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit_mapping(args: argparse.Namespace) -> dict[str, Any]:
    mapping = load_jsonl(args.mapping)
    records = pd.read_parquet(args.cross_asset / "avgflow/SOURCE_RECORDS.parquet")
    freeze = json.loads((args.cross_asset / "avgflow/COORDINATE_FREEZE.json").read_text(encoding="utf-8"))
    fidelity = json.loads((args.cross_asset / "avgflow/FIDELITY_COMPLETION.json").read_text(encoding="utf-8"))
    reconstruction = json.loads(args.mapping_audit.read_text(encoding="utf-8"))["audits"]["avgflow"]
    readiness = json.loads(args.readiness_audit.read_text(encoding="utf-8"))
    protocol = json.loads(args.cov_protocol.read_text(encoding="utf-8"))
    cohort = json.loads(args.cov_cohort.read_text(encoding="utf-8"))
    ids = records.record_id.astype(str).tolist()
    molecule_ids = records.molecule_id.astype(str).tolist()
    mapping_ids = [str(row["source_sample_id"]) for row in mapping]
    mapping_molecules = [str(row["source_molecule_id"]) for row in mapping]
    versions = sorted({str(row.get("mapping_version")) for row in mapping})
    legal = [row for row in mapping if row.get("identity_status") in VALID_IDENTITIES]
    valid_ensemble = [row for row in legal if row.get("reference_ensemble_status") == "VALID_ENSEMBLE"]

    sdf_hashes: dict[str, str] = {}
    suppliers = []
    for method in METHODS:
        spec = freeze["sdfs"][method]
        path = Path(spec["path"])
        actual = sha(path)
        if actual != spec["sha256"]:
            raise RuntimeError(f"current frozen SDF hash mismatch: {method}")
        sdf_hashes[method] = actual
        suppliers.append(Chem.ForwardSDMolSupplier(str(path), sanitize=False, removeHs=False))

    record_order = True
    molecule_identity = True
    atom_order_identity = True
    current_graph_identity = True
    stereo_annotation_mismatch_records = 0
    graph_hash_identity = True
    seen_molecules: set[str] = set()
    parsed = 0
    for index, molecules in enumerate(zip(*suppliers, strict=True)):
        if any(mol is None for mol in molecules):
            raise RuntimeError(f"current frozen SDF parse failure at {index}")
        raw = molecules[0]
        names = [str(mol.GetProp("sample_id") if mol.HasProp("sample_id") else mol.GetProp("_Name")) for mol in molecules]
        mids = [str(mol.GetProp("molecule_id")) if mol.HasProp("molecule_id") else "" for mol in molecules]
        record_order &= len(set(names)) == 1 and names[0] == ids[index] == mapping_ids[index]
        molecule_identity &= len(set(mids)) == 1 and mids[0] == molecule_ids[index] == mapping_molecules[index]
        raw_atom_order = atom_order_signature(raw)
        atom_order_identity &= all(atom_order_signature(mol) == raw_atom_order for mol in molecules[1:])
        raw_graph = normalized_graph_signature(raw)
        current_graph_identity &= all(normalized_graph_signature(mol) == raw_graph for mol in molecules[1:])
        stereo_annotation_mismatch_records += int(any(
            graph_signature(Chem.Mol(mol))[0] != graph_signature(Chem.Mol(raw))[0]
            for mol in molecules[1:]
        ))
        # The historical reconstruction deliberately bound one source graph
        # per molecule and copied that hash to all of its conformer rows.
        if molecule_ids[index] not in seen_molecules:
            graph_hash_identity &= graph_sha(raw) == str(mapping[index]["source_structure_hash"])
            seen_molecules.add(molecule_ids[index])
        parsed += 1
    reference_paths_available = all(Path(str(row["reference_source_path"])).is_file() for row in legal)
    reference_hashes_bound = all(bool(str(row.get("reference_identity_hash", ""))) for row in legal)
    checks = {
        "mapping_sha256_exact": sha(args.mapping) == EXPECTED_MAPPING_SHA,
        "mapping_rows_10000_unique": len(mapping) == 10000 and len(set(mapping_ids)) == 10000,
        "source_records_10000_unique": len(ids) == 10000 and len(set(ids)) == 10000,
        "record_ids_and_order_exact": record_order and ids == mapping_ids,
        "molecule_ids_exact": molecule_identity and molecule_ids == mapping_molecules,
        "atom_identity_and_order_all_current_methods": atom_order_identity,
        "current_method_graph_identity_normalized": current_graph_identity,
        "historical_first_record_graph_hash_per_molecule_exact": graph_hash_identity,
        "reference_identity_hashes_bound": reference_hashes_bound,
        "reference_paths_available": reference_paths_available,
        "mapping_version_exact": versions == ["cross-upstream-correspondence-v1"],
        "historical_mapping_audit_pass": reconstruction.get("status") == "PASS",
        "current_fidelity_uses_exact_mapping_hash": fidelity.get("correspondence_manifest_sha256") == EXPECTED_MAPPING_SHA,
        "current_fidelity_same_record_set": fidelity.get("same_record_set_all_methods") is True,
        "current_coordinate_freeze_exact_methods": tuple(freeze.get("methods", ())) == METHODS,
        "cov_amr_readiness_pass": readiness.get("AVG_COV_AMR_READY") == "YES",
        "cov_amr_cohort_bound_to_mapping": cohort["header"].get("correspondence_sha256") == EXPECTED_MAPPING_SHA,
        "cov_amr_protocol_frozen": protocol.get("status") == "FROZEN_NOT_EVALUATED",
        "cov_amr_threshold_frozen": protocol["coverage"].get("threshold_angstrom") == THRESHOLD,
        "historical_covmat_hash_exact": sha(args.covmat) == EXPECTED_COVMAT_SHA,
    }
    passed = all(checks.values()) and len(legal) == 7908 and len(valid_ensemble) == 7782 and parsed == 10000
    result = {
        "schema_version": "sixs-current-final-avgflow-reference-protocol-audit-v1",
        "status": "PASS_REUSED_EXACTLY" if passed else "FAIL_NOT_REUSABLE",
        "AVG_REFERENCE_MAPPING_REUSED": "YES" if passed else "NO",
        "AVG_REFERENCE_MAPPING_STATUS": "PASS_REUSED_EXACTLY" if passed else "FAIL_NOT_REUSABLE",
        "AVG_REFERENCE_EVAL_N": len(legal), "AVG_REFERENCE_ENSEMBLE_N": len(valid_ensemble),
        "source_records": len(ids), "sdf_records_checked": parsed,
        "mapping_version": versions, "mapping_sha256": sha(args.mapping),
        "frozen_cov_amr_cohort_sha256": sha(args.cov_cohort),
        "frozen_cov_amr_protocol_sha256": sha(args.cov_protocol),
        "current_coordinate_freeze_sha256": sha(args.cross_asset / "avgflow/COORDINATE_FREEZE.json"),
        "current_sdf_sha256": sdf_hashes, "checks": checks,
        "nonblocking_sdf_stereo_annotation_mismatch_records": stereo_annotation_mismatch_records,
        "stereo_annotation_note": "coordinate-derived SDF stereo tags differ on a small number of outputs; atom sequence and normalized typed connectivity remain the frozen identity gates",
        "reference_hash_verification_chain": "correspondence reconstruction PASS + frozen readiness cache-coordinate hash PASS",
        "bond_angle_reference_status": "NOT_EVALUATED__SYMMETRY_AMBIGUOUS_ROWS_LACK_UNIQUE_ATOM_PERMUTATION",
        "reference_rmsd_protocol": "heavy-atom symmetry-aware GetBestRMS; nearest frozen reference conformer",
        "cov_amr_protocol": "frozen 0.75A heavy-atom symmetry-aware molecule-equal ensemble protocol",
        "same_as_primary_all_atom_fixed_order_protocol": False,
        "model_training": False, "coordinate_regeneration": False, "sixs_reinference": False,
    }
    atomic_json(args.report_dir / "AVG_REFERENCE_PROTOCOL_AUDIT.json", result)
    if not passed:
        raise RuntimeError("current AvgFlow cohort is not an exact reuse of the frozen correspondence")
    return result


def add_conformers(template: Chem.Mol, coordinates: np.ndarray) -> Chem.Mol:
    molecule = Chem.Mol(template)
    molecule.RemoveAllConformers()
    for xyz in coordinates:
        if xyz.shape != (molecule.GetNumAtoms(), 3) or not np.isfinite(xyz).all():
            raise RuntimeError("reference coordinates incomplete or nonfinite")
        conformer = Chem.Conformer(molecule.GetNumAtoms())
        for atom_index, position in enumerate(xyz):
            conformer.SetAtomPosition(atom_index, position.tolist())
        molecule.AddConformer(conformer, assignId=True)
    return Chem.RemoveHs(molecule)


def reference_molecule(path: str) -> Chem.Mol:
    record = torch.load(Path(path), map_location="cpu", weights_only=False)
    adapted = adapt_formal_cache_record(record)
    return add_conformers(adapted["_formal_rdkit_mol"], np.asarray(record["x_ref_candidates"], dtype=np.float64))


def matrix_metrics(reference: Chem.Mol, generated: list[Chem.Mol]) -> dict[str, float]:
    matrix = np.empty((reference.GetNumConformers(), len(generated)), dtype=np.float64)
    for ref_id in range(reference.GetNumConformers()):
        for generated_id, molecule in enumerate(generated):
            matrix[ref_id, generated_id] = rdMolAlign.GetBestRMS(molecule, reference, prbId=0, refId=ref_id)
    if not np.isfinite(matrix).all():
        raise RuntimeError("nonfinite COV/AMR RMSD matrix")
    ref_min = matrix.min(axis=1)
    gen_min = matrix.min(axis=0)
    return {
        "COV_R": float(np.mean(ref_min < THRESHOLD)), "AMR_R": float(np.mean(ref_min)),
        "COV_P": float(np.mean(gen_min < THRESHOLD)), "AMR_P": float(np.mean(gen_min)),
    }


def evaluate_cov_amr(args: argparse.Namespace) -> pd.DataFrame:
    complete = args.asset_dir / "avgflow_reference/COV_AMR_COMPLETE.json"
    final_path = args.asset_dir / "avgflow_reference/COV_AMR_PER_MOLECULE.parquet"
    if complete.is_file() and final_path.is_file() and json.loads(complete.read_text(encoding="utf-8")).get("status") == "PASS":
        return pd.read_parquet(final_path)
    cohort = json.loads(args.cov_cohort.read_text(encoding="utf-8"))["molecules"]
    freeze = json.loads((args.cross_asset / "avgflow/COORDINATE_FREEZE.json").read_text(encoding="utf-8"))
    lookup = {sample_id: (i, j) for i, row in enumerate(cohort) for j, sample_id in enumerate(row["source_sample_ids"])}
    partial_path = args.asset_dir / "avgflow_reference/COV_AMR_PARTIAL.parquet"
    accumulated = pd.read_parquet(partial_path).to_dict("records") if partial_path.is_file() else []
    completed_indices = set(pd.DataFrame(accumulated).molecule_index.astype(int)) if accumulated else set()
    suppliers = [Chem.ForwardSDMolSupplier(str(freeze["sdfs"][m]["path"]), sanitize=True, removeHs=False) for m in METHODS]
    current_index: int | None = None
    current = {method: [] for method in METHODS}

    def checkpoint(index: int) -> None:
        nonlocal current, accumulated
        row = cohort[index]
        if index not in completed_indices:
            reference = reference_molecule(str(row["reference_source_path"]))
            if reference.GetNumConformers() != int(row["reference_conformer_count"]):
                raise RuntimeError("frozen reference ensemble count mismatch")
            for method in METHODS:
                if len(current[method]) != int(row["generated_conformer_count_per_method"]):
                    raise RuntimeError("frozen generated ensemble count mismatch")
                accumulated.append({
                    "molecule_index": index, "molecule_id": str(row["source_molecule_id"]),
                    "method": method, "generated_conformers": len(current[method]),
                    "reference_conformers": reference.GetNumConformers(),
                    **matrix_metrics(reference, current[method]),
                })
            completed_indices.add(index)
            if len(completed_indices) % 25 == 0 or len(completed_indices) == len(cohort):
                atomic_parquet(partial_path, pd.DataFrame(accumulated))
                atomic_json(args.report_dir / "AVG_REFERENCE_COV_AMR_STATUS.json", {
                    "status": "RUNNING", "completed_molecules": len(completed_indices),
                    "total_molecules": len(cohort), "pid": os.getpid(), "resume_supported": True,
                })
        current = {method: [] for method in METHODS}

    for molecules in zip(*suppliers, strict=True):
        if any(mol is None for mol in molecules):
            raise RuntimeError("current AvgFlow SDF parse failure during COV/AMR")
        ids = [str(mol.GetProp("sample_id")) for mol in molecules]
        if len(set(ids)) != 1:
            raise RuntimeError("current AvgFlow method identity mismatch during COV/AMR")
        location = lookup.get(ids[0])
        if location is None:
            continue
        molecule_index, within_index = location
        if current_index is None:
            current_index = molecule_index
        elif molecule_index != current_index:
            checkpoint(current_index)
            current_index = molecule_index
        if within_index != len(current[METHODS[0]]):
            raise RuntimeError("frozen AvgFlow ensemble order changed")
        for method, mol in zip(METHODS, molecules, strict=True):
            current[method].append(Chem.RemoveHs(Chem.Mol(mol)))
    if current_index is not None:
        checkpoint(current_index)
    frame = pd.DataFrame(accumulated).sort_values(["molecule_index", "method"], kind="stable").reset_index(drop=True)
    if len(frame) != 3431 * len(METHODS) or frame.molecule_index.nunique() != 3431:
        raise RuntimeError("current AvgFlow COV/AMR denominator mismatch")
    atomic_parquet(final_path, frame)
    atomic_json(complete, {"status": "PASS", "molecules": 3431, "methods": list(METHODS), "sha256": sha(final_path)})
    atomic_json(args.report_dir / "AVG_REFERENCE_COV_AMR_STATUS.json", {
        "status": "PASS", "completed_molecules": 3431, "total_molecules": 3431,
        "per_molecule_sha256": sha(final_path), "resume_supported": True,
    })
    return frame


def paired(values: pd.DataFrame, metric: str, candidate: str, seed: int) -> dict[str, Any]:
    base = values.loc[values.method == "AVGFLOW_RAW", ["molecule_id", metric]].rename(columns={metric: "baseline"})
    cand = values.loc[values.method == candidate, ["molecule_id", metric]].rename(columns={metric: "candidate"})
    joined = base.merge(cand, on="molecule_id", validate="one_to_one")
    effect = (joined.candidate.astype(float) - joined.baseline.astype(float)).to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.empty(10000)
    for start in range(0, 10000, 200):
        end = min(start + 200, 10000)
        indices = rng.integers(0, len(effect), size=(end - start, len(effect)))
        draws[start:end] = effect[indices].mean(axis=1)
    direction = effect if metric.startswith("COV") else -effect
    return {
        "comparison": f"{candidate}_minus_AVGFLOW_RAW", "metric": metric,
        "effect_definition": "candidate_minus_raw", "molecules": len(effect),
        "mean_paired_effect": float(effect.mean()), "ci95_low": float(np.quantile(draws, .025)),
        "ci95_high": float(np.quantile(draws, .975)), "median_molecule_effect": float(np.median(effect)),
        "wins": int((direction > 1e-12).sum()), "ties": int((np.abs(direction) <= 1e-12).sum()),
        "losses": int((direction < -1e-12).sum()), "cluster_unit": "molecule", "bootstrap_resamples": 10000,
    }


def finalize(args: argparse.Namespace, cov: pd.DataFrame) -> None:
    fidelity = pd.read_parquet(args.cross_asset / "avgflow/FIDELITY_PER_RECORD.parquet")
    source = pd.read_parquet(args.cross_asset / "avgflow/SOURCE_RECORDS.parquet")
    fidelity = fidelity.merge(source, on="record_id", validate="many_to_one")
    reference_molecule = fidelity.groupby(["method", "molecule_id"], sort=False).reference_rmsd_angstrom.mean().reset_index()
    rows = []
    for method in METHODS:
        rr = fidelity[fidelity.method == method].reference_rmsd_angstrom.astype(float)
        cc = cov[cov.method == method]
        rows.append({
            "method": method, "reference_rmsd_records": len(rr), "reference_rmsd_mean": float(rr.mean()),
            "reference_rmsd_median": float(rr.median()), "cov_amr_molecules": len(cc),
            **{metric: float(cc[metric].mean()) for metric in ("COV_P", "COV_R", "AMR_P", "AMR_R")},
            "bond_raw_mae": np.nan, "angle_cosine_mae": np.nan,
            "bond_angle_status": "NOT_EVALUATED__NO_UNIQUE_ATOM_PERMUTATION_FOR_ALL_SYMMETRY_MAPPINGS",
        })
    metrics = pd.DataFrame(rows)
    seed_rows = metrics[metrics.method != "AVGFLOW_RAW"]
    aggregate = {"method": "SIXS_U_THREE_SEED_MEAN_SD", "reference_rmsd_records": 7908, "cov_amr_molecules": 3431}
    for metric in ("reference_rmsd_mean", "COV_P", "COV_R", "AMR_P", "AMR_R"):
        aggregate[metric] = float(seed_rows[metric].mean())
        aggregate[f"{metric}_seed_sd"] = float(seed_rows[metric].std(ddof=1))
    metrics = pd.concat([metrics, pd.DataFrame([aggregate])], ignore_index=True, sort=False)
    atomic_csv(args.report_dir / "AVG_REFERENCE_METRICS.csv", metrics)

    paired_rows = []
    counter = 0
    for seed in SEEDS:
        candidate = f"AVGFLOW_SIXS_U_SEED{seed}"
        counter += 1
        paired_rows.append(paired(reference_molecule.rename(columns={"reference_rmsd_angstrom": "REFERENCE_RMSD"}), "REFERENCE_RMSD", candidate, 20262000 + counter))
        for metric in ("COV_P", "COV_R", "AMR_P", "AMR_R"):
            counter += 1
            paired_rows.append(paired(cov, metric, candidate, 20262000 + counter))
    stats = pd.DataFrame(paired_rows)
    atomic_csv(args.report_dir / "AVG_REFERENCE_PAIRED_STATS.csv", stats)
    significant_improvement = []
    significant_degradation = []
    for row in stats.itertuples(index=False):
        higher = str(row.metric).startswith("COV")
        significant_improvement.append(row.ci95_low > 0 if higher else row.ci95_high < 0)
        significant_degradation.append(row.ci95_high < 0 if higher else row.ci95_low > 0)
    fidelity_effect = (
        "IMPROVES" if any(significant_improvement) and not any(significant_degradation)
        else "DEGRADES" if any(significant_degradation) and not any(significant_improvement)
        else "MIXED" if any(significant_improvement) or any(significant_degradation)
        else "NO_MEASURABLE_CHANGE"
    )
    v3d = pd.read_parquet(args.cross_asset / "avgflow/VALIDITY3D.parquet")
    pb = pd.read_parquet(args.cross_asset / "avgflow/POSEBUSTERS.parquet")
    raw_v3d = float(v3d[v3d.method == "AVGFLOW_RAW"].validity3d.astype(bool).mean())
    final_v3d = [float(v3d[v3d.method == f"AVGFLOW_SIXS_U_SEED{s}"].validity3d.astype(bool).mean()) for s in SEEDS]
    raw_pb = float(pb[pb.method == "AVGFLOW_RAW"].PB.astype(bool).mean())
    final_pb = [float(pb[pb.method == f"AVGFLOW_SIXS_U_SEED{s}"].PB.astype(bool).mean()) for s in SEEDS]
    structural = "V3D_IMPROVES__PB_ESSENTIALLY_STABLE" if min(final_v3d) > raw_v3d and max(abs(x - raw_pb) for x in final_pb) <= .002 else "MIXED"
    energy_medians = []
    for seed in SEEDS:
        path = args.repo / f"reports/ecir_mvr/sixs_final_cross_upstream_unrestricted/avgflow/xtb_singlepoint/AVGFLOW_SIXS_U_SEED{seed}_DELTA_VS_SOURCE.csv"
        frame = pd.read_csv(path)
        energy_medians.append(float(frame.loc[frame.matched_success.astype(bool), "delta_e_kcal_mol"].median()))
    energy = "SMALL_POSITIVE_DELTA_E_SHIFT" if all(0 < x < .5 for x in energy_medians) else ("ENERGY_IMPROVEMENT" if all(x < 0 for x in energy_medians) else "MIXED")
    atomic_text(args.report_dir / "AVG_REFERENCE_FIDELITY_CONCLUSION.md", f"""# Current-final AvgFlow Reference fidelity audit

```text
AVG_REFERENCE_MAPPING_REUSED = YES
AVG_REFERENCE_MAPPING_STATUS = PASS_REUSED_EXACTLY
AVG_REFERENCE_EVAL_N = 7908
AVG_REFERENCE_ENSEMBLE_N = 7782
AVG_COV_AMR_MOLECULES = 3431
AVG_COV_AMR_THRESHOLD_ANGSTROM = 0.75
AVG_STRUCTURAL_VALIDITY_EFFECT = {structural}
AVG_REFERENCE_FIDELITY_EFFECT = {fidelity_effect}
AVG_ENERGY_EFFECT = {energy}
```

Reference RMSD uses all 7,908 legal mappings. COV-P/R and AMR-P/R use the pre-outcome frozen complete-ensemble subset (7,782 records in 3,431 molecule clusters). Both are heavy-atom, symmetry-aware AvgFlow protocols and are not numerically interchangeable with the primary prospective cohort's fixed-order all-atom Kabsch metric.

Bond/Angle errors are not reported because 5,255 legal symmetry-ambiguous records do not have a unique atom permutation. Structural validity, Reference fidelity, and energy are interpreted separately; closeness of aggregate V3D/PB/xTB values is not treated as Reference fidelity.
""")
    atomic_json(args.asset_dir / "avgflow_reference/COMPLETE.json", {
        "status": "PASS", "mapping_status": "PASS_REUSED_EXACTLY", "reference_eval_n": 7908,
        "reference_ensemble_n": 7782, "cov_amr_molecules": 3431,
        "structural_validity_effect": structural, "reference_fidelity_effect": fidelity_effect,
        "energy_effect": energy, "scientific_inputs_changed": False,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cross-asset", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--mapping-audit-only", action="store_true")
    args = parser.parse_args()
    for name in ("repo", "cross_asset", "asset_dir", "report_dir"):
        setattr(args, name, getattr(args, name).resolve())
    old = Path(r"E:\3dconformergenerationcode\4dadapter-lsgoba-v2-softplus-multiseed")
    corr = old / "reports/ecir_mvr/lsgoba_softplus_v2_final_cross_upstream/correspondence_reconstruction"
    ready = old / "reports/ecir_mvr/lsgoba_softplus_v2_final_cross_upstream/cov_amr_readiness"
    args.mapping = corr / "AVGFLOW_REFERENCE_CORRESPONDENCE_V1.jsonl"
    args.mapping_audit = corr / "CORRESPONDENCE_RECONSTRUCTION_AUDIT.json"
    args.readiness_audit = ready / "AVG_COV_AMR_READINESS_AUDIT.json"
    args.cov_protocol = ready / "AVG_COV_AMR_FROZEN_PROTOCOL.json"
    args.cov_cohort = ready / "AVG_COV_AMR_FROZEN_COHORT.json"
    args.covmat = old / "etflow/commons/covmat.py"
    return args


def main(args: argparse.Namespace) -> int:
    audit_path = args.report_dir / "AVG_REFERENCE_PROTOCOL_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    reusable = (
        audit.get("status") == "PASS_REUSED_EXACTLY"
        and audit.get("mapping_sha256") == sha(args.mapping)
        and audit.get("current_coordinate_freeze_sha256") == sha(args.cross_asset / "avgflow/COORDINATE_FREEZE.json")
        and audit.get("frozen_cov_amr_cohort_sha256") == sha(args.cov_cohort)
        and audit.get("frozen_cov_amr_protocol_sha256") == sha(args.cov_protocol)
    )
    if not reusable:
        audit = audit_mapping(args)
    if args.mapping_audit_only:
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0
    cov = evaluate_cov_amr(args)
    finalize(args, cov)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
