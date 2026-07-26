#!/usr/bin/env python3
"""Generate and atomically freeze Source/Reference/B/A/BA diagnostic coordinates."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
from etflow.ecir.bat_refinement import TorsionHead, canonical_rotatable_torsions, dihedral_angles, torsion_nll
from etflow.ecir.learned_geometry import LearnedGeometryObjective, distribution_parameters, prepare_graph, safety_accept
from etflow.ecir.lsgo_io import atomic_json, file_sha256, nearest_reference_metrics
from etflow.ecir.lsgo_mechanism import ba_abnormality, masked_gradient_update, primitive_z

OUT = ROOT / "reports/ecir_mvr/lsgo_mechanism"
CONFIG_PATH = ROOT / "configs/ecir_mvr_lsgo_mechanism.yaml"


def atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary); os.replace(temporary, path)


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False); os.replace(temporary, path)


def coordinate_sha(coordinates) -> str:
    if torch.is_tensor(coordinates): coordinates = coordinates.detach().cpu()
    value = np.asarray(coordinates, dtype=np.float64)
    import hashlib
    return hashlib.sha256(f"float64|{value.shape}|".encode() + value.tobytes()).hexdigest()


def load_ba(row, device):
    path = Path(row["path"])
    if file_sha256(path) != row["sha256"]: raise RuntimeError("BA checkpoint SHA mismatch")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = LearnedGeometryObjective(hidden_dim=128, layers=3, learned_sigma=False).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model


def molecule_identity(record):
    params = Chem.SmilesParserParams(); params.removeHs = False; params.sanitize = True
    molecule = Chem.MolFromSmiles(str(record["smiles"]), params)
    if molecule is None: raise RuntimeError("SMILES parsing failed")
    cache_to_rdkit = {atom.GetAtomMapNum(): atom.GetIdx() for atom in molecule.GetAtoms()}
    if sorted(cache_to_rdkit) != list(range(molecule.GetNumAtoms())): raise RuntimeError("atom map identity failed")
    elements = [None] * molecule.GetNumAtoms()
    for cache_index, rdkit_index in cache_to_rdkit.items(): elements[cache_index] = molecule.GetAtomWithIdx(rdkit_index).GetSymbol()
    charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    uhf = sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
    return molecule, cache_to_rdkit, elements, charge, uhf


def main() -> int:
    started = time.time(); config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    identity = json.loads((OUT / "DATASET_IDENTITY.json").read_text(encoding="utf-8"))
    threshold = json.loads((OUT / "manifests/THRESHOLD_FREEZE.json").read_text(encoding="utf-8"))
    if threshold["status"] != "FROZEN_BEFORE_XTB" or threshold["xtb_energy_records_read"] or threshold["xtb_force_records_read"]:
        raise RuntimeError("thresholds were not frozen before xTB")
    if identity["formal_test_records_read"] or identity["frozen_holdout_records_read"]: raise RuntimeError("protected split access")
    compact_path = Path(identity["compact_path"])
    if file_sha256(compact_path) != identity["compact_sha256"]: raise RuntimeError("mechanism compact SHA mismatch")
    items = list(torch.load(compact_path, map_location="cpu", weights_only=False)["items"])
    calibration = json.loads(Path(config["dataset"]["drcsr_calibration"]).read_text(encoding="utf-8"))
    ba_manifest_path = Path(config["ba_anchor"]["manifest"]); ba_manifest = json.loads(ba_manifest_path.read_text(encoding="utf-8"))
    ba_rows = {int(row["seed"]): row for row in ba_manifest["checkpoints"] if row["variant"] == "B" and int(row["seed"]) in config["ba_seeds"]}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = {seed: load_ba(row, device) for seed, row in ba_rows.items()}
    torsion_path = Path(config["torsion_anchor"]["checkpoint"])
    if file_sha256(torsion_path) != config["torsion_anchor"]["checkpoint_sha256"]: raise RuntimeError("torsion checkpoint SHA mismatch")
    torsion_checkpoint = torch.load(torsion_path, map_location=device, weights_only=False)
    torsion_head = TorsionHead(hidden_dim=128, components=3, learned_kappa=False).to(device)
    torsion_head.load_state_dict(torsion_checkpoint["model_state"], strict=True); torsion_head.eval()
    for parameter in torsion_head.parameters(): parameter.requires_grad_(False)
    representative = int(config["representative_ba_seed"]); rms_budget = float(config["ba_anchor"]["rms_budget_angstrom"]); atom_cap = float(config["ba_anchor"]["atom_cap_angstrom"])
    entry_lookup = {row["molecule_id"]: row for row in identity["entries"]}
    tasks, coordinate_rows, primitive_rows = [], [], []
    for molecule_index, item in enumerate(items):
        graph = prepare_graph(item["record"], calibration).to(device)
        molecule, cache_to_rdkit, elements, charge, uhf = molecule_identity(item["record"])
        refs = torch.as_tensor(item["references"], dtype=torch.float64)
        torsions = canonical_rotatable_torsions(item["record"])[0].to(device)
        parameters_by_seed, embedding = {}, None
        for seed, model in models.items():
            with torch.no_grad():
                prediction = model(graph); parameters_by_seed[seed] = distribution_parameters(graph, model=model, variant="B")
                if seed == representative: embedding = prediction["node_embedding"]
        torsion_prediction = torsion_head(embedding, torsions, fixed_kappa=float(config["torsion_anchor"]["fixed_kappa"])) if torsions.numel() else None
        metadata = entry_lookup[item["molecule_id"]]
        base = {"molecule_id": item["molecule_id"], "elements": elements, "charge": charge, "uhf": uhf, **{key: metadata[key] for key in ("flex_bin", "rotatable_bond_count", "heavy_atom_count", "aromatic", "ring", "amide_like")}}
        for reference_index, reference in enumerate(refs):
            task = {**base, "condition": "Reference", "seed": None, "sample_id": f"{item['molecule_id']}|reference{reference_index:03d}", "source_sample_id": None, "source_index": None, "reference_index": reference_index, "coordinates": reference.tolist()}
            task["coordinate_sha256"] = coordinate_sha(reference); tasks.append(task)
        for source_index, source_cpu in enumerate(item["sources"]):
            source = torch.as_tensor(source_cpu, dtype=torch.float64, device=device); sample_id = str(item["sample_ids"][source_index])
            source_task = {**base, "condition": "Source", "seed": None, "sample_id": sample_id, "source_sample_id": sample_id, "source_index": source_index, "reference_index": None, "coordinates": source.cpu().tolist()}
            source_task["coordinate_sha256"] = coordinate_sha(source.cpu()); tasks.append(source_task)
            rep_params = parameters_by_seed[representative]; b_z, a_z = primitive_z(source, graph, rep_params); scores = ba_abnormality(source, graph, rep_params)
            source_mode, source_rmsd = nearest_reference_metrics(source.cpu().float(), refs.float())
            torsion_source_nll = float(torsion_nll(dihedral_angles(source, torsions), torsion_prediction).mean()) if torsions.numel() else np.nan
            torsion_ref_nll = float(np.mean([float(torsion_nll(dihedral_angles(reference.to(device), torsions), torsion_prediction).mean()) for reference in refs])) if torsions.numel() else np.nan
            common = {**{key: base[key] for key in base if key not in {"elements"}}, "sample_id": sample_id, "source_index": source_index,
                      "source_b_abnormality": float(b_z.square().mean()), "source_a_abnormality": float(a_z.square().mean()), "source_ba_abnormality": float(scores["ba"]),
                      "source_torsion_nll": torsion_source_nll, "reference_mean_torsion_nll": torsion_ref_nll, "source_nearest_reference_mode": source_mode, "source_nearest_reference_rmsd": source_rmsd}
            for bond_index, (left, right) in enumerate(graph.bonds.t().tolist()):
                rd_bond = molecule.GetBondBetweenAtoms(cache_to_rdkit[left], cache_to_rdkit[right])
                primitive_rows.append({**common, "family": "B", "primitive_index": bond_index, "z": float(b_z[bond_index]), "abs_z": float(abs(b_z[bond_index])), "bond_type": str(rd_bond.GetBondType()), "hybridization": f"{molecule.GetAtomWithIdx(cache_to_rdkit[left]).GetHybridization()}-{molecule.GetAtomWithIdx(cache_to_rdkit[right]).GetHybridization()}"})
            for angle_index, (_, center, _) in enumerate(graph.angles.tolist()):
                atom = molecule.GetAtomWithIdx(cache_to_rdkit[center])
                primitive_rows.append({**common, "family": "A", "primitive_index": angle_index, "z": float(a_z[angle_index]), "abs_z": float(abs(a_z[angle_index])), "bond_type": "ANGLE", "hybridization": str(atom.GetHybridization())})
            for seed, parameters in parameters_by_seed.items():
                for family in ("B", "A", "BA"):
                    raw = masked_gradient_update(source, graph, parameters, family, rms_budget=rms_budget, atom_cap=atom_cap)
                    output, safety = safety_accept(source, raw["coordinates"], graph)
                    output_mode, output_rmsd = nearest_reference_metrics(output.cpu().float(), refs.float())
                    delta = output - source
                    output_ba = ba_abnormality(output, graph, parameters)
                    output_torsion_nll = float(torsion_nll(dihedral_angles(output, torsions), torsion_prediction).mean()) if seed == representative and torsions.numel() else np.nan
                    condition = f"{family}_seed{seed}"
                    task = {**base, "condition": condition, "seed": seed, "sample_id": sample_id, "source_sample_id": sample_id, "source_index": source_index, "reference_index": None, "coordinates": output.cpu().tolist()}
                    task["coordinate_sha256"] = coordinate_sha(output.cpu()); tasks.append(task)
                    coordinate_rows.append({**common, "condition": condition, "family": family, "seed": seed, "accepted": safety["accepted"], "chirality_preserved": safety["chirality_preserved"], "ring_nonregression": safety["ring_nonregression"], "catastrophic_clash_nonregression": safety["catastrophic_clash_nonregression"],
                        "movement_rms": float(torch.sqrt(delta.square().sum(-1).mean())), "movement_max": float(torch.linalg.vector_norm(delta, dim=-1).max()), "output_nearest_reference_mode": output_mode, "output_nearest_reference_rmsd": output_rmsd, "mode_switch": output_mode != source_mode,
                        "output_b_abnormality": float(output_ba["bond"]), "output_a_abnormality": float(output_ba["angle"]), "output_ba_abnormality": float(output_ba["ba"]), "output_torsion_nll": output_torsion_nll})
            bond_mask = b_z.abs() > float(threshold["bond_abs_z_reference_p95"]); angle_mask = a_z.abs() > float(threshold["angle_abs_z_reference_p95"])
            raw = masked_gradient_update(source, graph, rep_params, "BA", rms_budget=rms_budget, atom_cap=atom_cap, bond_mask=bond_mask, angle_mask=angle_mask)
            output, safety = safety_accept(source, raw["coordinates"], graph); delta = output - source
            output_mode, output_rmsd = nearest_reference_metrics(output.cpu().float(), refs.float()); output_ba = ba_abnormality(output, graph, rep_params)
            output_torsion_nll = float(torsion_nll(dihedral_angles(output, torsions), torsion_prediction).mean()) if torsions.numel() else np.nan
            condition = f"BA_abnormal_seed{representative}"; task = {**base, "condition": condition, "seed": representative, "sample_id": sample_id, "source_sample_id": sample_id, "source_index": source_index, "reference_index": None, "coordinates": output.cpu().tolist()}; task["coordinate_sha256"] = coordinate_sha(output.cpu()); tasks.append(task)
            coordinate_rows.append({**common, "condition": condition, "family": "BA_abnormal", "seed": representative, "accepted": safety["accepted"], "chirality_preserved": safety["chirality_preserved"], "ring_nonregression": safety["ring_nonregression"], "catastrophic_clash_nonregression": safety["catastrophic_clash_nonregression"], "active_bonds": int(bond_mask.sum()), "active_angles": int(angle_mask.sum()), "no_op": bool(raw.get("no_op", False)),
                "movement_rms": float(torch.sqrt(delta.square().sum(-1).mean())), "movement_max": float(torch.linalg.vector_norm(delta, dim=-1).max()), "output_nearest_reference_mode": output_mode, "output_nearest_reference_rmsd": output_rmsd, "mode_switch": output_mode != source_mode,
                "output_b_abnormality": float(output_ba["bond"]), "output_a_abnormality": float(output_ba["angle"]), "output_ba_abnormality": float(output_ba["ba"]), "output_torsion_nll": output_torsion_nll})
        print(f"MECHANISM COORD {molecule_index + 1}/{len(items)}", flush=True)
    if len(tasks) != 2441 or len(coordinate_rows) != 1440: raise RuntimeError(f"coordinate denominator changed: {len(tasks)}/{len(coordinate_rows)}")
    coordinate_path = OUT / "manifests/FROZEN_COORDINATES.pt"; atomic_torch(coordinate_path, {"schema_version": "mcvr-lsgo-mechanism-coordinates-v1", "tasks": tasks, "formal_test_records_read": 0, "frozen_holdout_records_read": 0})
    rows_path = OUT / "per_record/COORDINATE_DIAGNOSTICS.parquet"; atomic_frame(rows_path, pd.DataFrame(coordinate_rows))
    primitive_path = OUT / "per_record/PRIMITIVE_CONTEXT.parquet"; atomic_frame(primitive_path, pd.DataFrame(primitive_rows))
    manifest = {"schema_version": "mcvr-lsgo-mechanism-coordinate-freeze-v1", "status": "FROZEN_BEFORE_XTB", "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "conditions": sorted({row["condition"] for row in tasks}), "tasks": len(tasks), "source_records": 144, "reference_records": 857, "coordinate_diagnostic_rows": len(coordinate_rows), "primitive_rows": len(primitive_rows),
        "coordinate_path": str(coordinate_path), "coordinate_sha256": file_sha256(coordinate_path), "coordinate_diagnostics_path": str(rows_path), "coordinate_diagnostics_sha256": file_sha256(rows_path), "primitive_context_path": str(primitive_path), "primitive_context_sha256": file_sha256(primitive_path),
        "threshold_freeze_sha256": file_sha256(OUT / "manifests/THRESHOLD_FREEZE.json"), "dataset_identity_sha256": file_sha256(OUT / "DATASET_IDENTITY.json"), "config_sha256": file_sha256(CONFIG_PATH), "runtime_seconds": time.time() - started,
        "xtb_used_for_coordinates": False, "xtb_energy_records_read": 0, "xtb_force_records_read": 0, "formal_test_records_read": 0, "frozen_holdout_records_read": 0}
    atomic_json(OUT / "COORDINATE_FREEZE_MANIFEST.json", manifest)
    print("LSGO_MECHANISM_COORDINATES_FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
