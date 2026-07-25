#!/usr/bin/env python3
"""Generate and atomically freeze prospective Source/BA/BA+C coordinates."""

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
from etflow.ecir.bat_refinement import combined_gradient_update, frozen_ba_update, prepare_bat_graph, steric_metrics
from etflow.ecir.learned_geometry import LearnedGeometryObjective, distribution_parameters, prepare_graph
from etflow.ecir.lsgo_io import atomic_json, file_sha256, nearest_reference_metrics

OUT = ROOT / "reports/ecir_mvr/bat_refinement"
CONFIG_PATH = ROOT / "configs/ecir_mvr_bat_refinement.yaml"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False); os.replace(temporary, path)


def load_model(row, device):
    path = Path(row["path"])
    if file_sha256(path) != row["sha256"]:
        raise RuntimeError("BA checkpoint SHA mismatch")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = LearnedGeometryObjective(hidden_dim=128, layers=3, learned_sigma=False).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model


def base_molecule(item, sample_id):
    params = Chem.SmilesParserParams(); params.removeHs = False; params.sanitize = True
    molecule = Chem.MolFromSmiles(str(item["record"]["smiles"]), params)
    if molecule is None: raise RuntimeError(f"SMILES failed: {sample_id}")
    maps = [atom.GetAtomMapNum() for atom in molecule.GetAtoms()]
    if sorted(maps) != list(range(molecule.GetNumAtoms())): raise RuntimeError("atom map identity failed")
    molecule.SetProp("_Name", sample_id); molecule.SetProp("sample_id", sample_id); molecule.SetProp("partition", "BAT_EXTERNAL_CONFIRM")
    return molecule, maps


def write_sdf(path, items, frame, method):
    lookup = {str(item["sample_ids"][index]): item for item in items for index in range(3)}
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    writer = Chem.SDWriter(str(temporary)); writer.SetKekulize(False)
    try:
        for row in frame.itertuples(index=False):
            molecule, maps = base_molecule(lookup[str(row.sample_id)], str(row.sample_id))
            nested = row.source_coordinates if method == "Source" else row.output_coordinates
            coordinates = np.asarray(nested, dtype=np.float64)
            conformer = Chem.Conformer(molecule.GetNumAtoms())
            for atom_index, cache_index in enumerate(maps): conformer.SetAtomPosition(atom_index, coordinates[cache_index].tolist())
            molecule.RemoveAllConformers(); molecule.AddConformer(conformer, assignId=True)
            molecule.SetProp("method", method); molecule.SetProp("candidate", "primary")
            writer.write(molecule)
    finally:
        writer.close()
    os.replace(temporary, path)


def coordinate_rows(items, calibration, model, seed, method, device, config):
    records = []
    for molecule_index, item in enumerate(items):
        base = prepare_graph(item["record"], calibration).to(device)
        bat = prepare_bat_graph(base, item["record"], safe_factor_nonbonded=config["steric"]["safe_factor_nonbonded"], safe_factor_1_4=config["steric"]["safe_factor_1_4"], catastrophic_factor=config["steric"]["catastrophic_factor"]).to(device)
        with torch.no_grad(): parameters = distribution_parameters(base, model=model, variant="B")
        references = torch.as_tensor(item["references"], dtype=torch.float32)
        for source_index, source_cpu in enumerate(item["sources"]):
            source = torch.as_tensor(source_cpu, dtype=torch.float64, device=device)
            ba = frozen_ba_update(source, bat, parameters, rms_budget=.003, atom_cap=.03)
            if method == "BA":
                output, diagnostics = ba["coordinates"], {"fallback": ba["safety"]["fallback"], "rejected": False, "active_steric_count": 0, "backtracking_fraction": 1.0, "safety": ba["safety"]}
            else:
                diagnostics = combined_gradient_update(source, bat, parameters, rms_budget=.003, atom_cap=.03, tau=config["steric"]["tau_angstrom"], fallback_coordinates=ba["coordinates"], backtracking_fractions=config["steric"]["backtracking_fractions"])
                output = diagnostics["coordinates"]
            delta = output - source
            source_mode, source_rmsd = nearest_reference_metrics(source.cpu().float(), references)
            output_mode, output_rmsd = nearest_reference_metrics(output.cpu().float(), references)
            before, after = steric_metrics(source, bat), steric_metrics(output, bat)
            records.append({
                "method": f"{method}_seed{seed}", "seed": seed, "molecule_id": item["molecule_id"], "sample_id": item["sample_ids"][source_index], "source_index": source_index,
                "source_coordinates": source.cpu().tolist(), "output_coordinates": output.cpu().tolist(),
                "accepted": diagnostics["safety"]["accepted"], "fallback": diagnostics["fallback"], "rejected": diagnostics["rejected"],
                "chirality_preserved": diagnostics["safety"]["chirality_preserved"], "ring_nonregression": diagnostics["safety"]["ring_nonregression"],
                "graph_rms_movement": float(torch.sqrt(delta.square().sum(-1).mean())), "max_atom_movement": float(torch.linalg.vector_norm(delta, dim=-1).max()),
                "source_nearest_reference_mode": source_mode, "output_nearest_reference_mode": output_mode, "source_nearest_reference_rmsd": source_rmsd, "output_nearest_reference_rmsd": output_rmsd, "mode_switch": source_mode != output_mode,
                "source_steric_violations": before["violation_count"], "output_steric_violations": after["violation_count"], "source_penetration_sum": before["penetration_sum"], "output_penetration_sum": after["penetration_sum"], "active_steric_count": diagnostics["active_steric_count"], "backtracking_fraction": diagnostics["backtracking_fraction"],
            })
        if (molecule_index + 1) % 25 == 0: print(f"BAT COORD {method} seed{seed} {molecule_index + 1}/{len(items)}", flush=True)
    return pd.DataFrame(records)


def main() -> int:
    started = time.time(); config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    prereg_path = OUT / "PREREGISTRATION.json"; prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if prereg["status"] != "FROZEN" or prereg["eligible_variant"] != "BA+C": raise RuntimeError("formal candidate not frozen")
    identity = json.loads((OUT / "DATASET_IDENTITY.json").read_text(encoding="utf-8"))
    if identity["formal_test_records_read"] or identity["frozen_holdout_records_read"]: raise RuntimeError("protected split access")
    compact_path = OUT / "manifests/BAT_EXTERNAL_CONFIRM_COMPACT.pt"
    sha = json.loads((OUT / "DATASET_IDENTITY_SHA256.json").read_text(encoding="utf-8"))
    if file_sha256(compact_path) != sha["external_compact_sha256"]: raise RuntimeError("external compact SHA mismatch")
    compact = torch.load(compact_path, map_location="cpu", weights_only=False); items = list(compact["items"])
    if len(items) != 200 or sum(len(item["sample_ids"]) for item in items) != 600: raise RuntimeError("external denominator changed")
    calibration = json.loads(Path(config["dataset"]["drcsr_calibration"]).read_text(encoding="utf-8"))
    checkpoints = json.loads((OUT / "CHECKPOINT_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))["ba_checkpoints"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exports, source_written = [], False
    expected = [str(item["sample_ids"][index]) for item in items for index in range(3)]
    for row in checkpoints:
        seed = int(row["seed"]); model = load_model(row, device)
        for method in ("BA", "BA+C"):
            frame = coordinate_rows(items, calibration, model, seed, method, device, config)
            if not source_written:
                source_path = OUT / "sdf/Source__primary.sdf"; write_sdf(source_path, items, frame, "Source")
                exports.append({"method": "Source", "seed": None, "records": 600, "sdf_path": str(source_path), "sdf_sha256": file_sha256(source_path)})
                source_written = True
            name = f"{method}_seed{seed}"; coordinate_path = OUT / f"per_record/coordinates/{name}__primary.parquet"; atomic_frame(coordinate_path, frame)
            sdf_path = OUT / f"sdf/{name}__primary.sdf"; write_sdf(sdf_path, items, frame, name)
            exports.append({"method": name, "variant": method, "seed": seed, "records": 600, "checkpoint_sha256": row["sha256"], "coordinate_path": str(coordinate_path), "coordinate_sha256": file_sha256(coordinate_path), "sdf_path": str(sdf_path), "sdf_sha256": file_sha256(sdf_path)})
    for export in exports:
        molecules = [mol for mol in Chem.ForwardSDMolSupplier(export["sdf_path"], sanitize=False, removeHs=False)]
        if [mol.GetProp("sample_id") for mol in molecules if mol is not None] != expected: raise RuntimeError(f"SDF order changed: {export['method']}")
    manifest = {
        "schema_version": "mcvr-bat-coordinate-freeze-v1", "status": "FROZEN", "head": git("rev-parse", "HEAD"),
        "preregistration_sha256": file_sha256(prereg_path), "dataset_identity_sha256": file_sha256(OUT / "DATASET_IDENTITY.json"), "external_compact_sha256": file_sha256(compact_path),
        "conditions": [row["method"] for row in exports], "exports": exports, "records_per_condition": 600,
        "stopped_conditions": {"BA+T": "TORSION_NO_GO", "BAT+C": "TORSION_NO_GO"}, "runtime_seconds": time.time() - started,
        "external_evaluation_unlocked": True, "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
        "posebusters_used_for_selection": False, "xtb_used_for_selection": False,
    }
    atomic_json(OUT / "COORDINATE_FREEZE_MANIFEST.json", manifest)
    print("BAT_EXTERNAL_COORDINATES_FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
