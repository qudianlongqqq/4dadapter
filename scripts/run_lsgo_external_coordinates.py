#!/usr/bin/env python3
"""Freeze the one-shot LSGO external coordinates after the internal gate."""

from __future__ import annotations

import json
import os
import sys
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
OUT = ROOT / "reports/ecir_mvr/learned_geometry"
CONFIG_PATH = ROOT / "configs/ecir_mvr_lsgo_pilot.yaml"

from etflow.ecir.learned_geometry import (  # noqa: E402
    direct_gradient_update,
    distribution_parameters,
    prepare_graph,
    safety_accept,
)
from etflow.ecir.lsgo_io import atomic_json, file_sha256, nearest_reference_metrics  # noqa: E402
from scripts.run_mcvr_lsgo import load_selected, verify_calibration  # noqa: E402


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def base_molecule(item: dict, sample_id: str) -> tuple[Chem.Mol, list[int]]:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = True
    molecule = Chem.MolFromSmiles(str(item["record"]["smiles"]), params)
    if molecule is None:
        raise RuntimeError(f"SMILES reconstruction failed: {sample_id}")
    atom_maps = [atom.GetAtomMapNum() for atom in molecule.GetAtoms()]
    if sorted(atom_maps) != list(range(molecule.GetNumAtoms())):
        raise RuntimeError(f"atom-map identity failed: {sample_id}")
    if molecule.GetNumAtoms() != int(item["sources"].shape[1]):
        raise RuntimeError(f"atom denominator changed: {sample_id}")
    molecule.SetProp("_Name", sample_id)
    molecule.SetProp("sample_id", sample_id)
    molecule.SetProp("partition", "LSGO_EXTERNAL_CONFIRM")
    return molecule, atom_maps


def write_sdf(path: Path, items: list[dict], rows: pd.DataFrame, method: str) -> None:
    lookup = {str(item["sample_ids"][index]): item for item in items for index in range(3)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    writer = Chem.SDWriter(str(temporary))
    writer.SetKekulize(False)
    try:
        for row in rows.itertuples(index=False):
            sample_id = str(row.sample_id)
            molecule, atom_maps = base_molecule(lookup[sample_id], sample_id)
            nested = row.source_coordinates if method == "Source" else row.output_coordinates
            coordinates = np.stack([np.asarray(value, dtype=np.float64) for value in nested])
            if coordinates.shape != (molecule.GetNumAtoms(), 3) or not np.isfinite(coordinates).all():
                raise RuntimeError(f"coordinate identity failed: {method}/{sample_id}")
            conformer = Chem.Conformer(molecule.GetNumAtoms())
            for atom_index, cache_index in enumerate(atom_maps):
                conformer.SetAtomPosition(atom_index, coordinates[cache_index].tolist())
            molecule.RemoveAllConformers()
            molecule.AddConformer(conformer, assignId=True)
            molecule.SetProp("method", method)
            molecule.SetProp("candidate", "primary")
            molecule.SetIntProp("accepted", int(row.accepted if method != "Source" else 1))
            molecule.SetIntProp("fallback", int(row.fallback if method != "Source" else 0))
            writer.write(molecule)
    finally:
        writer.close()
    os.replace(temporary, path)


def coordinate_rows(
    items: list[dict], calibration: dict, *, variant: str, model, seed: int | None,
    budget: float, atom_cap: float, device: torch.device,
) -> pd.DataFrame:
    rows: list[dict] = []
    for molecule_index, item in enumerate(items):
        graph = prepare_graph(item["record"], calibration).to(device)
        parameters = distribution_parameters(graph, model=model, variant=variant)
        references = torch.as_tensor(item["references"], dtype=torch.float32)
        for source_index, source_cpu in enumerate(torch.as_tensor(item["sources"])):
            source = source_cpu.to(device=device, dtype=torch.float64)
            update = direct_gradient_update(
                source, graph, parameters, rms_budget=budget, atom_cap=atom_cap, steps=1,
            )
            output, safety = safety_accept(source, update["coordinates"], graph)
            delta = output - source
            source_mode, source_rmsd = nearest_reference_metrics(source.cpu().float(), references)
            output_mode, output_rmsd = nearest_reference_metrics(output.cpu().float(), references)
            rows.append({
                "method": f"{variant}-G", "variant": variant, "seed": seed,
                "molecule_id": str(item["molecule_id"]),
                "sample_id": str(item["sample_ids"][source_index]),
                "source_index": source_index,
                "source_coordinates": source.cpu().tolist(),
                "output_coordinates": output.cpu().tolist(),
                "finite": bool(update["finite"]), "accepted": bool(safety["accepted"]),
                "fallback": bool(safety["fallback"]),
                "chirality_preserved": bool(safety["chirality_preserved"]),
                "ring_nonregression": bool(safety["ring_nonregression"]),
                "catastrophic_clash_nonregression": bool(safety["catastrophic_clash_nonregression"]),
                "graph_rms_movement": float(torch.sqrt(delta.square().sum(-1).mean())),
                "max_atom_movement": float(torch.linalg.vector_norm(delta, dim=-1).max()),
                "source_nearest_reference_mode": source_mode,
                "output_nearest_reference_mode": output_mode,
                "source_nearest_reference_rmsd": source_rmsd,
                "output_nearest_reference_rmsd": output_rmsd,
                "mode_switch": bool(source_mode != output_mode),
                "budget_angstrom": budget, "steps": 1,
            })
        if (molecule_index + 1) % 25 == 0:
            print(f"LSGO EXTERNAL {variant}-G seed={seed} {molecule_index + 1}/{len(items)}", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    started = time.time()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    decision_path = OUT / "INTERNAL_SELECTION_REPORT.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") != "INTERNAL_GATE_PASSED" or not decision.get("external_evaluation_authorized"):
        raise RuntimeError("external evaluation remains locked")
    if decision.get("primary_variant") != "B" or decision.get("uncertainty_meaningful") is not False:
        raise RuntimeError("unexpected internal selection identity")
    if set(decision.get("uncertainty_by_seed", {}).values()) != {"SIGMA_INFLATION"}:
        raise RuntimeError("Variant C stop-rule identity changed")
    budget = float(decision["selected_primary_budget_angstrom"])
    if budget != float(config["external_evaluation"]["movement_max_angstrom"]):
        raise RuntimeError("selected movement budget changed")

    identity = json.loads((OUT / "DATASET_IDENTITY.json").read_text(encoding="utf-8"))
    if int(identity["formal_test_records_read"]) != 0 or int(identity["frozen_holdout_records_read"]) != 0:
        raise RuntimeError("protected split access detected")
    compact_path = ROOT / config["dataset"]["external_compact"]
    if file_sha256(compact_path) != identity["external_compact_sha256"]:
        raise RuntimeError("external-confirm compact SHA changed")
    compact = torch.load(compact_path, map_location="cpu", weights_only=False)
    items = list(compact["items"])
    if len(items) != 200 or sum(len(item["sample_ids"]) for item in items) != 600:
        raise RuntimeError("external-confirm denominator changed")
    calibration_path = Path(config["dataset"]["drcsr_calibration"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    verify_calibration(calibration)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    atom_cap = float(config["internal_selection"]["atom_cap_angstrom"])
    exports: list[dict] = []

    a_frame = coordinate_rows(
        items, calibration, variant="A", model=None, seed=None,
        budget=budget, atom_cap=atom_cap, device=device,
    )
    source_sdf = OUT / "sdf/Source__primary.sdf"
    write_sdf(source_sdf, items, a_frame, "Source")
    exports.append({
        "method": "Source", "variant": "Source", "seed": None, "records": 600,
        "sdf_path": str(source_sdf), "sdf_sha256": file_sha256(source_sdf),
    })
    a_path = OUT / "per_record/coordinates/A-G__primary.parquet"
    atomic_frame(a_path, a_frame)
    a_sdf = OUT / "sdf/A-G__primary.sdf"
    write_sdf(a_sdf, items, a_frame, "A-G")
    exports.append({
        "method": "A-G", "variant": "A", "seed": None, "records": 600,
        "coordinate_path": str(a_path), "coordinate_sha256": file_sha256(a_path),
        "sdf_path": str(a_sdf), "sdf_sha256": file_sha256(a_sdf),
    })

    checkpoint_manifest = json.loads((OUT / "CHECKPOINT_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    frozen_lookup = {(row["variant"], int(row["seed"])): row for row in checkpoint_manifest["checkpoints"]}
    for seed in config["seeds"]:
        model, _, checkpoint_path = load_selected(config, "B", int(seed), device)
        frozen = frozen_lookup[("B", int(seed))]
        if file_sha256(checkpoint_path) != frozen["sha256"]:
            raise RuntimeError(f"checkpoint SHA changed: B/{seed}")
        frame = coordinate_rows(
            items, calibration, variant="B", model=model, seed=int(seed),
            budget=budget, atom_cap=atom_cap, device=device,
        )
        coordinate_path = OUT / f"per_record/coordinates/B-G_seed{seed}__primary.parquet"
        atomic_frame(coordinate_path, frame)
        sdf_path = OUT / f"sdf/B-G_seed{seed}__primary.sdf"
        write_sdf(sdf_path, items, frame, f"B-G_seed{seed}")
        exports.append({
            "method": f"B-G_seed{seed}", "variant": "B", "seed": int(seed), "records": 600,
            "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": frozen["sha256"],
            "coordinate_path": str(coordinate_path), "coordinate_sha256": file_sha256(coordinate_path),
            "sdf_path": str(sdf_path), "sdf_sha256": file_sha256(sdf_path),
        })

    expected = [str(item["sample_ids"][index]) for item in items for index in range(3)]
    for export in exports:
        molecules = [molecule for molecule in Chem.ForwardSDMolSupplier(export["sdf_path"], sanitize=False, removeHs=False)]
        sample_ids = [molecule.GetProp("sample_id") for molecule in molecules if molecule is not None]
        if sample_ids != expected:
            raise RuntimeError(f"SDF paired order changed: {export['method']}")
    payload = {
        "schema_version": "mcvr-lsgo-coordinate-freeze-v1", "status": "FROZEN",
        "internal_selection_path": str(decision_path),
        "internal_selection_sha256": file_sha256(decision_path),
        "dataset_identity_sha256": identity["identity_sha256"],
        "external_compact_path": str(compact_path), "external_compact_sha256": file_sha256(compact_path),
        "primary_variant": "B", "primary_budget_angstrom": budget, "primary_steps": 1,
        "secondary_steps": 2, "secondary_external_evaluation_run": False,
        "conditions": ["Source", "A-G", "B-G_seed173", "B-G_seed181", "B-G_seed193"],
        "stopped_conditions": {
            "C-G": "Variant C stopped before external evaluation: SIGMA_INFLATION and DEV_B likelihood failure",
            "C-P": "Variant C stopped before external evaluation: SIGMA_INFLATION and DEV_B likelihood failure",
        },
        "exports": exports, "coordinate_condition_count": 4, "source_plus_condition_count": 5,
        "records_per_condition": 600, "runtime_seconds": time.time() - started,
        "external_evaluation_unlocked": True,
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
        "full10k_used_for_tuning": False, "posebusters_used_for_selection": False,
        "xtb_used_for_selection": False, "weighted_bac_used_for_selection": False,
    }
    atomic_json(OUT / "COORDINATE_FREEZE_MANIFEST.json", payload)
    print("LSGO_EXTERNAL_COORDINATES_FROZEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
