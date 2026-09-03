#!/usr/bin/env python
"""Generate the frozen prospective ETFlow NFE=10 primary-final sources.

This runner performs source generation only.  It never evaluates a refiner or
computes scientific endpoints.  Each record has a fixed seed derived from the
frozen protocol seed, molecule identity hash, and record index.  Record files
are written atomically and validated before reuse, so an interrupted formal run
can be resumed without changing membership, protocol, or random state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import pickle
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from rdkit import Chem


EXPECTED_PRIMARY_SHA256 = (
    "2a1d07af8c9e3150d1f2f3719d0bd43bd33819ca7674c364d0770c010cb86ee1"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "a24ae9a1fed2708696929308ed1dc10ab167fd66a2d51c44a4afb6c11badccb2"
)
RECORDS_PER_MOLECULE = 2
MAX_ENGINEERING_ATTEMPTS = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def record_seed(protocol_seed: int, identity_sha256: str, record_index: int) -> int:
    token = f"{protocol_seed}\0{identity_sha256}\0{record_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") & (2**63 - 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonical_identity(mol: Chem.Mol) -> str:
    normalized = Chem.Mol(mol)
    for atom in normalized.GetAtoms():
        atom.SetAtomMapNum(0)
    normalized = Chem.RemoveHs(normalized, sanitize=True)
    Chem.SanitizeMol(normalized)
    return Chem.MolToSmiles(normalized, canonical=True, isomericSmiles=True)


def atom_order_identity(mol: Chem.Mol) -> str:
    atoms = [
        {
            "index": atom.GetIdx(),
            "atomic_number": atom.GetAtomicNum(),
            "formal_charge": atom.GetFormalCharge(),
            "isotope": atom.GetIsotope(),
            "chiral_tag": str(atom.GetChiralTag()),
        }
        for atom in mol.GetAtoms()
    ]
    bonds = [
        {
            "begin": min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            "end": max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            "bond_type": str(bond.GetBondType()),
            "stereo": str(bond.GetStereo()),
        }
        for bond in mol.GetBonds()
    ]
    bonds.sort(key=lambda row: (row["begin"], row["end"], row["bond_type"], row["stereo"]))
    return canonical_json_sha256({"atoms": atoms, "bonds": bonds})


class ReferenceLoader:
    def __init__(self, source_root: Path):
        self.source_root = source_root.resolve()
        self._path: Path | None = None
        self._payload: Mapping[str, Any] | None = None

    def load(self, locator: str) -> tuple[Chem.Mol, Path, str]:
        relative, separator, raw_key = locator.partition("::")
        if not separator or not relative or not raw_key:
            raise ValueError(f"Invalid reference locator: {locator}")
        path = (self.source_root / Path(relative)).resolve()
        if self.source_root not in path.parents:
            raise ValueError(f"Reference locator escapes source root: {locator}")
        if path != self._path:
            with path.open("rb") as stream:
                payload = pickle.load(stream)
            if not isinstance(payload, Mapping):
                raise TypeError(f"Standardized pickle is not a mapping: {path}")
            self._path, self._payload = path, payload
        assert self._payload is not None
        entry = self._payload.get(raw_key)
        if not isinstance(entry, Mapping):
            raise KeyError(f"Molecule key absent from {path}: {raw_key}")
        conformers = entry.get("conformers")
        if not isinstance(conformers, list) or not conformers:
            raise ValueError(f"Reference conformer absent: {locator}")
        first = conformers[0]
        mol = first.get("rd_mol") if isinstance(first, Mapping) else None
        if mol is None:
            raise ValueError(f"Reference rd_mol absent: {locator}")
        return Chem.Mol(mol), path, raw_key


def load_runtime(etflow_root: Path):
    root = etflow_root.resolve()
    scripts = root / "scripts"
    sys.path.insert(0, str(scripts))
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        "_sixs_prospective_etflow_utils", scripts / "utils.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load ETFlow scripts/utils.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    feature_module = importlib.import_module("etflow.commons.featurization")
    batch_module = importlib.import_module("torch_geometric.data")
    return module, feature_module.MoleculeFeaturizer, feature_module.MoleculeData, batch_module.Batch


def featurize(mol: Chem.Mol, featurizer: Any, molecule_data: Any) -> Any:
    graph_mol = Chem.Mol(mol)
    graph_mol.RemoveAllConformers()
    atomic_numbers = featurizer.get_atomic_numbers_from_mol(graph_mol)
    edge_index, edge_attr = featurizer.get_edge_index_from_mol(graph_mol, False)
    node_attr = featurizer.get_atom_features_from_mol(graph_mol)
    chiral_index, chiral_nbr_index, chiral_tag = (
        featurizer.get_chiral_centers_from_mol(graph_mol)
    )
    rotatable_bond_index, atom_bond_influence_index = (
        featurizer.get_rotatable_bond_features_from_mol(graph_mol)
    )
    return molecule_data(
        num_nodes=int(atomic_numbers.numel()),
        atomic_numbers=atomic_numbers,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_attr=node_attr,
        chiral_index=chiral_index,
        chiral_nbr_index=chiral_nbr_index,
        chiral_tag=chiral_tag,
        rotatable_bond_index=rotatable_bond_index,
        atom_bond_influence_index=atom_bond_influence_index,
    )


def load_model(utils: Any, config: Mapping[str, Any], checkpoint: Path, device: torch.device):
    if config.get("model") != "BaseFlow":
        raise ValueError("Protocol requires official BaseFlow")
    model = utils.instantiate_model(config["model"], dict(config["model_args"]))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "state_dict" not in payload:
        raise ValueError("Checkpoint has no state_dict")
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()


def sample_record(
    model: Any,
    data: Any,
    batch_class: Any,
    sampler_args: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    set_seed(seed)
    batch = batch_class.from_data_list([data])
    with torch.inference_mode():
        coordinates = model.sample(
            batch["atomic_numbers"].to(device),
            batch["edge_index"].to(device),
            batch["batch"].to(device),
            node_attr=batch["node_attr"].to(device),
            chiral_index=batch["chiral_index"].to(device),
            chiral_nbr_index=batch["chiral_nbr_index"].to(device),
            chiral_tag=batch["chiral_tag"].to(device),
            rotatable_bond_index=batch["rotatable_bond_index"].to(device),
            atom_bond_influence_index=batch["atom_bond_influence_index"].to(device),
            **dict(sampler_args),
        )
    coordinates = coordinates.reshape(int(data.num_nodes), 3).detach().cpu().float()
    if tuple(coordinates.shape) != (int(data.num_nodes), 3):
        raise ValueError(f"Unexpected coordinate shape: {tuple(coordinates.shape)}")
    if not bool(torch.isfinite(coordinates).all()):
        raise ValueError("Generated coordinates contain NaN or Inf")
    return coordinates


def coordinate_sha256(coordinates: torch.Tensor) -> str:
    array = np.ascontiguousarray(coordinates.numpy(), dtype="<f4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def record_path(output_dir: Path, row: Mapping[str, Any], record_index: int) -> Path:
    return output_dir / "records" / (
        f"{int(row['selection_rank']):06d}_"
        f"{row['molecule_identity_sha256'][:16]}_r{record_index}.pt"
    )


def validate_saved_record(
    path: Path,
    row: Mapping[str, Any],
    record_index: int,
    protocol_sha256: str,
    primary_sha256: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "protocol_sha256": protocol_sha256,
        "primary_manifest_sha256": primary_sha256,
        "molecule_identity_sha256": row["molecule_identity_sha256"],
        "final_molecule_index": int(row["selection_rank"]),
        "etflow_record_index": record_index,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Existing record mismatch for {key}: {path}")
    coordinates = torch.as_tensor(payload["source_coordinates"], dtype=torch.float32)
    atomic_numbers = torch.as_tensor(payload["atomic_numbers"], dtype=torch.int32)
    if tuple(coordinates.shape) != (int(atomic_numbers.numel()), 3):
        raise ValueError(f"Existing record shape mismatch: {path}")
    if not bool(torch.isfinite(coordinates).all()):
        raise ValueError(f"Existing record is nonfinite: {path}")
    if coordinate_sha256(coordinates) != payload.get("coordinate_sha256"):
        raise ValueError(f"Existing coordinate hash mismatch: {path}")
    return manifest_row(path, payload)


def manifest_row(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "record_id": payload["record_id"],
        "final_molecule_index": payload["final_molecule_index"],
        "canonical_identity": payload["canonical_identity"],
        "molecule_identity_sha256": payload["molecule_identity_sha256"],
        "etflow_record_index": payload["etflow_record_index"],
        "record_seed": payload["record_seed"],
        "num_atoms": int(torch.as_tensor(payload["atomic_numbers"]).numel()),
        "atom_order_identity_sha256": payload["atom_order_identity_sha256"],
        "coordinate_sha256": payload["coordinate_sha256"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "protocol_sha256": payload["protocol_sha256"],
        "record_asset": str(path.resolve()),
        "record_asset_sha256": sha256_file(path),
    }


def main(args: argparse.Namespace) -> int:
    started = time.time()
    manifest_path = args.manifest.resolve()
    protocol_path = args.protocol.resolve()
    checkpoint = args.checkpoint.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_sha = sha256_file(manifest_path)
    checkpoint_sha = sha256_file(checkpoint)
    if primary_sha != EXPECTED_PRIMARY_SHA256:
        raise RuntimeError(f"Primary manifest SHA256 mismatch: {primary_sha}")
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint SHA256 mismatch: {checkpoint_sha}")
    protocol_sha = sha256_file(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["primary_final_manifest_sha256"] != primary_sha:
        raise RuntimeError("Protocol/primary manifest binding mismatch")
    if protocol["checkpoint_sha256"] != checkpoint_sha:
        raise RuntimeError("Protocol/checkpoint binding mismatch")
    if protocol["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("Protocol/config binding mismatch")
    if protocol["generation_runner_sha256"] != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Protocol/generation-runner binding mismatch")
    observed_commit = subprocess.check_output(
        ["git", "-C", str(args.etflow_root.resolve()), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if observed_commit != protocol["etflow_code"]["git_commit"]:
        raise RuntimeError("ETFlow code commit differs from frozen protocol")
    expected_source_root = Path(protocol["input_preprocessing"]["source_root"]).resolve()
    if args.source_root.resolve() != expected_source_root:
        raise RuntimeError("Source root differs from frozen protocol")
    expected_output_root = Path(protocol["output"]["root"]).resolve()
    if args.mode == "formal" and output_dir != expected_output_root:
        raise RuntimeError("Formal output root differs from frozen protocol")

    primary = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_rows = list(primary["rows"])
    if len(all_rows) != 2500 or int(primary["target_records"]) != 5000:
        raise RuntimeError("Frozen primary manifest count mismatch")
    rows = all_rows[: args.smoke_molecules] if args.mode == "smoke" else all_rows

    utils, featurizer_class, molecule_data, batch_class = load_runtime(args.etflow_root)
    config = utils.read_yaml(str(config_path))
    sampler_args = dict(config["eval_args"]["sampler_args"])
    if sampler_args != protocol["sampler_args"] or sampler_args.get("n_timesteps") != 10:
        raise RuntimeError("Runtime sampler arguments differ from frozen protocol")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; ETFlow generation did not start")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = load_model(utils, config, checkpoint, device)
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("ETFlow model is not on CUDA")
    featurizer = featurizer_class()
    references = ReferenceLoader(args.source_root)

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    existing_validated = 0
    first_generated: tuple[Any, Any, int, torch.Tensor] | None = None
    for molecule_position, row in enumerate(rows, start=1):
        try:
            mol, source_path, source_key = references.load(row["reference_identity"])
            observed_identity = canonical_identity(mol)
            if observed_identity != row["molecule_id"]:
                raise ValueError("Canonical identity differs from frozen manifest")
            topology_sha = atom_order_identity(mol)
            data = featurize(mol, featurizer, molecule_data)
            atomic_numbers = torch.tensor(
                [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.int32
            )
            if not torch.equal(atomic_numbers, data.atomic_numbers.to(torch.int32)):
                raise ValueError("Featurization changed atom ordering")
            for record_index in range(RECORDS_PER_MOLECULE):
                path = record_path(output_dir, row, record_index)
                if path.exists():
                    completed.append(
                        validate_saved_record(
                            path, row, record_index, protocol_sha, primary_sha
                        )
                    )
                    existing_validated += 1
                    continue
                seed = record_seed(
                    int(protocol["randomness"]["protocol_seed"]),
                    row["molecule_identity_sha256"],
                    record_index,
                )
                last_error = None
                for attempt in range(1, MAX_ENGINEERING_ATTEMPTS + 1):
                    try:
                        coordinates = sample_record(
                            model, data, batch_class, sampler_args, device, seed
                        )
                        payload = {
                            "schema_version": "sixs-prospective-etflow-source-record-v1",
                            "record_id": (
                                f"primary-final-{int(row['selection_rank']):06d}-"
                                f"etflow-{record_index}"
                            ),
                            "final_molecule_index": int(row["selection_rank"]),
                            "canonical_identity": row["molecule_id"],
                            "molecule_identity_sha256": row["molecule_identity_sha256"],
                            "reference_identity": row["reference_identity"],
                            "reference_source_path": str(source_path),
                            "reference_source_key": source_key,
                            "etflow_record_index": record_index,
                            "record_seed": seed,
                            "atomic_numbers": atomic_numbers,
                            "atom_order_identity_sha256": topology_sha,
                            "source_coordinates": coordinates,
                            "coordinate_sha256": coordinate_sha256(coordinates),
                            "coordinate_dtype": "float32",
                            "coordinate_units": "angstrom",
                            "checkpoint_sha256": checkpoint_sha,
                            "config_sha256": sha256_file(config_path),
                            "primary_manifest_sha256": primary_sha,
                            "protocol_sha256": protocol_sha,
                            "sampler_args": sampler_args,
                            "engineering_attempt": attempt,
                        }
                        atomic_torch_save(path, payload)
                        completed.append(
                            validate_saved_record(
                                path, row, record_index, protocol_sha, primary_sha
                            )
                        )
                        if first_generated is None:
                            first_generated = (data, row, seed, coordinates.clone())
                        last_error = None
                        break
                    except Exception as exc:  # finite, predeclared engineering retry
                        last_error = exc
                if last_error is not None:
                    raise last_error
        except Exception as exc:
            failures.append(
                {
                    "final_molecule_index": int(row["selection_rank"]),
                    "canonical_identity": row["molecule_id"],
                    "molecule_identity_sha256": row["molecule_identity_sha256"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        atomic_json(
            output_dir / "RUN_STATUS.json",
            {
                "status": "RUNNING",
                "mode": args.mode,
                "molecules_visited": molecule_position,
                "molecules_target": len(rows),
                "records_complete": len(completed),
                "failures": len(failures),
                "protocol_sha256": protocol_sha,
                "pid": os.getpid(),
            },
        )
        if molecule_position % 25 == 0 or molecule_position == len(rows):
            print(
                f"ETFLOW_SOURCE_PROGRESS molecules={molecule_position}/{len(rows)} "
                f"records={len(completed)} failures={len(failures)}",
                flush=True,
            )

    determinism_exact = None
    determinism_within_tolerance = None
    determinism_max_abs = None
    if args.mode == "smoke" and completed:
        first_row = rows[0]
        first_mol, _, _ = references.load(first_row["reference_identity"])
        first_data = featurize(first_mol, featurizer, molecule_data)
        first_seed = record_seed(
            int(protocol["randomness"]["protocol_seed"]),
            first_row["molecule_identity_sha256"],
            0,
        )
        regenerated = sample_record(
            model, first_data, batch_class, sampler_args, device, first_seed
        )
        saved_path = record_path(output_dir, first_row, 0)
        saved = torch.load(saved_path, map_location="cpu", weights_only=False)[
            "source_coordinates"
        ]
        delta = torch.max(torch.abs(regenerated - saved)).item()
        determinism_max_abs = float(delta)
        determinism_exact = bool(torch.equal(regenerated, saved))
        determinism_within_tolerance = bool(
            delta <= float(protocol["randomness"]["repeatability_atol_angstrom"])
        )
        if not determinism_within_tolerance:
            failures.append(
                {
                    "error_type": "DeterminismMismatch",
                    "error": f"Repeated seeded inference max_abs={delta}",
                }
            )

    completed.sort(key=lambda item: (item["final_molecule_index"], item["etflow_record_index"]))
    atomic_jsonl(output_dir / "SOURCE_RECORD_MANIFEST.jsonl", completed)
    atomic_json(output_dir / "GENERATION_FAILURES.json", failures)
    expected_records = len(rows) * RECORDS_PER_MOLECULE
    status = "PASS" if len(completed) == expected_records and not failures else "FAILED"
    unique_molecules = len({row["final_molecule_index"] for row in completed})
    final = {
        "schema_version": "sixs-prospective-etflow-source-status-v1",
        "status": status,
        "mode": args.mode,
        "pid": os.getpid(),
        "input_molecules": len(rows),
        "target_records": expected_records,
        "generated_or_validated_records": len(completed),
        "unique_molecules_with_records": unique_molecules,
        "missing_molecules": len(rows) - unique_molecules,
        "failed_molecules": len(failures),
        "existing_records_validated": existing_validated,
        "nfe": sampler_args["n_timesteps"],
        "sampler_args": sampler_args,
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": sha256_file(config_path),
        "primary_manifest_sha256": primary_sha,
        "protocol_sha256": protocol_sha,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "model_device": str(next(model.parameters()).device),
        "seed_rule": protocol["randomness"]["seed_rule"],
        "determinism_repeat_exact": determinism_exact,
        "determinism_repeat_within_tolerance": determinism_within_tolerance,
        "determinism_repeat_atol_angstrom": protocol["randomness"][
            "repeatability_atol_angstrom"
        ],
        "determinism_repeat_max_abs": determinism_max_abs,
        "elapsed_seconds": time.time() - started,
        "scientific_outcome_read": False,
        "primary_membership_changed": False,
    }
    atomic_json(output_dir / "FINAL_STATUS.json", final)
    return 0 if status == "PASS" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--etflow-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smoke-molecules", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        raise SystemExit(main(arguments))
    except SystemExit:
        raise
    except Exception as error:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            arguments.output_dir / "FINAL_STATUS.json",
            {
                "status": "ENGINEERING_FAILURE",
                "mode": arguments.mode,
                "pid": os.getpid(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "scientific_outcome_read": False,
                "primary_membership_changed": False,
            },
        )
        raise
