#!/usr/bin/env python
"""Generate deterministic, resumable ETFlow train/val upstream molecules."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch_geometric.data import Batch


GENERATOR_NAME = "ETFlow drugs-o3"
MANIFEST_VERSION = "etflow-formal-large-generation-v1"
STATE_VERSION = "etflow-formal-large-state-v1"
FORMAL_DEFAULTS = {
    "train": {"max_molecules": 50_000, "samples_per_molecule": 3},
    "val": {"max_molecules": 5_000, "samples_per_molecule": 2},
}
REQUIRED_RECORD_FIELDS = {
    "pos_gen",
    "pos_ref",
    "smiles",
    "atomic_numbers",
    "rdmol",
    "dataset_index",
    "split",
    "mol_id",
    "source_mol_id",
    "generator_name",
    "generator_checkpoint",
    "checkpoint_sha256",
    "config_sha256",
    "global_seed",
    "molecule_seed",
    "processed_source_path",
    "processed_source_identity",
    "generation_manifest_sha256",
    "topology_sha256",
    "record_content_sha256",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: Any) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def molecule_seed(global_seed: int, split: str, source_mol_id: str) -> int:
    payload = f"{int(global_seed)}\0{split}\0{source_mol_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        2**63 - 1
    )


def set_molecule_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_stable_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")[:140]
    if not safe:
        safe = "molecule"
    suffix = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]
    return f"{safe}__{suffix}"


def _relative_source(path: Path, processed_data: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(processed_data.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Processed source is outside processed_data: {resolved}"
        ) from exc


def select_molecules(
    data_files: Sequence[Path],
    *,
    processed_data: Path,
    split: str,
    max_molecules: int,
    seed: int,
) -> tuple[list[dict[str, Any]], str]:
    if max_molecules < 1:
        raise ValueError("max_molecules must be positive")
    candidates = []
    source_ids = set()
    stable_ids = set()
    split_index_digest = hashlib.sha256()
    for dataset_index, path_value in enumerate(data_files):
        path = Path(path_value)
        source_id = path.stem
        if source_id in source_ids:
            raise ValueError(f"Duplicate stable source_mol_id {source_id!r}")
        source_ids.add(source_id)
        stable_id = _safe_stable_id(source_id)
        if stable_id in stable_ids:
            raise ValueError(f"Stable output filename collision for {source_id!r}")
        stable_ids.add(stable_id)
        relative = _relative_source(path, processed_data)
        size = path.stat().st_size
        identity_row = {
            "dataset_index": dataset_index,
            "source_mol_id": source_id,
            "relative_source_path": relative,
            "source_file_size": size,
        }
        split_index_digest.update(
            json.dumps(identity_row, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        rank = hashlib.sha256(
            f"{int(seed)}\0{split}\0{source_id}".encode("utf-8")
        ).hexdigest()
        candidates.append(
            (
                rank,
                source_id,
                {
                    **identity_row,
                    "stable_id": stable_id,
                    "output_file": f"{stable_id}.pt",
                    "processed_source_path": str(path.resolve()),
                    "molecule_seed": molecule_seed(seed, split, source_id),
                },
            )
        )
    if len(candidates) < max_molecules:
        raise ValueError(
            f"Requested {max_molecules} {split} molecules, found {len(candidates)}"
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in candidates[:max_molecules]], split_index_digest.hexdigest()


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format_version": manifest["format_version"],
        "split": manifest["split"],
        "seed": manifest["seed"],
        "target_molecules": manifest["target_molecules"],
        "samples_per_molecule": manifest["samples_per_molecule"],
        "generator_name": manifest["generator_name"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "config_sha256": manifest["config_sha256"],
        "processed_split_identity_sha256": manifest[
            "processed_split_identity_sha256"
        ],
        "records": [
            {
                key: row[key]
                for key in (
                    "dataset_index",
                    "source_mol_id",
                    "stable_id",
                    "output_file",
                    "relative_source_path",
                    "source_file_size",
                    "molecule_seed",
                )
            }
            for row in manifest["records"]
        ],
    }


def build_generation_manifest(
    *,
    data_files: Sequence[Path],
    processed_data: Path,
    split: str,
    max_molecules: int,
    samples_per_molecule: int,
    seed: int,
    checkpoint_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    selected, processed_identity = select_molecules(
        data_files,
        processed_data=processed_data,
        split=split,
        max_molecules=max_molecules,
        seed=seed,
    )
    manifest = {
        "format_version": MANIFEST_VERSION,
        "created_at": utc_now(),
        "split": split,
        "seed": int(seed),
        "target_molecules": int(max_molecules),
        "samples_per_molecule": int(samples_per_molecule),
        "generator_name": GENERATOR_NAME,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "processed_data": str(processed_data.resolve()),
        "processed_split_identity_sha256": processed_identity,
        "selection": "sha256(seed, split, stable source_mol_id)",
        "records": selected,
    }
    manifest["manifest_sha256"] = canonical_sha256(_manifest_identity(manifest))
    return manifest


def validate_generation_manifest(
    manifest: Mapping[str, Any], expected: Mapping[str, Any] | None = None
) -> None:
    if str(manifest.get("format_version")) != MANIFEST_VERSION:
        raise ValueError("Unsupported generation manifest format")
    actual_hash = canonical_sha256(_manifest_identity(manifest))
    if str(manifest.get("manifest_sha256")) != actual_hash:
        raise ValueError("Generation manifest hash mismatch")
    records = list(manifest.get("records") or [])
    if len(records) != int(manifest.get("target_molecules", -1)):
        raise ValueError("Generation manifest target count mismatch")
    for key in ("source_mol_id", "dataset_index", "stable_id", "output_file"):
        values = [str(row[key]) for row in records]
        if len(values) != len(set(values)):
            raise ValueError(f"Generation manifest contains duplicate {key}")
    if expected is not None and actual_hash != str(expected.get("manifest_sha256")):
        raise ValueError("Existing generation manifest belongs to different inputs")


def atomic_json_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError(f"Refusing to write through symlink: {destination}")
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def atomic_torch_save(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError(f"Refusing to write through symlink: {destination}")
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    value = getattr(record, name, default)
    return value


def _topology_sha256(rdmol: Any, atomic_numbers: torch.Tensor) -> str:
    atoms = [int(value) for value in atomic_numbers.tolist()]
    if rdmol is None or not callable(getattr(rdmol, "GetAtoms", None)):
        raise ValueError("Processed molecule has no valid RDKit topology")
    mol_atoms = [int(atom.GetAtomicNum()) for atom in rdmol.GetAtoms()]
    if atoms != mol_atoms:
        raise ValueError("RDKit atom order differs from atomic_numbers")
    bonds = sorted(
        (
            min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            str(bond.GetBondType()),
            bool(bond.GetIsAromatic()),
        )
        for bond in rdmol.GetBonds()
    )
    return canonical_sha256({"atomic_numbers": atoms, "bonds": bonds})


def _record_content_identity(record: Any) -> dict[str, Any]:
    return {
        "source_mol_id": str(_field(record, "source_mol_id")),
        "mol_id": str(_field(record, "mol_id")),
        "dataset_index": int(_field(record, "dataset_index")),
        "split": str(_field(record, "split")),
        "generator_name": str(_field(record, "generator_name")),
        "smiles": str(_field(record, "smiles")),
        "checkpoint_sha256": str(_field(record, "checkpoint_sha256")),
        "config_sha256": str(_field(record, "config_sha256")),
        "global_seed": int(_field(record, "global_seed")),
        "molecule_seed": int(_field(record, "molecule_seed")),
        "processed_source_identity": str(
            _field(record, "processed_source_identity")
        ),
        "manifest_sha256": str(_field(record, "generation_manifest_sha256")),
        "topology_sha256": str(_field(record, "topology_sha256")),
        "atomic_numbers_sha256": tensor_sha256(_field(record, "atomic_numbers")),
        "pos_gen_sha256": tensor_sha256(_field(record, "pos_gen")),
        "pos_ref_sha256": tensor_sha256(_field(record, "pos_ref")),
    }


def record_content_sha256(record: Any) -> str:
    return canonical_sha256(_record_content_identity(record))


def validate_generated_record(
    record: Any,
    *,
    manifest: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
) -> dict[str, Any]:
    keys = set(record.keys()) if callable(getattr(record, "keys", None)) else set()
    if keys:
        missing = sorted(REQUIRED_RECORD_FIELDS.difference(keys))
    else:
        missing = sorted(
            key for key in REQUIRED_RECORD_FIELDS if _field(record, key) is None
        )
    if missing:
        raise ValueError(f"Generated molecule is missing fields: {missing}")
    expected = {
        "source_mol_id": manifest_row["source_mol_id"],
        "mol_id": manifest_row["source_mol_id"],
        "dataset_index": manifest_row["dataset_index"],
        "split": manifest["split"],
        "generator_name": GENERATOR_NAME,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "config_sha256": manifest["config_sha256"],
        "global_seed": manifest["seed"],
        "molecule_seed": manifest_row["molecule_seed"],
        "generation_manifest_sha256": manifest["manifest_sha256"],
        "processed_source_identity": canonical_sha256(
            {
                "relative_path": manifest_row["relative_source_path"],
                "size": manifest_row["source_file_size"],
            }
        ),
    }
    for key, value in expected.items():
        if _field(record, key) != value:
            raise ValueError(f"Generated molecule {key} identity mismatch")
    atomic_numbers = torch.as_tensor(_field(record, "atomic_numbers")).long().view(-1)
    if atomic_numbers.numel() < 1:
        raise ValueError("Generated molecule has no atoms")
    if str(_field(record, "smiles")).strip() in {"", "None"}:
        raise ValueError("Generated molecule has no valid SMILES")
    pos_gen = torch.as_tensor(_field(record, "pos_gen"), dtype=torch.float32)
    pos_ref = torch.as_tensor(_field(record, "pos_ref"), dtype=torch.float32)
    expected_samples = int(manifest["samples_per_molecule"])
    if tuple(pos_gen.shape) != (expected_samples, atomic_numbers.numel(), 3):
        raise ValueError(f"pos_gen shape mismatch: {tuple(pos_gen.shape)}")
    if pos_ref.ndim != 3 or pos_ref.size(0) < 1 or tuple(pos_ref.shape[1:]) != (
        atomic_numbers.numel(),
        3,
    ):
        raise ValueError(f"pos_ref shape mismatch: {tuple(pos_ref.shape)}")
    if not bool(torch.isfinite(pos_gen).all()) or not bool(
        torch.isfinite(pos_ref).all()
    ):
        raise ValueError("Generated or reference coordinates are non-finite")
    topology = _topology_sha256(_field(record, "rdmol"), atomic_numbers)
    if topology != str(_field(record, "topology_sha256")):
        raise ValueError("Generated molecule topology hash mismatch")
    content_hash = record_content_sha256(record)
    if content_hash != str(_field(record, "record_content_sha256")):
        raise ValueError("Generated molecule content hash mismatch")
    return {
        "source_mol_id": expected["source_mol_id"],
        "dataset_index": int(expected["dataset_index"]),
        "generated_conformers": int(pos_gen.size(0)),
        "num_atoms": int(atomic_numbers.numel()),
        "record_content_sha256": content_hash,
    }


def load_and_validate_generated_file(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Generated molecule is not a regular file: {path}")
    try:
        record = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Cannot load generated molecule {path}: {exc}") from exc
    return validate_generated_record(
        record, manifest=manifest, manifest_row=manifest_row
    )


def _reference_positions(source_path: Path) -> torch.Tensor:
    raw = torch.load(source_path, map_location="cpu", weights_only=False)
    value = _field(raw, "pos")
    if value is None:
        raise ValueError(f"Processed source has no pos: {source_path}")
    pos_ref = torch.as_tensor(value, dtype=torch.float32).cpu()
    if pos_ref.ndim == 2:
        pos_ref = pos_ref.unsqueeze(0)
    if pos_ref.ndim != 3 or pos_ref.size(0) < 1 or pos_ref.size(-1) != 3:
        raise ValueError(f"Invalid pos_ref shape {tuple(pos_ref.shape)}")
    if not bool(torch.isfinite(pos_ref).all()):
        raise ValueError("Reference coordinates contain NaN or Inf")
    return pos_ref


def _current_source_path(
    row: Mapping[str, Any], *, dataset: Any, processed_data: Path
) -> Path:
    """Resolve an audited source in the current checkout, not an old absolute path."""

    index = int(row["dataset_index"])
    try:
        source_path = Path(dataset.data_files[index]).expanduser().resolve()
    except (IndexError, TypeError) as exc:
        raise ValueError(f"Dataset index {index} is no longer available") from exc
    actual = {
        "source_mol_id": source_path.stem,
        "relative_source_path": _relative_source(source_path, processed_data),
        "source_file_size": source_path.stat().st_size,
    }
    for key, value in actual.items():
        if value != row[key]:
            raise ValueError(
                f"Processed source identity mismatch for dataset index {index}: {key}"
            )
    return source_path


def _verified_object_path(obj: Any, *, label: str, etflow_root: Path) -> Path:
    try:
        source = Path(inspect.getfile(obj)).expanduser().resolve()
    except (TypeError, OSError) as exc:
        raise RuntimeError(f"Cannot determine source path for {label}") from exc
    try:
        source.relative_to(etflow_root)
    except ValueError as exc:
        raise RuntimeError(
            f"{label} resolved outside requested ETFlow root: {source} "
            f"(expected under {etflow_root})"
        ) from exc
    return source


def _load_etflow_runtime(etflow_root: Path) -> SimpleNamespace:
    root = etflow_root.expanduser().resolve()
    package_dir = root / "etflow"
    utils_candidates = (root / "scripts" / "utils.py", root / "utils.py")
    utils_path = next((path for path in utils_candidates if path.is_file()), None)
    if not package_dir.is_dir() or utils_path is None:
        searched = ", ".join(str(path) for path in utils_candidates)
        raise FileNotFoundError(
            f"Invalid original ETFlow root: {root}; expected {package_dir} "
            f"and one of [{searched}]"
        )

    scripts_dir = root / "scripts"
    original_sys_path = list(sys.path)
    saved_etflow_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "etflow" or name.startswith("etflow.")
    }
    module_name = (
        "_formal_large_etflow_utils_"
        + hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    )
    saved_utils_module = sys.modules.get(module_name)
    closed = False

    def restore_import_state() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        for name in list(sys.modules):
            if name == "etflow" or name.startswith("etflow."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_etflow_modules)
        if saved_utils_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = saved_utils_module
        sys.path[:] = original_sys_path
        importlib.invalidate_caches()

    try:
        for name in saved_etflow_modules:
            sys.modules.pop(name, None)
        sys.path[:] = [str(root), str(scripts_dir)] + [
            entry
            for entry in original_sys_path
            if entry not in {str(root), str(scripts_dir)}
        ]
        importlib.invalidate_caches()
        spec = importlib.util.spec_from_file_location(module_name, utils_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import original ETFlow utils from {utils_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        dataset_module = importlib.import_module("etflow.data.dataset")
        dataset_class = getattr(dataset_module, "EuclideanDataset")
        model_class = getattr(module, "BaseFlow", None)
        if model_class is None:
            model_module = importlib.import_module("etflow.models.model")
            model_class = getattr(model_module, "BaseFlow")

        provenance = {
            "utils_module_path": str(
                _verified_object_path(module, label="utils module", etflow_root=root)
            ),
            "dataset_class_path": str(
                _verified_object_path(
                    dataset_class,
                    label="EuclideanDataset",
                    etflow_root=root,
                )
            ),
            "model_class_path": str(
                _verified_object_path(
                    model_class,
                    label="BaseFlow",
                    etflow_root=root,
                )
            ),
            "instantiate_model_path": str(
                _verified_object_path(
                    module.instantiate_model,
                    label="instantiate_model",
                    etflow_root=root,
                )
            ),
            "etflow_root": str(root),
        }
        print(
            "ETFLOW_RUNTIME_PROVENANCE="
            + json.dumps(provenance, sort_keys=True),
            flush=True,
        )
    except Exception:
        restore_import_state()
        raise

    return SimpleNamespace(
        read_yaml=module.read_yaml,
        instantiate_model=module.instantiate_model,
        dataset_class=dataset_class,
        model_class=model_class,
        batch_class=Batch,
        etflow_root=root,
        provenance=provenance,
        close=restore_import_state,
    )


def _load_model(
    runtime: SimpleNamespace,
    config: Mapping[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> Any:
    if str(config.get("model")) != "BaseFlow":
        raise ValueError("Formal upstream generation requires the original BaseFlow")
    model_args = dict(config.get("model_args") or {})
    if bool(model_args.get("use_jacobian_4d_correction", False)):
        raise ValueError("Jacobian/adapter models are forbidden for upstream generation")
    model = runtime.instantiate_model(config["model"], model_args)
    runtime_root = getattr(runtime, "etflow_root", None)
    if runtime_root is not None:
        actual_model_path = _verified_object_path(
            type(model),
            label=f"instantiated model class {type(model).__qualname__}",
            etflow_root=Path(runtime_root),
        )
        runtime.provenance["model_class_path"] = str(actual_model_path)
        runtime.provenance["model_class_name"] = (
            f"{type(model).__module__}.{type(model).__qualname__}"
        )
        print(
            "ETFLOW_RUNTIME_PROVENANCE="
            + json.dumps(runtime.provenance, sort_keys=True),
            flush=True,
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or "state_dict" not in checkpoint:
        raise ValueError("ETFlow checkpoint has no state_dict")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def _sample_one(
    *,
    model: Any,
    data: Any,
    pos_ref: torch.Tensor,
    samples_per_molecule: int,
    max_batch_size: int,
    sampler_args: Mapping[str, Any],
    device: torch.device,
    batch_class: Any,
) -> torch.Tensor:
    atomic_numbers = torch.as_tensor(_field(data, "atomic_numbers")).long().view(-1)
    num_atoms = int(atomic_numbers.numel())
    if pos_ref.size(1) != num_atoms:
        raise ValueError("Reference and model graph atom counts differ")
    chunks = []
    for batch_start in range(0, samples_per_molecule, max_batch_size):
        batch_size = min(max_batch_size, samples_per_molecule - batch_start)
        batched = batch_class.from_data_list([data] * batch_size)
        with torch.inference_mode():
            generated = model.sample(
                batched["atomic_numbers"].to(device),
                batched["edge_index"].to(device),
                batched["batch"].to(device),
                node_attr=batched["node_attr"].to(device),
                chiral_index=batched["chiral_index"].to(device),
                chiral_nbr_index=batched["chiral_nbr_index"].to(device),
                chiral_tag=batched["chiral_tag"].to(device),
                rotatable_bond_index=batched["rotatable_bond_index"].to(device),
                atom_bond_influence_index=batched[
                    "atom_bond_influence_index"
                ].to(device),
                **dict(sampler_args),
            )
        generated = generated.reshape(batch_size, num_atoms, 3).detach().cpu()
        if not bool(torch.isfinite(generated).all()):
            raise ValueError("Generated positions contain NaN or Inf")
        chunks.append(generated)
    result = torch.cat(chunks, dim=0)
    if tuple(result.shape) != (samples_per_molecule, num_atoms, 3):
        raise ValueError(f"Unexpected generated shape {tuple(result.shape)}")
    return result


def _state_payload(
    *,
    status: str,
    manifest: Mapping[str, Any],
    completed: int,
    next_position: int,
    started_at: str,
    elapsed: float,
    generated_this_run: int,
    latest_error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rate = generated_this_run / elapsed if elapsed > 0 else 0.0
    remaining = int(manifest["target_molecules"]) - completed
    return {
        "format_version": STATE_VERSION,
        "status": status.upper(),
        "split": manifest["split"],
        "target_molecules": int(manifest["target_molecules"]),
        "completed_molecules": int(completed),
        "next_position": int(next_position),
        "seed": int(manifest["seed"]),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "config_sha256": manifest["config_sha256"],
        "processed_split_identity_sha256": manifest[
            "processed_split_identity_sha256"
        ],
        "manifest_sha256": manifest["manifest_sha256"],
        "started_at": started_at,
        "updated_at": utc_now(),
        "latest_error": dict(latest_error) if latest_error else None,
        "molecules_per_second": rate,
        "eta_seconds": remaining / rate if rate else None,
    }


def _run_generation_with_runtime(
    args: argparse.Namespace,
    *,
    runtime: SimpleNamespace,
) -> dict[str, Any]:
    defaults = FORMAL_DEFAULTS[args.split]
    max_molecules = int(
        defaults["max_molecules"]
        if args.max_molecules is None
        else args.max_molecules
    )
    samples_per_molecule = int(
        defaults["samples_per_molecule"]
        if args.samples_per_molecule is None
        else args.samples_per_molecule
    )
    if max_molecules < 1 or samples_per_molecule < 1:
        raise ValueError("Molecule and conformer counts must be positive")
    if args.save_every_molecules < 1:
        raise ValueError("save_every_molecules must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    molecules_dir = output_dir / "molecules"
    if output_dir.is_symlink() or molecules_dir.is_symlink():
        raise ValueError("Output directories must not be symlinks")
    molecules_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "generation_manifest.json"
    state_path = (args.state_path or output_dir / "generation_state.json").resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    processed_data = args.processed_data.expanduser().resolve()
    for label, path in (
        ("checkpoint", checkpoint_path),
        ("config", config_path),
        ("processed_data", processed_data),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    config = runtime.read_yaml(str(config_path))
    dataset = runtime.dataset_class(
        partition=config["datamodule_args"]["partition"],
        split=args.split,
        data_dir=processed_data,
    )
    expected_manifest = build_generation_manifest(
        data_files=[Path(path) for path in dataset.data_files],
        processed_data=processed_data,
        split=args.split,
        max_molecules=max_molecules,
        samples_per_molecule=samples_per_molecule,
        seed=args.seed,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(
                "Generation manifest already exists; pass --resume after validation"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_generation_manifest(manifest, expected_manifest)
    else:
        if any(molecules_dir.iterdir()) or state_path.exists():
            raise ValueError("Partial outputs exist without a generation manifest")
        manifest = expected_manifest
        atomic_json_save(manifest, manifest_path)

    expected_output_files = {
        str((molecules_dir / str(row["output_file"])).resolve())
        for row in manifest["records"]
    }
    unexpected_output_files = sorted(
        str(path)
        for path in molecules_dir.rglob("*.pt")
        if str(path.resolve()) not in expected_output_files
    )
    if unexpected_output_files:
        preview = unexpected_output_files[:10]
        raise ValueError(
            f"Molecule directory contains unmanifested .pt files: {preview}"
        )

    completed_positions = set()
    for position, row in enumerate(manifest["records"]):
        path = molecules_dir / row["output_file"]
        if path.exists():
            load_and_validate_generated_file(
                path, manifest=manifest, manifest_row=row
            )
            completed_positions.add(position)
    completed = len(completed_positions)
    next_position = next(
        (index for index in range(len(manifest["records"])) if index not in completed_positions),
        len(manifest["records"]),
    )
    prior_state = {}
    if state_path.is_file():
        try:
            prior_state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior_state = {}
        if prior_state and str(prior_state.get("manifest_sha256")) != str(
            manifest["manifest_sha256"]
        ):
            raise ValueError("Generation state belongs to a different manifest")
    started_at = str(prior_state.get("started_at") or utc_now())
    started = time.perf_counter()
    generated_this_run = 0
    atomic_json_save(
        _state_payload(
            status="COMPLETED" if completed == max_molecules else "RUNNING",
            manifest=manifest,
            completed=completed,
            next_position=next_position,
            started_at=started_at,
            elapsed=0.0,
            generated_this_run=0,
        ),
        state_path,
    )
    if completed == max_molecules:
        return {"completed_molecules": completed, "generated_this_run": 0}

    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        model = _load_model(runtime, config, checkpoint_path, device)
        eval_args = dict(config.get("eval_args") or {})
        max_batch_size = int(eval_args.get("batch_size", 32))
        sampler_args = dict(eval_args.get("sampler_args") or {})
        if max_batch_size < 1:
            raise ValueError("eval_args.batch_size must be positive")
    except Exception as exc:
        row = manifest["records"][next_position]
        atomic_json_save(
            _state_payload(
                status="FAILED",
                manifest=manifest,
                completed=completed,
                next_position=next_position,
                started_at=started_at,
                elapsed=time.perf_counter() - started,
                generated_this_run=0,
                latest_error={
                    "position": next_position,
                    "source_mol_id": row["source_mol_id"],
                    "error": repr(exc),
                },
            ),
            state_path,
        )
        raise

    for position, row in enumerate(manifest["records"]):
        if position in completed_positions:
            continue
        destination = molecules_dir / row["output_file"]
        try:
            source_path = _current_source_path(
                row, dataset=dataset, processed_data=processed_data
            )
            set_molecule_seed(int(row["molecule_seed"]))
            data = dataset[int(row["dataset_index"])]
            pos_ref = _reference_positions(source_path)
            atomic_numbers = torch.as_tensor(_field(data, "atomic_numbers")).long().view(-1)
            rdmol = _field(data, "mol")
            topology_hash = _topology_sha256(rdmol, atomic_numbers)
            pos_gen = _sample_one(
                model=model,
                data=data,
                pos_ref=pos_ref,
                samples_per_molecule=samples_per_molecule,
                max_batch_size=max_batch_size,
                sampler_args=sampler_args,
                device=device,
                batch_class=runtime.batch_class,
            )
            record = {
                "pos_gen": pos_gen,
                "pos_ref": pos_ref,
                "smiles": str(_field(data, "smiles")),
                "atomic_numbers": atomic_numbers.cpu(),
                "rdmol": rdmol,
                "dataset_index": int(row["dataset_index"]),
                "split": args.split,
                "mol_id": row["source_mol_id"],
                "source_mol_id": row["source_mol_id"],
                "generator_name": GENERATOR_NAME,
                "generator_checkpoint": str(checkpoint_path),
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "config_sha256": manifest["config_sha256"],
                "global_seed": int(args.seed),
                "molecule_seed": int(row["molecule_seed"]),
                "processed_source_path": str(source_path),
                "processed_source_identity": canonical_sha256(
                    {
                        "relative_path": row["relative_source_path"],
                        "size": row["source_file_size"],
                    }
                ),
                "generation_manifest_sha256": manifest["manifest_sha256"],
                "topology_sha256": topology_hash,
            }
            record["record_content_sha256"] = record_content_sha256(record)
            validate_generated_record(record, manifest=manifest, manifest_row=row)
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite {destination}")
            atomic_torch_save(record, destination)
            load_and_validate_generated_file(
                destination, manifest=manifest, manifest_row=row
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            atomic_json_save(
                _state_payload(
                    status="FAILED",
                    manifest=manifest,
                    completed=completed,
                    next_position=position,
                    started_at=started_at,
                    elapsed=elapsed,
                    generated_this_run=generated_this_run,
                    latest_error={
                        "position": position,
                        "source_mol_id": row["source_mol_id"],
                        "error": repr(exc),
                    },
                ),
                state_path,
            )
            raise
        completed_positions.add(position)
        completed += 1
        generated_this_run += 1
        next_position = next(
            (
                index
                for index in range(position + 1, len(manifest["records"]))
                if index not in completed_positions
            ),
            len(manifest["records"]),
        )
        if (
            generated_this_run % int(args.save_every_molecules) == 0
            or completed == max_molecules
        ):
            atomic_json_save(
                _state_payload(
                    status="COMPLETED" if completed == max_molecules else "RUNNING",
                    manifest=manifest,
                    completed=completed,
                    next_position=next_position,
                    started_at=started_at,
                    elapsed=time.perf_counter() - started,
                    generated_this_run=generated_this_run,
                ),
                state_path,
            )
        print(
            f"[{completed}/{max_molecules}] {row['source_mol_id']} "
            f"seed={row['molecule_seed']}",
            flush=True,
        )
    return {
        "completed_molecules": completed,
        "generated_this_run": generated_this_run,
    }


def run_generation(
    args: argparse.Namespace,
    *,
    runtime_loader: Callable[[Path], SimpleNamespace] = _load_etflow_runtime,
) -> dict[str, Any]:
    runtime = runtime_loader(args.etflow_root)
    try:
        result = _run_generation_with_runtime(args, runtime=runtime)
        provenance = getattr(runtime, "provenance", None)
        if provenance is not None:
            result["runtime_provenance"] = dict(provenance)
        return result
    finally:
        close = getattr(runtime, "close", None)
        if callable(close):
            close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--etflow_root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--processed_data", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--max_molecules", type=int)
    parser.add_argument("--samples_per_molecule", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every_molecules", type=int, default=100)
    parser.add_argument("--state_path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_generation(args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
