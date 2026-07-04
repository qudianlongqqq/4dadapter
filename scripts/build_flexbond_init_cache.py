#!/usr/bin/env python
"""Build generator-agnostic FlexBond cache files from sampled outputs.

The most direct ETFlow path is a packed ``.pkl``/``.pt`` list whose entries
contain ``pos_gen``, ``pos_ref``, ``smiles``, and ``atomic_numbers``.  A
separate processed reference directory is also supported and is matched by
mol_id first, then exact ordered SMILES.  Positional/index-only pairing is
intentionally not used.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from etflow.commons.featurization import (
    MoleculeFeaturizer,
    get_sample_field,
    mol_to_ordered_smiles,
    recover_mol_from_sample,
)
from etflow.data.flexbond_optimizer_dataset import validate_cache_record


def _load(path: Path) -> Any:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            return pickle.load(handle)
    return torch.load(path, map_location="cpu", weights_only=False)


def _is_record(value: Any) -> bool:
    return isinstance(value, Mapping) or callable(getattr(value, "keys", None))


def _records(path: Path) -> list[tuple[str, Any]]:
    if path.is_dir():
        output = []
        for item in sorted(path.rglob("*.pt")):
            value = _load(item)
            if _is_record(value):
                output.append((item.stem, value))
        return output
    value = _load(path)
    if isinstance(value, Mapping) and isinstance(value.get("molecules"), list):
        value = value["molecules"]
    if _is_record(value):
        return [(path.stem, value)]
    if isinstance(value, (list, tuple)):
        return [(str(index), item) for index, item in enumerate(value)]
    raise TypeError(f"Unsupported sampled-output container: {type(value).__name__}.")


def _first(record: Any, names: Iterable[str], default=None):
    for name in names:
        value = get_sample_field(record, name)
        if value is not None:
            return value
    return default


def _positions(record: Any, generated: bool) -> torch.Tensor:
    names = ("x_init", "pos_gen", "generated_positions", "positions") if generated else (
        "x_ref_candidates",
        "pos_ref",
        "pos",
        "reference_positions",
    )
    value = _first(record, names)
    if value is None:
        raise ValueError(f"Record has none of the coordinate fields {names}.")
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.size(-1) != 3:
        raise ValueError(f"Coordinates must be [C, N, 3], got {tuple(tensor.shape)}.")
    return tensor


def _identity(record: Any, fallback: str) -> tuple[str, str]:
    mol_id = str(_first(record, ("mol_id", "molecule_id", "id"), fallback))
    smiles = str(_first(record, ("smiles", "canonical_smiles", "smi"), ""))
    return mol_id, smiles


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:160]


def _reference_lookup(
    records: list[tuple[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_id, by_smiles = {}, {}
    for fallback, record in records:
        mol_id, smiles = _identity(record, fallback)
        if mol_id in by_id:
            raise ValueError(f"Duplicate reference mol_id: {mol_id!r}.")
        by_id[mol_id] = record
        if smiles:
            if smiles in by_smiles:
                raise ValueError(f"Duplicate reference SMILES: {smiles!r}.")
            by_smiles[smiles] = record
    return by_id, by_smiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_path", required=True, type=Path)
    parser.add_argument(
        "--reference_path",
        type=Path,
        help="Optional reference file/directory; omit when init records contain pos_ref.",
    )
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--generator_name", default="ETFlow")
    parser.add_argument("--max_molecules", type=int)
    args = parser.parse_args()

    init_records = _records(args.init_path)
    if args.max_molecules is not None:
        init_records = init_records[: args.max_molecules]
    ref_by_id: dict[str, Any] = {}
    ref_by_smiles: dict[str, Any] = {}
    if args.reference_path is not None:
        ref_by_id, ref_by_smiles = _reference_lookup(_records(args.reference_path))

    output_dir = args.output_dir / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    featurizer = MoleculeFeaturizer()
    written = 0
    for fallback, init_record in init_records:
        mol_id, smiles = _identity(init_record, fallback)
        init_explicit_id = _first(init_record, ("mol_id", "molecule_id", "id"))
        reference_record = init_record
        try:
            refs = _positions(reference_record, generated=False)
        except ValueError:
            reference_record = ref_by_id.get(mol_id)
            if reference_record is None:
                reference_record = ref_by_smiles.get(smiles)
            if reference_record is None:
                raise ValueError(
                    f"No reference matched mol_id={mol_id!r}, smiles={smiles!r}."
                )
            refs = _positions(reference_record, generated=False)
        reference_explicit_id = _first(
            reference_record, ("mol_id", "molecule_id", "id")
        )
        if (
            init_explicit_id is not None
            and reference_explicit_id is not None
            and str(init_explicit_id) != str(reference_explicit_id)
        ):
            raise ValueError(
                "Molecule-id mismatch between generated and reference records: "
                f"{init_explicit_id!r} vs {reference_explicit_id!r}."
            )

        atomic_numbers = _first(init_record, ("atomic_numbers", "z"))
        if atomic_numbers is None:
            atomic_numbers = _first(reference_record, ("atomic_numbers", "z"))
        recovery_source = init_record if smiles or any(
            get_sample_field(init_record, key) is not None
            for key in ("mol", "rdmol", "rdkit_mol", "rd_mol", "mol_block")
        ) else reference_record
        if atomic_numbers is None:
            preview = recover_mol_from_sample(recovery_source)
            atomic_numbers = torch.tensor(
                [atom.GetAtomicNum() for atom in preview.mol.GetAtoms()],
                dtype=torch.long,
            )
        atomic_numbers = torch.as_tensor(atomic_numbers, dtype=torch.long).view(-1)
        recovery = recover_mol_from_sample(
            recovery_source, expected_atomic_numbers=atomic_numbers
        )
        mol = recovery.mol
        ordered_smiles = mol_to_ordered_smiles(mol)
        featurizer.cache_recovered_mol(ordered_smiles, mol)
        node_attr = featurizer.get_atom_features_from_mol(mol)
        edge_index, edge_attr = featurizer.get_edge_index_from_mol(
            mol, use_edge_feat=True
        )
        rotatable, influence = featurizer.get_rotatable_bond_features_from_mol(mol)
        rotatable_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
        for bond_idx in range(rotatable.size(1)):
            atom_a, atom_b = rotatable[:, bond_idx]
            rotatable_mask |= (
                ((edge_index[0] == atom_a) & (edge_index[1] == atom_b))
                | ((edge_index[0] == atom_b) & (edge_index[1] == atom_a))
            )
        ref_numbers = torch.as_tensor(
            _first(reference_record, ("atomic_numbers", "z"), atomic_numbers),
            dtype=torch.long,
        ).view(-1)

        for gen_index, x_init in enumerate(_positions(init_record, generated=True)):
            sample_id = f"{mol_id}__gen{gen_index:04d}"
            record = {
                "mol_id": sample_id,
                "source_mol_id": mol_id,
                "smiles": ordered_smiles,
                "atomic_numbers": atomic_numbers,
                "x_init_atomic_numbers": atomic_numbers,
                "x_ref_atomic_numbers": ref_numbers,
                "node_attr": node_attr,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "rotatable_bond_mask": rotatable_mask,
                "rotatable_bond_index": rotatable,
                "atom_bond_influence_index": influence,
                "x_init": x_init,
                "x_ref_candidates": refs,
                "num_rotatable_bonds": int(rotatable.size(1)),
                "metadata": {
                    "generator_name": args.generator_name,
                    "init_path": str(args.init_path),
                    "reference_path": str(args.reference_path or args.init_path),
                    "split": args.split,
                    "generated_conformer_index": gen_index,
                    "molecule_recovery_source": recovery.source,
                    "reference_mol_id": (
                        str(reference_explicit_id)
                        if reference_explicit_id is not None
                        else None
                    ),
                },
            }
            matched = validate_cache_record(record)
            record.update(
                {
                    "x_ref": matched["x_ref"],
                    "x_ref_aligned": matched["x_ref_aligned"],
                    "selected_reference_index": matched["selected_reference_index"],
                    "selected_reference_rmsd": matched["selected_rmsd"],
                }
            )
            destination = output_dir / f"{_safe_name(sample_id)}.pt"
            if destination.exists():
                raise FileExistsError(
                    f"Refusing to overwrite duplicate cache record: {destination}"
                )
            torch.save(record, destination)
            written += 1
    print(f"Wrote {written} validated FlexBond cache files to {output_dir}")


if __name__ == "__main__":
    main()
