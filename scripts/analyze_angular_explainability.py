"""Measure how much Flow Matching target velocity is explained by bond rotations.

For every molecule this script builds a molecule-local linear system

    u_i ~= sum_b m_{i,b} * omega_b x (x_i - c_b)

where ``c_b`` is the bond midpoint and ``m_{i,b}`` selects the smaller component
created by cutting bond ``b``.  No model or training code is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from loguru import logger as log
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from utils import instantiate_model, read_yaml

from etflow.commons.rotatable_motion import rotatable_bond_sides
from etflow.data import EuclideanDataset
from etflow.models import BaseFlow
from etflow.models.utils import rmsd_align

torch.set_float32_matmul_precision("high")


REQUIRED_FIELDS = ("pos", "edge_index", "batch", "smiles")
MOLECULE_COLUMNS = [
    "mol_id",
    "num_atoms",
    "num_rotatable_bonds",
    "target_velocity_norm",
    "angular_velocity_norm",
    "residual_velocity_norm",
    "angular_explain_ratio",
    "single_bond_explain_ratio",
    "multi_bond_explain_ratio",
    "random_bond_explain_ratio",
]
RATIO_COLUMNS = [
    "angular_explain_ratio",
    "single_bond_explain_ratio",
    "multi_bond_explain_ratio",
    "random_bond_explain_ratio",
]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _as_graph_list(
    value,
    num_graphs: int,
    field_name: str,
    allow_none: bool = False,
) -> List:
    if value is None:
        if allow_none:
            return [None] * num_graphs
        raise ValueError(f"Batch field '{field_name}' is missing for {num_graphs} graphs.")
    if isinstance(value, (list, tuple)):
        return list(value)
    if num_graphs == 1:
        return [value]
    raise ValueError(
        f"Batch field '{field_name}' is not a list for {num_graphs} graphs. "
        f"Received type {type(value)!r}."
    )


def _validate_batch(batch: Batch) -> None:
    keys = set(batch.keys())
    missing = [field for field in REQUIRED_FIELDS if field not in keys]
    if missing:
        raise ValueError(
            f"Batch is missing required fields {missing}. Available fields: {sorted(keys)}."
        )


def _load_model(config: dict, checkpoint_path: str, device: torch.device) -> BaseFlow:
    model = instantiate_model(config["model"], config["model_args"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def _build_dataloader(args, config: dict) -> DataLoader:
    dataset = EuclideanDataset(
        data_dir=args.data_dir,
        partition=config["datamodule_args"]["partition"],
        split=args.split,
    )
    dataloader_args = dict(config["datamodule_args"].get("dataloader_args", {}))
    dataloader_args.update(
        {
            "batch_size": args.batch_size,
            "shuffle": False,
            "num_workers": args.num_workers,
            "pin_memory": False,
        }
    )
    if args.num_workers == 0:
        dataloader_args.pop("persistent_workers", None)
    return DataLoader(dataset, **dataloader_args)


@torch.no_grad()
def _flow_matching_batch(
    model: BaseFlow,
    batch_data: Batch,
    device: torch.device,
    split: str,
    t_low: float,
    t_high: float,
) -> Dict[str, torch.Tensor]:
    """Construct x_t and u_t through the same BaseFlow methods used in training."""

    pos = batch_data["pos"].to(device)
    edge_index = batch_data["edge_index"].to(device)
    graph_batch = batch_data["batch"].to(device)
    smiles = batch_data.get("smiles", None)
    batch_size = int(graph_batch.max().item()) + 1 if graph_batch.numel() else 1

    x0 = model.sample_base_dist(
        pos.shape,
        edge_index=edge_index,
        batch=graph_batch,
        smiles=smiles,
    )
    time_stage = "val" if split in {"val", "test"} else "train"
    t = model.sample_time(
        num_samples=batch_size,
        low=t_low,
        high=t_high,
        stage=time_stage,
    )
    if model.prior_type == "harmonic":
        x0 = rmsd_align(pos=x0, ref_pos=pos, batch=graph_batch)

    x_t, target_velocity = model.compute_conditional_vector_field(
        x0=x0,
        x1=pos,
        t=t,
        batch=graph_batch,
    )
    return {"x_t": x_t, "target_velocity": target_velocity, "batch": graph_batch}


def _neighbors_without_bond(
    mol: Mol,
    atom_idx: int,
    cut_a: int,
    cut_b: int,
) -> Iterable[int]:
    for neighbor in mol.GetAtomWithIdx(int(atom_idx)).GetNeighbors():
        neighbor_idx = int(neighbor.GetIdx())
        if {int(atom_idx), neighbor_idx} != {int(cut_a), int(cut_b)}:
            yield neighbor_idx


def _component_after_cut(mol: Mol, start_idx: int, cut_a: int, cut_b: int) -> List[int]:
    visited = {int(start_idx)}
    queue: deque[int] = deque([int(start_idx)])
    while queue:
        atom_idx = queue.popleft()
        for neighbor_idx in _neighbors_without_bond(mol, atom_idx, cut_a, cut_b):
            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                queue.append(neighbor_idx)
    return sorted(visited)


def _smaller_side(side_a: Sequence[int], side_b: Sequence[int]) -> List[int]:
    """Choose one rotating side deterministically, preferring the smaller component."""

    key_a = (len(side_a), tuple(side_a))
    key_b = (len(side_b), tuple(side_b))
    return list(side_a if key_a <= key_b else side_b)


def _bond_descriptors_from_rotatable_sides(
    bond_sides: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    descriptors = []
    for bond in bond_sides:
        descriptors.append(
            {
                "bond_index": int(bond["bond_index"]),
                "bond_atom_a": int(bond["bond_atom_a"]),
                "bond_atom_b": int(bond["bond_atom_b"]),
                "affected_atoms": _smaller_side(
                    bond["side_a_atoms"],
                    bond["side_b_atoms"],
                ),
            }
        )
    return descriptors


def _all_bridge_bond_descriptors(mol: Mol) -> List[Dict[str, object]]:
    """Return all bonds whose removal splits a connected molecule in two."""

    num_atoms = int(mol.GetNumAtoms())
    descriptors = []
    for bond in mol.GetBonds():
        atom_a = int(bond.GetBeginAtomIdx())
        atom_b = int(bond.GetEndAtomIdx())
        side_a = _component_after_cut(mol, atom_a, atom_a, atom_b)
        side_b = _component_after_cut(mol, atom_b, atom_a, atom_b)
        if set(side_a).intersection(side_b) or len(side_a) + len(side_b) != num_atoms:
            continue
        descriptors.append(
            {
                "bond_index": int(bond.GetIdx()),
                "bond_atom_a": atom_a,
                "bond_atom_b": atom_b,
                "affected_atoms": _smaller_side(side_a, side_b),
            }
        )
    descriptors.sort(key=lambda item: int(item["bond_index"]))
    return descriptors


def _single_bond_masks(
    mol: Mol,
    bond_descriptors: Sequence[Dict[str, object]],
    num_atoms: int,
) -> List[List[int]]:
    """Assign atoms in overlapping masks to their topologically nearest bond."""

    masks = [set(map(int, bond["affected_atoms"])) for bond in bond_descriptors]
    assigned: List[List[int]] = [[] for _ in bond_descriptors]
    if not masks:
        return assigned

    distances = np.asarray(Chem.GetDistanceMatrix(mol), dtype=np.float64)
    for atom_idx in range(num_atoms):
        candidates = [bond_idx for bond_idx, mask in enumerate(masks) if atom_idx in mask]
        if not candidates:
            continue
        selected = min(
            candidates,
            key=lambda bond_idx: (
                min(
                    distances[atom_idx, int(bond_descriptors[bond_idx]["bond_atom_a"])],
                    distances[atom_idx, int(bond_descriptors[bond_idx]["bond_atom_b"])],
                ),
                int(bond_descriptors[bond_idx]["bond_index"]),
            ),
        )
        assigned[selected].append(atom_idx)
    return assigned


def _cross_matrix_for_omega_cross_r(r: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(r[:, 0])
    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    return torch.stack(
        [
            torch.stack([zeros, rz, -ry], dim=1),
            torch.stack([-rz, zeros, rx], dim=1),
            torch.stack([ry, -rx, zeros], dim=1),
        ],
        dim=1,
    )


def _build_design_matrix(
    x: torch.Tensor,
    bond_descriptors: Sequence[Dict[str, object]],
    atom_masks: Optional[Sequence[Sequence[int]]] = None,
) -> torch.Tensor:
    """Build only the current molecule's [3N, 3B] design matrix."""

    num_atoms = x.size(0)
    num_bonds = len(bond_descriptors)
    design = torch.zeros(
        (num_atoms, 3, 3 * num_bonds),
        dtype=x.dtype,
        device=x.device,
    )
    masks = atom_masks
    if masks is None:
        masks = [bond["affected_atoms"] for bond in bond_descriptors]

    for bond_idx, (bond, atom_indices) in enumerate(zip(bond_descriptors, masks)):
        if not atom_indices:
            continue
        index = torch.as_tensor(atom_indices, dtype=torch.long, device=x.device)
        atom_a = int(bond["bond_atom_a"])
        atom_b = int(bond["bond_atom_b"])
        center = 0.5 * (x[atom_a] + x[atom_b])
        r = x.index_select(0, index) - center
        design[index, :, 3 * bond_idx : 3 * (bond_idx + 1)] = (
            _cross_matrix_for_omega_cross_r(r)
        )
    return design.reshape(3 * num_atoms, 3 * num_bonds)


def _fit_angular_system(
    x: torch.Tensor,
    target_velocity: torch.Tensor,
    bond_descriptors: Sequence[Dict[str, object]],
    atom_masks: Optional[Sequence[Sequence[int]]] = None,
    eps: float = 1.0e-12,
) -> Dict[str, object]:
    """Least-squares fit u ~= A omega for one molecule."""

    # These systems are small; CPU float64 gives stable rank-deficient least squares
    # even when overlapping masks create dependent columns.
    x_fit = x.detach().to(device="cpu", dtype=torch.float64)
    target_fit = target_velocity.detach().to(device="cpu", dtype=torch.float64)
    design = _build_design_matrix(x_fit, bond_descriptors, atom_masks=atom_masks)
    target_flat = target_fit.reshape(-1)

    if design.size(1) == 0:
        omega = torch.empty(0, dtype=x_fit.dtype, device=x_fit.device)
        angular_flat = torch.zeros_like(target_flat)
    else:
        try:
            omega = torch.linalg.lstsq(design, target_flat).solution
        except RuntimeError:
            omega = torch.linalg.pinv(design) @ target_flat
        angular_flat = design @ omega

    residual_flat = target_flat - angular_flat
    target_norm = torch.linalg.norm(target_flat)
    angular_norm = torch.linalg.norm(angular_flat)
    residual_norm = torch.linalg.norm(residual_flat)
    if target_norm <= eps:
        explain_ratio = float("nan")
    else:
        explain_ratio = float(
            (1.0 - residual_norm.pow(2) / target_norm.pow(2)).item()
        )

    return {
        "omega": omega,
        "angular_velocity": angular_flat.reshape_as(target_fit),
        "residual_velocity": residual_flat.reshape_as(target_fit),
        "target_velocity_norm": float(target_norm.item()),
        "angular_velocity_norm": float(angular_norm.item()),
        "residual_velocity_norm": float(residual_norm.item()),
        "explain_ratio": explain_ratio,
    }


def _sample_random_bonds(
    mol: Mol,
    num_bonds: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    candidates = _all_bridge_bond_descriptors(mol)
    if num_bonds > len(candidates):
        raise ValueError(
            f"Cannot sample {num_bonds} bridge bonds from only {len(candidates)} candidates."
        )
    if num_bonds == 0:
        return []
    selected = rng.sample(candidates, k=num_bonds)
    selected.sort(key=lambda item: int(item["bond_index"]))
    return selected


def _save_summary(molecule_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    aggregation = {
        "num_molecules": ("mol_id", "size"),
        "mean_target_velocity_norm": ("target_velocity_norm", "mean"),
        "mean_angular_velocity_norm": ("angular_velocity_norm", "mean"),
        "mean_residual_velocity_norm": ("residual_velocity_norm", "mean"),
    }
    for column in RATIO_COLUMNS:
        aggregation[f"mean_{column}"] = (column, "mean")
        aggregation[f"std_{column}"] = (column, "std")
        aggregation[f"median_{column}"] = (column, "median")

    summary = (
        molecule_df.groupby("num_rotatable_bonds", as_index=False, dropna=False)
        .agg(**aggregation)
        .sort_values("num_rotatable_bonds")
    )
    summary.to_csv(output_path, index=False)
    return summary


def analyze(args) -> None:
    if args.num_molecules <= 0:
        raise ValueError("--num_molecules must be positive.")

    _seed_everything(args.seed)
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_yaml(args.config)
    checkpoint_path = os.path.expanduser(os.path.expandvars(args.checkpoint))
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    device = _device_from_arg(args.device)
    log.info(f"Using device: {device}")
    log.info(f"Loading checkpoint: {checkpoint_path}")
    dataloader = _build_dataloader(args, config)
    model = _load_model(config, checkpoint_path, device)

    molecule_rows: List[Dict[str, object]] = []
    for batch_idx, batch_data in enumerate(
        tqdm(dataloader, desc="Analyzing angular explainability")
    ):
        _validate_batch(batch_data)
        if batch_idx == 0:
            log.info(f"First batch keys: {sorted(batch_data.keys())}")

        flow = _flow_matching_batch(
            model=model,
            batch_data=batch_data,
            device=device,
            split=args.split,
            t_low=args.t_low,
            t_high=args.t_high,
        )
        graph_batch = flow["batch"]
        num_graphs = int(graph_batch.max().item()) + 1 if graph_batch.numel() else 1
        smiles_list = _as_graph_list(batch_data.get("smiles", None), num_graphs, "smiles")
        mol_list = _as_graph_list(
            batch_data.get("mol", None),
            num_graphs,
            "mol",
            allow_none=True,
        )

        for graph_idx in range(num_graphs):
            if len(molecule_rows) >= args.num_molecules:
                break

            atom_mask = graph_batch == graph_idx
            local_x = flow["x_t"][atom_mask]
            local_target = flow["target_velocity"][atom_mask]
            num_atoms = int(atom_mask.sum().item())
            smiles = smiles_list[graph_idx]
            mol = mol_list[graph_idx]

            bond_sides = rotatable_bond_sides(
                smiles=smiles,
                mol=mol,
                expected_num_atoms=num_atoms,
            )
            if mol is None:
                # rotatable_bond_sides parsed the same SMILES, so recreating here preserves order.
                mol = Chem.MolFromSmiles(smiles)
            rotatable_bonds = _bond_descriptors_from_rotatable_sides(bond_sides)

            multi_fit = _fit_angular_system(local_x, local_target, rotatable_bonds)
            single_masks = _single_bond_masks(mol, rotatable_bonds, num_atoms)
            single_fit = _fit_angular_system(
                local_x,
                local_target,
                rotatable_bonds,
                atom_masks=single_masks,
            )
            random_bonds = _sample_random_bonds(mol, len(rotatable_bonds), rng)
            random_fit = _fit_angular_system(local_x, local_target, random_bonds)

            molecule_rows.append(
                {
                    "mol_id": len(molecule_rows),
                    "num_atoms": num_atoms,
                    "num_rotatable_bonds": len(rotatable_bonds),
                    "target_velocity_norm": multi_fit["target_velocity_norm"],
                    "angular_velocity_norm": multi_fit["angular_velocity_norm"],
                    "residual_velocity_norm": multi_fit["residual_velocity_norm"],
                    "angular_explain_ratio": multi_fit["explain_ratio"],
                    "single_bond_explain_ratio": single_fit["explain_ratio"],
                    "multi_bond_explain_ratio": multi_fit["explain_ratio"],
                    "random_bond_explain_ratio": random_fit["explain_ratio"],
                }
            )

        if len(molecule_rows) >= args.num_molecules:
            break

    if not molecule_rows:
        raise RuntimeError("No molecules were analyzed; check the split and data directory.")

    molecule_df = pd.DataFrame(molecule_rows, columns=MOLECULE_COLUMNS)
    molecule_df.to_csv(output_dir / "angular_explain_by_molecule.csv", index=False)
    summary_df = _save_summary(
        molecule_df,
        output_dir / "angular_explain_summary.csv",
    )
    manifest = {
        "config": args.config,
        "checkpoint": checkpoint_path,
        "split": args.split,
        "num_molecules": args.num_molecules,
        "analyzed_molecules": len(molecule_df),
        "mask_side": "smaller_component_after_cut",
        "bond_center": "midpoint_at_x_t",
        "random_bond_pool": "all_bridge_bonds",
        "t_low": args.t_low,
        "t_high": args.t_high,
        "seed": args.seed,
    }
    with open(output_dir / "analysis_manifest.json", "w") as file:
        json.dump(manifest, file, indent=2)

    log.info(f"Saved molecule rows: {len(molecule_df)}")
    log.info(f"Saved summary rows: {len(summary_df)}")
    log.info(f"Output directory: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit rotatable-bond angular velocities to Flow Matching targets."
    )
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--checkpoint", "-k", type=str, required=True)
    parser.add_argument("--split", choices=["val", "test", "train"], default="test")
    parser.add_argument("--num_molecules", "--max_molecules", type=int, default=200)
    parser.add_argument("--output_dir", "-o", type=str, default="logs_angular_explainability")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t_low", type=float, default=1.0e-4)
    parser.add_argument("--t_high", type=float, default=0.9999)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
