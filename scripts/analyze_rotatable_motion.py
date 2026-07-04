"""Analyze relative rigid motion across rotatable bonds."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger as log
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from utils import instantiate_model, read_yaml

from etflow.commons.rotatable_motion import (
    count_rotatable_bonds,
    decompose_rotatable_bond_motion,
    rotatable_bond_sides,
)
from etflow.data import EuclideanDataset
from etflow.models import BaseFlow
from etflow.models.utils import rmsd_align

torch.set_float32_matmul_precision("high")


REQUIRED_FIELDS = (
    "atomic_numbers",
    "pos",
    "edge_index",
    "node_attr",
    "batch",
    "smiles",
)
VALID_STATUS = "ok"
MOTION_COLUMNS = [
    "molecule_id",
    "smiles",
    "num_atoms",
    "num_rotatable_bonds",
    "rdkit_num_rotatable_bonds",
    "candidate_rotatable_bonds",
    "velocity_source",
    "bond_index",
    "bond_atom_a",
    "bond_atom_b",
    "side_a_size",
    "side_b_size",
    "fit_status_a",
    "fit_status_b",
    "fit_rank_a",
    "fit_rank_b",
    "omega_a_x",
    "omega_a_y",
    "omega_a_z",
    "omega_b_x",
    "omega_b_y",
    "omega_b_z",
    "delta_omega_x",
    "delta_omega_y",
    "delta_omega_z",
    "relative_rotation_norm",
    "bond_axis_x",
    "bond_axis_y",
    "bond_axis_z",
    "torsion_velocity",
    "abs_torsion_velocity",
    "side_a_residual_ratio",
    "side_b_residual_ratio",
    "side_a_rigid_explain_ratio",
    "side_b_rigid_explain_ratio",
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
        raise ValueError(
            f"Batch field '{field_name}' is missing for {num_graphs} graphs."
        )
    if isinstance(value, (list, tuple)):
        return list(value)
    if num_graphs == 1:
        return [value]
    raise ValueError(
        f"Batch field '{field_name}' is not a list for {num_graphs} graphs. "
        f"Received type {type(value)!r}."
    )


def _optional_tensor(batch: Batch, key: str, device: torch.device) -> Optional[torch.Tensor]:
    value = batch.get(key, None)
    if value is None:
        return None
    return value.to(device)


def _validate_batch(batch: Batch) -> None:
    keys = set(batch.keys())
    missing = [field for field in REQUIRED_FIELDS if field not in keys]
    if missing:
        raise ValueError(
            "Batch is missing required fields "
            f"{missing}. Available fields: {sorted(keys)}. "
            "Inspect the processed Data object and update the analysis script mapping."
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
    z = batch_data["atomic_numbers"].to(device)
    pos = batch_data["pos"].to(device)
    bond_index = batch_data["edge_index"].to(device)
    graph_batch = batch_data["batch"].to(device)
    node_attr = _optional_tensor(batch_data, "node_attr", device)
    edge_attr = _optional_tensor(batch_data, "edge_attr", device)
    smiles = batch_data.get("smiles", None)

    batch_size = int(graph_batch.max().item()) + 1 if graph_batch.numel() else 1
    x0 = model.sample_base_dist(
        pos.shape,
        edge_index=bond_index,
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
    pred_velocity = model(
        z=z,
        t=t,
        pos=x_t,
        bond_index=bond_index,
        edge_attr=edge_attr,
        node_attr=node_attr,
        batch=graph_batch,
        rotatable_bond_index=_optional_tensor(
            batch_data, "rotatable_bond_index", device
        ),
        atom_bond_influence_index=_optional_tensor(
            batch_data, "atom_bond_influence_index", device
        ),
    )

    return {
        "atomic_numbers": z,
        "x_t": x_t,
        "target_velocity": target_velocity,
        "pred_velocity": pred_velocity,
        "batch": graph_batch,
        "t": t,
    }


def _is_valid_row(df: pd.DataFrame) -> pd.Series:
    return (df["fit_status_a"] == VALID_STATUS) & (df["fit_status_b"] == VALID_STATUS)


def _add_side_means(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mean_side_residual_ratio"] = out[
        ["side_a_residual_ratio", "side_b_residual_ratio"]
    ].mean(axis=1)
    out["mean_side_rigid_explain_ratio"] = out[
        ["side_a_rigid_explain_ratio", "side_b_rigid_explain_ratio"]
    ].mean(axis=1)
    return out


def _top3_mean(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return float("nan")
    return float(values.nlargest(3).mean())


def _save_molecule_summary(bond_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    valid = _add_side_means(bond_df.loc[_is_valid_row(bond_df)].copy())
    if valid.empty:
        out = pd.DataFrame(
            columns=[
                "molecule_id",
                "smiles",
                "num_atoms",
                "num_rotatable_bonds",
                "rdkit_num_rotatable_bonds",
                "candidate_rotatable_bonds",
                "velocity_source",
                "valid_rotatable_bonds",
                "mean_abs_torsion_velocity",
                "median_abs_torsion_velocity",
                "sum_abs_torsion_velocity",
                "max_abs_torsion_velocity",
                "top3_mean_abs_torsion_velocity",
                "mean_relative_rotation_norm",
                "sum_relative_rotation_norm",
                "max_relative_rotation_norm",
                "mean_side_residual_ratio",
                "mean_side_rigid_explain_ratio",
            ]
        )
        out.to_csv(output_path, index=False)
        return out

    grouped = valid.groupby(
        [
            "molecule_id",
            "smiles",
            "num_atoms",
            "num_rotatable_bonds",
            "rdkit_num_rotatable_bonds",
            "candidate_rotatable_bonds",
            "velocity_source",
        ],
        as_index=False,
    )
    out = grouped.agg(
        valid_rotatable_bonds=("bond_index", "size"),
        mean_abs_torsion_velocity=("abs_torsion_velocity", "mean"),
        median_abs_torsion_velocity=("abs_torsion_velocity", "median"),
        sum_abs_torsion_velocity=("abs_torsion_velocity", "sum"),
        max_abs_torsion_velocity=("abs_torsion_velocity", "max"),
        top3_mean_abs_torsion_velocity=("abs_torsion_velocity", _top3_mean),
        mean_relative_rotation_norm=("relative_rotation_norm", "mean"),
        sum_relative_rotation_norm=("relative_rotation_norm", "sum"),
        max_relative_rotation_norm=("relative_rotation_norm", "max"),
        mean_side_residual_ratio=("mean_side_residual_ratio", "mean"),
        mean_side_rigid_explain_ratio=("mean_side_rigid_explain_ratio", "mean"),
    )
    out.to_csv(output_path, index=False)
    return out


def _save_summary(bond_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    valid = _add_side_means(bond_df.loc[_is_valid_row(bond_df)].copy())
    if valid.empty:
        out = pd.DataFrame(
            columns=[
                "velocity_source",
                "count",
                "mean_abs_torsion_velocity",
                "median_abs_torsion_velocity",
                "std_abs_torsion_velocity",
                "mean_relative_rotation_norm",
                "median_relative_rotation_norm",
                "mean_side_residual_ratio",
                "mean_side_rigid_explain_ratio",
            ]
        )
        out.to_csv(output_path, index=False)
        return out

    out = valid.groupby("velocity_source", as_index=False).agg(
        count=("abs_torsion_velocity", "size"),
        mean_abs_torsion_velocity=("abs_torsion_velocity", "mean"),
        median_abs_torsion_velocity=("abs_torsion_velocity", "median"),
        std_abs_torsion_velocity=("abs_torsion_velocity", "std"),
        mean_relative_rotation_norm=("relative_rotation_norm", "mean"),
        median_relative_rotation_norm=("relative_rotation_norm", "median"),
        mean_side_residual_ratio=("mean_side_residual_ratio", "mean"),
        mean_side_rigid_explain_ratio=("mean_side_rigid_explain_ratio", "mean"),
    )
    out.to_csv(output_path, index=False)
    return out


def _save_target_pred_errors(bond_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    valid = bond_df.loc[_is_valid_row(bond_df)].copy()
    key_cols = [
        "molecule_id",
        "bond_index",
        "bond_atom_a",
        "bond_atom_b",
        "smiles",
        "num_atoms",
        "num_rotatable_bonds",
        "rdkit_num_rotatable_bonds",
        "candidate_rotatable_bonds",
    ]
    metric_cols = [
        "torsion_velocity",
        "abs_torsion_velocity",
        "relative_rotation_norm",
    ]
    target = valid.loc[valid["velocity_source"] == "target", key_cols + metric_cols]
    pred = valid.loc[valid["velocity_source"] == "pred", key_cols + metric_cols]
    joined = target.merge(pred, on=key_cols, suffixes=("_target", "_pred"), how="inner")

    if joined.empty:
        out = pd.DataFrame(
            columns=[
                "molecule_id",
                "smiles",
                "num_atoms",
                "num_rotatable_bonds",
                "rdkit_num_rotatable_bonds",
                "candidate_rotatable_bonds",
                "bond_index",
                "target_torsion_velocity",
                "pred_torsion_velocity",
                "abs_error_torsion_velocity",
                "target_abs_torsion_velocity",
                "pred_abs_torsion_velocity",
                "target_relative_rotation_norm",
                "pred_relative_rotation_norm",
                "abs_error_relative_rotation_norm",
            ]
        )
        out.to_csv(output_path, index=False)
        return out

    joined = joined.rename(
        columns={
            "torsion_velocity_target": "target_torsion_velocity",
            "torsion_velocity_pred": "pred_torsion_velocity",
            "abs_torsion_velocity_target": "target_abs_torsion_velocity",
            "abs_torsion_velocity_pred": "pred_abs_torsion_velocity",
            "relative_rotation_norm_target": "target_relative_rotation_norm",
            "relative_rotation_norm_pred": "pred_relative_rotation_norm",
        }
    )
    joined["abs_error_torsion_velocity"] = (
        joined["pred_torsion_velocity"] - joined["target_torsion_velocity"]
    ).abs()
    joined["abs_error_relative_rotation_norm"] = (
        joined["pred_relative_rotation_norm"] - joined["target_relative_rotation_norm"]
    ).abs()
    out_cols = [
        "molecule_id",
        "smiles",
        "num_atoms",
        "num_rotatable_bonds",
        "rdkit_num_rotatable_bonds",
        "candidate_rotatable_bonds",
        "bond_index",
        "target_torsion_velocity",
        "pred_torsion_velocity",
        "abs_error_torsion_velocity",
        "target_abs_torsion_velocity",
        "pred_abs_torsion_velocity",
        "target_relative_rotation_norm",
        "pred_relative_rotation_norm",
        "abs_error_relative_rotation_norm",
    ]
    joined[out_cols].to_csv(output_path, index=False)
    return joined[out_cols]


def _save_molecule_error_summary(error_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    columns = [
        "molecule_id",
        "smiles",
        "num_atoms",
        "num_rotatable_bonds",
        "rdkit_num_rotatable_bonds",
        "candidate_rotatable_bonds",
        "matched_rotatable_bonds",
        "mean_abs_torsion_error",
        "sum_abs_torsion_error",
        "max_abs_torsion_error",
        "mean_abs_relative_rotation_error",
        "sum_abs_relative_rotation_error",
    ]
    if error_df.empty:
        out = pd.DataFrame(columns=columns)
        out.to_csv(output_path, index=False)
        return out

    out = error_df.groupby(
        [
            "molecule_id",
            "smiles",
            "num_atoms",
            "num_rotatable_bonds",
            "rdkit_num_rotatable_bonds",
            "candidate_rotatable_bonds",
        ],
        as_index=False,
    ).agg(
        matched_rotatable_bonds=("bond_index", "size"),
        mean_abs_torsion_error=("abs_error_torsion_velocity", "mean"),
        sum_abs_torsion_error=("abs_error_torsion_velocity", "sum"),
        max_abs_torsion_error=("abs_error_torsion_velocity", "max"),
        mean_abs_relative_rotation_error=(
            "abs_error_relative_rotation_norm",
            "mean",
        ),
        sum_abs_relative_rotation_error=(
            "abs_error_relative_rotation_norm",
            "sum",
        ),
    )
    out.to_csv(output_path, index=False)
    return out


def _motion_rows(
    molecule_id: int,
    smiles: str,
    num_atoms: int,
    rdkit_num_rotatable_bonds: int,
    candidate_rotatable_bonds: int,
    velocity_source: str,
    rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    out = []
    for row in rows:
        item = {
            "molecule_id": molecule_id,
            "smiles": smiles,
            "num_atoms": num_atoms,
            "num_rotatable_bonds": rdkit_num_rotatable_bonds,
            "rdkit_num_rotatable_bonds": rdkit_num_rotatable_bonds,
            "candidate_rotatable_bonds": candidate_rotatable_bonds,
            "velocity_source": velocity_source,
        }
        item.update(row)
        out.append(item)
    return out


def analyze(args) -> None:
    _seed_everything(args.seed)
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

    motion_rows: List[Dict[str, object]] = []
    total_bonds = 0
    skipped_small_side = 0
    num_seen = 0

    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Analyzing rotatable motion")):
        _validate_batch(batch_data)
        if batch_idx == 0:
            log.info(f"First batch keys: {sorted(batch_data.keys())}")

        flow = _flow_matching_batch(
            model,
            batch_data,
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
            if num_seen >= args.max_molecules:
                break

            atom_mask = graph_batch == graph_idx
            local_x = flow["x_t"][atom_mask]
            local_target = flow["target_velocity"][atom_mask]
            local_pred = flow["pred_velocity"][atom_mask]
            smiles = smiles_list[graph_idx]
            mol = mol_list[graph_idx]
            molecule_id = num_seen
            num_atoms = int(atom_mask.sum().item())

            rdkit_num_rotatable_bonds = count_rotatable_bonds(smiles=smiles, mol=mol)
            bond_sides = rotatable_bond_sides(
                smiles=smiles,
                mol=mol,
                expected_num_atoms=num_atoms,
            )
            candidate_rotatable_bonds = len(bond_sides)
            total_bonds += candidate_rotatable_bonds
            skipped_small_side += sum(
                1
                for bond in bond_sides
                if len(bond["side_a_atoms"]) < args.min_side_atoms
                or len(bond["side_b_atoms"]) < args.min_side_atoms
            )

            for source, velocity in (("target", local_target), ("pred", local_pred)):
                rows = decompose_rotatable_bond_motion(
                    local_x,
                    velocity,
                    bond_sides,
                    min_side_atoms=args.min_side_atoms,
                )
                motion_rows.extend(
                    _motion_rows(
                        molecule_id=molecule_id,
                        smiles=smiles,
                        num_atoms=num_atoms,
                        rdkit_num_rotatable_bonds=rdkit_num_rotatable_bonds,
                        candidate_rotatable_bonds=candidate_rotatable_bonds,
                        velocity_source=source,
                        rows=rows,
                    )
                )

            num_seen += 1

        if num_seen >= args.max_molecules:
            break

    bond_df = pd.DataFrame(motion_rows, columns=MOTION_COLUMNS)
    bond_path = output_dir / "rotatable_bond_relative_motion.csv"
    bond_df.to_csv(bond_path, index=False)

    molecule_df = _save_molecule_summary(
        bond_df,
        output_dir / "rotatable_motion_by_molecule.csv",
    )
    summary_df = _save_summary(bond_df, output_dir / "rotatable_motion_summary.csv")
    error_df = _save_target_pred_errors(
        bond_df,
        output_dir / "rotatable_motion_target_pred_errors.csv",
    )
    molecule_error_df = _save_molecule_error_summary(
        error_df,
        output_dir / "rotatable_motion_error_by_molecule.csv",
    )

    if bond_df.empty:
        valid_rows = 0
        valid_bonds = 0
        skipped_fitting_failure = 0
    else:
        valid_mask = _is_valid_row(bond_df)
        valid_rows = int(valid_mask.sum())
        valid_bonds = int(
            bond_df.loc[valid_mask, ["molecule_id", "bond_index"]]
            .drop_duplicates()
            .shape[0]
        )
        skipped_fitting_failure = int(
            bond_df.loc[
                ~valid_mask
                & (bond_df["side_a_size"] >= args.min_side_atoms)
                & (bond_df["side_b_size"] >= args.min_side_atoms)
                & bond_df["bond_index"].notna(),
                ["molecule_id", "bond_index"],
            ]
            .drop_duplicates()
            .shape[0]
        )

    manifest = {
        "config": args.config,
        "checkpoint": checkpoint_path,
        "split": args.split,
        "max_molecules": args.max_molecules,
        "analyzed_molecules": int(num_seen),
        "min_side_atoms": args.min_side_atoms,
        "t_low": args.t_low,
        "t_high": args.t_high,
        "seed": args.seed,
    }
    with open(output_dir / "analysis_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"total candidate rotatable bonds: {total_bonds}")
    log.info(f"valid rotatable bonds: {valid_bonds}")
    log.info(f"valid rotatable bond/source rows: {valid_rows}")
    log.info(f"skipped bonds due to small side: {skipped_small_side}")
    log.info(f"skipped bonds due to fitting failure: {skipped_fitting_failure}")
    log.info(f"Saved rotatable bond rows: {len(bond_df)}")
    log.info(f"Saved molecule rows: {len(molecule_df)}")
    log.info(f"Saved summary rows: {len(summary_df)}")
    log.info(f"Saved target/pred error rows: {len(error_df)}")
    log.info(f"Saved molecule error rows: {len(molecule_error_df)}")
    log.info(f"output directory: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--checkpoint", "-k", type=str, required=True)
    parser.add_argument("--split", choices=["val", "test", "train"], default="test")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--max_molecules", type=int, default=200)
    parser.add_argument("--output_dir", "-o", type=str, default="logs_rotatable_motion_first")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t_low", type=float, default=1.0e-4)
    parser.add_argument("--t_high", type=float, default=0.9999)
    parser.add_argument("--min_side_atoms", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
