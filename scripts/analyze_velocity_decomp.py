"""Analyze fragment-level rigid-flexible velocity decomposition."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger as log
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from utils import instantiate_model, read_yaml

from etflow.commons.fragmentation import FragmentationResult, fragment_molecule
from etflow.commons.velocity_decomposition import (
    decompose_velocity_by_fragment,
    iter_atom_decomposition_rows,
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
FRAGMENT_TYPES = ("aromatic_ring", "ring", "rotatable_region", "other")
MIN_FRAGMENT_ATOMS_FOR_SUMMARY = 3


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
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
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


def _fragment_rows(
    molecule_id: int,
    smiles: str,
    num_atoms: int,
    num_rotatable_bonds: int,
    velocity_source: str,
    fragments: Iterable[Dict[str, object]],
) -> Iterable[Dict[str, object]]:
    for row in fragments:
        out = {
            "molecule_id": molecule_id,
            "smiles": smiles,
            "num_atoms": num_atoms,
            "num_rotatable_bonds": num_rotatable_bonds,
            "velocity_source": velocity_source,
        }
        out.update(
            {
                key: value
                for key, value in row.items()
                if key not in {"center", "translation_velocity", "omega"}
            }
        )
        out["center_x"], out["center_y"], out["center_z"] = [
            float(v) for v in row["center"].tolist()
        ]
        out["translation_x"], out["translation_y"], out["translation_z"] = [
            float(v) for v in row["translation_velocity"].tolist()
        ]
        out["omega_x"], out["omega_y"], out["omega_z"] = [
            float(v) for v in row["omega"].tolist()
        ]
        yield out


def _add_atom_rows(
    atom_rows: List[Dict[str, object]],
    molecule_id: int,
    smiles: str,
    atomic_numbers: torch.Tensor,
    fragmentation: FragmentationResult,
    source: str,
    decomp: Dict[str, object],
    x: torch.Tensor,
    v: torch.Tensor,
) -> None:
    for row in iter_atom_decomposition_rows(
        source,
        x.detach().cpu(),
        v.detach().cpu(),
        decomp["rigid_velocity"].detach().cpu(),
        decomp["residual_velocity"].detach().cpu(),
        fragmentation.atom_to_fragment_id.cpu(),
        fragmentation.fragment_types,
    ):
        row.update(
            {
                "molecule_id": molecule_id,
                "smiles": smiles,
                "atomic_number": int(atomic_numbers[row["atom_index"]].item()),
            }
        )
        atom_rows.append(row)


def _save_molecule_summary(fragment_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    index_cols = ["molecule_id", "smiles", "num_atoms", "num_rotatable_bonds"]
    base = fragment_df[index_cols].drop_duplicates().set_index("molecule_id")

    source_mean = fragment_df.pivot_table(
        index="molecule_id",
        columns="velocity_source",
        values="residual_ratio",
        aggfunc="mean",
    )
    for source in ("target", "pred"):
        base[f"mean_{source}_residual_ratio"] = source_mean.get(source)

    type_mean_all = fragment_df.pivot_table(
        index="molecule_id",
        columns="fragment_type",
        values="residual_ratio",
        aggfunc="mean",
    )
    for fragment_type in FRAGMENT_TYPES:
        base[f"mean_{fragment_type}_residual_ratio"] = type_mean_all.get(fragment_type)

    type_source_mean = fragment_df.pivot_table(
        index="molecule_id",
        columns=["velocity_source", "fragment_type"],
        values="residual_ratio",
        aggfunc="mean",
    )
    for source in ("target", "pred"):
        for fragment_type in FRAGMENT_TYPES:
            col = (source, fragment_type)
            base[f"mean_{source}_{fragment_type}_residual_ratio"] = (
                type_source_mean[col] if col in type_source_mean else np.nan
            )

    out = base.reset_index()
    out.to_csv(output_path, index=False)
    return out


def _save_summary(fragment_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    summary_df = fragment_df.copy()
    if "fit_status" in summary_df.columns and "omega_norm" in summary_df.columns:
        summary_df.loc[summary_df["fit_status"] != "ok", "omega_norm"] = np.nan

    grouped = summary_df.groupby(["fragment_type", "velocity_source"], dropna=False)
    summary = grouped.agg(
        count=("residual_ratio", "size"),
        mean_residual_ratio=("residual_ratio", "mean"),
        std_residual_ratio=("residual_ratio", "std"),
        median_residual_ratio=("residual_ratio", "median"),
        mean_rigid_explain_ratio=("rigid_explain_ratio", "mean"),
        std_rigid_explain_ratio=("rigid_explain_ratio", "std"),
        median_rigid_explain_ratio=("rigid_explain_ratio", "median"),
        mean_omega_norm=("omega_norm", "mean"),
    ).reset_index()
    summary.to_csv(output_path, index=False)
    return summary


def _filtered_fragment_df(fragment_df: pd.DataFrame) -> pd.DataFrame:
    return fragment_df.loc[
        (fragment_df["fit_status"] == "ok")
        & (fragment_df["num_fragment_atoms"] >= MIN_FRAGMENT_ATOMS_FOR_SUMMARY)
    ].copy()


def _save_target_pred_errors(fragment_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    key_cols = [
        "molecule_id",
        "smiles",
        "num_atoms",
        "num_rotatable_bonds",
        "fragment_id",
        "fragment_type",
        "num_fragment_atoms",
    ]
    metric_cols = [
        "residual_ratio",
        "rigid_explain_ratio",
        "residual_norm",
        "velocity_norm",
        "translation_x",
        "translation_y",
        "translation_z",
        "omega_x",
        "omega_y",
        "omega_z",
        "omega_norm",
    ]

    target = fragment_df.loc[fragment_df["velocity_source"] == "target", key_cols + metric_cols]
    pred = fragment_df.loc[fragment_df["velocity_source"] == "pred", key_cols + metric_cols]
    joined = target.merge(pred, on=key_cols, suffixes=("_target", "_pred"), how="inner")

    if joined.empty:
        joined.to_csv(output_path, index=False)
        return joined

    translation_delta = joined[
        ["translation_x_pred", "translation_y_pred", "translation_z_pred"]
    ].to_numpy() - joined[
        ["translation_x_target", "translation_y_target", "translation_z_target"]
    ].to_numpy()
    omega_delta = joined[["omega_x_pred", "omega_y_pred", "omega_z_pred"]].to_numpy() - joined[
        ["omega_x_target", "omega_y_target", "omega_z_target"]
    ].to_numpy()

    joined["translation_error_norm"] = np.linalg.norm(translation_delta, axis=1)
    joined["omega_error_norm"] = np.linalg.norm(omega_delta, axis=1)
    joined["residual_ratio_error"] = (
        joined["residual_ratio_pred"] - joined["residual_ratio_target"]
    )
    joined["rigid_explain_ratio_error"] = (
        joined["rigid_explain_ratio_pred"] - joined["rigid_explain_ratio_target"]
    )
    joined["residual_norm_error"] = joined["residual_norm_pred"] - joined[
        "residual_norm_target"
    ]
    joined.to_csv(output_path, index=False)
    return joined


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

    fragment_rows: List[Dict[str, object]] = []
    molecule_examples: List[Dict[str, object]] = []
    atom_rows: List[Dict[str, object]] = []
    num_seen = 0

    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Analyzing")):
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
            local_z = flow["atomic_numbers"][atom_mask].detach().cpu()
            smiles = smiles_list[graph_idx]
            mol = mol_list[graph_idx]
            molecule_id = num_seen
            num_atoms = int(atom_mask.sum().item())

            fragmentation = fragment_molecule(
                smiles=smiles,
                mol=mol,
                expected_num_atoms=num_atoms,
            )
            fragment_ids = fragmentation.atom_to_fragment_id.to(device)

            target_decomp = decompose_velocity_by_fragment(
                local_x,
                local_target,
                fragment_ids,
                fragmentation.fragment_types,
            )
            pred_decomp = decompose_velocity_by_fragment(
                local_x,
                local_pred,
                fragment_ids,
                fragmentation.fragment_types,
            )

            for source, decomp in (("target", target_decomp), ("pred", pred_decomp)):
                fragment_rows.extend(
                    _fragment_rows(
                        molecule_id=molecule_id,
                        smiles=smiles,
                        num_atoms=num_atoms,
                        num_rotatable_bonds=fragmentation.num_rotatable_bonds,
                        velocity_source=source,
                        fragments=decomp["fragments"],
                    )
                )
                velocity = local_target if source == "target" else local_pred
                _add_atom_rows(
                    atom_rows=atom_rows,
                    molecule_id=molecule_id,
                    smiles=smiles,
                    atomic_numbers=local_z,
                    fragmentation=fragmentation,
                    source=source,
                    decomp=decomp,
                    x=local_x,
                    v=velocity,
                )

            if len(molecule_examples) < args.save_examples:
                molecule_examples.append(
                    {
                        "molecule_id": molecule_id,
                        "smiles": smiles,
                        "num_atoms": num_atoms,
                        "t": flow["t"][graph_idx].detach().cpu(),
                        "x_t": local_x.detach().cpu(),
                        "target_velocity": local_target.detach().cpu(),
                        "pred_velocity": local_pred.detach().cpu(),
                        "atom_to_fragment_id": fragmentation.atom_to_fragment_id.cpu(),
                        "fragment_types": fragmentation.fragment_types,
                        "ring_atom_mask": fragmentation.ring_atom_mask,
                        "aromatic_atom_mask": fragmentation.aromatic_atom_mask,
                        "rotatable_bond_atom_mask": fragmentation.rotatable_bond_atom_mask,
                        "target_rigid_velocity": target_decomp["rigid_velocity"].detach().cpu(),
                        "target_residual_velocity": target_decomp[
                            "residual_velocity"
                        ].detach().cpu(),
                        "pred_rigid_velocity": pred_decomp["rigid_velocity"].detach().cpu(),
                        "pred_residual_velocity": pred_decomp[
                            "residual_velocity"
                        ].detach().cpu(),
                    }
                )

            num_seen += 1

        if num_seen >= args.max_molecules:
            break

    if not fragment_rows:
        raise RuntimeError("No molecules were analyzed; check the split and max_molecules.")

    fragment_df = pd.DataFrame(fragment_rows)
    atom_df = pd.DataFrame(atom_rows)
    fragment_df.to_csv(output_dir / "decomp_by_fragment.csv", index=False)
    atom_df.to_csv(output_dir / "decomp_by_atom.csv", index=False)
    molecule_df = _save_molecule_summary(fragment_df, output_dir / "decomp_by_molecule.csv")
    summary_df = _save_summary(fragment_df, output_dir / "decomp_summary.csv")
    filtered_summary_df = _save_summary(
        _filtered_fragment_df(fragment_df),
        output_dir / "decomp_summary_filtered.csv",
    )
    error_df = _save_target_pred_errors(
        fragment_df,
        output_dir / "decomp_target_pred_errors.csv",
    )
    torch.save(molecule_examples, output_dir / "decomp_examples.pt")

    manifest = {
        "config": args.config,
        "checkpoint": checkpoint_path,
        "split": args.split,
        "max_molecules": args.max_molecules,
        "analyzed_molecules": int(num_seen),
        "t_low": args.t_low,
        "t_high": args.t_high,
        "seed": args.seed,
    }
    with open(output_dir / "analysis_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Saved fragment rows: {len(fragment_df)}")
    log.info(f"Saved atom rows: {len(atom_df)}")
    log.info(f"Saved molecule rows: {len(molecule_df)}")
    log.info(f"Saved summary rows: {len(summary_df)}")
    log.info(f"Saved filtered summary rows: {len(filtered_summary_df)}")
    log.info(f"Saved target/pred error rows: {len(error_df)}")
    log.info(f"Output directory: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--checkpoint", "-k", type=str, required=True)
    parser.add_argument("--split", choices=["val", "test", "train"], default="test")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--max_molecules", type=int, default=200)
    parser.add_argument("--output_dir", "-o", type=str, default="logs_velocity_decomp_first")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t_low", type=float, default=1.0e-4)
    parser.add_argument("--t_high", type=float, default=0.9999)
    parser.add_argument("--save_examples", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
