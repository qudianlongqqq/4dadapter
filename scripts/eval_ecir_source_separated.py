#!/usr/bin/env python
"""Source- and flexibility-separated diagnostics for a frozen ECIR checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import numpy as np
import pandas as pd
import torch

from etflow.ecir.audit import displacement_metrics
from etflow.ecir.dataset import ECIRMixedDataset
from etflow.ecir.model import ECIRFlowSystem


METRICS = (
    "bond_violation",
    "angle_violation",
    "torsion_circular_error",
    "ring_invalidity",
    "clash_score",
    "chirality_error",
    "aligned_RMSD",
)
SET_METRICS = ("COV_P", "COV_R", "MAT_P", "MAT_R", "diversity")


def _subset_names(row: pd.Series) -> list[str]:
    rotatable = int(row["rotatable_bond_count"])
    return [
        "all",
        "rotatable_le_2" if rotatable <= 2 else ("rotatable_3_5" if rotatable <= 5 else "rotatable_ge_6"),
        "ring" if bool(row["has_ring"]) else "non_ring",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--atlas_path", type=Path, required=True)
    parser.add_argument("--target_cache_dir", type=Path, required=True)
    parser.add_argument("--reproduction_dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_records", type=int, default=100)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=Path, default=Path("diagnostics/ecir_mvr/root_cause"))
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ECIRFlowSystem(**dict(payload["config"].get("model") or {})).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    dataset = ECIRMixedDataset(
        args.cache_dir,
        args.split,
        atlas_path=args.atlas_path,
        target_cache_dir=args.target_cache_dir,
        real_error_ratio=1.0,
        synthetic_error_ratio=0.0,
        clean_identity_ratio=0.0,
        max_records=args.max_records,
    )

    long_frame = pd.read_csv(args.reproduction_dir / "per_conformer.csv")
    baseline = long_frame[long_frame.method == "upstream"].set_index("sample_id")
    candidate = long_frame[long_frame.method == "ECIR_4step_teacher"].set_index("sample_id")
    rows = []
    with torch.inference_mode():
        for index, source_path in enumerate(dataset.files):
            record = torch.load(source_path, map_location="cpu", weights_only=False)
            entry = dataset.entries[index]
            data = dataset.get(index)
            sample_id = str(record.get("sample_id", record.get("mol_id")))
            upstream = torch.as_tensor(record[entry.get("coordinate_key", "x_init")], dtype=torch.float32)
            refined, diagnostics = model.refine(
                data.to(device), coordinates=upstream.to(device), steps=args.steps
            )
            refined = refined.cpu()
            gate_values = np.asarray([step["gate_mean"] for step in diagnostics], dtype=float)
            displacement = displacement_metrics(upstream, refined)
            row = {
                "molecule_id": str(record.get("source_mol_id", record.get("mol_id"))),
                "sample_id": sample_id,
                "source_type": str(entry.get("source_type", "unknown")),
                "rotatable_bond_count": int(record.get("num_rotatable_bonds", 0)),
                "has_ring": bool(torch.as_tensor(record.get("bond_is_in_ring", [])).any()),
                "gate_mean": float(gate_values.mean()),
                "gate_p50": float(np.quantile(gate_values, 0.50)),
                "gate_p90": float(np.quantile(gate_values, 0.90)),
                "gate_p95": float(np.quantile(gate_values, 0.95)),
                **displacement,
            }
            for metric in METRICS:
                row[f"upstream_{metric}"] = float(baseline.loc[sample_id, metric])
                row[f"refined_{metric}"] = float(candidate.loc[sample_id, metric])
                row[f"delta_{metric}"] = row[f"refined_{metric}"] - row[f"upstream_{metric}"]
            row["delta_internal_error_sum"] = sum(
                row[f"delta_{metric}"]
                for metric in ("bond_violation", "angle_violation", "torsion_circular_error", "ring_invalidity", "clash_score")
            )
            row["repair_outcome"] = (
                "improved" if row["delta_internal_error_sum"] < -1.0e-6
                else ("worsened" if row["delta_internal_error_sum"] > 1.0e-6 else "nearly_unchanged")
            )
            row["subsets"] = ";".join(_subset_names(pd.Series(row)))
            rows.append(row)

    conformer = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conformer.to_csv(args.output_dir / "source_separated_per_conformer.csv", index=False)

    numeric = [column for column in conformer.select_dtypes(include=[np.number]).columns]
    molecule = conformer.groupby(["molecule_id", "source_type"], as_index=False)[numeric].mean()
    molecule_outcome = conformer.groupby(["molecule_id", "source_type"])["repair_outcome"].agg(
        lambda values: values.value_counts().index[0]
    ).reset_index()
    molecule = molecule.merge(molecule_outcome, on=["molecule_id", "source_type"], how="left")
    old_molecule = pd.read_csv(args.reproduction_dir / "per_molecule.csv")
    source_by_molecule = conformer[["molecule_id", "source_type"]].drop_duplicates()
    old_molecule = old_molecule.merge(source_by_molecule, on="molecule_id", how="left")
    pivot = old_molecule.pivot(index=["molecule_id", "source_type"], columns="method", values=list(SET_METRICS))
    for metric in SET_METRICS:
        molecule = molecule.merge(
            pd.DataFrame({
                "molecule_id": pivot.index.get_level_values(0),
                "source_type": pivot.index.get_level_values(1),
                f"upstream_{metric}": pivot[(metric, "upstream")].values,
                f"refined_{metric}": pivot[(metric, "ECIR_4step_teacher")].values,
                f"delta_{metric}": pivot[(metric, "ECIR_4step_teacher")].values - pivot[(metric, "upstream")].values,
            }),
            on=["molecule_id", "source_type"], how="left",
        )
    molecule.to_csv(args.output_dir / "source_separated_per_molecule.csv", index=False)

    summary_rows = []
    for source in sorted(conformer.source_type.unique()):
        source_frame = conformer[conformer.source_type == source]
        for subset in ("all", "rotatable_le_2", "rotatable_3_5", "rotatable_ge_6", "ring", "non_ring"):
            selected_ids = source_frame.loc[
                source_frame.subsets.str.split(";").apply(lambda values: subset in values), "molecule_id"
            ].unique()
            selected = molecule[(molecule.source_type == source) & molecule.molecule_id.isin(selected_ids)]
            if selected.empty:
                continue
            summary = {
                "source_type": source,
                "subset": subset,
                "molecules": int(selected.molecule_id.nunique()),
                "conformers": int(source_frame[source_frame.molecule_id.isin(selected_ids)].shape[0]),
                "fraction_improved": float((selected.repair_outcome == "improved").mean()),
                "fraction_worsened": float((selected.repair_outcome == "worsened").mean()),
                "fraction_nearly_unchanged": float((selected.repair_outcome == "nearly_unchanged").mean()),
            }
            for column in selected.select_dtypes(include=[np.number]).columns:
                if column not in {"rotatable_bond_count"}:
                    summary[column] = float(selected[column].mean())
            summary_rows.append(summary)
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(args.output_dir / "source_separated_summary.csv", index=False)
    print(json.dumps(summary_frame[summary_frame.subset == "all"].to_dict("records"), indent=2))


if __name__ == "__main__":
    main()
