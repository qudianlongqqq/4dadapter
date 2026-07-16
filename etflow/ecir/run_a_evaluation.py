"""Validation-only evaluation primitives for MCVR Stage 2b Run A."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from .acceptance import evaluate_candidate, select_trajectory_candidate
from .audit import displacement_metrics, torsion_change_metrics
from .mvr_dataset import deterministic_error_features


CHEMICAL_METRICS = (
    "bond_outlier_rate",
    "bond_outlier_magnitude",
    "angle_outlier_rate",
    "angle_outlier_magnitude",
    "ring_bond_outlier_rate",
    "ring_planarity_outlier_rate",
    "clash_penetration",
    "severe_clash_rate",
    "chirality_error",
    "stereocenter_degenerate_rate",
    "torsion_prior_outlier_score",
    "total_thresholded_validity_score",
)
ACCURACY_METRICS = ("aligned_RMSD", "MAT_P", "MAT_R", "COV_P", "COV_R", "diversity")
BOOTSTRAP_METRICS = (*CHEMICAL_METRICS, *ACCURACY_METRICS)
GROUP_NAMES = (
    "all", "ETFlow_normal", "Cartesian_mild", "Cartesian_medium", "Cartesian_severe",
    "unseen_update_scale_0.35", "rotatable_le_2", "rotatable_3_5",
    "rotatable_ge_6", "ring", "non_ring", "clean_valid",
)


def rmsd_matrix(generated: Sequence[Tensor], references: Tensor) -> Tensor:
    mobile = torch.stack([torch.as_tensor(value, dtype=torch.float64) for value in generated])
    target = torch.as_tensor(references, dtype=torch.float64)
    if target.ndim == 2:
        target = target.unsqueeze(0)
    x = mobile - mobile.mean(1, keepdim=True)
    y = target - target.mean(1, keepdim=True)
    covariance = torch.einsum("gni,rnj->grij", x, y)
    u, singular, vh = torch.linalg.svd(covariance)
    determinant = torch.linalg.det(vh.transpose(-2, -1) @ u.transpose(-2, -1))
    trace = singular.sum(-1) - torch.where(determinant < 0, 2.0 * singular[..., -1], 0.0)
    squared = (
        x.square().sum((1, 2))[:, None]
        + y.square().sum((1, 2))[None, :]
        - 2.0 * trace
    ) / x.size(1)
    return squared.clamp_min(0.0).sqrt().to(torch.float32)


def nearest_rmsd(coordinates: Tensor, references: Tensor) -> float:
    return float(rmsd_matrix([coordinates], references).min())


def _load_source_coordinates(row) -> tuple[dict[str, Any], Tensor]:
    record = torch.load(Path(row.source_path), map_location="cpu", weights_only=False)
    if row.coordinate_path is not None and not pd.isna(row.coordinate_path):
        payload = torch.load(Path(row.coordinate_path), map_location="cpu", weights_only=False)
        coordinates = torch.as_tensor(payload[row.coordinate_key], dtype=torch.float32)
    else:
        coordinates = torch.as_tensor(record[row.coordinate_key], dtype=torch.float32)
    return record, coordinates


def graph_data(
    record: Mapping[str, Any], coordinates: Tensor, row,
    *, active_mode_mask: Tensor | None = None,
) -> Data:
    edge_index = torch.as_tensor(record["edge_index"], dtype=torch.long)
    return Data(
        num_nodes=coordinates.size(0),
        node_attr=torch.as_tensor(record["node_attr"], dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.as_tensor(
            record.get("edge_attr", torch.ones(edge_index.size(1), 1)), dtype=torch.float32
        ),
        bond_is_in_ring=torch.as_tensor(
            record.get("bond_is_in_ring", torch.zeros(edge_index.size(1))), dtype=torch.bool
        ),
        rotatable_bond_index=torch.as_tensor(
            record.get("rotatable_bond_index", torch.empty(2, 0)), dtype=torch.long
        ),
        atom_bond_influence_index=torch.as_tensor(
            record.get("atom_bond_influence_index", torch.empty(2, 0)), dtype=torch.long
        ),
        x_init=coordinates,
        upstream_metadata=torch.tensor([[
            min(float(row.NFE) / 10.0, 1.0), float(row.update_scale),
            (float(row.seed) % 10_000.0) / 10_000.0, 1.0,
        ]], dtype=torch.float32),
        active_mode_mask=(
            torch.as_tensor(active_mode_mask, dtype=torch.float32).reshape(1, 6)
            if active_mode_mask is not None else torch.zeros(1, 6)
        ),
    )


def build_items(source_path: str | Path, target_path: str | Path, validity) -> list[dict[str, Any]]:
    source = pd.read_parquet(source_path).sort_values(["molecule_id", "sample_id"])
    if set(source.split.unique()) != {"val"}:
        raise ValueError("Run A evaluation requires validation sources only")
    targets = pd.read_parquet(target_path).set_index("sample_id")
    items = []
    for row in source.itertuples(index=False):
        record, coordinates = _load_source_coordinates(row)
        target_row = targets.loc[row.sample_id]
        payload = torch.load(Path(target_row.target_cache_path), map_location="cpu", weights_only=False)
        minimal = torch.as_tensor(payload["x_target"], dtype=torch.float32)
        references = torch.as_tensor(
            record.get("x_ref_candidates", record.get("x_ref_aligned")), dtype=torch.float32
        )
        if references.ndim == 2:
            references = references.unsqueeze(0)
        input_validity = validity.evaluate(coordinates, record, baseline_coordinates=coordinates)
        rotatable = int(record.get("num_rotatable_bonds", 0))
        has_ring = bool(torch.as_tensor(record.get("bond_is_in_ring", [])).any())
        clean = all(input_validity[name] <= 0.0 for name in (
            "bond_outlier_rate", "angle_outlier_rate", "ring_bond_outlier_rate",
            "ring_planarity_outlier_rate", "clash_penetration", "severe_clash_rate",
            "stereocenter_degenerate_rate",
        ))
        active = torch.tensor([
            float(input_validity["bond_outlier_rate"] > 0),
            float(input_validity["angle_outlier_rate"] > 0),
            float(input_validity["ring_bond_outlier_rate"] > 0 or input_validity["ring_planarity_outlier_rate"] > 0),
            float(input_validity["clash_penetration"] > 0 or input_validity["severe_clash_rate"] > 0),
            float(input_validity["torsion_prior_outlier_score"] > 4.0),
            float(clean),
        ])
        groups = ["all"]
        if row.generator_name == "ETFlow_formal_upstream":
            groups.append("ETFlow_normal")
        elif row.source_severity in {"mild", "medium", "severe"}:
            groups.append(f"Cartesian_{row.source_severity}")
        if row.generator_name == "Cartesian_teacher_100k" and abs(float(row.update_scale) - 0.35) < 1e-12:
            groups.append("unseen_update_scale_0.35")
        groups.append("rotatable_le_2" if rotatable <= 2 else (
            "rotatable_3_5" if rotatable <= 5 else "rotatable_ge_6"
        ))
        groups.append("ring" if has_ring else "non_ring")
        if clean:
            groups.append("clean_valid")
        items.append({
            "row": row, "record": record, "input": coordinates,
            "minimal_target": minimal, "references": references,
            "input_validity": input_validity, "input_rmsd": nearest_rmsd(coordinates, references),
            "data": graph_data(record, coordinates, row, active_mode_mask=active), "groups": groups,
            "rotatable": rotatable, "has_ring": has_ring, "clean": clean,
        })
    return items


def build_clean_control_items(
    items: Sequence[dict[str, Any]], validity, *, limit: int = 20
) -> list[dict[str, Any]]:
    """Build validation-only identity controls from chemically clean references."""
    controls = []
    seen_molecules: set[str] = set()
    clean_metrics = (
        "bond_outlier_rate", "angle_outlier_rate", "ring_bond_outlier_rate",
        "ring_planarity_outlier_rate", "clash_penetration", "severe_clash_rate",
        "stereocenter_degenerate_rate",
    )
    for item in items:
        molecule = str(item["row"].molecule_id)
        if molecule in seen_molecules:
            continue
        for reference in item["references"]:
            reference = torch.as_tensor(reference, dtype=torch.float32).clone()
            values = validity.evaluate(reference, item["record"], baseline_coordinates=reference)
            if any(float(values[name]) > 0.0 for name in clean_metrics):
                continue
            row = item["row"]._replace(sample_id=f"{item['row'].sample_id}__clean_control")
            control = {
                **item,
                "row": row,
                "input": reference,
                "minimal_target": reference.clone(),
                "input_validity": values,
                "input_rmsd": nearest_rmsd(reference, item["references"]),
                "data": graph_data(
                    item["record"], reference, row,
                    active_mode_mask=torch.tensor([0, 0, 0, 0, 0, 1]),
                ),
                "groups": ["clean_valid"],
                "clean": True,
            }
            controls.append(control)
            seen_molecules.add(molecule)
            break
        if len(controls) >= int(limit):
            break
    return controls


@torch.inference_mode()
def infer_mvr(
    model,
    items: Sequence[dict[str, Any]],
    validity,
    *,
    device: torch.device,
    steps: int = 4,
    step_size: float = 0.25,
    batch_size: int = 32,
    acceptance_mode: str = "best_of_trajectory",
    acceptance_config: Mapping[str, float] | None = None,
) -> tuple[list[Tensor], list[Tensor], list[dict[str, Any]]]:
    model.eval()
    raw_coordinates: list[Tensor] = []
    accepted_coordinates: list[Tensor] = []
    metadata: list[dict[str, Any]] = []
    schedule = torch.linspace(0.0, 1.0, int(steps)).tolist()
    for start in range(0, len(items), batch_size):
        selected = list(items[start:start + batch_size])
        batch = Batch.from_data_list([item["data"] for item in selected]).to(device)
        current = torch.cat([item["input"] for item in selected]).to(device)
        ptr = batch.ptr.detach().cpu().tolist()
        trajectories: list[list[Tensor]] = [[] for _ in selected]
        uncertainties: list[list[float]] = [[] for _ in selected]
        gates: list[list[float]] = [[] for _ in selected]
        safety: list[list[float]] = [[] for _ in selected]
        velocity_norms: list[list[float]] = [[] for _ in selected]
        torsion_gates: list[list[float]] = [[] for _ in selected]
        torsion_active: list[list[float]] = [[] for _ in selected]
        torsion_norms: list[list[float]] = [[] for _ in selected]
        torsion_fractions: list[list[float]] = [[] for _ in selected]
        max_torsion_gate = 0.0
        max_torsion_contribution = 0.0
        for time_value in schedule:
            current_cpu = current.detach().cpu()
            features = []
            for local, item in enumerate(selected):
                left, right = ptr[local], ptr[local + 1]
                values = validity.evaluate(
                    current_cpu[left:right], item["record"], baseline_coordinates=item["input"]
                )
                features.append(deterministic_error_features(
                    values, item["record"], str(item["row"].source_severity)
                ))
            trust_remaining = []
            settings = dict(acceptance_config or {})
            for local, item in enumerate(selected):
                left, right = ptr[local], ptr[local + 1]
                changed = torsion_change_metrics(
                    item["input"], current_cpu[left:right], item["record"]
                )["max_rotatable_torsion_change"]
                limit = float(
                    settings.get("max_high_flex_torsion_change_rad", 0.35)
                    if item["rotatable"] >= 6
                    else settings.get("max_torsion_change_rad", 0.70)
                )
                trust_remaining.append(max(0.0, limit - float(changed)))
            output = model(
                batch, current, current.new_full((len(selected),), float(time_value)),
                deterministic_features=torch.stack(features).to(device),
                torsion_trust_remaining=current.new_tensor(trust_remaining),
            )
            max_torsion_gate = max(max_torsion_gate, float(output["torsion_gate"].abs().max()))
            max_torsion_contribution = max(
                max_torsion_contribution,
                float(output["v_torsion_contribution"].abs().max()),
            )
            current = current + float(step_size) * output["v_final"]
            snapshot = current.detach().cpu()
            for local in range(len(selected)):
                left, right = ptr[local], ptr[local + 1]
                trajectories[local].append(snapshot[left:right].clone())
                uncertainties[local].append(float(output["uncertainty"][local]))
                gates[local].append(float(output["rigid_gate"][local]))
                safety[local].append(float(output["global_safety_gate"][local]))
                velocity_norms[local].append(float(
                    torch.linalg.vector_norm(output["v_final"][left:right], dim=-1).mean()
                ))
                torsion_gates[local].append(float(output["torsion_gate"][local]))
                torsion_active[local].append(float(output["torsion_gate_active"][local]))
                torsion_norm = float(torch.linalg.vector_norm(
                    output["v_torsion_contribution"][left:right], dim=-1
                ).mean())
                rigid_norm = float(torch.linalg.vector_norm(
                    output["v_rigid_contribution"][left:right], dim=-1
                ).mean())
                torsion_norms[local].append(torsion_norm)
                torsion_fractions[local].append(torsion_norm / max(torsion_norm + rigid_norm, 1e-12))
        if getattr(model, "torsion_gate_fixed_zero", False) and (
            max_torsion_gate != 0.0 or max_torsion_contribution != 0.0
        ):
            raise RuntimeError("Run A torsion branch contributed to inference")
        for local, item in enumerate(selected):
            raw = trajectories[local][-1]
            accepted, decision = select_trajectory_candidate(
                item["input"], trajectories[local], item["record"], validity,
                mode=acceptance_mode, uncertainties=uncertainties[local],
                config=acceptance_config,
            )
            raw_coordinates.append(raw)
            accepted_coordinates.append(accepted)
            metadata.append({
                "accepted": decision.accepted,
                "selected_step": decision.selected_step,
                "reject_reasons": ";".join(decision.reject_reasons),
                "validity_gain": decision.validity_gain,
                "uncertainty": decision.uncertainty,
                "rigid_gate_mean": float(np.mean(gates[local])),
                "global_safety_gate_mean": float(np.mean(safety[local])),
                "velocity_norm_mean": float(np.mean(velocity_norms[local])),
                "torsion_gate_max": max_torsion_gate,
                "torsion_contribution_max": max_torsion_contribution,
                "torsion_gate_mean": float(np.mean(torsion_gates[local])),
                "torsion_gate_active_fraction": float(np.mean(torsion_active[local])),
                "torsion_velocity_norm": float(np.mean(torsion_norms[local])),
                "torsion_velocity_fraction": float(np.mean(torsion_fractions[local])),
            })
    return raw_coordinates, accepted_coordinates, metadata


def _validity_values(values: Mapping[str, float]) -> dict[str, float]:
    return {
        **{name: float(values[name]) for name in CHEMICAL_METRICS if name != "chirality_error"},
        "chirality_error": 1.0 - float(values["chirality_preserved"]),
    }


def method_rows(
    items: Sequence[dict[str, Any]],
    method_coordinates: Mapping[str, Sequence[Tensor]],
    validity,
    method_metadata: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> pd.DataFrame:
    rows = []
    method_metadata = method_metadata or {}
    for method, coordinates in method_coordinates.items():
        metadata = method_metadata.get(method, [{} for _ in items])
        for item, candidate, extra in zip(items, coordinates, metadata):
            candidate = torch.as_tensor(candidate, dtype=torch.float32)
            values = validity.evaluate(candidate, item["record"], baseline_coordinates=item["input"])
            displacement = displacement_metrics(item["input"], candidate)
            torsion = torsion_change_metrics(item["input"], candidate, item["record"])
            rmsd = nearest_rmsd(candidate, item["references"])
            row = {
                "method": method,
                "molecule_id": str(item["row"].molecule_id),
                "sample_id": str(item["row"].sample_id),
                "source": str(item["row"].generator_name),
                "severity": str(item["row"].source_severity),
                "update_scale": float(item["row"].update_scale),
                "rotatable_bond_count": item["rotatable"],
                "has_ring": item["has_ring"],
                "clean_valid": item["clean"],
                "groups": ";".join(item["groups"]),
                "aligned_RMSD": rmsd,
                "delta_aligned_RMSD": rmsd - item["input_rmsd"],
                **_validity_values(values),
                "mean_displacement": displacement["mean_atom_displacement"],
                "molecule_rms_displacement": displacement["aligned_rms_displacement"],
                "max_displacement": displacement["max_atom_displacement"],
                "coordinate_unchanged": float(torch.equal(candidate, item["input"])),
                "mean_torsion_change": torsion["torsion_circular_change"],
                "high_flex_torsion_change": (
                    torsion["max_rotatable_torsion_change"] if item["rotatable"] >= 6 else 0.0
                ),
                "accepted": float(extra.get("accepted", method != "run_a_accepted")),
                "selected_step": float(extra.get("selected_step", 0)),
                "uncertainty": float(extra.get("uncertainty", 0.0)),
                "reject_reasons": str(extra.get("reject_reasons", "")),
                "rigid_gate_mean": float(extra.get("rigid_gate_mean", 0.0)),
                "global_safety_gate_mean": float(extra.get("global_safety_gate_mean", 0.0)),
                "velocity_norm_mean": float(extra.get("velocity_norm_mean", 0.0)),
                "torsion_gate_max": float(extra.get("torsion_gate_max", 0.0)),
                "torsion_contribution_max": float(extra.get("torsion_contribution_max", 0.0)),
                "torsion_gate_mean": float(extra.get("torsion_gate_mean", 0.0)),
                "torsion_gate_active_fraction": float(extra.get("torsion_gate_active_fraction", 0.0)),
                "torsion_velocity_norm": float(extra.get("torsion_velocity_norm", 0.0)),
                "torsion_velocity_fraction": float(extra.get("torsion_velocity_fraction", 0.0)),
            }
            for name in CHEMICAL_METRICS:
                input_value = (
                    1.0 - item["input_validity"]["chirality_preserved"]
                    if name == "chirality_error" else item["input_validity"][name]
                )
                row[f"delta_{name}"] = row[name] - float(input_value)
            rows.append(row)
    return pd.DataFrame(rows)


def _set_metrics(items, coordinates, selected_indices) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in selected_indices:
        grouped[str(items[index]["row"].molecule_id)].append(index)
    result = {}
    for molecule, indices in grouped.items():
        generated = [coordinates[index] for index in indices]
        references = items[indices[0]]["references"]
        matrix = rmsd_matrix(generated, references)
        if len(generated) > 1:
            self_matrix = rmsd_matrix(generated, torch.stack(generated))
            upper = torch.triu(
                torch.ones(len(generated), len(generated), dtype=torch.bool), diagonal=1
            )
            diversity = float(self_matrix[upper].mean())
        else:
            diversity = 0.0
        result[molecule] = {
            "COV_P": float((matrix.min(1).values < 1.25).float().mean()),
            "COV_R": float((matrix.min(0).values < 1.25).float().mean()),
            "MAT_P": float(matrix.min(1).values.mean()),
            "MAT_R": float(matrix.min(0).values.mean()),
            "diversity": diversity,
        }
    return result


def molecule_rows(
    record_rows: pd.DataFrame,
    items: Sequence[dict[str, Any]],
    method_coordinates: Mapping[str, Sequence[Tensor]],
    *,
    group: str = "all",
) -> pd.DataFrame:
    selected_indices = [index for index, item in enumerate(items) if group in item["groups"]]
    selected_ids = {str(items[index]["row"].sample_id) for index in selected_indices}
    frame = record_rows[record_rows.sample_id.isin(selected_ids)]
    rows = []
    numeric = [
        name for name in frame.select_dtypes(include=[np.number]).columns
        if name not in {"update_scale"}
    ]
    for method, coordinates in method_coordinates.items():
        set_values = _set_metrics(items, coordinates, selected_indices) if selected_indices else {}
        method_frame = frame[frame.method == method]
        for molecule, subset in method_frame.groupby("molecule_id"):
            row = {"group": group, "method": method, "molecule_id": molecule}
            row.update({name: float(subset[name].mean()) for name in numeric})
            row.update(set_values[molecule])
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_groups(
    record_rows: pd.DataFrame,
    items: Sequence[dict[str, Any]],
    method_coordinates: Mapping[str, Sequence[Tensor]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    molecule_frames = []
    summary = []
    for group in GROUP_NAMES:
        molecules = molecule_rows(record_rows, items, method_coordinates, group=group)
        if molecules.empty:
            for method in method_coordinates:
                summary.append({"group": group, "method": method, "molecules": 0, "records": 0})
            continue
        molecule_frames.append(molecules)
        record_count = sum(group in item["groups"] for item in items)
        for method, subset in molecules.groupby("method"):
            row = {
                "group": group, "method": method,
                "molecules": int(subset.molecule_id.nunique()), "records": int(record_count),
            }
            row.update({
                name: float(subset[name].mean())
                for name in subset.select_dtypes(include=[np.number]).columns
            })
            row["accepted_fraction"] = float(subset.accepted.mean())
            row["rejected_fraction"] = 1.0 - row["accepted_fraction"]
            row["unchanged_fraction"] = float(subset.coordinate_unchanged.mean())
            row["validity_improved_fraction"] = float(
                (subset.delta_total_thresholded_validity_score < -1e-6).mean()
            )
            row["validity_worsened_fraction"] = float(
                (subset.delta_total_thresholded_validity_score > 1e-6).mean()
            )
            row["RMSD_improved_fraction"] = float((subset.delta_aligned_RMSD < -1e-6).mean())
            row["RMSD_worsened_fraction"] = float((subset.delta_aligned_RMSD > 1e-6).mean())
            row["p95_displacement"] = float(subset.molecule_rms_displacement.quantile(0.95))
            row["max_displacement"] = float(subset.max_displacement.max())
            row["p95_torsion_change"] = float(subset.mean_torsion_change.quantile(0.95))
            high_values = subset.loc[subset.rotatable_bond_count >= 6, "high_flex_torsion_change"]
            row["high_flex_p95_torsion_change"] = float(high_values.quantile(0.95)) if not high_values.empty else 0.0
            summary.append(row)
    return pd.DataFrame(summary), pd.concat(molecule_frames, ignore_index=True) if molecule_frames else pd.DataFrame()


def paired_bootstrap(
    molecule_frame: pd.DataFrame,
    *,
    candidate: str,
    baseline: str = "upstream",
    draws: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    all_frame = molecule_frame[molecule_frame.group == "all"]
    result = {}
    for metric in BOOTSTRAP_METRICS:
        pivot = all_frame.pivot(index="molecule_id", columns="method", values=metric).dropna()
        delta = pivot[candidate].to_numpy() - pivot[baseline].to_numpy()
        if not delta.size:
            result[metric] = {"mean": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
            continue
        rng = np.random.default_rng(seed)
        means = np.asarray([
            rng.choice(delta, size=delta.size, replace=True).mean() for _ in range(draws)
        ])
        result[metric] = {
            "mean": float(delta.mean()),
            "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975)),
        }
    return result


def accuracy_gate(
    summary: pd.DataFrame,
    bootstrap: Mapping[str, Mapping[str, float]],
    margins: Mapping[str, float],
    *,
    method: str = "run_a_accepted",
) -> dict[str, bool]:
    all_rows = summary[summary.group == "all"].set_index("method")
    candidate, baseline = all_rows.loc[method], all_rows.loc["upstream"]
    return {
        "rmsd_mean": bootstrap["aligned_RMSD"]["mean"] <= float(margins["rmsd_mean_delta_max"]),
        "rmsd_ci": bootstrap["aligned_RMSD"]["ci95_high"] <= float(margins["rmsd_ci_upper_max"]),
        "mat_p_mean": bootstrap["MAT_P"]["mean"] <= float(margins["mat_p_mean_delta_max"]),
        "mat_p_ci": bootstrap["MAT_P"]["ci95_high"] <= float(margins["mat_p_ci_upper_max"]),
        "mat_r_mean": bootstrap["MAT_R"]["mean"] <= float(margins["mat_r_mean_delta_max"]),
        "mat_r_ci": bootstrap["MAT_R"]["ci95_high"] <= float(margins["mat_r_ci_upper_max"]),
        "cov_p": candidate.COV_P >= baseline.COV_P - float(margins["cov_p_absolute_drop_max"]),
        "cov_r": candidate.COV_R >= baseline.COV_R - float(margins["cov_r_absolute_drop_max"]),
    }


def evaluate_run_a_only(
    model,
    items,
    validity,
    *,
    device,
    inference: Mapping[str, Any],
    margins: Mapping[str, float],
    bootstrap_draws: int = 500,
    clean_control_items: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    raw, accepted, metadata = infer_mvr(
        model, items, validity, device=device,
        steps=int(inference["teacher_steps"]), step_size=float(inference["step_size"]),
        acceptance_mode=str(inference["acceptance_mode"]),
    )
    coordinates = {
        "upstream": [item["input"] for item in items],
        "run_a_raw": raw,
        "run_a_accepted": accepted,
    }
    rows = method_rows(
        items, coordinates, validity,
        method_metadata={"run_a_raw": metadata, "run_a_accepted": metadata},
    )
    summary, molecules = summarize_groups(rows, items, coordinates)
    bootstrap = paired_bootstrap(
        molecules, candidate="run_a_accepted", draws=bootstrap_draws
    )
    gates = accuracy_gate(summary, bootstrap, margins)
    all_rows = summary[summary.group == "all"].set_index("method")
    candidate = all_rows.loc["run_a_accepted"]
    upstream = all_rows.loc["upstream"]
    clean_controls = list(clean_control_items or build_clean_control_items(items, validity))
    clean_identity_fraction = math.nan
    clean_summary = pd.DataFrame()
    if clean_controls:
        _, clean_accepted, clean_metadata = infer_mvr(
            model, clean_controls, validity, device=device,
            steps=int(inference["teacher_steps"]), step_size=float(inference["step_size"]),
            acceptance_mode=str(inference["acceptance_mode"]),
        )
        clean_coordinates = {
            "upstream": [item["input"] for item in clean_controls],
            "run_a_accepted": clean_accepted,
        }
        clean_rows = method_rows(
            clean_controls, clean_coordinates, validity,
            method_metadata={"run_a_accepted": clean_metadata},
        )
        clean_summary, _ = summarize_groups(
            clean_rows, clean_controls, clean_coordinates
        )
        clean_match = clean_summary[
            (clean_summary.group == "clean_valid")
            & (clean_summary.method == "run_a_accepted")
        ]
        if not clean_match.empty:
            clean_identity_fraction = float(clean_match.unchanged_fraction.iloc[0])
    return {
        "record_rows": rows,
        "summary": summary,
        "molecule_rows": molecules,
        "bootstrap": bootstrap,
        "accuracy_gate": gates,
        "accuracy_noninferior": all(gates.values()),
        "validity_delta": float(candidate.total_thresholded_validity_score - all_rows.loc["upstream"].total_thresholded_validity_score),
        "mean_displacement": float(candidate.molecule_rms_displacement),
        "acceptance_fraction": float(candidate.accepted_fraction),
        "validity_worsened_fraction": float(candidate.validity_worsened_fraction),
        "chirality_delta": float(candidate.chirality_error - upstream.chirality_error),
        "severe_clash_delta": float(candidate.severe_clash_rate - upstream.severe_clash_rate),
        "high_flex_torsion_change": float(
            summary[
                (summary.group == "rotatable_ge_6")
                & (summary.method == "run_a_accepted")
            ].high_flex_torsion_change.iloc[0]
        ) if not summary[
            (summary.group == "rotatable_ge_6")
            & (summary.method == "run_a_accepted")
        ].empty else math.nan,
        "identity_fraction": clean_identity_fraction,
        "clean_control_summary": clean_summary,
        "torsion_gate_max": float(rows[rows.method.str.startswith("run_a")].torsion_gate_max.max()),
        "torsion_contribution_max": float(rows[rows.method.str.startswith("run_a")].torsion_contribution_max.max()),
    }
