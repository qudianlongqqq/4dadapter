"""Validation-only inference and metrics for unified BAC candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Batch

from .audit import torsion_change_metrics
from .bac_constraints import canonical_constraint_fields
from .bac_safety import (
    BACSafetyConfig,
    evaluate_bac_proposal,
    select_safe_bac_proposal,
)
from .mvr_dataset import deterministic_error_features
from .run_a_evaluation import method_rows, paired_bootstrap, summarize_groups


def _coordinate_update_stats(update: Tensor) -> dict[str, float]:
    update = torch.as_tensor(update, dtype=torch.float32)
    if not update.numel():
        return {"rms": 0.0, "max": 0.0}
    norms = torch.linalg.vector_norm(update, dim=-1)
    return {
        "rms": float(norms.square().mean().sqrt()),
        "max": float(norms.max()),
    }


def attach_canonical_constraints(
    items: Sequence[dict[str, Any]],
    validity: Any,
    *,
    source_identity_sha256: str,
) -> None:
    for item in items:
        fields = canonical_constraint_fields(
            validity,
            item["record"],
            source_identity_sha256=source_identity_sha256,
        )
        for name, value in fields.items():
            setattr(item["data"], name, value)


@torch.inference_mode()
def infer_bac(
    model: torch.nn.Module,
    items: Sequence[dict[str, Any]],
    validity: Any,
    *,
    device: torch.device,
    steps: int = 4,
    step_size: float = 0.25,
    batch_size: int = 64,
    safety_config: BACSafetyConfig | None = None,
    trajectory_semantics: str = "legacy_bac",
) -> tuple[list[Tensor], list[dict[str, Any]]]:
    if trajectory_semantics not in {"legacy_bac", "formal_d1b"}:
        raise ValueError("unknown BAC trajectory semantics")
    config = safety_config or BACSafetyConfig()
    coordinates: list[Tensor] = []
    metadata: list[dict[str, Any]] = []
    model.eval()
    for start in range(0, len(items), int(batch_size)):
        selected = items[start : start + int(batch_size)]
        batch = Batch.from_data_list([item["data"] for item in selected]).to(device)
        current = batch.x_init.clone()
        ptr = batch.ptr.detach().cpu().tolist()
        trajectories: list[list[Tensor]] = [[] for _ in selected]
        step_diagnostics: list[list[dict[str, float]]] = [[] for _ in selected]
        schedule = (
            torch.linspace(0.0, 1.0, int(steps)).tolist()
            if trajectory_semantics == "formal_d1b"
            else [1.0 - step / max(int(steps), 1) for step in range(int(steps))]
        )
        for step, time_value in enumerate(schedule):
            model_kwargs: dict[str, Tensor] = {}
            if trajectory_semantics == "formal_d1b":
                current_cpu = current.detach().cpu()
                features = []
                trust_remaining = []
                for local, item in enumerate(selected):
                    left, right = ptr[local], ptr[local + 1]
                    values = validity.evaluate(
                        current_cpu[left:right],
                        item["record"],
                        baseline_coordinates=item["input"],
                    )
                    severity = str(getattr(item.get("row"), "source_severity", "normal"))
                    features.append(
                        deterministic_error_features(values, item["record"], severity)
                    )
                    changed = torsion_change_metrics(
                        item["input"], current_cpu[left:right], item["record"]
                    )["max_rotatable_torsion_change"]
                    limit = 0.35 if int(item.get("rotatable", 0)) >= 6 else 0.70
                    trust_remaining.append(max(0.0, limit - float(changed)))
                model_kwargs = {
                    "deterministic_features": torch.stack(features).to(device),
                    "torsion_trust_remaining": current.new_tensor(trust_remaining),
                }
            t = current.new_full((batch.num_graphs,), float(time_value))
            output = model(batch, current, t, **model_kwargs)
            current = current + float(step_size) * output["v_final"]
            for local in range(len(selected)):
                left, right = ptr[local], ptr[local + 1]
                trajectories[local].append(current[left:right].detach().cpu().clone())
                step_diagnostics[local].append(
                    {
                        "step": step + 1,
                        "global_safety_gate_mean": float(
                            output["global_safety_gate"][local].mean()
                        ),
                        "velocity_norm_mean": float(
                            torch.linalg.vector_norm(
                                output["v_final"][left:right], dim=-1
                            ).mean()
                        ),
                        "angle_gate_mean": float(
                            output.get("angle_gate", current.new_zeros(1)).mean()
                        ),
                        "clash_gate_mean": float(
                            output.get("clash_gate", current.new_zeros(1)).mean()
                        ),
                        "neural_delta": _coordinate_update_stats(
                            float(step_size)
                            * output.get(
                                "v_bond_cartesian",
                                output.get(
                                    "v_neural_prior",
                                    output.get("v_raw", output["v_final"]),
                                ),
                            )[left:right]
                        ),
                        "angle_delta": _coordinate_update_stats(
                            output.get(
                                "v_angle_jacobian_coordinate",
                                float(step_size)
                                * output.get(
                                    "v_jacobian_geometry", torch.zeros_like(current)
                                ),
                            )[left:right]
                        ),
                        "clash_delta": _coordinate_update_stats(
                            output.get(
                                "v_clash_repulsion_coordinate",
                                torch.zeros_like(current),
                            )[left:right]
                        ),
                        "fused_delta": _coordinate_update_stats(
                            float(step_size) * output["v_final"][left:right]
                        ),
                    }
                )
        for local, item in enumerate(selected):
            source = item["input"]
            candidates = []
            for step, proposal in enumerate(trajectories[local], start=1):
                if config.enable_backtracking:
                    checked, decision = select_safe_bac_proposal(
                        source,
                        proposal - source,
                        item["record"],
                        validity,
                        config,
                    )
                else:
                    checked = proposal
                    decision = evaluate_bac_proposal(
                        source, checked, item["record"], validity, config
                    )
                candidates.append((step, checked, decision))
            safe = [value for value in candidates if value[2]["accepted"]]
            if safe:
                selected_step, accepted, decision = max(
                    safe,
                    key=lambda value: (
                        value[2]["bac_gain"],
                        -value[2]["displacement"]["aligned_rms_displacement"],
                        -value[0],
                    ),
                )
                rolled_back = False
            else:
                selected_step, accepted = 0, source.clone()
                decision = candidates[-1][2] if candidates else {
                    "accepted": False,
                    "reasons": ["no_candidate"],
                }
                rolled_back = True
            coordinates.append(torch.as_tensor(accepted).clone())
            diag_index = selected_step - 1 if selected_step > 0 else -1
            diag = step_diagnostics[local][diag_index]
            proposal = trajectories[local][-1]
            proposal_decision = evaluate_bac_proposal(
                source, proposal, item["record"], validity, config
            )
            final_decision = evaluate_bac_proposal(
                source, accepted, item["record"], validity, config
            )
            all_failed_checks = sorted(
                {
                    reason
                    for candidate in candidates
                    for reason in candidate[2].get("reasons", [])
                }
            )
            reasons = list(decision.get("reasons", []))
            metadata.append(
                {
                    "accepted": not rolled_back,
                    "selected_step": selected_step,
                    "reject_reasons": ";".join(decision.get("reasons", [])),
                    "rolled_back": rolled_back,
                    "bac_gain": float(decision.get("bac_gain", 0.0)),
                    "backtracking_enabled": bool(config.enable_backtracking),
                    "selected_scale": float(decision.get("selected_scale", 1.0)),
                    "backtracking_attempts": len(decision.get("attempts", [])),
                    "trajectory_semantics": trajectory_semantics,
                    "primary_reject_reason": reasons[0] if reasons else "",
                    "all_failed_checks": ";".join(all_failed_checks),
                    "source_metrics": proposal_decision.get("before", {}),
                    "proposal_metrics": proposal_decision.get("after", {}),
                    "final_metrics": final_decision.get("after", {}),
                    "proposal_displacement": proposal_decision.get(
                        "displacement", _coordinate_update_stats(proposal - source)
                    ),
                    "accepted_displacement": final_decision.get(
                        "displacement", _coordinate_update_stats(accepted - source)
                    ),
                    "final_coordinate_equals_source": bool(torch.equal(accepted, source)),
                    "final_coordinate_equals_proposal": bool(
                        torch.equal(accepted, proposal)
                    ),
                    **diag,
                }
            )
    return coordinates, metadata


def evaluate_bac_candidate(
    model: torch.nn.Module,
    items: Sequence[dict[str, Any]],
    validity: Any,
    *,
    device: torch.device,
    inference: Mapping[str, Any],
    source_identity_sha256: str,
    bootstrap_draws: int = 500,
    trajectory_semantics: str = "legacy_bac",
    safety_objective_mode: str = "legacy_rate_sum",
) -> dict[str, Any]:
    attach_canonical_constraints(
        items, validity, source_identity_sha256=source_identity_sha256
    )
    safety_settings = dict(inference.get("safety", {}))
    safety_settings["objective_mode"] = safety_objective_mode
    accepted, metadata = infer_bac(
        model,
        items,
        validity,
        device=device,
        steps=int(inference.get("teacher_steps", 4)),
        step_size=float(inference.get("step_size", 0.25)),
        batch_size=int(inference.get("batch_size", 64)),
        safety_config=BACSafetyConfig(**safety_settings),
        trajectory_semantics=trajectory_semantics,
    )
    methods = {
        "upstream": [item["input"] for item in items],
        "v2_bac_accepted": accepted,
    }
    records = method_rows(
        items,
        methods,
        validity,
        method_metadata={"v2_bac_accepted": metadata},
    )
    summary, molecules = summarize_groups(records, items, methods)
    bootstrap = paired_bootstrap(
        molecules,
        candidate="v2_bac_accepted",
        draws=int(bootstrap_draws),
    )
    all_rows = summary[summary.group == "all"].set_index("method")
    candidate = all_rows.loc["v2_bac_accepted"]
    baseline = all_rows.loc["upstream"]
    return {
        "records": records,
        "molecules": molecules,
        "summary": summary,
        "bootstrap": bootstrap,
        "metadata": metadata,
        "metrics": {
            "bond_delta": float(
                candidate.bond_outlier_rate - baseline.bond_outlier_rate
            ),
            "angle_delta": float(
                candidate.angle_outlier_rate - baseline.angle_outlier_rate
            ),
            "clash_delta": float(
                candidate.clash_penetration - baseline.clash_penetration
            ),
            "ring_delta": float(
                candidate.ring_bond_outlier_rate
                - baseline.ring_bond_outlier_rate
            ),
            "rmsd_delta": float(candidate.aligned_RMSD - baseline.aligned_RMSD),
            "mat_p_delta": float(candidate.MAT_P - baseline.MAT_P),
            "mat_r_delta": float(candidate.MAT_R - baseline.MAT_R),
            "cov_p_delta": float(candidate.COV_P - baseline.COV_P),
            "cov_r_delta": float(candidate.COV_R - baseline.COV_R),
            "accepted_fraction": float(candidate.accepted_fraction),
            "rollback_fraction": float(candidate.rejected_fraction),
            "mean_displacement": float(candidate.molecule_rms_displacement),
            "failure_rate": 0.0,
        },
        "test_records_read": 0,
        "test_assets_opened": False,
        "validation_only": True,
    }


def summary_json(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "metrics": dict(result["metrics"]),
        "bootstrap": dict(result["bootstrap"]),
        "test_records_read": 0,
        "test_assets_opened": False,
        "validation_only": True,
    }


def write_evaluation(result: Mapping[str, Any], output: str) -> None:
    path = pd.io.common.stringify_path(output)
    result["records"].to_csv(f"{path}_per_record.csv", index=False)
    result["molecules"].to_csv(f"{path}_per_molecule.csv", index=False)
    result["summary"].to_csv(f"{path}_summary.csv", index=False)
