#!/usr/bin/env python3
"""Read-only module-level forensic of SIXS belief/post gradients.

No optimizer is constructed. No backward() or optimizer.step() is called.
Only torch.autograd.grad(create_graph=False) is used on frozen TRAIN batches.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\3dconformergenerationcode\4dadapter-lsgoba-musigma-reliability-factorial")
sys.path.insert(0, str(ROOT))

import scripts.run_sixs_j1r1_full_joint_adaptive_ba_movement as restricted
import scripts.run_sixs_j1r1_full_joint_unrestricted_movement as unrestricted


REPORT = ROOT / "reports/ecir_mvr/sixs_objective_module_gradient_forensic_audit"
BATCHES = 128
BATCH_MOLECULES = 64
PREFLIGHT_SEED = 308
NUMERICAL_TOLERANCE = 1.0e-12
EXPECTED_SAMPLE_SHA256 = "9133d40f40da53c2abea0e1211f1bf7b0b0efde848b48b6224c4957f7748834d"
BACKBONE_LR = 1.5e-4
HEAD_LR = 3.0e-4


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def module_parameters(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    result = {
        "BACKBONE": model.belief.backbone_parameters(),
        "BOND_MU": list(model.belief.geometry.bond_head.parameters()),
        "ANGLE_MU": list(model.belief.geometry.angle_head.parameters()),
        "J1_SIGMA": model.belief.sigma_head_parameters(),
        "RELIABILITY": list(model.reliability.parameters()),
        "ADAPTIVE_BA": list(model.adaptive_ba.parameters()),
        "MAGNITUDE": list(model.magnitude.parameters()),
    }
    ids = [id(parameter) for values in result.values() for parameter in values]
    model_ids = [id(parameter) for parameter in model.parameters()]
    if len(ids) != len(set(ids)):
        raise RuntimeError("module parameter sets overlap")
    if set(ids) != set(model_ids):
        raise RuntimeError("module parameter sets do not cover the model exactly")
    if not all(parameter.requires_grad for values in result.values() for parameter in values):
        raise RuntimeError("a trainable module parameter has requires_grad=False")
    return result


def gradient_metrics(
    gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
    learning_rate: float,
) -> dict[str, float | int]:
    squared = 0.0
    nonzero = 0
    above = 0
    count = 0
    tensors_with_gradient = 0
    for gradient, parameter in zip(gradients, parameters, strict=True):
        count += parameter.numel()
        if gradient is None:
            continue
        tensors_with_gradient += 1
        value = gradient.detach()
        squared += float(value.double().square().sum())
        nonzero += int((value != 0).sum())
        above += int((value.abs() > NUMERICAL_TOLERANCE).sum())
    norm = math.sqrt(squared)
    rms = norm / math.sqrt(max(count, 1))
    return {
        "parameter_count": count,
        "parameter_tensors_with_gradient": tensors_with_gradient,
        "gradient_l2": norm,
        "gradient_rms_per_parameter": rms,
        "nonzero_parameter_count_exact": nonzero,
        "above_tolerance_parameter_count": above,
        "above_tolerance_fraction": above / max(count, 1),
        "effective_first_order_update_scale": learning_rate * norm,
    }


def quantile_summary(series: pd.Series) -> dict[str, float]:
    values = series.to_numpy(dtype=np.float64)
    return {
        "median": float(np.median(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
    }


def run_branch(
    branch: str,
    builder: Callable[[torch.device], torch.nn.Module],
    loss_function: Callable[..., tuple[Any, Any, Mapping[str, torch.Tensor], Any, Any]],
    prepared: Mapping[str, Any],
    sources: Mapping[str, list[dict[str, Any]]],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Exact sampling protocol from the preceding audit and the established
    # TRAIN-only loss-scale preflight.
    generator = restricted.seed_all(PREFLIGHT_SEED)
    model = builder(device)
    model.train()
    modules = module_parameters(model)
    flat = [parameter for values in modules.values() for parameter in values]
    offsets: dict[str, tuple[int, int]] = {}
    offset = 0
    for name, parameters in modules.items():
        offsets[name] = (offset, offset + len(parameters))
        offset += len(parameters)
    initial_parameter_sha256 = parameter_sha256(model)
    sampled: list[int] = []
    rows: list[dict[str, Any]] = []

    for batch_index in range(BATCHES):
        graphs, batch_graph, source, reference, chosen = restricted.frozen.sample_batch(
            prepared["train"], sources, generator, BATCH_MOLECULES, device
        )
        sampled.extend(chosen)
        model.zero_grad(set_to_none=True)
        _, _, losses, _, _ = loss_function(model, graphs, batch_graph, source, reference)
        belief = losses["belief"]
        post = losses["post"]
        if not bool(torch.isfinite(belief)) or not bool(torch.isfinite(post)):
            raise RuntimeError(f"{branch} batch {batch_index + 1}: nonfinite loss")
        belief_gradients = torch.autograd.grad(
            belief, flat, retain_graph=True, create_graph=False, allow_unused=True
        )
        post_gradients = torch.autograd.grad(
            post, flat, retain_graph=False, create_graph=False, allow_unused=True
        )
        for loss_name, gradients in (
            ("BELIEF", belief_gradients),
            ("POST", post_gradients),
        ):
            for module_name, parameters in modules.items():
                start, end = offsets[module_name]
                learning_rate = BACKBONE_LR if module_name == "BACKBONE" else HEAD_LR
                metrics = gradient_metrics(gradients[start:end], parameters, learning_rate)
                rows.append(
                    {
                        "branch": branch,
                        "batch": batch_index + 1,
                        "batch_molecules": BATCH_MOLECULES,
                        "loss": loss_name,
                        "module": module_name,
                        "loss_value": float((belief if loss_name == "BELIEF" else post).detach()),
                        "learning_rate": learning_rate,
                        "numerical_tolerance": NUMERICAL_TOLERANCE,
                        **metrics,
                    }
                )
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError(f"{branch}: autograd.grad unexpectedly populated .grad")
        del belief_gradients, post_gradients, losses, belief, post, batch_graph, source, reference
        if (batch_index + 1) % 16 == 0:
            print(json.dumps({"branch": branch, "completed_batches": batch_index + 1}), flush=True)

    final_parameter_sha256 = parameter_sha256(model)
    if initial_parameter_sha256 != final_parameter_sha256:
        raise RuntimeError(f"{branch}: model parameters changed during read-only audit")
    sample_sha256 = canonical_sha(sampled)
    if sample_sha256 != EXPECTED_SAMPLE_SHA256:
        raise RuntimeError(f"{branch}: sampled TRAIN molecule identity changed")
    metadata = {
        "sampled_molecule_index_sha256": sample_sha256,
        "initial_parameter_sha256": initial_parameter_sha256,
        "final_parameter_sha256": final_parameter_sha256,
        "parameters_unchanged": True,
        "all_parameters_requires_grad": True,
    }
    del model
    torch.cuda.empty_cache()
    return rows, metadata


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for read-only gradient audit")
    device = torch.device("cuda:0")
    prepared, source_payload = restricted.frozen.load_inputs()
    sources = restricted.frozen.source_index(source_payload["train"])

    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for branch, builder, losses in (
        ("RESTRICTED", restricted.build_model, restricted.batch_losses),
        ("UNRESTRICTED", unrestricted.build_model, unrestricted.batch_losses),
    ):
        local_rows, local_metadata = run_branch(
            branch, builder, losses, prepared, sources, device
        )
        rows.extend(local_rows)
        metadata[branch] = local_metadata

    frame = pd.DataFrame(rows)
    batch_path = REPORT / "05_PER_BATCH_MODULE_GRADIENTS.csv"
    atomic_csv(batch_path, frame)

    summary_rows: list[dict[str, Any]] = []
    for (branch, module_name, loss_name), local in frame.groupby(
        ["branch", "module", "loss"], sort=False
    ):
        row: dict[str, Any] = {
            "branch": branch,
            "module": module_name,
            "loss": loss_name,
            "parameter_count": int(local["parameter_count"].iloc[0]),
            "learning_rate": float(local["learning_rate"].iloc[0]),
            "numerical_tolerance": NUMERICAL_TOLERANCE,
        }
        for field in (
            "gradient_l2",
            "gradient_rms_per_parameter",
            "nonzero_parameter_count_exact",
            "above_tolerance_parameter_count",
            "above_tolerance_fraction",
            "effective_first_order_update_scale",
        ):
            for statistic, value in quantile_summary(local[field]).items():
                row[f"{field}_{statistic}"] = value
        summary_rows.append(row)
    summary_frame = pd.DataFrame(summary_rows)
    summary_path = REPORT / "03_MODULE_GRADIENT_SUMMARY.csv"
    atomic_csv(summary_path, summary_frame)

    pressure_columns = [
        "branch", "module", "loss", "parameter_count", "learning_rate",
        "gradient_l2_median", "gradient_l2_p25", "gradient_l2_p75", "gradient_l2_p95",
        "effective_first_order_update_scale_median",
        "effective_first_order_update_scale_p25",
        "effective_first_order_update_scale_p75",
        "effective_first_order_update_scale_p95",
    ]
    pressure_path = REPORT / "04_LR_WEIGHTED_UPDATE_PRESSURE.csv"
    atomic_csv(pressure_path, summary_frame[pressure_columns])

    route_table = pd.DataFrame(
        [
            ("BACKBONE", "INDIRECT", "INDIRECT_MICROSCOPIC_AT_INITIALIZATION", "YES", "YES", "YES"),
            ("BOND_MU", "DIRECT", "NONE", "YES", "NO", "YES"),
            ("ANGLE_MU", "DIRECT", "NONE", "YES", "NO", "YES"),
            ("J1_SIGMA", "DIRECT", "NONE", "YES", "NO", "YES"),
            ("RELIABILITY", "NONE", "DIRECT", "NO", "YES", "YES"),
            ("ADAPTIVE_BA", "NONE", "DIRECT", "NO", "YES", "YES"),
            ("MAGNITUDE", "NONE", "DIRECT", "NO", "YES", "YES"),
        ],
        columns=[
            "MODULE", "BELIEF_PATH", "POST_PATH", "UPDATED_BY_BELIEF",
            "UPDATED_BY_POST", "UPDATED_BY_TOTAL_LOSS",
        ],
    )
    route_path = REPORT / "02_AUTOGRAD_MODULE_ROUTES.csv"
    atomic_csv(route_path, route_table)

    # Effective modules use observed median gradient norm above the absolute
    # tolerance and at least one scalar gradient above that tolerance.
    classifications: dict[str, Any] = {}
    for branch in ("RESTRICTED", "UNRESTRICTED"):
        branch_frame = summary_frame[summary_frame["branch"] == branch]
        effective: dict[str, list[str]] = {"BELIEF": [], "POST": []}
        for loss_name in ("BELIEF", "POST"):
            for _, row in branch_frame[branch_frame["loss"] == loss_name].iterrows():
                if (
                    float(row["gradient_l2_median"]) > NUMERICAL_TOLERANCE
                    and float(row["above_tolerance_fraction_median"]) > 0.0
                ):
                    effective[loss_name].append(str(row["module"]))
        shared = sorted(set(effective["BELIEF"]) & set(effective["POST"]))
        classifications[branch] = {
            "belief_primary_modules": effective["BELIEF"],
            "post_primary_modules": effective["POST"],
            "shared_effective_modules": shared,
        }

    if classifications["RESTRICTED"] != classifications["UNRESTRICTED"]:
        raise RuntimeError("Restricted and Unrestricted effective module routing differs")
    expected = {
        "belief_primary_modules": ["BACKBONE", "BOND_MU", "ANGLE_MU", "J1_SIGMA"],
        "post_primary_modules": ["RELIABILITY", "ADAPTIVE_BA", "MAGNITUDE"],
        "shared_effective_modules": [],
    }
    if classifications["RESTRICTED"] != expected:
        raise RuntimeError(
            f"observed effective module classification changed: {classifications['RESTRICTED']}"
        )

    autograd_report = """# Exact autograd graph and detach forensic

## Module routes

- `BACKBONE`: belief is indirect through the Bond/Angle mu and J1 sigma
  predictions. Post has a theoretical indirect route only through the
  non-detached primitive features passed to Reliability.
- `BOND_MU`, `ANGLE_MU`, `J1_SIGMA`: direct belief path; no post path because
  action-time mu/sigma and post standardizers are detached.
- `RELIABILITY`, `ADAPTIVE_BA`, `MAGNITUDE`: direct post supervision through
  proposal coordinates; no belief path.

## Exact graph operations

| operation | current location | effect |
|---|---|---|
| beta stop-gradient | `musigma_reliability.py:278-283` | only `sigma^2` beta weight detached; NLL sigma remains differentiable |
| J1 sigma feature route | restricted runner `:274`; unrestricted `:168` | `detach_sigma_features=False`, so belief sigma reaches backbone |
| source q, mu, sigma detach | `j1r1_full_joint.py:110-130`; unrestricted `:82-99` | prevents post from updating mu/sigma heads through action coefficients |
| primitive Cartesian derivative detach | `musigma_reliability.py:301-340` | first-order action; no coordinate Hessian |
| rigid projection coordinates detach | `j1r1_full_joint.py:156`; unrestricted `:121` | geometry frame does not create a source-coordinate derivative route |
| graph embedding detach for Adaptive BA | `j1r1_full_joint.py:89-95` | BA is post-trained, but BA cannot send post gradient to backbone |
| source-state detach | `j1r1_full_joint.py:187`; unrestricted `:143` | magnitude state cannot send post gradient upstream |
| magnitude graph embedding detach | `j1r1_full_joint.py:188`; unrestricted `:144` | magnitude cannot send post gradient to backbone |
| action-loss sigma detach | `musigma_reliability.py:458-465` | post standardization does not update sigma |
| clone | none in belief/post objective path | no effect |
| no_grad | unrestricted initialization copy `j1r1_full_joint_unrestricted.py:65-68` only | no forward/autograd disconnection |
| requires_grad | dynamically checked for every trainable parameter | all true |
| create_graph | false in this audit; ordinary training backward also does not request a higher-order graph | no second-order graph |

## Backbone post-gradient cause

Reliability receives `bond_features` and `angle_features` without detach, so the
backbone is not accidentally disconnected. However, the shared Reliability
output layer is initialized with weight standard deviation `1e-10`
(`musigma_reliability.py:168-174`), and its sigmoid is initialized near 0.999.
The post-to-feature Jacobian therefore contains both the microscopic final
weight and a small sigmoid derivative. All alternative backbone routes are
intentionally detached by Adaptive-BA and Magnitude boundaries. The observed
post-backbone norm near `8e-15` is thus expected at initialization, not a
cancellation, no-grad block, or implementation bug.
"""
    graph_path = REPORT / "01_EXACT_AUTOGRAD_FORENSIC.md"
    atomic_text(graph_path, autograd_report)

    invalidation = """# Previous coefficient invalidation

`PREVIOUS_ALPHA_GRAD approximately 2.255e12` is not valid for training. Its
denominator is the numerically near-zero post gradient on a backbone that is
only theoretically shared at the scientific initialization. Multiplying the
whole post loss by this number would also multiply the non-shared Reliability,
Adaptive-BA, and Magnitude gradients and is not a valid common-gradient
equalization.

`ALPHA_LOSS approximately -0.1069` is also not a valid positive loss weight.
The beta/Gaussian NLL can be negative and has an arbitrary additive zero,
whereas post is a nonnegative squared geometry objective.

Therefore `LOSS_MAGNITUDE_BALANCING_VALID=NO` and the previous alpha-based
training recommendation is superseded without running any experiment.
"""
    invalidation_path = REPORT / "00_PREVIOUS_ALPHA_INVALIDATION.md"
    atomic_text(invalidation_path, invalidation)

    result = {
        "schema_version": "sixs-objective-module-gradient-forensic-v1",
        "AUDIT_STATUS": "COMPLETE_READ_ONLY",
        "PREVIOUS_ALPHA_GRAD": "APPROXIMATELY_2.255E12",
        "PREVIOUS_ALPHA_GRAD_VALID_FOR_TRAINING": "NO",
        "LOSS_MAGNITUDE_BALANCING_VALID": "NO",
        "NUMERICAL_TOLERANCE": NUMERICAL_TOLERANCE,
        "POST_BACKBONE_NEAR_ZERO_CAUSE": "EXPECTED_BY_DESIGN",
        "POST_BACKBONE_CAUSE_DETAIL": "MICROSCOPIC_RELIABILITY_FINAL_WEIGHT_INITIALIZATION_PLUS_INTENTIONAL_BA_AND_MAGNITUDE_DETACH_BOUNDARIES",
        "BELIEF_PRIMARY_MODULES": expected["belief_primary_modules"],
        "POST_PRIMARY_MODULES": expected["post_primary_modules"],
        "SHARED_EFFECTIVE_MODULES": expected["shared_effective_modules"],
        "BELIEF_POST_WEIGHT_COUPLING": "MOSTLY_DISJOINT",
        "FULL_JOINT_MEANING": "JOINT_OPTIMIZATION_WITH_SPECIALIZED_LOSS_PATHS",
        "GRADIENT_BALANCING_APPROACH_VALID": "NO",
        "OBJECTIVE_RATIO_SENSITIVITY_PRIORITY": "MEDIUM",
        "OBJECTIVE_RATIO_PRIORITY_REASON": "LOSS_COEFFICIENTS_PRIMARILY_SCALE_SPECIALIZED_MODULE_UPDATES_RATHER_THAN_BALANCE_A_SHARED_EFFECTIVE_GRADIENT",
        "IMPLEMENTATION_BUG_FOUND": "NO",
        "BETA_NLL_BETA": 0.5,
        "BETA_NLL_PRIMARY_MODULES": ["BACKBONE", "BOND_MU", "ANGLE_MU", "J1_SIGMA"],
        "BETA_NLL_EFFECTIVE_COMPETITION_WITH_POST": "NO_AT_SCIENTIFIC_INITIALIZATION",
        "BRANCH_METADATA": metadata,
        "SAMPLING": {
            "split_indexed": "TRAIN_ONLY",
            "batches_per_branch": BATCHES,
            "batch_molecules": BATCH_MOLECULES,
            "molecule_draws_per_branch": BATCHES * BATCH_MOLECULES,
            "seed": PREFLIGHT_SEED,
            "sample_sha256": EXPECTED_SAMPLE_SHA256,
        },
        "GUARDS": {
            "optimizer_constructed": False,
            "optimizer_step_called": False,
            "backward_called": False,
            "create_graph": False,
            "parameters_updated": False,
            "new_training": False,
            "dev_read": False,
            "formal_outcome_read": False,
            "large_holdout_outcome_read": False,
            "seed331_started": False,
            "seed353_started": False,
        },
    }
    report_path = REPORT / "06_FORENSIC_CONCLUSION.json"
    atomic_json(report_path, result)

    artifact_paths = [
        invalidation_path,
        graph_path,
        route_path,
        summary_path,
        pressure_path,
        batch_path,
        report_path,
    ]
    final = {
        **result,
        "ARTIFACT_SHA256": {path.name: file_sha256(path) for path in artifact_paths},
    }
    final_path = REPORT / "FINAL_STATUS.json"
    atomic_json(final_path, final)
    print(json.dumps(final, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
