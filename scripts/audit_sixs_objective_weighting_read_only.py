#!/usr/bin/env python3
"""Read-only TRAIN-only audit of SIXS beta-NLL and belief/post weighting.

This program deliberately constructs no optimizer, never calls backward(), and
never mutates model parameters.  It uses torch.autograd.grad only to measure
the two frozen objective gradients at the scientific initialization.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\3dconformergenerationcode\4dadapter-lsgoba-musigma-reliability-factorial")
sys.path.insert(0, str(ROOT))

import scripts.run_sixs_j1r1_full_joint_adaptive_ba_movement as restricted
import scripts.run_sixs_j1r1_full_joint_unrestricted_movement as unrestricted


REPORT = ROOT / "reports/ecir_mvr/sixs_objective_read_only_audit"
BATCH_CSV = REPORT / "04_TRAIN_ONLY_BATCH_GRADIENTS.csv"
SUMMARY_JSON = REPORT / "05_ROBUST_SUMMARY.json"
FINAL_JSON = REPORT / "FINAL_STATUS.json"
BATCHES = 128
BATCH_MOLECULES = 64
PREFLIGHT_SEED = 308  # established TRAIN-only loss-scale preflight: seed307 + 1


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def parameter_norm(grads: Iterable[torch.Tensor | None], reference: torch.Tensor) -> float:
    total = reference.new_zeros((), dtype=torch.float64)
    for grad in grads:
        if grad is not None:
            total = total + grad.detach().double().square().sum()
    return float(total.sqrt())


def common_metrics(
    belief_grads: Sequence[torch.Tensor | None],
    post_grads: Sequence[torch.Tensor | None],
    reference: torch.Tensor,
) -> tuple[float, float, float]:
    belief_sq = reference.new_zeros((), dtype=torch.float64)
    post_sq = reference.new_zeros((), dtype=torch.float64)
    dot = reference.new_zeros((), dtype=torch.float64)
    for belief_grad, post_grad in zip(belief_grads, post_grads, strict=True):
        if belief_grad is not None:
            belief_sq = belief_sq + belief_grad.detach().double().square().sum()
        if post_grad is not None:
            post_sq = post_sq + post_grad.detach().double().square().sum()
        if belief_grad is not None and post_grad is not None:
            dot = dot + (belief_grad.detach().double() * post_grad.detach().double()).sum()
    belief_norm = belief_sq.sqrt()
    post_norm = post_sq.sqrt()
    denominator = belief_norm * post_norm
    cosine = dot / denominator if float(denominator) > 0 else reference.new_tensor(float("nan"))
    return float(belief_norm), float(post_norm), float(cosine)


def summarize(values: pd.Series, quantiles: Sequence[float]) -> dict[str, float]:
    array = values.to_numpy(dtype=np.float64)
    result = {"mean": float(np.mean(array)), "median": float(np.median(array))}
    for quantile in quantiles:
        result[f"p{int(round(100 * quantile)):02d}"] = float(np.quantile(array, quantile))
    return result


def classify_balance(alpha: float) -> str:
    if 0.90 <= alpha <= 1.10:
        return "YES"
    if (2.0 / 3.0) <= alpha <= 1.50:
        return "APPROXIMATELY"
    return "NO"


def classify_relation(cosine: Mapping[str, float]) -> str:
    median = cosine["median"]
    negative = cosine["fraction_lt_0"]
    strong_negative = cosine["fraction_lt_minus_0_25"]
    if median >= 0.50 and negative < 0.10:
        return "ALIGNED"
    if median > 0.0 and negative < 0.25:
        return "MOSTLY_ALIGNED"
    if median < 0.0 and (negative > 0.50 or strong_negative > 0.35):
        return "CONFLICTING"
    return "MIXED"


def route_label(norm: float) -> str:
    return "RECEIVES_GRADIENT" if norm > 0.0 and math.isfinite(norm) else "NO_GRADIENT"


def run_branch(
    branch: str,
    build_model: Callable[[torch.device], torch.nn.Module],
    batch_losses: Callable[..., tuple[Any, Any, Mapping[str, torch.Tensor], Any, Any]],
    prepared: Mapping[str, Any],
    sources: Mapping[str, list[dict[str, Any]]],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]], list[int], dict[str, int]]:
    # Match the established preflight ordering: seed the explicit sampler, then
    # build the scientific initialization (which independently resets global RNG).
    generator = restricted.seed_all(PREFLIGHT_SEED)
    model = build_model(device)
    model.train()
    groups = model.parameter_groups()
    all_parameters = [parameter for values in groups.values() for parameter in values]
    all_ids = [id(parameter) for parameter in all_parameters]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError(f"{branch}: parameter groups overlap")
    group_counts = {name: sum(parameter.numel() for parameter in values) for name, values in groups.items()}
    common = groups["backbone"]
    common_count = len(common)
    if all_parameters[:common_count] != common:
        raise RuntimeError(f"{branch}: backbone is not the leading flattened group")

    rows: list[dict[str, Any]] = []
    sampled: list[int] = []
    route_accumulator = {
        loss: {name: [] for name in groups}
        for loss in ("belief", "post")
    }
    for batch_index in range(BATCHES):
        graphs, batch_graph, source, reference, chosen = restricted.frozen.sample_batch(
            prepared["train"], sources, generator, BATCH_MOLECULES, device
        )
        sampled.extend(chosen)
        model.zero_grad(set_to_none=True)
        _, _, losses, _, _ = batch_losses(model, graphs, batch_graph, source, reference)
        belief = losses["belief"]
        post = losses["post"]
        if not bool(torch.isfinite(belief)) or not bool(torch.isfinite(post)):
            raise RuntimeError(f"{branch} batch {batch_index + 1}: nonfinite loss")

        belief_grads = torch.autograd.grad(
            belief, all_parameters, retain_graph=True, allow_unused=True, create_graph=False
        )
        post_grads = torch.autograd.grad(
            post, all_parameters, retain_graph=False, allow_unused=True, create_graph=False
        )
        belief_common = belief_grads[:common_count]
        post_common = post_grads[:common_count]
        belief_norm, post_norm, cosine = common_metrics(belief_common, post_common, belief)
        if not all(math.isfinite(value) for value in (belief_norm, post_norm, cosine)):
            raise RuntimeError(f"{branch} batch {batch_index + 1}: nonfinite common gradient metric")

        offset = 0
        for group_name, parameters in groups.items():
            end = offset + len(parameters)
            route_accumulator["belief"][group_name].append(
                parameter_norm(belief_grads[offset:end], belief)
            )
            route_accumulator["post"][group_name].append(
                parameter_norm(post_grads[offset:end], post)
            )
            offset = end

        rows.append(
            {
                "branch": branch,
                "batch": batch_index + 1,
                "batch_molecules": BATCH_MOLECULES,
                "belief_loss": float(belief.detach()),
                "post_loss": float(post.detach()),
                "common_belief_grad_norm": belief_norm,
                "common_post_grad_norm": post_norm,
                "common_gradient_cosine": cosine,
            }
        )
        del belief_grads, post_grads, losses, belief, post, batch_graph, source, reference
        if (batch_index + 1) % 16 == 0:
            print(json.dumps({"branch": branch, "completed_batches": batch_index + 1}), flush=True)

    route_norms = {
        loss: {group: float(np.median(values)) for group, values in by_group.items()}
        for loss, by_group in route_accumulator.items()
    }
    # autograd.grad does not populate .grad; this is an explicit no-update guard.
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError(f"{branch}: unexpected populated parameter.grad")
    del model
    torch.cuda.empty_cache()
    return rows, route_norms, sampled, group_counts


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this read-only preflight execution")
    device = torch.device("cuda:0")

    restricted_config = restricted.cfg()
    unrestricted_config = unrestricted.cfg()
    if restricted_config["model"]["beta_nll_beta"] != unrestricted_config["model"]["beta_nll_beta"]:
        raise RuntimeError("Restricted and Unrestricted beta differ")
    if restricted_config["model"]["beta_nll_beta"] != 0.5:
        raise RuntimeError("unexpected current beta")

    # This loader verifies and materializes the frozen TRAIN/VAL container used
    # by prior TRAIN-only preflights. Only prepared['train'] and source['train']
    # are indexed below; no DEV manifest or DEV outcome artifact is opened.
    prepared, source_payload = restricted.frozen.load_inputs()
    sources = restricted.frozen.source_index(source_payload["train"])

    all_rows: list[dict[str, Any]] = []
    branch_data: dict[str, Any] = {}
    branch_specs = (
        ("RESTRICTED", restricted.build_model, restricted.batch_losses),
        ("UNRESTRICTED", unrestricted.build_model, unrestricted.batch_losses),
    )
    for branch, builder, losses in branch_specs:
        rows, routes, sampled, counts = run_branch(
            branch, builder, losses, prepared, sources, device
        )
        all_rows.extend(rows)
        branch_data[branch] = {
            "routes": routes,
            "sampled_molecule_index_sha256": canonical_sha(sampled),
            "parameter_counts": counts,
        }

    frame = pd.DataFrame(all_rows)
    atomic_csv(BATCH_CSV, frame)
    summaries: dict[str, Any] = {}
    relations: dict[str, str] = {}
    for branch in ("RESTRICTED", "UNRESTRICTED"):
        local = frame[frame["branch"] == branch]
        belief_summary = summarize(local["belief_loss"], (0.25, 0.75, 0.95))
        post_summary = summarize(local["post_loss"], (0.25, 0.75, 0.95))
        belief_grad_summary = summarize(local["common_belief_grad_norm"], (0.25, 0.75, 0.95))
        post_grad_summary = summarize(local["common_post_grad_norm"], (0.25, 0.75, 0.95))
        cosine_summary = summarize(local["common_gradient_cosine"], (0.10, 0.25, 0.75, 0.90))
        cosine_summary.update(
            {
                "fraction_lt_0": float((local["common_gradient_cosine"] < 0.0).mean()),
                "fraction_lt_minus_0_25": float((local["common_gradient_cosine"] < -0.25).mean()),
                "fraction_gt_0": float((local["common_gradient_cosine"] > 0.0).mean()),
            }
        )
        alpha_grad = belief_grad_summary["median"] / post_grad_summary["median"]
        alpha_loss = belief_summary["median"] / post_summary["median"]
        relation = classify_relation(cosine_summary)
        relations[branch] = relation
        summaries[branch] = {
            "belief_loss": belief_summary,
            "post_loss": post_summary,
            "belief_common_gradient_norm": belief_grad_summary,
            "post_common_gradient_norm": post_grad_summary,
            "common_gradient_cosine": cosine_summary,
            "alpha_grad": alpha_grad,
            "alpha_loss": alpha_loss,
            "current_1_to_1_gradient_balanced": classify_balance(alpha_grad),
            "current_1_to_1_loss_balanced": classify_balance(alpha_loss),
            "belief_post_relation": relation,
            **branch_data[branch],
        }

    alpha_relative_difference = abs(
        summaries["RESTRICTED"]["alpha_grad"] - summaries["UNRESTRICTED"]["alpha_grad"]
    ) / max(abs(summaries["RESTRICTED"]["alpha_grad"]), 1.0e-300)
    alpha_branch_difference = "SIGNIFICANT" if alpha_relative_difference > 0.05 else "NOT_SIGNIFICANT"
    any_conflicting = "CONFLICTING" in relations.values()
    any_unbalanced = any(
        summaries[branch]["current_1_to_1_gradient_balanced"] == "NO"
        for branch in summaries
    )
    experiment_needed = "YES" if any_conflicting or any_unbalanced else (
        "OPTIONAL" if "MIXED" in relations.values() else "NO"
    )
    overall_relation = relations["RESTRICTED"] if len(set(relations.values())) == 1 else (
        f"RESTRICTED_{relations['RESTRICTED']}__UNRESTRICTED_{relations['UNRESTRICTED']}"
    )

    summary = {
        "schema_version": "sixs-objective-read-only-audit-v1",
        "audit_status": "COMPLETE_READ_ONLY",
        "sampling_protocol": {
            "split_indexed": "TRAIN_ONLY",
            "batches_per_branch": BATCHES,
            "batch_molecules": BATCH_MOLECULES,
            "molecule_draws_per_branch": BATCHES * BATCH_MOLECULES,
            "seed": PREFLIGHT_SEED,
            "matches_existing_train_only_loss_scale_preflight": True,
            "expected_existing_sample_sha256": "9133d40f40da53c2abea0e1211f1bf7b0b0efde848b48b6224c4957f7748834d",
        },
        "branches": summaries,
        "alpha_grad_relative_branch_difference": alpha_relative_difference,
        "alpha_grad_branch_difference": alpha_branch_difference,
        "overall_belief_post_relation": overall_relation,
        "scalar_weighting_may_be_insufficient": "YES" if any_conflicting else "NO",
        "objective_weight_experiment_needed": experiment_needed,
        "recommended_objective_weight_test": "A_CURRENT_1_TO_1_VS_B_TRAIN_DERIVED_ALPHA_GRAD" if experiment_needed != "NO" else "NONE",
        "guards": {
            "optimizer_constructed": False,
            "optimizer_step_called": False,
            "backward_called": False,
            "create_graph": False,
            "parameters_updated": False,
            "checkpoint_created": False,
            "new_training": False,
            "dev_read": False,
            "formal_outcome_read": False,
            "large_holdout_outcome_read": False,
            "seed331_started": False,
            "seed353_started": False,
        },
    }
    atomic_json(SUMMARY_JSON, summary)

    routes = summaries["RESTRICTED"]["routes"]
    route_rows = []
    for group in ("backbone", "mu", "j1_sigma", "reliability", "adaptive_ba", "magnitude"):
        belief_norm = routes["belief"][group]
        post_norm = routes["post"][group]
        if belief_norm > 0 and post_norm > 0:
            category = "BOTH"
        elif belief_norm > 0:
            category = "BELIEF_ONLY"
        elif post_norm > 0:
            category = "POST_ONLY"
        else:
            category = "NEITHER"
        route_rows.append(
            {
                "parameter_group": group,
                "belief_route": route_label(belief_norm),
                "post_route": route_label(post_norm),
                "category": category,
                "belief_gradient_norm_median": belief_norm,
                "post_gradient_norm_median": post_norm,
            }
        )
    atomic_csv(REPORT / "03_PARAMETER_GRADIENT_ROUTING.csv", pd.DataFrame(route_rows))

    atomic_text(
        REPORT / "01_EXACT_OBJECTIVE.md",
        """# Exact current objective (read-only audit)

## Restricted

`L_total = 1.0 * L_belief + 1.0 * L_post + 0.40793421960700144 * L_move`.

## Unrestricted

`L_total = 1.0 * L_belief + 1.0 * L_post`.

For each molecule `m`, `L_belief` is the equal-family average of the J1
beta-NLL primitive values: one mean over its Bond primitives, one mean over its
Angle primitives, followed by `0.5 * (Bond + Angle)` and then a mean over
molecules. `L_post` applies the same molecule/equal-family aggregation to
squared Reference errors of the proposal geometry standardized by detached
predictive sigma.

Code locations: `configs/sixs_j1r1_full_joint_adaptive_ba_movement.json:25-27`,
`configs/sixs_j1r1_full_joint_unrestricted_movement.json:27-29`,
`scripts/run_sixs_j1r1_full_joint_adaptive_ba_movement.py:272-288`,
`scripts/run_sixs_j1r1_full_joint_unrestricted_movement.py:166-178`, and
`etflow/ecir/musigma_reliability.py:239-285,449-466`.
""",
    )
    atomic_text(
        REPORT / "02_BETA_NLL_AND_ROUTING_AUDIT.md",
        """# beta-NLL and gradient routing audit

The current exponent is `beta=0.5`. For primitive target `y`, mean `mu`, and
scale `sigma`, the implemented value is

`[0.5 * ((y-mu)/sigma)^2 + log(sigma)] * stopgrad(sigma^2)^beta`.

The Gaussian additive constant is omitted; because the beta factor is detached,
this omission does not alter the implemented parameter gradients. The detached
factor is exactly `sigma.square().detach().pow(beta)`. The NLL's sigma remains
differentiable. Thus the mu derivative scales as
`(mu-y) * sigma^(2*beta-2)` (at beta 0.5, `(mu-y)/sigma`), while the sigma
derivative is the NLL sigma derivative times the detached factor. Since current
J1 calls `detach_sigma_features=False`, the sigma route reaches both sigma heads
and the shared backbone. This is faithful to the core Seitzer et al. beta-NLL
stop-gradient reweighting idea.

The exact beta was historically fixed as a human scientific design constant.
The J0/J1/J2 factorial supported J1 as a method, but did not isolate or select
the value 0.5. Therefore `BETA_ORIGIN=HISTORICAL_DESIGN`, not TRAIN-derived or
DEV-selected.

Belief routes to backbone, mu heads, and J1 sigma heads. Post routes to
backbone (only through non-detached primitive features entering Reliability),
Reliability, Adaptive BA, and Magnitude. Post sees detached mu, sigma,
source-state magnitude inputs, graph embedding for Adaptive BA/Magnitude, and
detached Cartesian primitive derivatives. Consequently, mu and sigma heads get
no post gradient, and Reliability/Adaptive-BA/Magnitude get no belief gradient.
The exact common set is the shared backbone. Restricted `L_move` is separate
and routes only to Magnitude.

Code locations: `etflow/ecir/musigma_reliability.py:66-145,148-227,256-298,
301-340,449-466`; `etflow/ecir/j1r1_full_joint.py:44-71,89-208`; and
`etflow/ecir/j1r1_full_joint_unrestricted.py:54-168`.
""",
    )

    restricted_sample_match = (
        summaries["RESTRICTED"]["sampled_molecule_index_sha256"]
        == summary["sampling_protocol"]["expected_existing_sample_sha256"]
    )
    unrestricted_sample_match = (
        summaries["UNRESTRICTED"]["sampled_molecule_index_sha256"]
        == summary["sampling_protocol"]["expected_existing_sample_sha256"]
    )
    if not restricted_sample_match or not unrestricted_sample_match:
        raise RuntimeError("TRAIN-only sampling identity does not match established preflight")

    artifacts = [
        REPORT / "01_EXACT_OBJECTIVE.md",
        REPORT / "02_BETA_NLL_AND_ROUTING_AUDIT.md",
        REPORT / "03_PARAMETER_GRADIENT_ROUTING.csv",
        BATCH_CSV,
        SUMMARY_JSON,
    ]
    final = {
        "schema_version": "sixs-objective-read-only-audit-final-v1",
        "AUDIT_STATUS": "COMPLETE_READ_ONLY",
        "BETA_NLL_BETA": 0.5,
        "BETA_ORIGIN": "HISTORICAL_DESIGN",
        "LAMBDA_BELIEF": 1.0,
        "LAMBDA_POST": 1.0,
        "COMMON_PARAMETER_SET": "BACKBONE",
        "BRANCHES": {
            branch: {
                "BELIEF_GRAD_NORM_MEDIAN": summaries[branch]["belief_common_gradient_norm"]["median"],
                "POST_GRAD_NORM_MEDIAN": summaries[branch]["post_common_gradient_norm"]["median"],
                "GRADIENT_COSINE_MEDIAN": summaries[branch]["common_gradient_cosine"]["median"],
                "GRADIENT_CONFLICT_FRACTION": summaries[branch]["common_gradient_cosine"]["fraction_lt_0"],
                "ALPHA_GRAD_TRAIN_DERIVED": summaries[branch]["alpha_grad"],
                "ALPHA_LOSS_TRAIN_DERIVED": summaries[branch]["alpha_loss"],
                "CURRENT_1_TO_1_GRADIENT_BALANCED": summaries[branch]["current_1_to_1_gradient_balanced"],
                "CURRENT_1_TO_1_LOSS_BALANCED": summaries[branch]["current_1_to_1_loss_balanced"],
                "BELIEF_POST_RELATION": summaries[branch]["belief_post_relation"],
            }
            for branch in ("RESTRICTED", "UNRESTRICTED")
        },
        "ALPHA_GRAD_BRANCH_DIFFERENCE": alpha_branch_difference,
        "BELIEF_POST_RELATION": overall_relation,
        "OBJECTIVE_WEIGHT_EXPERIMENT_NEEDED": experiment_needed,
        "RECOMMENDED_OBJECTIVE_WEIGHT_TEST": summary["recommended_objective_weight_test"],
        "NEW_TRAINING": "NO",
        "DEV_READ": "NO",
        "FORMAL_OUTCOME_READ": "NO",
        "LARGE_HOLDOUT_OUTCOME_READ": "NO",
        "ARTIFACT_SHA256": {path.name: sha256(path) for path in artifacts},
    }
    atomic_json(FINAL_JSON, final)
    print(json.dumps(final, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
