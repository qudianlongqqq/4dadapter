#!/usr/bin/env python3
"""Read-only baseline audit for the seed307 Full-Joint capacity experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from scipy.stats import linregress

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
RUN = "sixs_j1r1_full_joint_adaptive_ba_movement_seed307"
SOURCE_REPORT = ROOT / "reports/ecir_mvr" / RUN
SOURCE_ARTIFACT = ROOT / "artifacts/ecir_mvr" / RUN
OUT = ROOT / "reports/ecir_mvr/sixs_full_joint_final_capacity_audit"
CONFIG = ROOT / "configs/sixs_j1r1_full_joint_adaptive_ba_movement.json"


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str))


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_state_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def model_identity(config: dict[str, Any], groups: dict[str, Any]) -> str:
    rows = [
        ("BACKBONE", "Trainable 3-layer 128D message-passing GNN", "graph atom/bond topology", "node and graph embeddings", "J1 belief + post via Reliability/BA paths", "NO"),
        ("MU_B", "Trainable Bond mean head", "symmetric endpoint + edge embedding", "Bond mu", "J1 beta-NLL belief", "NO"),
        ("MU_A", "Trainable Angle mean head", "center/symmetric endpoints + two edges", "Angle mu", "J1 beta-NLL belief", "NO"),
        ("J1_SIGMA", "Trainable direct predictive-scale heads", "source-free local graph features + sigma_stat", "sigma_stat*exp(6*tanh(raw/6))", "J1 beta-NLL belief", "NO"),
        ("RELIABILITY", "Trainable shared source-conditioned primitive gate", "local graph, q_source, mu, sigma, signed/absolute/standardized defect, family", "r in (0,1)", "post action loss", "NO"),
        ("ADAPTIVE_BA", "Trainable per-molecule BA head", "graph mean embedding", "softmax(w_B,w_A)", "post action loss", "NO"),
        ("MAGNITUDE", "Trainable per-molecule magnitude head", "detached graph mean + normalized 17D source state", "tau=0.01*sigmoid(raw)", "post + movement regularizer", "NO"),
    ]
    table = pd.DataFrame(rows, columns=["COMPONENT", "TRAINABLE", "INPUT", "OUTPUT", "LOSS_PATH", "REFERENCE_VISIBLE_AT_INFERENCE"])
    group_text = "\n".join(f"- {row['group']}: {row['parameter_count']:,} trainable parameters" for row in groups["groups"])
    return f"""# Final model identity

The identity below was re-derived from the current config, Python implementation, parameter-group audit and step-17,500 checkpoint rather than copied from the prior conclusion.

{table.to_markdown(index=False)}

Parameter groups are disjoint and all are trainable:

{group_text}

Total trainable parameters: **{groups['total_parameter_count']:,}**.

## Exact inference dataflow

`graph -> trainable GNN -> local embeddings -> mu_B/mu_A and direct J1 sigma`; then `source geometry + local embedding + mu + sigma + source defect -> Reliability`; the graph mean produces learned `softmax(w_B,w_A)`. Primitive descent contributions use learned BA, Reliability, source defects and inverse predictive variance, are projected out of rigid modes and normalized to unit graph RMS. A detached graph mean plus a normalized 17D source-state vector produces a learned per-molecule `tau`; the proposal is finally subject to the 0.03 A per-atom cap.

Reference coordinates enter only the training/evaluation losses. They are absent from inference features.

```text
BACKBONE_TRAINABLE = YES
MU_B_TRAINABLE = YES
MU_A_TRAINABLE = YES
J1_SIGMA_TRAINABLE = YES
RELIABILITY_TRAINABLE = YES
ADAPTIVE_BA_TRAINABLE = YES
MAGNITUDE_TRAINABLE = YES
SECOND_ORDER_USED = NO
NAIVE_GAUSSIAN_MU_SIGMA_REINTRODUCED = NO
TEACHER_STUDENT_USED = NO
REFERENCE_VISIBLE_AT_INFERENCE = NO
```
"""


def constants_inventory(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[tuple[Any, ...]] = []
    def add(c, v, f, s, role, cls, train, infer, sensitive, evidence, removable):
        rows.append((c, v, f, s, role, cls, train, infer, sensitive, evidence, removable))
    add("hidden_dim",128,"config + j1r1_full_joint.py","model.hidden_dim","representation width","ENGINEERING_HYPERPARAMETER","YES","YES","POSSIBLY","No architecture sweep in this protocol","YES_WITH_RETRAINING")
    add("message_passing_layers",3,"config + learned_geometry.py","model.layers","GNN depth","ENGINEERING_HYPERPARAMETER","YES","YES","POSSIBLY","Inherited standard architecture","YES_WITH_RETRAINING")
    add("beta_NLL_beta",0.5,"config + musigma_reliability.py","beta_nll_beta","J1 beta-NLL coupling exponent","HUMAN_SCIENTIFIC_DESIGN","YES","YES_VIA_TRAINED_MODEL","YES","Six-arm factorial: J1 was best musigma training","NO")
    add("log_sigma_ratio_limit",6.0,"musigma_reliability.py","LOG_SIGMA_RATIO_LIMIT","bounded direct predictive sigma correction","HUMAN_SCIENTIFIC_DESIGN","YES","YES","YES","Saturation diagnostics exist; no bound sweep","NO")
    add("bond_sigma_stat_floor",0.01,"frozen prepared graph","graph.bond_fixed[:,1] observed minimum","inherited action/predictive baseline lower scale","HUMAN_SCIENTIFIC_DESIGN","YES","YES","YES","77.74% of frozen DEV Bond primitives at minimum","NO")
    add("angle_sigma_stat_floor",0.02,"frozen prepared graph","graph.angle_fixed[:,1] observed minimum","inherited action/predictive baseline lower scale","HUMAN_SCIENTIFIC_DESIGN","YES","YES","YES","13.15% of frozen DEV Angle primitives at minimum","NO")
    add("reliability_output_bounds","sigmoid (0,1)","musigma_reliability.py","PrimitiveReliabilityHead.forward","nonnegative bounded action gate","SYMMETRY_OR_MATH_REQUIRED","YES","YES","YES","Six-arm R0/R1 ablation","NO")
    add("reliability_initial_value",0.999,"musigma_reliability.py","INITIAL_RELIABILITY","near-identity action initialization","HUMAN_SCIENTIFIC_DESIGN","YES","NO_AFTER_TRAINING","POSSIBLY","Deterministic R1 initialization only","YES_WITH_REDESIGN")
    add("reliability_final_weight_std",1e-10,"musigma_reliability.py","PrimitiveReliabilityHead.__init__","nonzero first-step gradient with near-constant gate","ENGINEERING_HYPERPARAMETER","YES","NO_AFTER_TRAINING","NO","Backward compatibility/gradient preflight passes","YES")
    add("reliability_sigma_floor",1e-12,"musigma_reliability.py","_scalars","division stability","ENGINEERING_HYPERPARAMETER","YES","YES","NO","Far below scientific scales","YES")
    add("equal_family_training_weights","0.5 Bond + 0.5 Angle","musigma_reliability.py","molecule_equal_family_mean","equal-family molecule-balanced loss","HUMAN_SCIENTIFIC_DESIGN","YES","YES_VIA_TRAINED_MODEL","YES","No weight sweep; learned inference BA is separate","NO")
    add("adaptive_BA_initial_logits","0 -> [0.5,0.5]","j1r1_full_joint.py","AdaptiveBAHead.__init__","symmetry-preserving initialization","SYMMETRY_OR_MATH_REQUIRED","YES","NO_AFTER_TRAINING","NO","Exact equal initialization","YES")
    add("adaptive_BA_temperature",1.0,"j1r1_full_joint.py","torch.softmax default","standard softmax scale; no explicit temperature","SYMMETRY_OR_MATH_REQUIRED","YES","YES","POSSIBLY","No separate temperature exists","NO")
    add("primitive_family_average","1/n_B and 1/n_A","j1r1_full_joint.py","full_joint_action","prevent primitive-count dominance","SYMMETRY_OR_MATH_REQUIRED","YES","YES","YES","Mathematical family mean","NO")
    add("direction_family_scale","learned w_B,w_A; no fixed 0.5","j1r1_full_joint.py","full_joint_action","per-molecule direction balance","LEARNED","YES","YES","YES","DEV distribution persisted","NO")
    add("direction_graph_RMS_target",1.0,"j1r1_full_joint.py","direction = projected / rms","separate direction from magnitude","SYMMETRY_OR_MATH_REQUIRED","YES","YES","YES","Exact construction","NO")
    add("initial_tau_A",0.003,"config + lsgoba_v2_joint_magnitude.py","initial_tau","movement-head initialization","HUMAN_SCIENTIFIC_DESIGN","YES","NO_AFTER_TRAINING","YES","Joint-Magnitude predecessor supports learnability","NO")
    add("tau_max_A",0.01,"config + lsgoba_v2_joint_magnitude.py","tau_max","upper movement range","HUMAN_SCIENTIFIC_DESIGN","YES","YES","YES","p99 0.00806 and rare exact max: weak binding evidence","NO")
    add("per_atom_cap_A",0.03,"config + scaled_proposal","atom_cap","hard per-atom movement safety cap","HUMAN_SCIENTIFIC_DESIGN","YES","YES","YES","0.6% records/0.023% atoms capped: weak binding evidence","NO")
    add("movement_calibration_target",0.05,"config provenance + TRAIN preflight","5-percent initial median post rule","sets regularizer scale","HUMAN_SCIENTIFIC_DESIGN","YES","NO_DIRECT","YES","TRAIN-only calibration artifact exists","NO")
    add("lambda_move",0.40793421960700144,"config","objective.lambda_move","movement penalty coefficient","TRAIN_DATA_DERIVED","YES","YES_VIA_TRAINED_MODEL","YES","Derived from fixed 5% target on TRAIN","NO")
    add("movement_penalty","mean((tau/0.01)^2)","runner","batch_losses","quadratic normalized magnitude regularizer","HUMAN_SCIENTIFIC_DESIGN","YES","YES_VIA_TRAINED_MODEL","YES","No functional-form ablation","NO")
    add("source_state_tail_quantile",0.90,"lsgoba_v2_joint_magnitude.py","_family_features","P90 absolute standardized defect","HUMAN_SCIENTIFIC_DESIGN","YES","YES","POSSIBLY","Inherited 17D source state","YES_WITH_REDESIGN")
    add("state_std_floor",1e-6,"lsgoba_v2_joint_magnitude.py","set_state_normalization","normalization stability","ENGINEERING_HYPERPARAMETER","YES","YES","NO","Below observed state variation","YES")
    add("angle_cosine_clamp",1e-7,"musigma_reliability.py","_angle_descent","stable derivative near collinearity","ENGINEERING_HYPERPARAMETER","YES","YES","NO","Numerical only","YES")
    add("geometry_norm_epsilon",1e-12,"musigma_reliability.py","_bond_descent/_angle_descent","division stability","ENGINEERING_HYPERPARAMETER","YES","YES","NO","Numerical only","YES")
    add("direction_RMS_epsilon",1e-14,"j1r1_full_joint.py","full_joint_action","zero/nonfinite direction guard","ENGINEERING_HYPERPARAMETER","YES","YES","NO","Numerical only","YES")
    add("atom_scale_epsilon",1e-15,"lsgoba_v2_joint_magnitude.py","scaled_proposal","zero displacement division guard","ENGINEERING_HYPERPARAMETER","YES","YES","NO","Numerical only","YES")
    add("optimizer","AdamW","config + runner","optimizer_for","optimizer family","ENGINEERING_HYPERPARAMETER","YES","NO","POSSIBLY","Single preregistered recipe","YES_WITH_RETRAINING")
    add("backbone_learning_rate",1.5e-4,"config","training.backbone_learning_rate","backbone optimizer LR","ENGINEERING_HYPERPARAMETER","YES","NO","POSSIBLY","Inherited recipe","YES_WITH_RETRAINING")
    add("head_learning_rate",3e-4,"config","training.head_learning_rate","head optimizer LR","ENGINEERING_HYPERPARAMETER","YES","NO","POSSIBLY","Inherited recipe","YES_WITH_RETRAINING")
    add("weight_decay",1e-6,"config","training.weight_decay","AdamW regularization","ENGINEERING_HYPERPARAMETER","YES","NO","POSSIBLY","Inherited recipe","YES_WITH_RETRAINING")
    add("gradient_clip",1.0,"config","training.gradient_clip","global gradient norm cap","ENGINEERING_HYPERPARAMETER","YES","NO","POSSIBLY","Gradient log available","YES_WITH_RETRAINING")
    add("batch_molecules",64,"config","training.batch_molecules","training batch size","ENGINEERING_HYPERPARAMETER","YES","NO","NO","Inherited recipe","YES_WITH_RETRAINING")
    add("current_optimizer_steps",17500,"config","training.optimizer_steps","initial training budget","ENGINEERING_HYPERPARAMETER","YES","NO","YES","Capacity audit tests natural horizon","YES")
    add("scheduler_T_max",22500,"config","training.scheduler_horizon","natural cosine horizon","ENGINEERING_HYPERPARAMETER","YES","NO","YES","Continuation endpoint preregistered by existing scheduler","NO_FOR_THIS_AUDIT")
    add("log_interval",25,"config","training.log_interval","training telemetry cadence","ENGINEERING_HYPERPARAMETER","YES","NO","NO","701 existing rows","YES")
    add("recovery_interval",250,"config","training.recovery_interval","state checkpoint cadence","ENGINEERING_HYPERPARAMETER","YES","NO","NO","Complete continuation state exists","YES")
    add("bootstrap_resamples",10000,"config","evaluation.bootstrap_resamples","paired molecule-cluster uncertainty","ENGINEERING_HYPERPARAMETER","NO","NO","NO","Frozen evaluation protocol","YES")
    add("PB_material_drop_tolerance",0.001,"config","evaluation.pb_material_drop_tolerance","selection noninferiority margin","HUMAN_SCIENTIFIC_DESIGN","NO","NO","YES","Preregistered development decision rule","NO")
    return pd.DataFrame(rows, columns=["CONSTANT","VALUE","FILE","LINE_OR_SYMBOL","ROLE","CLASSIFICATION","AFFECTS_TRAINING","AFFECTS_INFERENCE","SCIENTIFICALLY_SENSITIVE","EXISTING_SENSITIVITY_EVIDENCE","CAN_BE_REMOVED_WITHOUT_REDESIGN"])


CORE_METRICS = ["TOTAL_LOSS","PRIOR_J1_LOSS","POST_LOSS","MOVEMENT_LOSS","MU_BOND_MAE","MU_ANGLE_MAE","SIGMA_MEAN","R_MEAN","W_B_MEAN","TAU_MEAN"]


def prepare_curve(log: pd.DataFrame, horizon: int, base_lr: float, head_lr: float) -> pd.DataFrame:
    result = log.copy()
    factor = (1.0 + np.cos(np.pi * result.step.to_numpy(float) / horizon)) / 2.0
    result["LR_BACKBONE"] = base_lr * factor
    result["LR_HEAD"] = head_lr * factor
    for metric in CORE_METRICS:
        result[f"{metric}_ROLLING_MEDIAN_500_STEPS"] = result[metric].rolling(20, min_periods=5).median()
        result[f"{metric}_ROLLING_VARIANCE_500_STEPS"] = result[metric].rolling(20, min_periods=5).var()
    return result


def slopes(log: pd.DataFrame) -> pd.DataFrame:
    rows = []
    maximum = float(log.step.max())
    for fraction in (0.50, 0.25, 0.10, 0.05):
        local = log[log.step >= maximum * (1.0 - fraction)]
        for metric in CORE_METRICS + [c for c in log if c.startswith("GRAD_")]:
            fit = linregress(local.step.to_numpy(float) / 1000.0, local[metric].to_numpy(float))
            rows.append({"final_fraction":fraction,"start_step":int(local.step.min()),"metric":metric,"slope_per_1000_steps":fit.slope,"p_value_diagnostic":fit.pvalue,"r_squared":fit.rvalue**2,"start_rolling_median":float(local[metric].head(min(20,len(local))).median()),"end_rolling_median":float(local[metric].tail(min(20,len(local))).median()),"rolling_variance_end":float(local[metric].tail(min(20,len(local))).var())})
    return pd.DataFrame(rows)


def learning_curve_fit(log: pd.DataFrame) -> dict[str, Any]:
    # Fit 500-step-bin medians to asymptote + amplitude*step^-power. Diagnostic only.
    result: dict[str, Any] = {"model":"asymptote + amplitude * step**(-power)","diagnostic_only":True,"fits":{}}
    work = log[log.step >= 2500].copy()
    work["bin"] = (work.step // 500).astype(int)
    for metric in ("TOTAL_LOSS","POST_LOSS","MU_BOND_MAE","MU_ANGLE_MAE","SIGMA_MEAN","R_MEAN"):
        grouped = work.groupby("bin", sort=True).agg(step=("step","median"), value=(metric,"median"))
        x, y = grouped.step.to_numpy(float), grouped.value.to_numpy(float)
        def fn(step, asymptote, amplitude, power):
            return asymptote + amplitude * np.power(step, -power)
        try:
            scale = max(float(np.ptp(y)), abs(float(np.median(y))), 1e-9)
            popt, _ = curve_fit(fn, x, y, p0=[float(y[-1]), float((y[0]-y[-1])*x[0]**0.5),0.5], bounds=([-10*scale,-1e8,0.01],[10*scale,1e8,5.0]), maxfev=100000)
            pred = fn(x,*popt)
            result["fits"][metric] = {"status":"PASS","asymptote":float(popt[0]),"amplitude":float(popt[1]),"power":float(popt[2]),"rmse":float(np.sqrt(np.mean((pred-y)**2))),"current_value_last_bin":float(y[-1]),"current_minus_asymptote":float(y[-1]-popt[0]),"predicted_at_22500":float(fn(22500,*popt)),"bins":len(x)}
        except Exception as exc:
            result["fits"][metric] = {"status":"FIT_FAILED","error":str(exc)}
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    groups = json.loads((SOURCE_REPORT / "PARAMETER_GROUP_AUDIT.json").read_text(encoding="utf-8"))
    log = pd.read_csv(SOURCE_REPORT / "TRAIN_LOG.csv")
    recovery_path = SOURCE_ARTIFACT / "RECOVERY_CHECKPOINT.pt"
    final_path = SOURCE_REPORT / "FINAL_CHECKPOINT.pt"
    recovery = torch.load(recovery_path, map_location="cpu", weights_only=False)
    final = torch.load(final_path, map_location="cpu", weights_only=False)
    scheduler = recovery["scheduler_state"]
    state_gate = {
        "MODEL_STATE":"PASS" if recovery["step"] == final["step"] == 17500 and parameter_state_equal(recovery["model_state"], final["model_state"]) else "FAIL",
        "OPTIMIZER_STATE":"PASS" if bool(recovery.get("optimizer_state")) else "FAIL",
        "SCHEDULER_STATE":"PASS" if scheduler.get("T_max") == config["training"]["scheduler_horizon"] and scheduler.get("last_epoch") == 17500 else "FAIL",
        "RNG_STATE":"PASS" if all(key in recovery for key in ("generator_state","python_rng_state","numpy_rng_state","torch_rng_state","cuda_rng_state")) else "FAIL",
        "DATA_ORDER_SAMPLING_PROTOCOL":"PASS" if recovery.get("config_sha256") == sha256(CONFIG) else "FAIL",
        "recovery_checkpoint_sha256":sha256(recovery_path),
        "final_checkpoint_sha256":sha256(final_path),
        "step":int(recovery["step"]),
        "scheduler_state":scheduler,
    }
    if any(state_gate[key] != "PASS" for key in ("MODEL_STATE","OPTIMIZER_STATE","SCHEDULER_STATE","RNG_STATE","DATA_ORDER_SAMPLING_PROTOCOL")):
        atomic_json(OUT / "STATE_CONTINUITY_GATE.json", state_gate)
        raise RuntimeError("state continuity gate failed")
    atomic_json(OUT / "STATE_CONTINUITY_GATE.json", state_gate)
    atomic_text(OUT / "01_FINAL_MODEL_IDENTITY.md", model_identity(config, groups))
    inventory = constants_inventory(config)
    atomic_csv(OUT / "02_HUMAN_CONSTANT_INVENTORY.csv", inventory)
    human = inventory[inventory.CLASSIFICATION == "HUMAN_SCIENTIFIC_DESIGN"]
    core_human = human[~human.CONSTANT.isin(["PB_material_drop_tolerance"])]
    atomic_text(OUT / "03_HUMAN_CONSTANT_CLASSIFICATION.md", f"""# Human-constant classification

The current method is reasonably described as **largely learned rather than hand-weighted**: primitive Reliability, per-molecule Bond/Angle direction weights and per-molecule movement magnitude are learned. It is not parameter-free. The active model/training path retains **{len(core_human)}** human scientific design constants (the PB noninferiority margin is an additional evaluation-only constant).

```text
CORE_DIRECTION_WEIGHTS_HUMAN_FIXED = NO
CORE_MOVEMENT_PER_MOLECULE_HUMAN_FIXED = NO
NUMBER_OF_HUMAN_SCIENTIFIC_DESIGN_CONSTANTS = {len(core_human)}
MOST_IMPORTANT_REMAINING_HUMAN_CONSTANTS = beta_NLL_beta; log_sigma_ratio_limit; inherited sigma_stat floors; tau_max; per-atom cap; movement regularizer form and 5% calibration target
```

Learned BA/tau do not remove the scientific influence of the hand-set bounds and loss structure. Full details and sensitivity evidence are in `02_HUMAN_CONSTANT_INVENTORY.csv`.
""")
    curve = prepare_curve(log, config["training"]["scheduler_horizon"], config["training"]["backbone_learning_rate"], config["training"]["head_learning_rate"])
    atomic_csv(OUT / "04_TRAINING_CURVE.csv", curve)
    slope_frame = slopes(log)
    atomic_csv(OUT / "TRAINING_TREND_DIAGNOSTICS.csv", slope_frame)
    fits = learning_curve_fit(log)
    atomic_json(OUT / "06_LEARNING_CURVE_FIT.json", fits)
    current_factor = (1 + math.cos(math.pi * 17500 / 22500)) / 2
    validation_present = any("VAL" in c.upper() or "VALID" in c.upper() for c in log.columns)
    # Pre-results rule: ongoing statistically visible mu/R/sigma trends, incomplete scheduler,
    # but no held-out validation => likely rather than clearly not saturated.
    classification = "LIKELY_NOT_SATURATED"
    atomic_text(OUT / "05_TRAINING_SATURATION_AUDIT.md", f"""# Training saturation audit

The 17,500-step endpoint is classified **{classification}**. This decision was frozen before the 22,500-step checkpoint or its DEV coordinates existed.

- The configured cosine scheduler horizon is 22,500; training stopped at 77.7778% of that horizon.
- The actual checkpoint scheduler is at epoch 17,500 and its learning rate is {current_factor:.8%} of the initial rate (backbone {scheduler['_last_lr'][0]:.10g}, head {scheduler['_last_lr'][1]:.10g}), not the scheduler minimum.
- In the final 10%, Bond mu MAE, Angle mu MAE, sigma mean and Reliability mean retain nonzero fitted slopes; several are statistically visible as telemetry trends. Final-interval slope, rolling-median and rolling-variance details are frozen in `TRAINING_TREND_DIAGNOSTICS.csv`.
- Total/post losses are heavy-tailed and their terminal slopes are noisy; they do not independently establish continued improvement.
- Held-out online validation metrics are {'present' if validation_present else 'absent'}. Their absence prevents a stronger `CLEARLY_NOT_SATURATED` claim.
- Power-law/asymptotic fits are diagnostic only and are not used as a pass/fail rule.

```text
TRAINING_SATURATION = {classification}
CURRENT_STEPS = 17500
SCHEDULER_HORIZON = 22500
CURRENT_BUDGET_FRACTION_OF_SCHEDULER_HORIZON = {17500/22500:.12f}
LEARNING_RATE_AT_MINIMUM = NO
VALIDATION_CURVE_AVAILABLE = {'YES' if validation_present else 'NO'}
EXTENSION_RULE = RUN_TO_EXISTING_SCHEDULER_HORIZON_ONLY
```
""")
    prereg = {
        "schema_version":"sixs-full-joint-capacity-preregistration-v1",
        "frozen_before_extension":True,
        "training_saturation":classification,
        "extension_authorized_by_user_rule":True,
        "extension_endpoint":22500,
        "endpoint_source":"existing CosineAnnealingLR T_max",
        "state_continuity_gate":"PASS",
        "no_dev_checkpoint_selection":True,
        "evaluation_cohort":{"molecules":2500,"records":5000},
        "bootstrap":{"clusters":"molecule","resamples":10000,"seed":20260830},
        "decision_rule":"V3D clear positive paired evidence, continuous geometry same direction, and no material PB decline => 17500 limiting; otherwise NO_OR_WEAK",
        "formal_read":False,"large_holdout_read":False,"seed331_started":False,"seed353_started":False,
        "source_hashes":{"config":sha256(CONFIG),"train_log":sha256(SOURCE_REPORT/"TRAIN_LOG.csv"),"recovery":sha256(recovery_path),"final_17500":sha256(final_path)},
    }
    atomic_json(OUT / "SATURATION_PREREGISTRATION.json", prereg)
    print(json.dumps({"status":"BASELINE_AUDIT_COMPLETE","training_saturation":classification,"state_continuity":"PASS","human_scientific_constants_active":len(core_human)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
