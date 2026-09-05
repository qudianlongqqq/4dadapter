#!/usr/bin/env python3
"""Read-only SIXS final-candidate selection audit.

Consumes only frozen DEV/evidence/audit artifacts.  It does not train, generate
coordinates, run evaluators, or access protected Formal/large-holdout outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/ecir_mvr/sixs_final_candidate_selection_audit"
INTEGRITY = ROOT / "reports/ecir_mvr/sixs_final_project_integrity_generalization_audit"
R_REPORT = ROOT / "reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307"
R_EVAL = ROOT / "artifacts/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/dev_evaluation"
U_REPORT = ROOT / "reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT"
U_EVAL = ROOT / "artifacts/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/dev_evaluation"
CAPACITY = ROOT / "reports/ecir_mvr/sixs_full_joint_final_capacity_audit"
CURRENT_EVIDENCE = ROOT / "reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/01_CURRENT_FINAL_EVIDENCE"

REPORTS = [
    "01_RESTRICTED_VS_UNRESTRICTED.md",
    "02_XTB_ROBUST_COMPARISON.md",
    "03_MOVEMENT_CONCLUSION_AUDIT.md",
    "04_REMAINING_HUMAN_CONSTANTS.csv",
    "05_CONSTANT_SENSITIVITY_PRIORITY.md",
    "06_DATA_INTEGRITY_RECONFIRMATION.md",
    "07_DEV_BIAS_INTERPRETATION.md",
    "08_CROSS_MOLECULE_GENERALIZATION.md",
    "09_GENERALIZATION_LIMIT.md",
    "10_ARCHITECTURE_COMPLETENESS.md",
    "11_DO_NOT_CONTINUE_LIST.md",
    "12_REMAINING_EXPERIMENT_PLAN.md",
    "13_MULTISEED_DECISION.md",
    "14_SCIENTIFIC_RISK_RANKING.md",
    "FINAL_CANDIDATE_AUDIT.md",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(name: str, content: str) -> None:
    path = OUT / name
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(content.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(name: str, frame: pd.DataFrame) -> None:
    path = OUT / name
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def markdown(frame: pd.DataFrame) -> str:
    cols = [str(column) for column in frame.columns]
    def value(x: Any) -> str:
        if pd.isna(x):
            return ""
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.12g}"
        return str(x).replace("|", "\\|").replace("\n", " ")
    rows = [[value(x) for x in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def distribution(values: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    return {
        "mean": float(np.mean(x)), "median": float(np.median(x)),
        "p95": float(np.quantile(x, .95)), "p99": float(np.quantile(x, .99)),
        "max": float(np.max(x)),
    }


def trimmed_mean(values: pd.Series, fraction: float) -> float:
    x = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy(float))
    trim = int(math.floor(fraction * len(x)))
    return float(np.mean(x[trim:len(x)-trim])) if trim else float(np.mean(x))


def xtb_profile(values: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "records": int(len(x)), "mean": float(x.mean()), "median": float(x.median()),
        "trimmed_mean_5pct": trimmed_mean(x, .05), "trimmed_mean_10pct": trimmed_mean(x, .10),
        "fraction_deltaE_lt_0": float((x < 0).mean()),
        "p90": float(x.quantile(.90)), "p95": float(x.quantile(.95)),
        "p99": float(x.quantile(.99)), "max": float(x.max()),
        "count_gt_25": int((x > 25).sum()), "count_gt_50": int((x > 50).sum()),
        "count_gt_100": int((x > 100).sum()),
    }


def component_summary(restricted: pd.DataFrame, unrestricted: pd.DataFrame, components: list[str]) -> pd.DataFrame:
    r = restricted.set_index("record_id")
    u = unrestricted.set_index("record_id")
    if set(r.index) != set(u.index):
        raise RuntimeError("component record alignment failed")
    rows = []
    for component in components:
        rv = r.loc[u.index, component].astype(bool)
        uv = u[component].astype(bool)
        rows.append({
            "component": component, "restricted_pass_rate": float(rv.mean()),
            "unrestricted_pass_rate": float(uv.mean()),
            "delta_unrestricted_minus_restricted": float(uv.mean() - rv.mean()),
            "unrestricted_wins": int((uv & ~rv).sum()),
            "restricted_wins": int((rv & ~uv).sum()), "ties": int((uv == rv).sum()),
        })
    return pd.DataFrame(rows)


def verify_integrity_audit() -> tuple[dict[str, Any], int, list[str]]:
    status = json.loads((INTEGRITY / "FINAL_STATUS.json").read_text(encoding="utf-8"))
    mismatches = []
    for name, expected in status["REPORT_SHA256"].items():
        path = INTEGRITY / name
        if not path.is_file() or sha256(path) != expected:
            mismatches.append(name)
    return status, len(status["REPORT_SHA256"]), mismatches


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    integrity, integrity_hash_count, integrity_mismatches = verify_integrity_audit()
    if integrity_mismatches:
        raise RuntimeError(f"integrity audit hash mismatch: {integrity_mismatches}")

    r = pd.read_parquet(R_EVAL / "PER_RECORD.parquet").sort_values("record_id").reset_index(drop=True)
    u = pd.read_parquet(U_EVAL / "PER_RECORD.parquet").sort_values("record_id").reset_index(drop=True)
    if not r.record_id.equals(u.record_id):
        raise RuntimeError("restricted/unrestricted record alignment failed")

    paired = pd.read_csv(U_REPORT / "CURRENT_VS_UNRESTRICTED_PAIRED.csv").sort_values("record_id").reset_index(drop=True)
    if not paired.record_id.equals(r.record_id):
        raise RuntimeError("Reference/xTB paired artifact alignment failed")

    rv = pd.read_parquet(R_EVAL / "VALIDITY3D.parquet")
    uv = pd.read_parquet(U_EVAL / "VALIDITY3D.parquet")
    rp = pd.read_parquet(R_EVAL / "POSEBUSTERS.parquet")
    up = pd.read_parquet(U_EVAL / "POSEBUSTERS.parquet")
    v3d_components = ["bond_geometry_valid", "angle_geometry_valid", "aromatic_ring_valid", "intramolecular_steric_clash_valid", "validity3d"]
    pb_components = ["mol_pred_loaded", "sanitization", "inchi_convertible", "all_atoms_connected", "no_radicals", "bond_lengths", "bond_angles", "internal_steric_clash", "aromatic_ring_flatness", "non-aromatic_ring_non-flatness", "double_bond_flatness", "PB"]
    v3d_table = component_summary(rv, uv, v3d_components)
    pb_table = component_summary(rp, up, pb_components)

    scalar_rows = []
    scalar_sources = {
        "Bond raw MAE": (r.bond_raw_mae, u.bond_raw_mae, "LOWER"),
        "Angle raw MAE": (r.angle_raw_mae, u.angle_raw_mae, "LOWER"),
        "internal post": (r.internal_post, u.internal_post, "LOWER"),
        "direction improvement": (r.direction_improvement, u.direction_improvement, "HIGHER"),
        "Source RMSD (native evaluation payload)": (r.source_rmsd, u.source_rmsd, "LOWER"),
        "Source RMSD (matched serialized-coordinate evidence)": (paired.current_source_rmsd, paired.source_rmsd, "LOWER"),
        "Reference RMSD": (paired.current_reference_rmsd, paired.reference_rmsd, "LOWER"),
        "tau": (r.tau, u.tau, "DESCRIPTIVE"),
    }
    for metric, (r_values, u_values, direction) in scalar_sources.items():
        rd, ud = distribution(r_values), distribution(u_values)
        scalar_rows.append({"metric": metric, "preferred_direction": direction, **{f"restricted_{k}": v for k, v in rd.items()}, **{f"unrestricted_{k}": v for k, v in ud.items()}, "delta_mean_unrestricted_minus_restricted": ud["mean"] - rd["mean"]})
    frozen_r = pd.read_csv(CURRENT_EVIDENCE / "03_MMFF_XTB_MATCHED_COMPARISON.csv")
    frozen_r = frozen_r[frozen_r.method.eq("SIXS_FULL_JOINT_STEP17500")].iloc[0]
    frozen_u = pd.read_csv(U_REPORT / "UNRESTRICTED_MOVEMENT_DISTRIBUTION.csv")
    frozen_u = frozen_u[frozen_u.object.eq("source_rmsd_angstrom")].iloc[0]
    scalar_rows.insert(4, {
        "metric": "Source RMSD (candidate frozen summaries)", "preferred_direction": "LOWER",
        "restricted_mean": frozen_r.source_rmsd_mean, "restricted_median": frozen_r.source_rmsd_median,
        "restricted_p95": frozen_r.source_rmsd_p95, "restricted_p99": frozen_r.source_rmsd_p99, "restricted_max": frozen_r.source_rmsd_max,
        "unrestricted_mean": frozen_u["mean"], "unrestricted_median": frozen_u["median"],
        "unrestricted_p95": frozen_u.p95, "unrestricted_p99": frozen_u.p99, "unrestricted_max": frozen_u["max"],
        "delta_mean_unrestricted_minus_restricted": frozen_u["mean"] - frozen_r.source_rmsd_mean,
    })
    scalar_table = pd.DataFrame(scalar_rows)

    existing_bootstrap = pd.read_csv(U_REPORT / "CURRENT_VS_UNRESTRICTED_BOOTSTRAP.csv")
    existing_bootstrap["provenance"] = "FROZEN_EXISTING_2500_MOLECULE_PAIRED_BOOTSTRAP"
    x_r = xtb_profile(paired.current_deltaE)
    x_u = xtb_profile(paired.deltaE_kcal_mol)
    xtb_table = pd.DataFrame([{"method": "RESTRICTED", **x_r}, {"method": "UNRESTRICTED", **x_u}])

    atomic_text("01_RESTRICTED_VS_UNRESTRICTED.md", f"""# Restricted versus Unrestricted complete evidence

Both candidates are step-17,500 seed307 models evaluated on the identical 2,500-molecule/5,000-record DEV cohort. All values below were recomputed from the frozen per-record artifacts; no coordinate or evaluator was rerun.

## Continuous and movement endpoints

{markdown(scalar_table)}

## V3D components

{markdown(v3d_table)}

## PoseBusters components

{markdown(pb_table)}

## Existing paired molecule-bootstrap confidence intervals

{markdown(existing_bootstrap)}

The three Source-RMSD rows retain all existing scientific representations: each candidate's frozen summary, the native evaluation payload, and the matched SDF-serialized evidence used by the frozen paired bootstrap. Their sub-microangstrom discrepancies are serialization/recomputation distinctions, not candidate reversals. The Reference-RMSD advantage of Unrestricted is tiny but its CI excludes zero. V3D is numerically +0.003 for Unrestricted but unresolved; PB is identical. Internal objective/Bond/Angle endpoints favor Unrestricted descriptively but do not establish a broad winner without an existing paired CI or preregistered cross-formulation selection rule.

```text
RESTRICTED_VS_UNRESTRICTED_SCIENTIFIC_CLASSIFICATION = PARETO_NEAR_TIE
```
""")

    atomic_text("02_XTB_ROBUST_COMPARISON.md", f"""# xTB robust comparison

Median is the primary location statistic because matched DeltaE has a positive heavy tail. Mean is retained as a secondary expectation statistic.

{markdown(xtb_table)}

Unrestricted has the more negative median and 5%/10% trimmed means. Restricted has the larger lower-energy fraction, slightly better P95/P99, the slightly more negative arithmetic mean, and a much smaller worst positive outlier (53.1 versus 570.0 kcal/mol). These bulk-versus-tail signals conflict.

```text
XTB_PRIMARY_LOCATION_STATISTIC = MEDIAN
XTB_MEAN_RETAINED_AS_SECONDARY = YES
RESTRICTED_XTB_ROBUST_PROFILE = SLIGHTLY_WEAKER_MEDIAN_BULK__BETTER_LOWER_ENERGY_FRACTION_AND_CATASTROPHIC_TAIL
UNRESTRICTED_XTB_ROBUST_PROFILE = SLIGHTLY_BETTER_MEDIAN_AND_TRIMMED_BULK__WORSE_CATASTROPHIC_POSITIVE_TAIL
XTB_PREFERENCE = MIXED
```
""")

    unrestricted_gt_001 = float((u.tau > .01).mean())
    restricted_cap_fraction = float(r.atom_cap_active.astype(bool).mean())
    unrestricted_cap_fraction = float(u.atom_cap_active.astype(bool).mean())
    atomic_text("03_MOVEMENT_CONCLUSION_AUDIT.md", f"""# Movement conclusion audit

Unrestricted jointly removes `L_move`, finite `tau_max`, the per-atom cap and rollback, and uses `softplus(raw)`. It cannot identify the causal effect of any one removed constraint.

- Unrestricted tau median: {u.tau.median():.12g} A; P95: {u.tau.quantile(.95):.12g} A; P99: {u.tau.quantile(.99):.12g} A; max: {u.tau.max():.12g} A.
- Fraction of Unrestricted records with tau > 0.01 A: {unrestricted_gt_001:.6g}.
- Restricted atom-cap active fraction: {restricted_cap_fraction:.6g}; Unrestricted: {unrestricted_cap_fraction:.6g}.
- The bulk remains small without explicit penalties. Rare tail movement exists, but there is no population-level movement explosion.
- Because V3D/PB show no clear loss and Reference RMSD slightly improves, all constraints can be removed without a clear aggregate performance loss on this DEV cohort. The worse single xTB tail prevents interpreting removal as uniformly safer or automatically superior.

```text
SMALL_MOVEMENT_EMERGES_NATURALLY = YES
MOVEMENT_EXPLOSION_OBSERVED = NO
CURRENT_001_BOUND_WAS_BINDING_FOR_BULK = NO
CURRENT_001_BOUND_BINDING_EVIDENCE = WEAK_TAIL_ONLY
UNRESTRICTED_PROVES_LMOVE_ALONE_UNNECESSARY = NO
CAN_WE_DELETE_ALL_THREE_CONSTRAINTS_WITHOUT_CLEAR_PERFORMANCE_LOSS = YES
```
""")

    constants = pd.read_csv(INTEGRITY / "03_HUMAN_CONSTANTS.csv")
    active = constants[(constants.CLASS == "HUMAN_SCIENTIFIC_DESIGN") & constants.ACTIVE_IN_UNRESTRICTED.astype(bool)].copy()
    priority_map = {
        "beta NLL exponent": ("HIGH PRIORITY SENSITIVITY", "SAFE_BUT_UNDERTESTED", "Can alter the J1 mean/sigma gradient tradeoff; J0/J1/J2 does not isolate exact beta."),
        "log sigma ratio limit": ("MEDIUM", "SAFE_AND_JUSTIFIED", "Historical exponential pathology motivates a bound; DEV upper-saturation fractions are zero, so exact 6 is not frequently binding."),
        "Reliability initialization": ("LOW", "SAFE_AND_JUSTIFIED", "Optimization start only; final Bond/Angle Reliability means 0.049/0.097 are far from 0.999 and neither head collapses."),
        "equal Bond/Angle family aggregation": ("MEDIUM", "SAFE_BUT_UNDERTESTED", "Acts in the molecule-balanced training loss; Adaptive BA acts in inference/action weighting, so there is no logical conflict."),
        "belief/post objective coefficients": ("HIGH PRIORITY SENSITIVITY", "SENSITIVITY_EVIDENCE_WEAK", "The 1:1 balance is the largest remaining unisolated scientific objective choice."),
        "initial tau": ("LOW", "SAFE_AND_JUSTIFIED", "Optimization start only; final medians 0.0054/0.0059 differ materially from 0.003."),
    }
    active["SENSITIVITY_PRIORITY"] = active.CONSTANT.map(lambda x: priority_map[x][0])
    active["CURRENT_CLASSIFICATION"] = active.CONSTANT.map(lambda x: priority_map[x][1])
    active["EVIDENCE_INTERPRETATION"] = active.CONSTANT.map(lambda x: priority_map[x][2])
    active["OUTCOME_TUNED"] = "NO"
    active["TRAINING_ROLE"] = active.PURPOSE
    active["INFERENCE_ROLE"] = np.where(active.INFERENCE_ACTIVE.astype(bool), active.PURPOSE, "NONE_DIRECT")
    active = active[["CONSTANT", "VALUE", "PATH", "SYMBOL", "TRAINING_ROLE", "INFERENCE_ROLE", "OUTCOME_TUNED", "CURRENT_CLASSIFICATION", "SENSITIVITY_PRIORITY", "EVIDENCE_INTERPRETATION"]]
    atomic_csv("04_REMAINING_HUMAN_CONSTANTS.csv", active)

    atomic_text("05_CONSTANT_SENSITIVITY_PRIORITY.md", f"""# Remaining constant sensitivity priority

{markdown(active)}

`beta=0.5` could change the learned J1 sigma conclusion because it changes the relative sigma/mean gradient pressure; existing factorial evidence selects J1 but does not isolate beta. The sigma log-ratio bound is not frequently binding in DEV (both family upper-saturation fractions are zero), so its exact value is medium rather than high priority. Reliability initialization and initial tau have both been learned far away from their starts. Equal-family aggregation is a training-loss normalization and does not conflict with Adaptive BA's action-time family allocation.

The most important remaining human objective choice is the 1:1 belief/post coefficient ratio. Nevertheless, no current constant is unsafe or supported as a blocking defect.

```text
HIGH_PRIORITY_HUMAN_CONSTANTS = beta NLL exponent; belief/post objective coefficients
SENSITIVITY_BEFORE_MULTISEED = OPTIONAL
UNSAFE_HUMAN_CONSTANT_FOUND = NO
```
""")

    data_status = "PASS" if not integrity_mismatches and integrity["DATA_LEAKAGE_FOUND"] == "NO" else "ISSUE_FOUND"
    atomic_text("06_DATA_INTEGRITY_RECONFIRMATION.md", f"""# Data integrity reconfirmation

This audit reused the completed full-project integrity audit and verified all {integrity_hash_count} recorded report hashes. It did not rescan raw coordinates because no internal contradiction was found.

```text
TRAIN_DEV_MOLECULE_OVERLAP = {integrity['TRAIN_DEV_MOLECULE_OVERLAP']}
TRAIN_DEV_CONFORMER_LEAKAGE = {integrity['TRAIN_DEV_CONFORMER_LEAKAGE']}
REFERENCE_LEAKAGE = {integrity['REFERENCE_LEAKAGE']}
ALL_INFERENCE_STATISTICS_TRAIN_ONLY = {integrity['ALL_INFERENCE_STATISTICS_TRAIN_ONLY']}
DATA_LEAKAGE_FOUND = {integrity['DATA_LEAKAGE_FOUND']}
DATA_INTEGRITY_STATUS = {data_status}
```
""")

    atomic_text("07_DEV_BIAS_INTERPRETATION.md", """# DEV bias interpretation

DEV is molecule-disjoint from TRAIN and therefore supplies valid development-stage cross-molecule evidence. It has also guided repeated formulation decisions, so it is not an unbiased final estimate and must not be relabeled as a test set.

```text
DEV_USED_FOR_MODEL_DEVELOPMENT = YES
FINAL_DEV_IS_UNBIASED_TEST_SET = NO
DEV_ROLE = DEVELOPMENT_SET
DEV_CROSS_MOLECULE_EVIDENCE_VALID = YES
FINAL_GENERALIZATION_REQUIRES_PROTECTED_DATA = YES
```
""")

    atomic_text("08_CROSS_MOLECULE_GENERALIZATION.md", f"""# Cross-molecule generalization

The completed ECFP4/scaffold/size/flexibility/Source-quality audit is reused without recomputation.

| Question | Restricted | Unrestricted |
| --- | --- | --- |
| Exact unseen molecule | SUPPORTED (development evidence) | SUPPORTED (development evidence) |
| Distant from TRAIN | degradation NOT_SUPPORTED | degradation NOT_SUPPORTED |
| TRAIN-unseen scaffold | SUPPORTED | SUPPORTED |
| High flexibility | PARTIAL | PARTIAL |
| Large molecule | PARTIAL | PARTIAL |
| Poor Source | PARTIAL | PARTIAL |

Median maximum TRAIN similarity is {integrity['DEV_MAX_TRAIN_SIMILARITY_MEDIAN']:.10g}; DEV molecules seen in TRAIN are {integrity['DEV_MOLECULES_SEEN_IN_TRAIN']}. Similarity correlations and subgroup patterns do not establish a clear formulation-specific generalization difference.

```text
CROSS_MOLECULE_SPLIT = PASS
FORMULATION_GENERALIZATION_DIFFERENCE = NOT_DETECTED
```
""")

    atomic_text("09_GENERALIZATION_LIMIT.md", """# Generalization limit

The maximum safe claim is unseen-molecule development performance within the same ETFlow Source distribution. Current Restricted/Unrestricted checkpoints have no zero-shot AvgFlow or DiTMC result; protected Formal and large-holdout outcomes remain unread.

```text
CURRENT_MODEL_CROSS_UPSTREAM = NOT_YET_TESTED
MAXIMUM_SAFE_GENERALIZATION_CLAIM = UNSEEN_MOLECULE_WITHIN_SAME_ETFLOW_SOURCE_DISTRIBUTION
UNIVERSAL_GENERALIZATION_CLAIM = NOT_SUPPORTED
```
""")

    sigma = pd.read_csv(R_REPORT / "SIGMA_DIAGNOSTICS.csv")
    reliability = pd.read_csv(R_REPORT / "RELIABILITY_DIAGNOSTICS.csv")
    ba = pd.read_csv(R_REPORT / "BA_DISTRIBUTION.csv")
    atomic_text("10_ARCHITECTURE_COMPLETENESS.md", f"""# Architecture completeness

| Module | Finding |
| --- | --- |
| mu | Functional; finite Bond/Angle prediction diagnostics and nonzero action gains. |
| sigma | Functional J1 predictive scale; zero lower/upper saturation in both families and extreme >10x inflation below {sigma.extreme_inflation_fraction_sigma_gt_10x_stat.max():.6g}. Predictive sigma alone was insufficient, but is not collapsed. |
| Reliability | Functional and noncollapsed; Bond/Angle means {reliability['mean'].iloc[0]:.4g}/{reliability['mean'].iloc[1]:.4g}, far from the 0.999 initialization. |
| Adaptive BA | Functional; weights are finite and learned around, but not exactly at, 0.5 (`w_B` range {ba['min'].iloc[0]:.4g}--{ba['max'].iloc[0]:.4g}). |
| VJP | Functional first-order action path; gradient audits pass and no second order is required. |
| rigid projection | Functional and retained in both candidates. |
| magnitude | Functional/noncollapsed; learned tau is nonzero and variable in both candidates. |
| finite movement bound | Not architecture-essential: Unrestricted removes it without clear aggregate DEV loss, although Restricted retains a better catastrophic xTB tail. |

```text
SIGMA_INFLATION_PATHOLOGY = NOT_OBSERVED
RELIABILITY_COLLAPSE = NO
BA_COLLAPSE = NO
MAGNITUDE_COLLAPSE = NO
KNOWN_ARCHITECTURAL_DEFECT = NO
ARCHITECTURE_MATURITY = MATURE_BUT_NOT_FINAL_VALIDATED
```
""")

    atomic_text("11_DO_NOT_CONTINUE_LIST.md", """# Work not supported by current evidence

- Do not extend routine training to 30k/40k: step22,500 was near plateau and failed the preregistered promotion criterion.
- Do not run a movement-cap or tau-multiplier sweep: joint removal already shows only a Pareto near-tie and bulk movement remains naturally small.
- Do not add another movement controller, Gauss-Newton, second-order training or trust-region machinery without a diagnosed architectural defect.
- Do not revive the unfinished Teacher-Student Sigma-v2 route merely because it exists; the current J1 sigma plus Reliability route is functional.
- Do not return to naive unbounded Gaussian sigma; its heavy-tail/exponential pathology is already documented.
- Do not tune against the repeatedly used DEV set or select a checkpoint/seed by protected outcomes.
""")

    plan = pd.DataFrame([
        ("P0", "before multiseed", "none required", "Current candidates are valid; targeted beta/objective sensitivity is optional, not blocking."),
        ("P0", "multiseed", "seed331/seed353 for both Restricted and Unrestricted", "Resolve formulation-by-seed stability and tail robustness."),
        ("P1", "after formulation freeze", "current-formulation cross-upstream evaluation", "ETFlow/DiTMC/AvgFlow transfer remains unknown."),
        ("P1", "after formulation freeze", "authorized Formal and large-holdout evaluation", "Required for unbiased final generalization."),
        ("P1", "after formulation freeze", "AMR/COV and matched conformer-quality reporting", "Complete downstream conformer-generation evidence."),
        ("P2", "after formulation freeze", "runtime and matched-compute profiling", "Quantify operational cost and fairness."),
        ("P2", "before final publication freeze", "targeted beta and belief/post sensitivity", "Clarify exact-number dependence if resources permit; no broad sweep."),
    ], columns=["priority", "timing", "experiment", "reason"])
    atomic_text("12_REMAINING_EXPERIMENT_PLAN.md", "# Remaining experiment plan\n\n" + markdown(plan) + "\n\nNo experiment in this table was started by this audit.\n")

    atomic_text("13_MULTISEED_DECISION.md", """# Multiseed decision

Restricted has the more conservative movement implementation and markedly better catastrophic xTB tail. Unrestricted removes three human scientific constants, learns small movement naturally, slightly improves Reference RMSD, and has a numerically higher but unresolved V3D. Their generalization patterns are not detectably different and neither is a scientific loser at seed307.

Selecting only one now would convert a seed307 Pareto near-tie into an unsupported formulation freeze. Multiseed should therefore include both formulations under matched seeds and compute, with a preregistered rule emphasizing V3D/PB, Reference RMSD and robust xTB median/tails.

```text
SENSITIVITY_BEFORE_MULTISEED = OPTIONAL
EXPERIMENTS_REQUIRED_BEFORE_MULTISEED = NONE
RECOMMENDED_MULTISEED_STRATEGY = RUN_BOTH_RESTRICTED_AND_UNRESTRICTED_WITH_MATCHED_SEEDS_AND_COMPUTE
```
""")

    risks = pd.DataFrame([
        ("HIGH", "DEV adaptation bias", "The same molecule-disjoint DEV repeatedly guided formulation choices."),
        ("HIGH", "seed instability", "Both final formulations currently have only seed307 evidence."),
        ("MEDIUM", "cross-upstream missing", "Current checkpoints are untested outside the ETFlow Source distribution."),
        ("MEDIUM", "protected evaluation missing", "Formal and large-holdout outcomes are required for unbiased final validation."),
        ("MEDIUM", "human constant sensitivity", "beta and belief/post=1:1 remain materially underisolated, but no unsafe value was found."),
        ("MEDIUM", "flexibility/size/poor-Source evidence", "These subgroup claims are only partial and development-stage."),
        ("LOW", "data leakage", "Exact molecule, conformer and Reference leakage checks pass."),
        ("LOW", "known architecture failure", "No sigma/Reliability/BA/magnitude collapse or movement explosion is observed."),
    ], columns=["rank", "risk", "evidence"])
    atomic_text("14_SCIENTIFIC_RISK_RANKING.md", "# Final scientific risk ranking\n\n" + markdown(risks) + "\n")

    final = {
        "AUDIT_STATUS": "COMPLETE_READ_ONLY",
        "DATA_INTEGRITY_STATUS": data_status,
        "KNOWN_ARCHITECTURAL_DEFECT": "NO",
        "ARCHITECTURE_MATURITY": "MATURE_BUT_NOT_FINAL_VALIDATED",
        "RESTRICTED_HUMAN_CONSTANTS": 9,
        "UNRESTRICTED_HUMAN_CONSTANTS": 6,
        "HIGH_PRIORITY_HUMAN_CONSTANTS": ["beta NLL exponent", "belief/post objective coefficients"],
        "XTB_PREFERENCE": "MIXED",
        "RESTRICTED_XTB_ROBUST_PROFILE": "SLIGHTLY_WEAKER_MEDIAN_BULK__BETTER_LOWER_ENERGY_FRACTION_AND_CATASTROPHIC_TAIL",
        "UNRESTRICTED_XTB_ROBUST_PROFILE": "SLIGHTLY_BETTER_MEDIAN_AND_TRIMMED_BULK__WORSE_CATASTROPHIC_POSITIVE_TAIL",
        "SMALL_MOVEMENT_EMERGES_NATURALLY": "YES",
        "MOVEMENT_EXPLOSION_OBSERVED": "NO",
        "CURRENT_001_BOUND_WAS_BINDING_FOR_BULK": "NO",
        "UNRESTRICTED_PROVES_LMOVE_ALONE_UNNECESSARY": "NO",
        "CAN_DELETE_MOVEMENT_CONSTRAINTS_WITHOUT_CLEAR_LOSS": "YES",
        "RESTRICTED_VS_UNRESTRICTED_SCIENTIFIC_CLASSIFICATION": "PARETO_NEAR_TIE",
        "DEV_ROLE": "DEVELOPMENT_SET",
        "DEV_CROSS_MOLECULE_EVIDENCE_VALID": "YES",
        "FINAL_GENERALIZATION_REQUIRES_PROTECTED_DATA": "YES",
        "MAXIMUM_SAFE_GENERALIZATION_CLAIM": "UNSEEN_MOLECULE_WITHIN_SAME_ETFLOW_SOURCE_DISTRIBUTION",
        "FORMULATION_GENERALIZATION_DIFFERENCE": "NOT_DETECTED",
        "SENSITIVITY_BEFORE_MULTISEED": "OPTIONAL",
        "RECOMMENDED_MULTISEED_STRATEGY": "RUN_BOTH_RESTRICTED_AND_UNRESTRICTED_WITH_MATCHED_SEEDS_AND_COMPUTE",
        "EXPERIMENTS_REQUIRED_BEFORE_MULTISEED": [],
        "EXPERIMENTS_REQUIRED_AFTER_FORMULATION_FREEZE": ["current-formulation cross-upstream evaluation", "authorized Formal evaluation", "authorized large-holdout evaluation", "AMR/COV", "runtime", "matched-compute analysis"],
        "TOP_SCIENTIFIC_RISKS": ["DEV adaptation bias", "seed instability", "cross-upstream evidence missing", "protected evaluation missing", "beta and belief/post coefficient sensitivity"],
        "NEW_TRAINING": "NO", "SEED331_STARTED": "NO", "SEED353_STARTED": "NO",
        "FORMAL_OUTCOME_READ": "NO", "LARGE_HOLDOUT_OUTCOME_READ": "NO",
    }
    final_lines = "\n".join(f"{key} = {'; '.join(value) if isinstance(value, list) else value}" for key, value in final.items())
    atomic_text("FINAL_CANDIDATE_AUDIT.md", "# SIXS final candidate selection audit\n\n" + final_lines + "\n")

    hashes = {name: sha256(OUT / name) for name in REPORTS}
    status = {
        "schema_version": "sixs-final-candidate-selection-audit-v1",
        **final,
        "SOURCE_ARTIFACTS": {
            "integrity_final_status_sha256": sha256(INTEGRITY / "FINAL_STATUS.json"),
            "restricted_per_record_sha256": sha256(R_EVAL / "PER_RECORD.parquet"),
            "unrestricted_per_record_sha256": sha256(U_EVAL / "PER_RECORD.parquet"),
            "existing_paired_bootstrap_sha256": sha256(U_REPORT / "CURRENT_VS_UNRESTRICTED_BOOTSTRAP.csv"),
            "capacity_final_status_sha256": sha256(CAPACITY / "FINAL_STATUS.json"),
        },
        "REPORT_SHA256": hashes,
    }
    temporary = OUT / f"FINAL_STATUS.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, OUT / "FINAL_STATUS.json")
    print(json.dumps({key: status[key] for key in ["AUDIT_STATUS", "DATA_INTEGRITY_STATUS", "XTB_PREFERENCE", "RESTRICTED_VS_UNRESTRICTED_SCIENTIFIC_CLASSIFICATION", "SENSITIVITY_BEFORE_MULTISEED", "RECOMMENDED_MULTISEED_STRATEGY"]}, indent=2))


if __name__ == "__main__":
    main()
