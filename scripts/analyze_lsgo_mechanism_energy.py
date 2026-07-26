#!/usr/bin/env python3
"""Analyze frozen xTB energies without changing coordinates or decisions."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
from etflow.ecir.lsgo_io import atomic_json, file_sha256

OUT = ROOT / "reports/ecir_mvr/lsgo_mechanism"
CONFIG = yaml.safe_load((ROOT / "configs/ecir_mvr_lsgo_mechanism.yaml").read_text(encoding="utf-8"))
KCAL = 627.509474


def atomic_text(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8"); os.replace(temporary, path)


def atomic_frame(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False); os.replace(temporary, path)


def ci_cluster(frame, column, statistic, *, replicates=5000, seed=47021):
    groups = [group[column].to_numpy(float) for _, group in frame.groupby("molecule_id", sort=True)]
    rng = np.random.default_rng(seed); values = np.empty(replicates)
    for index in range(replicates):
        chosen = rng.integers(0, len(groups), len(groups)); sample = np.concatenate([groups[value] for value in chosen])
        values[index] = statistic(sample)
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def correlation_ci(frame, x, y, kind, *, replicates=5000, seed=47021):
    ordered = frame.reset_index(drop=True)
    groups = [group.index.to_numpy() for _, group in ordered.groupby("molecule_id", sort=True)]
    x_values, y_values = ordered[x].to_numpy(float), ordered[y].to_numpy(float)
    rng = np.random.default_rng(seed); values = []
    function = spearmanr if kind == "spearman" else pearsonr
    for _ in range(replicates):
        indices = np.concatenate([groups[value] for value in rng.integers(0, len(groups), len(groups))])
        value = function(x_values[indices], y_values[indices]).statistic
        if np.isfinite(value): values.append(value)
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def energy_summary(frame, condition, diagnostics):
    values = frame.delta_energy_kcal_mol.to_numpy(float); movement = diagnostics[diagnostics.condition == condition]
    positive = values[values > 0]
    return {"condition": condition, "seed": frame.seed.dropna().iloc[0] if frame.seed.notna().any() else None, "records": len(values),
        "mean_delta_energy_kcal_mol": float(values.mean()), "median_delta_energy_kcal_mol": float(np.median(values)), "improved_fraction": float((values < 0).mean()),
        "p90": float(np.quantile(values, .90)), "p95": float(np.quantile(values, .95)), "p99": float(np.quantile(values, .99)), "max_harmful": float(values.max()), "positive_tail_mean": float(positive.mean()) if positive.size else 0.,
        "mean_movement_rms_angstrom": float(movement.movement_rms.mean()), "p95_movement_rms_angstrom": float(movement.movement_rms.quantile(.95)), "accepted_fraction": float(movement.accepted.mean()),
        "topology_guard_fraction": float(movement.catastrophic_clash_nonregression.mean()), "chirality_fraction": float(movement.chirality_preserved.mean()), "mode_switch_fraction": float(movement.mode_switch.mean())}


def main() -> int:
    manifest = json.loads((OUT / "manifests/XTB_SINGLE_POINT_COMPLETE.json").read_text(encoding="utf-8"))
    if manifest["status"] != "COMPLETED" or manifest["formal_test_records_read"] or manifest["frozen_holdout_records_read"]: raise RuntimeError("xTB manifest invalid")
    xtb = pd.read_parquet(manifest["result_path"]); diagnostics = pd.read_parquet(OUT / "per_record/COORDINATE_DIAGNOSTICS.parquet")
    references = xtb[xtb.condition == "Reference"].copy(); sources = xtb[xtb.condition == "Source"].copy()
    source_energy = sources.set_index("sample_id").energy_hartree
    output = xtb[~xtb.condition.isin(["Source", "Reference"])].copy()
    output["source_energy_hartree"] = output.source_sample_id.map(source_energy); output["delta_energy_kcal_mol"] = (output.energy_hartree - output.source_energy_hartree) * KCAL
    ref_groups = {key: group.sort_values("reference_index") for key, group in references.groupby("molecule_id")}
    nearest = diagnostics[diagnostics.condition == f"BA_seed{CONFIG['representative_ba_seed']}"][["sample_id", "source_nearest_reference_mode"]].drop_duplicates().set_index("sample_id").source_nearest_reference_mode
    ensemble_rows = []
    for row in sources.itertuples(index=False):
        ref = ref_groups[row.molecule_id]; energies = ref.energy_hartree.to_numpy(float); nearest_index = int(nearest.loc[row.sample_id])
        ensemble_rows.append({"molecule_id": row.molecule_id, "sample_id": row.sample_id, "source_index": row.source_index, "flex_bin": row.flex_bin, "source_energy_hartree": row.energy_hartree,
            "reference_min_energy_hartree": float(energies.min()), "reference_median_energy_hartree": float(np.median(energies)), "reference_nearest_energy_hartree": float(ref.loc[ref.reference_index == nearest_index, "energy_hartree"].iloc[0]), "reference_nearest_index": nearest_index,
            "source_minus_reference_min_kcal_mol": float((row.energy_hartree - energies.min()) * KCAL), "source_minus_reference_median_kcal_mol": float((row.energy_hartree - np.median(energies)) * KCAL), "source_minus_reference_nearest_kcal_mol": float((row.energy_hartree - ref.loc[ref.reference_index == nearest_index, "energy_hartree"].iloc[0]) * KCAL)})
    ensemble = pd.DataFrame(ensemble_rows)
    rep_ba = output[output.condition == f"BA_seed{CONFIG['representative_ba_seed']}"][["source_sample_id", "energy_hartree", "delta_energy_kcal_mol"]].rename(columns={"source_sample_id": "sample_id", "energy_hartree": "ba_energy_hartree", "delta_energy_kcal_mol": "ba_delta_energy_kcal_mol"})
    ensemble = ensemble.merge(rep_ba, on="sample_id", validate="one_to_one")
    atomic_frame(OUT / "per_record/SOURCE_REFERENCE_ENERGY.parquet", ensemble)
    tie = float(CONFIG["thresholds"]["energy_tie_kcal_mol"])
    fraction_rows = []
    for label, column in (("minimum", "source_minus_reference_min_kcal_mol"), ("median", "source_minus_reference_median_kcal_mol"), ("aligned_rmsd_nearest", "source_minus_reference_nearest_kcal_mol")):
        for relation, fn in (("source_lower", lambda x: x < -tie), ("approximately_tied", lambda x: np.abs(x) <= tie), ("source_higher", lambda x: x > tie)):
            local = ensemble.assign(indicator=fn(ensemble[column]).astype(float)); estimate = float(local.indicator.mean()); ci = ci_cluster(local, "indicator", np.mean)
            fraction_rows.append({"reference_statistic": label, "relation": relation, "fraction": estimate, "ci95_low": ci[0], "ci95_high": ci[1]})
    fraction_frame = pd.DataFrame(fraction_rows)
    case_rows = []
    for label, mask in (("Source above Reference median", ensemble.source_minus_reference_median_kcal_mol > tie), ("Source below Reference median", ensemble.source_minus_reference_median_kcal_mol < -tie), ("approximately tied", ensemble.source_minus_reference_median_kcal_mol.abs() <= tie)):
        local = ensemble[mask]
        case_rows.append({"case": label, "count": len(local), "ba_median_delta": float(local.ba_delta_energy_kcal_mol.median()) if len(local) else np.nan, "ba_improved_fraction": float((local.ba_delta_energy_kcal_mol < 0).mean()) if len(local) else np.nan, "ba_p95": float(local.ba_delta_energy_kcal_mol.quantile(.95)) if len(local) else np.nan})
    source_md = ["# Source–Reference ensemble xTB energy", "", "All values are GFN2-xTB single-point energies at frozen coordinates. Reference is a sampled plausible ensemble, not a unique coordinate target. Nearest means aligned-RMSD nearest under the inherited matcher.", "", "## Source relation fractions", "", fraction_frame.to_markdown(index=False, floatfmt=".4f"), "", "## LSGO-B behavior by Source/Reference relation", "", pd.DataFrame(case_rows).to_markdown(index=False, floatfmt=".4f"), "", "A negative BA ΔE means the frozen BA update lowers energy relative to Source. Thus improvement when Source is already below the Reference median is evidence against Reference-coordinate imitation."]
    atomic_text(OUT / "SOURCE_REFERENCE_ENERGY.md", "\n".join(source_md))
    summaries = []
    for condition, group in output.groupby("condition", sort=True): summaries.append(energy_summary(group, condition, diagnostics))
    summary_frame = pd.DataFrame(summaries); summary_frame.to_csv(OUT / "B_A_BA_ABLATION.csv", index=False)
    rep_diag = diagnostics[diagnostics.condition == f"BA_seed{CONFIG['representative_ba_seed']}"]
    analysis = ensemble.merge(rep_diag, on=["molecule_id", "sample_id", "source_index", "flex_bin"], validate="one_to_one")
    analysis["energy_gain_kcal_mol"] = -analysis.ba_delta_energy_kcal_mol
    threshold = json.loads((OUT / "manifests/THRESHOLD_FREEZE.json").read_text(encoding="utf-8"))["ba_abnormality_reference_median"]
    analysis["ba_level"] = np.where(analysis.source_ba_abnormality > threshold, "high_BA", "low_BA")
    analysis["source_energy_level"] = np.where(analysis.source_minus_reference_median_kcal_mol > 0, "Source_high_energy", "Source_low_energy")
    quadrant_rows = []
    for (ba_level, energy_level), group in analysis.groupby(["ba_level", "source_energy_level"]):
        quadrant_rows.append({"quadrant": f"{ba_level}/{energy_level}", "count": len(group), "median_delta_energy_kcal_mol": float(group.ba_delta_energy_kcal_mol.median()), "improved_fraction": float((group.ba_delta_energy_kcal_mol < 0).mean()), "p95": float(group.ba_delta_energy_kcal_mol.quantile(.95)), "p99": float(group.ba_delta_energy_kcal_mol.quantile(.99)), "mean_movement_rms_angstrom": float(group.movement_rms.mean()), "mode_switch_fraction": float(group.mode_switch.mean())})
    pd.DataFrame(quadrant_rows).to_csv(OUT / "SOURCE_REFERENCE_QUADRANTS.csv", index=False)
    association_rows = []
    for label, column in (("Bond", "source_b_abnormality"), ("Angle", "source_a_abnormality"), ("BA", "source_ba_abnormality")):
        rho = float(spearmanr(analysis[column], analysis.energy_gain_kcal_mol).statistic); pear = float(pearsonr(analysis[column], analysis.energy_gain_kcal_mol).statistic)
        association_rows.append({"score": label, "spearman": rho, "spearman_ci95": correlation_ci(analysis, column, "energy_gain_kcal_mol", "spearman"), "pearson_diagnostic": pear, "pearson_ci95": correlation_ci(analysis, column, "energy_gain_kcal_mol", "pearson")})
    association = pd.DataFrame(association_rows)
    atomic_text(OUT / "BA_ENERGY_ASSOCIATION.md", "# BA abnormality–energy association\n\nEnergy gain is `E_Source-E_Output`; positive is beneficial. Spearman is primary; Pearson is diagnostic. CIs resample molecules.\n\n" + association.to_markdown(index=False, floatfmt=".4f") + "\n\nThe coordinate diagnostic table also records ΔS_B and ΔS_A through the Source and Output abnormality columns.")
    strata = []
    analysis["heavy_atom_bin"] = pd.cut(analysis.heavy_atom_count, [-np.inf, 20, 35, np.inf], labels=["<=20", "21-35", ">35"])
    for variable in ("flex_bin", "heavy_atom_bin", "aromatic", "amide_like", "ring"):
        for value, group in analysis.groupby(variable, observed=True):
            strata.append({"stratum": variable, "value": str(value), "records": len(group), "mean_source_ba_abnormality": float(group.source_ba_abnormality.mean()), "median_energy_gain_kcal_mol": float(group.energy_gain_kcal_mol.median()), "improved_fraction": float((group.energy_gain_kcal_mol > 0).mean()), "p95_harmful_delta": float(group.ba_delta_energy_kcal_mol.quantile(.95))})
    primitive = pd.read_parquet(OUT / "per_record/PRIMITIVE_CONTEXT.parquet")
    primitive = primitive.merge(analysis[["sample_id", "energy_gain_kcal_mol"]], on="sample_id", validate="many_to_one")
    for variables in (("family", "bond_type"), ("family", "hybridization")):
        for keys, group in primitive.groupby(list(variables)):
            if len(group) >= 30: strata.append({"stratum": "/".join(variables), "value": "/".join(map(str, keys)), "records": len(group), "mean_source_ba_abnormality": float(group.abs_z.mean()), "median_energy_gain_kcal_mol": float(group.energy_gain_kcal_mol.median()), "improved_fraction": float((group.energy_gain_kcal_mol > 0).mean()), "p95_harmful_delta": np.nan})
    strata_frame = pd.DataFrame(strata).sort_values("median_energy_gain_kcal_mol", ascending=False)
    atomic_text(OUT / "CHEMICAL_STRATA.md", "# Chemical strata\n\nPositive energy gain means BA lowered xTB energy. Bond-type/hybridization rows report mean primitive `|z|`; molecule-level rows report BA score. Descriptive only; no post-hoc selection.\n\n" + strata_frame.to_markdown(index=False, floatfmt=".4f"))
    high = analysis.groupby("flex_bin").agg(records=("sample_id", "size"), median_ba_delta=("ba_delta_energy_kcal_mol", "median"), improved_fraction=("ba_delta_energy_kcal_mol", lambda x: (x < 0).mean()), median_torsion_surprise=("source_torsion_nll", "median"), remaining_vs_reference_median=("source_minus_reference_median_kcal_mol", lambda x: np.nan)).reset_index()
    for index, row in high.iterrows():
        local = analysis[analysis.flex_bin == row.flex_bin]; high.loc[index, "remaining_vs_reference_median"] = float(((local.ba_energy_hartree - local.reference_median_energy_hartree) * KCAL).median())
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    high.to_csv(OUT / "tables/HIGH_FLEX_ENERGY.csv", index=False)
    gate_condition = f"BA_abnormal_seed{CONFIG['representative_ba_seed']}"; gate_energy = output[output.condition == gate_condition][["source_sample_id", "delta_energy_kcal_mol"]].rename(columns={"source_sample_id": "sample_id", "delta_energy_kcal_mol": "abnormal_delta"})
    gate_diag = diagnostics[diagnostics.condition == gate_condition][["sample_id", "movement_rms", "no_op", "active_bonds", "active_angles"]].rename(columns={"movement_rms": "abnormal_movement", "no_op": "abnormal_no_op"})
    gate = analysis.merge(gate_energy, on="sample_id").merge(gate_diag, on="sample_id")
    move_reduction = 1 - float(gate.abnormal_movement.mean() / gate.movement_rms.mean()); benefit_retention = float((-gate.abnormal_delta.median()) / (-gate.ba_delta_energy_kcal_mol.median()))
    p99_increase = float(gate.abnormal_delta.quantile(.99) - gate.ba_delta_energy_kcal_mol.quantile(.99)); rules = CONFIG["decision_rules"]["abnormality_gate"]
    gate_candidate = move_reduction >= rules["minimum_movement_reduction_fraction"] and benefit_retention >= rules["minimum_benefit_retained_fraction"] and p99_increase <= rules["maximum_harmful_p99_increase_kcal_mol"]
    lower = gate[gate.source_minus_reference_median_kcal_mol < -tie]
    gate_table = pd.DataFrame([{"subset": "all", "records": len(gate), "no_op_fraction": gate.abnormal_no_op.mean(), "movement_reduction_fraction": move_reduction, "benefit_retained_fraction": benefit_retention, "ba_all_median_delta": gate.ba_delta_energy_kcal_mol.median(), "abnormal_median_delta": gate.abnormal_delta.median(), "p99_harmful_increase": p99_increase}, {"subset": "Source below Reference median", "records": len(lower), "no_op_fraction": lower.abnormal_no_op.mean(), "movement_reduction_fraction": 1-lower.abnormal_movement.mean()/lower.movement_rms.mean(), "benefit_retained_fraction": (-lower.abnormal_delta.median())/(-lower.ba_delta_energy_kcal_mol.median()), "ba_all_median_delta": lower.ba_delta_energy_kcal_mol.median(), "abnormal_median_delta": lower.abnormal_delta.median(), "p99_harmful_increase": lower.abnormal_delta.quantile(.99)-lower.ba_delta_energy_kcal_mol.quantile(.99)}])
    atomic_text(OUT / "ABNORMALITY_GATE_DIAGNOSTIC.md", "# BA abnormal-only / no-op diagnostic\n\nDecision: **" + ("ABNORMALITY_GATE_CANDIDATE" if gate_candidate else "KEEP_BA_ALL") + "**. The candidate is diagnostic and does not modify LSGO-B.\n\n" + gate_table.to_markdown(index=False, floatfmt=".4f"))
    safety = diagnostics.groupby("condition").agg(records=("sample_id", "size"), accepted=("accepted", "mean"), chirality=("chirality_preserved", "mean"), ring_nonregression=("ring_nonregression", "mean"), hard_steric_nonregression=("catastrophic_clash_nonregression", "mean")).reset_index()
    atomic_text(OUT / "CLASH_FINAL_POSITION.md", "# Clash final position\n\nNo soft clash optimization was run. The hard steric condition remains a do-no-harm acceptance guard. Existing Source clash is not claimed to be repaired; only newly introduced catastrophic penetration is rejected.\n\n" + safety.to_markdown(index=False, floatfmt=".4f"))
    energy_audit = {"source_reference_fractions": fraction_rows, "cases": case_rows, "associations": association_rows, "abnormality_gate_candidate": bool(gate_candidate), "abnormality_gate_metrics": gate_table.to_dict("records"), "formal_test_records_read": 0, "frozen_holdout_records_read": 0}
    atomic_json(OUT / "manifests/ENERGY_ANALYSIS.json", energy_audit)
    print("LSGO_MECHANISM_ENERGY_ANALYSIS_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
