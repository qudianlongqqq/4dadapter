#!/usr/bin/env python3
"""Project frozen xTB forces onto B/A/T internal-coordinate subspaces."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr, spearmanr

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
from etflow.ecir.bat_refinement import canonical_rotatable_torsions, dihedral_angles
from etflow.ecir.geometry import bond_lengths
from etflow.ecir.learned_geometry import prepare_graph, stable_angle_cosine
from etflow.ecir.lsgo_io import atomic_json, file_sha256
from etflow.ecir.lsgo_mechanism import circular_finite_difference, finite_difference_row, force_projection, internal_jacobians, joint_matrix

OUT = ROOT / "reports/ecir_mvr/lsgo_mechanism"
CONFIG = yaml.safe_load((ROOT / "configs/ecir_mvr_lsgo_mechanism.yaml").read_text(encoding="utf-8"))
KCAL = 627.509474


def atomic_text(path: Path, value: str):
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}"); temporary.write_text(value.rstrip() + "\n", encoding="utf-8"); os.replace(temporary, path)


def clean(matrix):
    matrix = torch.as_tensor(matrix, dtype=torch.float64)
    if not matrix.numel(): return matrix, 0
    valid = torch.isfinite(matrix).all(1) & (torch.linalg.vector_norm(matrix, dim=1) > 1e-12)
    return matrix[valid], int((~valid).sum())


def correlation(frame, x, y):
    finite = frame[[x, y]].dropna()
    return {"records": len(finite), "spearman": float(spearmanr(finite[x], finite[y]).statistic) if len(finite) > 2 else np.nan, "pearson_diagnostic": float(pearsonr(finite[x], finite[y]).statistic) if len(finite) > 2 else np.nan}


def main():
    force_manifest = json.loads((OUT / "manifests/XTB_FORCE_COMPLETE.json").read_text(encoding="utf-8")); protocol = json.loads((OUT / "manifests/XTB_FORCE_PROTOCOL.json").read_text(encoding="utf-8"))
    if force_manifest["status"] != "COMPLETED" or protocol["status"] != "AVAILABLE": raise RuntimeError("xTB force unavailable")
    force = pd.read_parquet(force_manifest["result_path"]); freeze = json.loads((OUT / "COORDINATE_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    tasks = torch.load(freeze["coordinate_path"], map_location="cpu", weights_only=False)["tasks"]
    task_lookup = {(row["condition"], row["sample_id"]): row for row in tasks}
    compact = torch.load(json.loads((OUT / "DATASET_IDENTITY.json").read_text(encoding="utf-8"))["compact_path"], map_location="cpu", weights_only=False)
    item_lookup = {row["molecule_id"]: row for row in compact["items"]}; calibration = json.loads(Path(CONFIG["dataset"]["drcsr_calibration"]).read_text(encoding="utf-8"))
    relative = float(CONFIG["force"]["svd_relative_rank_tolerance"]); absolute = float(CONFIG["force"]["svd_absolute_rank_tolerance"])
    rows, fd_rows = [], []
    for record in force.itertuples(index=False):
        task = task_lookup[(record.condition, record.sample_id)]; xyz = torch.as_tensor(task["coordinates"], dtype=torch.float64)
        item = item_lookup[record.molecule_id]; graph = prepare_graph(item["record"], calibration); torsions = canonical_rotatable_torsions(item["record"])[0]
        matrices = internal_jacobians(xyz, graph, torsions); dropped = {}
        for family in ("B", "A", "T"): matrices[family], dropped[family] = clean(matrices[family])
        ba, bat = joint_matrix(matrices["B"], matrices["A"]), joint_matrix(matrices["B"], matrices["A"], matrices["T"])
        xtb_force = -torch.from_numpy(np.asarray(record.gradient_hartree_per_bohr.tolist(), dtype=np.float64))
        projections = {family: force_projection(xtb_force, matrix, xyz, relative_tolerance=relative, absolute_tolerance=absolute) for family, matrix in {**matrices, "BA": ba, "BAT": bat}.items()}
        incremental = (projections["BAT"]["projection_norm"] - projections["BA"]["projection_norm"]) / max(projections["BA"]["force_norm"], 1e-15)
        rows.append({"condition": record.condition, "stage": "before" if record.condition == "Source" else "after", "molecule_id": record.molecule_id, "sample_id": record.sample_id, "flex_bin": record.flex_bin, "rotatable_bond_count": record.rotatable_bond_count,
            "force_norm_hartree_per_bohr": projections["BA"]["force_norm"], "B_fraction": projections["B"]["fraction"], "A_fraction": projections["A"]["fraction"], "T_fraction": projections["T"]["fraction"], "BA_union_fraction": projections["BA"]["fraction"], "BAT_union_fraction": projections["BAT"]["fraction"], "BAT_incremental_fraction": incremental,
            "B_projection_norm": projections["B"]["projection_norm"], "A_projection_norm": projections["A"]["projection_norm"], "T_projection_norm": projections["T"]["projection_norm"], "BA_projection_norm": projections["BA"]["projection_norm"], "BAT_projection_norm": projections["BAT"]["projection_norm"],
            "B_rank": projections["B"]["rank"], "A_rank": projections["A"]["rank"], "T_rank": projections["T"]["rank"], "BA_rank": projections["BA"]["rank"], "BAT_rank": projections["BAT"]["rank"], "B_dropped_rows": dropped["B"], "A_dropped_rows": dropped["A"], "T_dropped_rows": dropped["T"]})
        # Existing central-difference audit logic: first valid row per family on each Source.
        if record.condition == "Source":
            if graph.bonds.size(1):
                numeric = finite_difference_row(lambda value: bond_lengths(value, graph.bonds[:, :1]), xyz, step=1e-5); analytic = internal_jacobians(xyz, graph, torsions[:0])["B"][:1]
                fd_rows.append({"molecule_id": record.molecule_id, "family": "B", "max_abs_error": float((numeric - analytic).abs().max())})
            if graph.angles.size(0):
                numeric = finite_difference_row(lambda value: stable_angle_cosine(value, graph.angles[:1]), xyz, step=1e-5); analytic = internal_jacobians(xyz, graph, torsions[:0])["A"][:1]
                fd_rows.append({"molecule_id": record.molecule_id, "family": "A", "max_abs_error": float((numeric - analytic).abs().max())})
            if torsions.size(0):
                numeric = circular_finite_difference(lambda value: dihedral_angles(value, torsions[:1]), xyz, step=1e-5); analytic = internal_jacobians(xyz, graph, torsions[:1])["T"][:1]
                fd_rows.append({"molecule_id": record.molecule_id, "family": "T", "max_abs_error": float((numeric - analytic).abs().max())})
    projection = pd.DataFrame(rows); projection.to_csv(OUT / "XTB_FORCE_PROJECTION.csv", index=False)
    fd = pd.DataFrame(fd_rows); (OUT / "tables").mkdir(parents=True, exist_ok=True); fd.to_csv(OUT / "tables/INTERNAL_JACOBIAN_FD.csv", index=False)
    summary = projection.groupby(["stage", "flex_bin"]).agg(records=("sample_id", "size"), force_norm_median=("force_norm_hartree_per_bohr", "median"), B_fraction_median=("B_fraction", "median"), A_fraction_median=("A_fraction", "median"), T_fraction_median=("T_fraction", "median"), BA_union_fraction_median=("BA_union_fraction", "median"), BAT_union_fraction_median=("BAT_union_fraction", "median"), BAT_incremental_fraction_median=("BAT_incremental_fraction", "median"), BA_projection_norm_median=("BA_projection_norm", "median"), T_projection_norm_median=("T_projection_norm", "median")).reset_index()
    overall = projection.groupby("stage").agg(records=("sample_id", "size"), force_norm_median=("force_norm_hartree_per_bohr", "median"), B_fraction_median=("B_fraction", "median"), A_fraction_median=("A_fraction", "median"), T_fraction_median=("T_fraction", "median"), BA_union_fraction_median=("BA_union_fraction", "median"), BAT_union_fraction_median=("BAT_union_fraction", "median"), BAT_incremental_fraction_median=("BAT_incremental_fraction", "median"), BA_projection_norm_median=("BA_projection_norm", "median"), T_projection_norm_median=("T_projection_norm", "median")).reset_index(); overall.insert(1, "flex_bin", "all")
    summary = pd.concat([overall, summary], ignore_index=True); summary.to_csv(OUT / "tables/FORCE_PROJECTION_SUMMARY.csv", index=False)
    atomic_text(OUT / "XTB_FORCE_PROTOCOL.md", "# xTB force protocol\n\nStatus: **AVAILABLE**. GFN2-xTB 6.7.1 `--grad` was used at frozen coordinates with no optimization. The file reports the energy gradient in Hartree/Bohr; physical force is its negative. Energy identity error was `%.3g Eh`, repeat error `%.3g Eh/bohr`, and input-coordinate order error `%.3g bohr`. Central finite-difference relative errors for the first three Cartesian components were `%s`.\n\nThis diagnostic was never used for training, selection, coordinate generation or tuning." % (protocol["energy_identity_error_hartree"], protocol["gradient_repeat_max_error_hartree_per_bohr"], protocol["coordinate_order_max_error_bohr"], ", ".join(f"{row['relative_error']:.3g}" for row in protocol["finite_difference"])))
    atomic_text(OUT / "INTERNAL_SUBSPACE_AUDIT.md", "# Internal-coordinate subspace audit\n\nJacobians are analytic autograd derivatives checked by central finite differences; periodic wrap is used for torsions. Non-finite and norm≤1e-12 rows are excluded before fixed-tolerance SVD (`rtol=1e-8`, `atol=1e-10`). Raw B/A/T projections are descriptive because they overlap; BA and BAT are joint row-space SVD unions.\n\n" + summary.to_markdown(index=False, floatfmt=".5f") + "\n\nFinite-difference maximum absolute errors:\n\n" + fd.groupby("family").max_abs_error.agg(["count", "median", "max"]).to_markdown(floatfmt=".3g"))
    # Torsion detector associations.
    energy = pd.read_parquet(OUT / "per_record/SOURCE_REFERENCE_ENERGY.parquet")
    diagnostics = pd.read_parquet(OUT / "per_record/COORDINATE_DIAGNOSTICS.parquet"); rep = diagnostics[diagnostics.condition == f"BA_seed{CONFIG['representative_ba_seed']}"]
    torsion = energy.merge(rep[["sample_id", "source_torsion_nll", "output_torsion_nll"]], on="sample_id", validate="one_to_one")
    torsion["remaining_ba_excess_kcal_mol"] = (torsion.ba_energy_hartree - torsion.reference_median_energy_hartree) * KCAL
    before = projection[projection.stage == "before"][["sample_id", "T_fraction", "BAT_incremental_fraction"]]
    torsion_force = torsion.merge(before, on="sample_id", how="left")
    associations = {"torsion_surprise_vs_source_reference_median_excess": correlation(torsion_force, "source_torsion_nll", "source_minus_reference_median_kcal_mol"), "torsion_surprise_vs_remaining_ba_excess": correlation(torsion_force, "source_torsion_nll", "remaining_ba_excess_kcal_mol"), "torsion_surprise_vs_independent_t_force_fraction": correlation(torsion_force, "source_torsion_nll", "T_fraction"), "torsion_surprise_vs_incremental_bat_fraction": correlation(torsion_force, "source_torsion_nll", "BAT_incremental_fraction")}
    rules = CONFIG["decision_rules"]["torsion_actionability"]; before_projection = projection[projection.stage == "before"]
    overall_incremental = float(before_projection.BAT_incremental_fraction.median()); high_incremental = float(before_projection[before_projection.flex_bin == "high_ge_5"].BAT_incremental_fraction.median())
    actionable = overall_incremental >= rules["bat_incremental_projection_fraction_median"] or high_incremental >= rules["bat_incremental_projection_fraction_high_flex"]
    force_rho = associations["torsion_surprise_vs_incremental_bat_fraction"]["spearman"]; detector_weak = not np.isfinite(force_rho) or abs(force_rho) < rules["detector_weak_spearman_abs"]
    torsion_decision = "TORSION_REDESIGN_WARRANTED" if actionable else "TORSION_LOW_ACTIONABILITY"
    atomic_text(OUT / "TORSION_PHYSICAL_ROLE.md", "# Torsion physical role\n\nDecision: **%s**. The frozen K3 model's Reference likelihood success is retained; this audit tests whether its Source surprise is a useful physical-error detector.\n\n- median incremental BAT-union fraction before BA: `%.5f`; high-flex: `%.5f`;\n- surprise vs incremental BAT fraction Spearman: `%s` (%s detector under preregistered |ρ|<%.2f rule).\n\nAssociations:\n\n%s\n\nAn independent T projection is not additive with BA because subspaces overlap; the incremental BAT-union fraction is the sufficiency diagnostic." % (torsion_decision, overall_incremental, high_incremental, "nan" if not np.isfinite(force_rho) else f"{force_rho:.4f}", "weak" if detector_weak else "non-weak", rules["detector_weak_spearman_abs"], pd.DataFrame([{"association": key, **value} for key, value in associations.items()]).to_markdown(index=False, floatfmt=".4f")))
    high_energy = pd.read_csv(OUT / "tables/HIGH_FLEX_ENERGY.csv"); high_force = summary[summary.flex_bin != "all"]
    high_table = high_energy.merge(high_force, on="flex_bin", how="left")
    atomic_text(OUT / "HIGH_FLEX_ANALYSIS.md", "# Flexibility-stratified analysis\n\nBins (0–2, 3–4, ≥5 canonical free rotors) and six force records per bin were frozen before xTB access.\n\n" + high_table.to_markdown(index=False, floatfmt=".4f"))
    # Energy component and full mechanism decisions.
    ablation = pd.read_csv(OUT / "B_A_BA_ABLATION.csv"); medians = {}
    for family in ("B", "A", "BA"):
        medians[family] = float((-ablation[ablation.condition.str.match(f"^{family}_seed")].median_delta_energy_kcal_mol).median())
    b_retained, a_retained = medians["B"] / medians["BA"], medians["A"] / medians["BA"]
    dominance = CONFIG["decision_rules"]["component_dominance"]
    component = "B_DOMINANT" if b_retained >= dominance["retained_fraction_of_ba"] and a_retained <= dominance["other_component_max_fraction_of_ba"] else ("A_DOMINANT" if a_retained >= dominance["retained_fraction_of_ba"] and b_retained <= dominance["other_component_max_fraction_of_ba"] else None)
    synergy_fraction = (medians["BA"] - max(medians["B"], medians["A"])) / max(medians["B"], medians["A"])
    synergy = synergy_fraction >= CONFIG["decision_rules"]["ba_synergy"]["relative_gain_over_best_single"]
    energy_audit = json.loads((OUT / "manifests/ENERGY_ANALYSIS.json").read_text(encoding="utf-8")); gate = "ABNORMALITY_GATE_CANDIDATE" if energy_audit["abnormality_gate_candidate"] else "KEEP_BA_ALL"
    before_all = summary[(summary.stage == "before") & (summary.flex_bin == "all")].iloc[0]; after_all = summary[(summary.stage == "after") & (summary.flex_bin == "all")].iloc[0]
    local_strain = before_all.BA_union_fraction_median >= CONFIG["decision_rules"]["ba_local_strain"]["minimum_ba_union_force_fraction_median"] and bool((ablation[ablation.condition.str.match("^BA_seed")].p99 <= 1e-12).all())
    decisions = (["BA_LOCAL_STRAIN_CONFIRMED"] if local_strain else []) + ([component] if component else []) + (["BA_SYNERGISTIC"] if synergy else []) + [torsion_decision, gate]
    payload = {"schema_version": "mcvr-lsgo-mechanism-decision-v1", "status": "COMPLETED", "decisions": decisions, "component_median_energy_gain_kcal_mol": medians, "b_fraction_of_ba_gain": b_retained, "a_fraction_of_ba_gain": a_retained, "ba_synergy_fraction_over_best_single": synergy_fraction, "ba_force_before": before_all.to_dict(), "ba_force_after": after_all.to_dict(), "torsion_incremental_before_median": overall_incremental, "torsion_incremental_high_flex_before_median": high_incremental, "torsion_detector_associations": associations, "abnormality_gate": gate, "current_formal_method": "LSGO-B + topology/chirality/hard-steric/ring safety guards", "method_modified": False, "formal_test_records_read": 0, "frozen_holdout_records_read": 0}
    atomic_json(OUT / "FINAL_DECISION.json", payload)
    print("LSGO_MECHANISM_FORCE_ANALYSIS_COMPLETED")
    return 0


if __name__ == "__main__": raise SystemExit(main())
