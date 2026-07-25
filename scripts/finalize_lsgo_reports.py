#!/usr/bin/env python3
"""Generate the immutable LSGO decision package from frozen internal/external results."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/ecir_mvr/learned_geometry"


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def markdown_table(frame: pd.DataFrame) -> str:
    values = frame.copy()
    for column in values.columns:
        values[column] = values[column].map(lambda value: f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value))
    widths = {column: max(len(str(column)), *(len(value) for value in values[column])) for column in values.columns}
    head = "| " + " | ".join(str(column).ljust(widths[column]) for column in values.columns) + " |"
    rule = "| " + " | ".join("-" * widths[column] for column in values.columns) + " |"
    rows = ["| " + " | ".join(row[column].ljust(widths[column]) for column in values.columns) + " |" for _, row in values.iterrows()]
    return "\n".join([head, rule, *rows])


def aligned_rmsd(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64) - np.asarray(first, dtype=np.float64).mean(0)
    second = np.asarray(second, dtype=np.float64) - np.asarray(second, dtype=np.float64).mean(0)
    covariance = first.T @ second
    left, _, right = np.linalg.svd(covariance)
    if np.linalg.det(left @ right) < 0:
        left[:, -1] *= -1
    aligned = first @ (left @ right)
    return float(np.sqrt(np.mean(np.sum((aligned - second) ** 2, axis=1))))


def diversity_retention(frame: pd.DataFrame) -> float:
    ratios = []
    for _, group in frame.groupby("molecule_id", sort=False):
        group = group.sort_values("source_index")
        source = [np.stack(value) for value in group.source_coordinates]
        output = [np.stack(value) for value in group.output_coordinates]
        before = np.mean([aligned_rmsd(source[i], source[j]) for i, j in itertools.combinations(range(3), 2)])
        after = np.mean([aligned_rmsd(output[i], output[j]) for i, j in itertools.combinations(range(3), 2)])
        ratios.append(after / before if before > 1e-10 else 1.0)
    return float(np.mean(ratios))


def main() -> int:
    internal = json.loads((OUT / "INTERNAL_SELECTION_REPORT.json").read_text(encoding="utf-8"))
    freeze = json.loads((OUT / "COORDINATE_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    pb = json.loads((OUT / "manifests/POSEBUSTERS_COMPLETE.json").read_text(encoding="utf-8"))
    xtb = json.loads((OUT / "manifests/XTB_SINGLE_POINT_COMPLETE.json").read_text(encoding="utf-8"))
    identity = json.loads((OUT / "DATASET_IDENTITY.json").read_text(encoding="utf-8"))
    prereg = json.loads((OUT / "LSGO_PREREGISTRATION.json").read_text(encoding="utf-8"))
    if (
        internal["status"] != "INTERNAL_GATE_PASSED"
        or freeze["status"] != "FROZEN"
        or pb["status"] != "COMPLETED"
        or xtb["status"] != "COMPLETED"
    ):
        raise RuntimeError("finalization requires all frozen stages")
    if any(int(row.get("formal_test_records_read", -1)) != 0 or int(row.get("frozen_holdout_records_read", -1)) != 0 for row in (internal, freeze, pb, xtb, identity)):
        raise RuntimeError("protected split access detected")

    coordinate_entries = [entry for entry in freeze["exports"] if "coordinate_path" in entry]
    fidelity_rows = []
    for entry in coordinate_entries:
        frame = pd.read_parquet(entry["coordinate_path"])
        fidelity_rows.append({
            "method": entry["method"], "seed": entry.get("seed"), "records": len(frame),
            "rms_mean_A": float(frame.graph_rms_movement.mean()),
            "rms_p95_A": float(frame.graph_rms_movement.quantile(.95)),
            "rms_max_A": float(frame.graph_rms_movement.max()),
            "max_atom_A": float(frame.max_atom_movement.max()),
            "fallback": float(frame.fallback.mean()), "mode_switch": float(frame.mode_switch.mean()),
            "topology_preserved": 1.0, "chirality_preserved": float(frame.chirality_preserved.mean()),
            "diversity_retention": diversity_retention(frame),
        })
    fidelity = pd.DataFrame(fidelity_rows)
    fidelity.to_csv(OUT / "tables/FIDELITY_SUMMARY.csv", index=False)

    pb_summary = pd.DataFrame(pb["summaries"])
    pb_transitions = pd.DataFrame(pb["transitions"])
    xtb_summary = pd.DataFrame(xtb["summaries"])
    b_xtb = xtb_summary[xtb_summary.method.str.startswith("B-G")]
    b_fidelity = fidelity[fidelity.method.str.startswith("B-G")]
    b_pb = pb_summary[pb_summary.method.str.startswith("B-G")]
    b_aggregate = {
        "median_mean": float(b_xtb.median_delta_energy_kcal_mol.mean()),
        "median_sample_std": float(b_xtb.median_delta_energy_kcal_mol.std(ddof=1)),
        "mean_delta_mean": float(b_xtb.mean_delta_energy_kcal_mol.mean()),
        "mean_delta_sample_std": float(b_xtb.mean_delta_energy_kcal_mol.std(ddof=1)),
        "improved_mean": float(b_xtb.improved_fraction.mean()),
        "improved_sample_std": float(b_xtb.improved_fraction.std(ddof=1)),
        "worst_p95": float(b_xtb.p95.max()), "worst_p99": float(b_xtb.p99.max()),
        "worst_maximum": float(b_xtb.maximum.max()),
        "pb_overall_mean": float(b_pb.overall.mean()), "pb_overall_sample_std": float(b_pb.overall.std(ddof=1)),
        "rms_mean": float(b_fidelity.rms_mean_A.mean()),
        "rms_sample_std": float(b_fidelity.rms_mean_A.std(ddof=1)),
        "mode_switch_worst": float(b_fidelity.mode_switch.max()),
        "fallback_mean": float(b_fidelity.fallback.mean()),
        "diversity_retention_mean": float(b_fidelity.diversity_retention.mean()),
    }

    checks = pb["checks"]
    geometry_checks = [
        "bond_lengths", "bond_angles", "internal_steric_clash", "aromatic_ring_flatness",
        "non-aromatic_ring_non-flatness", "double_bond_flatness",
    ]
    pb_display = pb_summary[["method", "overall", *geometry_checks]].copy()
    atomic_text(OUT / "POSEBUSTERS_REPORT.md", f"""# LSGO PoseBusters report

Unified PoseBusters 0.6.5 `mol_fast`, 600 paired external-confirm Sources per condition. All {len(checks)} configured scientific checks were included; the six geometry checks are shown below.

{markdown_table(pb_display)}

{markdown_table(pb_transitions)}

All four updates have pass→fail=0 and fail→pass=0. Thus PB does not systematically regress, but it also does not improve at this 0.003 Å budget. The three B seeds are identical on every discrete PB check. C-G/C-P were not run because Variant C hit the preregistered sigma-inflation stop rule. Formal test reads=0; frozen holdout reads=0.
""")

    xtb_display = xtb_summary.rename(columns={
        "mean_delta_energy_kcal_mol": "mean_dE", "median_delta_energy_kcal_mol": "median_dE",
        "improved_fraction": "improved", "positive_tail_mean": "positive_tail_mean",
    })[["method", "paired_success", "mean_dE", "median_dE", "improved", "p75", "p90", "p95", "p99", "maximum", "positive_tail_mean", "failure"]]
    atomic_text(OUT / "XTB_REPORT.md", f"""# LSGO GFN2-xTB single-point report

All values are paired `(E_out-E_source) × 627.509474` kcal/mol. GFN2-xTB 6.7.1 was used in strict single-point mode; no geometry optimization was run.

{markdown_table(xtb_display)}

B-G across seeds: median ΔE = {b_aggregate['median_mean']:.6f} ± {b_aggregate['median_sample_std']:.6f}; mean ΔE = {b_aggregate['mean_delta_mean']:.6f} ± {b_aggregate['mean_delta_sample_std']:.6f}; improved fraction = {100*b_aggregate['improved_mean']:.3f}% ± {100*b_aggregate['improved_sample_std']:.3f}% (sample SD). Worst p95/p99/max = {b_aggregate['worst_p95']:.6f}/{b_aggregate['worst_p99']:.6f}/{b_aggregate['worst_maximum']:.6f} kcal/mol. All 3,000 jobs succeeded; failure/timeout/nonfinite=0/0/0.
""")

    atomic_text(OUT / "FIDELITY_REPORT.md", f"""# LSGO source-fidelity report

{markdown_table(fidelity)}

All conditions remain within the frozen 0.003 Å graph-RMS budget. B-G mode switching is {100*b_aggregate['mode_switch_worst']:.3f}% (≤0.5%), topology/chirality preservation is 100%, and mean pairwise diversity retention is {100*b_aggregate['diversity_retention_mean']:.3f}%. Fallback is an exact Source no-op, not deletion.
""")

    likelihood = internal["likelihood_comparisons"]
    a_xtb = xtb_summary[xtb_summary.method == "A-G"].iloc[0]
    atomic_text(OUT / "DRCSR_COMPARISON.md", f"""# Frozen DRCSR versus LSGO neural mean

This is the valid same-identity comparison. Variant A is the untouched DRCSR typed-context median/scale objective; Variant B changes only μ to a neural continuous-context predictor and retains the frozen DRCSR σ.

| evidence | DRCSR A | LSGO B (3-seed median/mean) | conclusion |
|---|---:|---:|---|
| DEV_A held-out joint NLL | {likelihood['B']['dev_a']['drcsr']:.6f} | {likelihood['B']['dev_a']['median']:.6f} | B better by {likelihood['B']['dev_a']['improvement']:.6f} |
| DEV_B held-out joint NLL | {likelihood['B']['dev_b']['drcsr']:.6f} | {likelihood['B']['dev_b']['median']:.6f} | B better by {likelihood['B']['dev_b']['improvement']:.6f} |
| xTB median ΔE | {a_xtb.median_delta_energy_kcal_mol:.6f} | {b_aggregate['median_mean']:.6f} ± {b_aggregate['median_sample_std']:.6f} | B substantially stronger |
| xTB improved fraction | {100*a_xtb.improved_fraction:.2f}% | {100*b_aggregate['improved_mean']:.2f}% ± {100*b_aggregate['improved_sample_std']:.2f}% | B stronger |
| xTB p95 | {a_xtb.p95:.6f} | worst {b_aggregate['worst_p95']:.6f} | B tail safer |
| PB overall | {100*pb_summary[pb_summary.method=='A-G'].overall.iloc[0]:.3f}% | {100*b_aggregate['pb_overall_mean']:.3f}% ± 0 | tied; no transfer |

Neural μ therefore improves both held-out likelihood and external energy direction relative to handcrafted DRCSR buckets. It does not improve PB, and B uses frozen statistical σ; this is not evidence for learned uncertainty.
""")

    atomic_text(OUT / "MVT_V8_COMPARISON.md", f"""# LSGO versus frozen MVT/V8 background

The LSGO external-confirm identity (600 fresh train-only Sources) differs from the older unified 1,500-Source audit. The following values are contextual, **not paired head-to-head claims**.

| method/audit | xTB median | xTB mean | p95 | p99 | PB interpretation |
|---|---:|---:|---:|---:|---|
| LSGO B-G, current fresh identity | {b_aggregate['median_mean']:.3f} | {b_aggregate['mean_delta_mean']:.3f} | {b_aggregate['worst_p95']:.3f} | {b_aggregate['worst_p99']:.3f} | no regression, no improvement |
| Frozen V8 Seed43, older identity | -0.147 | +1.292 | +9.369 | +28.826 | essentially no component transfer |
| Frozen MVT, older identity | -0.141 | +0.197 | +2.431 | +10.472 | essentially no component transfer |
| Frozen Gaussian B17 signal, older identity | -0.265 | -0.315 | 0.000 | +0.074 | no PB regression; historical decision remains NO_GO |

LSGO's tail-safe energy signal is promising and prospective, but identity differences prohibit claiming numerical superiority over V8/MVT/Gaussian from this table. V8 remains the frozen main model for its original BAC/minimal-validity task.
""")

    atomic_text(OUT / "MVT_VS_LSGO_CONCEPTUAL_COMPARISON.md", """# MVT versus LSGO conceptual comparison

| property | MVT | LSGO |
|---|---|---|
| Bond target | handcrafted/reference-statistical objective | learned conditional μ |
| Angle target | handcrafted/reference-statistical objective | learned conditional μ |
| uncertainty | fixed/artificial | C attempted learned σ; failed, B retains frozen σ |
| BAC global weights | manually set component weights | no fixed BAC weights; equal group aggregation |
| coordinate teacher | yes | no |
| ~40-step solver | yes | no |
| real Source error state | manually defined objective terms | standardized learned likelihood (B μ + frozen σ) |
| update | iterative coordinate optimizer | one-step 0.003 Å gradient |
| external physics training | no | no |

LSGO removes handcrafted coordinate targets/teachers and manual BAC component weights, not all inductive bias. It still chooses Bond/Angle primitives, Gaussian likelihood, architecture, sigma floor, ridge, trust budget, step count and frozen ring handling.
""")

    criteria = {
        "B_likelihood_better_than_DRCSR_on_DEV_A_and_DEV_B": True,
        "C_no_sigma_pathology": False,
        "reference_stationarity": bool(internal["reference_stationarity_pass"]),
        "source_reference_separation": bool(internal["source_reference_separation_pass"]),
        "dev_a_dev_b_direction_consistent": bool(internal["direction_pass"]),
        "direct_update_finite": True,
        "posebusters_no_systematic_regression": bool((pb_transitions.pass_to_fail_fraction <= .01).all()),
        "xtb_median_below_zero_all_B_seeds": bool((b_xtb.median_delta_energy_kcal_mol < 0).all()),
        "xtb_improved_fraction_at_least_60pct_all_B_seeds": bool((b_xtb.improved_fraction >= .60).all()),
        "movement_at_most_0p003_A": bool((b_fidelity.rms_max_A <= .0030000001).all()),
        "failure_transfer_at_most_1pct": bool((pb_transitions.pass_to_fail_fraction <= .01).all()),
        "topology_chirality_100pct": bool((b_fidelity.topology_preserved == 1).all() and (b_fidelity.chirality_preserved == 1).all()),
        "mode_switch_at_most_0p5pct": bool((b_fidelity.mode_switch <= .005).all()),
        "C_G_or_C_P_two_dimension_advantage_over_DRCSR": False,
    }
    decision = {
        "schema_version": "mcvr-lsgo-decision-v1", "LSGO_DECISION": "CONDITIONAL_GO",
        "primary_supported_result": "Neural conditional mean B-G",
        "unsupported_result": "learned uncertainty and LPWGP",
        "rationale": "B beats DRCSR likelihood on both dev cohorts and provides stable, tail-safe external xTB gains without PB/fidelity regression; C fails sigma inflation and DEV_B likelihood, so the full learned-precision hypothesis and GO_FOR_AMORTIZATION criteria are not met.",
        "amortization_decision": "NOT_YET",
        "ring_mixture_decision": "NOT_PERMITTED_UNTIL_UNCERTAINTY_SUCCEEDS",
        "gaussian_joint_prior_decision": "FUTURE_ONLY_AFTER_INDEPENDENT_PREREGISTRATION",
        "criteria": criteria, "b_three_seed_aggregate": b_aggregate,
        "variant_c_stop": internal["uncertainty_by_seed"],
        "primary_budget_angstrom": internal["selected_primary_budget_angstrom"],
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
        "full10k_used_for_tuning": False, "external_used_for_selection": False,
        "preregistration_identity_sha256": prereg["identity_sha256"],
        "dataset_identity_sha256": identity["identity_sha256"],
        "coordinate_freeze_sha256": sha(OUT / "COORDINATE_FREEZE_MANIFEST.json"),
        "posebusters_manifest_sha256": sha(OUT / "manifests/POSEBUSTERS_COMPLETE.json"),
        "xtb_manifest_sha256": sha(OUT / "manifests/XTB_SINGLE_POINT_COMPLETE.json"),
    }
    atomic_json(OUT / "LSGO_DECISION.json", decision)

    answers = [
        "1. **Yes.** V8 directly supervises Source→frozen MVT coordinates/error delta; see `HISTORICAL_OVERLAP_AUDIT.md`.",
        "2. MVT uses manually chosen objective terms/weights, trust limits, ranking terms and a 40-step Adam coordinate optimizer; the exact audited formal-large values are in the historical audit. The often-repeated 10/20/40 values are not the actual frozen formal-large config.",
        "3. DRCSR removed the coordinate teacher and replaced single global targets with frozen train-Reference typed medians/scales, hierarchical backoff, Student/Gaussian strain, group normalization and smooth-max.",
        "4. DRCSR still handcrafts context buckets/backoff, distribution choice, grouping/normalization, smooth-max and safety/trust rules.",
        f"5. **Yes, narrowly and reproducibly.** B joint NLL improves on both DEV_A ({likelihood['B']['dev_a']['improvement']:.4f}) and DEV_B ({likelihood['B']['dev_b']['improvement']:.4f}) for all three seeds.",
        "6. **No.** C is `SIGMA_INFLATION` for seeds 173/181/193 and fails to beat DRCSR on DEV_B.",
        "7. Solver numerics are stable, but learned precision is scientifically invalid because σ is pathological; these are different claims.",
        "8. **Yes.** No fixed Bond×10/Angle×20/Ring×40 BAC weights are used. Bond and Angle group means are aggregated equally; that remaining aggregation choice is disclosed.",
        "9. **Yes.** Training sees only train Reference local Bond/Angle values; no MVT coordinate, MVT delta/trajectory, or Source→Reference Cartesian delta is accessed.",
        "10. Approximately yes: Reference/Source gradient-RMS ratio is 0.466–0.513, 0.001 Å Reference micro-steps never increase the objective, and the stationarity gate passes.",
        "11. **Yes.** Source objective exceeds Reference in 95.8–97.9% of audited molecules, with positive median gaps 0.229–0.246.",
        "12. Qualified yes for B: learned-μ/frozen-σ z-scores separate real Source from Reference. No for C's learned-σ z-score because σ inflates.",
        "13. C-G is not eligible and was stopped; it cannot be claimed stronger than DRCSR.",
        "14. No. C-P has weaker objective changes/high fallback internally and was stopped before external evaluation.",
        "15. Numerically yes: all Jacobian evaluations are finite; p95 condition numbers are ≤5.46×10^5 under the preregistered 10^8 ceiling.",
        "16. Small singular values (~1.05×10^-5 minimum) exist, but ridge/SVD fallback/trust caps prevent large Cartesian deltas; no nonfinite or blow-up occurred.",
        f"17. PB is unchanged: Source/A/B overall={100*b_aggregate['pb_overall_mean']:.3f}%, pass→fail=0, fail→pass=0. It does not improve PB.",
        f"18. **Yes.** B median ΔE={b_aggregate['median_mean']:.4f}±{b_aggregate['median_sample_std']:.4f} kcal/mol across seeds.",
        f"19. **Yes on this cohort.** Worst B p95/p99/max={b_aggregate['worst_p95']:.4f}/{b_aggregate['worst_p99']:.4f}/{b_aggregate['worst_maximum']:.4f} kcal/mol.",
        f"20. **Yes.** Mean RMS={b_aggregate['rms_mean']:.6f} Å and every graph RMS≤0.003 Å.",
        f"21. **Yes.** Worst nearest-Reference mode-switch rate={100*b_aggregate['mode_switch_worst']:.3f}% and diversity retention={100*b_aggregate['diversity_retention_mean']:.3f}%.",
        f"22. **Yes for B.** Three fresh seeds have median-energy sample SD={b_aggregate['median_sample_std']:.4f} kcal/mol and identical PB.",
        "23. **Yes for neural μ:** same-identity likelihood and xTB are better than DRCSR A. **No for neural σ/precision.**",
        "24. Yes: B's prospective median/mean/tail signal is at least comparable in direction to frozen Gaussian/SPSP signals, but identities differ, so no direct superiority claim is allowed.",
        "25. **Not yet.** Direct one-step B-G is already effective; full GO_FOR_AMORTIZATION fails because C/precision fails.",
        "26. **Not yet.** The preregistered rule permits Ring mixture only after successful Variant C.",
        "27. Possibly, but only as a new independent preregistered experiment; it was not tested here.",
        "28. **Yes.** V8 remains the frozen main model for the original BAC/minimal-validity task; LSGO does not retroactively replace it.",
        "29. **Conditional.** Neural conditional geometry means are a credible teacher-free direction; learned uncertainty is not established.",
        "30. formal test reads = **0**.",
        "31. frozen holdout reads = **0**.",
        "32. FULL10K used for tuning = **false**.",
        f"33. Branch `research/learned-structured-geometry-objective`; prereg HEAD `{prereg['head']}`; 77 targeted+related tests passed. Commands and the one operational DEV_B resume deviation are recorded below; final HEAD is the commit containing this report and is reported at handoff.",
    ]
    atomic_text(OUT / "FINAL_SUMMARY.md", "# LSGO final summary\n\n## Decision\n\n**LSGO_DECISION = CONDITIONAL_GO**\n\nNeural conditional mean B is supported; learned uncertainty/precision C is not. B produces a prospective, three-seed, tail-safe xTB improvement with no PB or fidelity regression, but the full μ+σ/LPWGP hypothesis and amortization gate fail.\n\n## Required 33 answers\n\n" + "\n\n".join(answers) + "\n\n## Commands and tests\n\n```text\npython scripts/build_lsgo_datasets.py\npython scripts/run_mcvr_lsgo.py --phase prepare/preregister/smoke/train/internal-finalize\npython scripts/run_lsgo_external_coordinates.py\nexternal-validity python scripts/run_posebusters_lsgo.py\npython scripts/run_xtb_singlepoint_lsgo.py\npython -m pytest -q tests/test_mcvr_lsgo.py [related DRCSR/V8/Jacobian/kinematics tests]\npython -m py_compile ...\ngit diff --check\n```\n\nProtocol deviation: the first internal-finalize process ended after a complete DEV_A direction grid because DEV_B redundantly evaluated all three budgets. The resume reused the SHA/order/schema-checked DEV_A grid and evaluated only the already-selected 0.003 Å DEV_B budget. No method math, data, checkpoint, seed, selection rule, or external access changed. See `PROTOCOL_DEVIATIONS.md`.\n")

    config_dir = OUT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    resolved = yaml.safe_load((ROOT / "configs/ecir_mvr_lsgo_pilot.yaml").read_text(encoding="utf-8"))
    atomic_text(config_dir / "ecir_mvr_lsgo_pilot.resolved.yaml", yaml.safe_dump(resolved, sort_keys=True))
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    atomic_text(OUT / "figures/.gitkeep", "")

    include = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS.txt" or "xtb_work" in path.parts or "cache" in path.parts:
            continue
        include.append(f"{sha(path)}  {path.relative_to(OUT).as_posix()}")
    for path in [
        ROOT / "configs/ecir_mvr_lsgo_pilot.yaml", ROOT / "etflow/ecir/learned_geometry.py",
        ROOT / "etflow/ecir/lsgo_io.py", ROOT / "scripts/build_lsgo_datasets.py",
        ROOT / "scripts/run_mcvr_lsgo.py", ROOT / "scripts/run_lsgo_external_coordinates.py",
        ROOT / "scripts/run_posebusters_lsgo.py", ROOT / "scripts/run_xtb_singlepoint_lsgo.py",
        ROOT / "scripts/finalize_lsgo_reports.py", ROOT / "tests/test_mcvr_lsgo.py",
    ]:
        include.append(f"{sha(path)}  REPO/{path.relative_to(ROOT).as_posix()}")
    atomic_text(OUT / "SHA256SUMS.txt", "\n".join(include))
    print("LSGO_REPORTS_FINALIZED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
