#!/usr/bin/env python
"""Consolidate only already-completed SIXS ablations with explicit identity scope."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/ecir_mvr/sixs_final_ablation"


def values(row, mapping):
    return {target: row.get(source, np.nan) for target, source in mapping.items()}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    columns = {
        "V3D": "proposal_V3D", "PB": "proposal_PB", "Internal_post": "internal_post",
        "Direction_improve": "direction_improvement", "Bond_MAE": "bond_raw_mae",
        "Angle_MAE": "angle_raw_mae", "Source_RMSD": "source_rmsd",
    }
    rows = []
    factorial = pd.read_csv(ROOT / "reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/LIVE_RESULTS.csv")
    for arm in ("J1-R0", "J1-R1"):
        row = factorial[factorial.arm == arm].iloc[0]
        rows.append({"Variant": f"Reliability_{'OFF' if arm.endswith('R0') else 'ON'}__{arm}", "Question": "Reliability", "Evidence_scope": "HISTORICAL_J1_EQUAL_BA_FIXED_MOVEMENT_NOT_EXACT_FINAL", "Current_final_model_compatible": False, **values(row, columns)})
    joint = pd.read_csv(ROOT / "reports/ecir_mvr/sixs_j1r1_joint_magnitude_interaction_seed307/FOUR_CELL_SUMMARY.csv")
    for arm in ("J1-R1-FIXED", "J1-R1-JOINT"):
        row = joint[joint.arm == arm].iloc[0]
        rows.append({"Variant": arm, "Question": "Learned movement", "Evidence_scope": "HISTORICAL_J1_R1_EQUAL_BA_NESTED_CONTROL", "Current_final_model_compatible": False, **values(row, columns)})
    adaptive = pd.read_csv(ROOT / "reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/DEV_SUMMARY.csv").iloc[0]
    rows.append({"Variant": "J1_R1_ADAPTIVE_BA_RESTRICTED_SEED307", "Question": "Adaptive BA", "Evidence_scope": "CURRENT_FINAL_LINEAGE_RESTRICTED_SEED307", "Current_final_model_compatible": True, **values(adaptive, columns)})
    primary = pd.read_csv(ROOT / "reports/ecir_mvr/sixs_primary_final_evaluation/03_FINAL_METHOD_SUMMARY.csv")
    for formulation, prefix in (("Restricted", "restricted_seed"), ("Unrestricted", "unrestricted_seed")):
        part = primary[primary.method.str.startswith(prefix)]
        rows.append({
            "Variant": f"FINAL_{formulation.upper()}_THREE_SEED_MEAN", "Question": "Restricted vs Unrestricted",
            "Evidence_scope": "EXACT_FINAL_PRIMARY_2500_THREE_SEED", "Current_final_model_compatible": True,
            "V3D": part.v3d_overall.mean(), "PB": part.posebusters_overall.mean(),
            "Internal_post": np.nan, "Direction_improve": np.nan,
            "Bond_MAE": part.bond_raw_mae.mean(), "Angle_MAE": part.angle_cosine_raw_mae.mean(),
            "Source_RMSD": part.raw_source_displacement_rms.mean(), "Reference_RMSD": part.reference_rmsd.mean(),
            "xTB_median_DeltaE": part.xtb_delta_median.mean(), "DeltaE_lower_fraction": part.xtb_delta_lower_fraction.mean(),
        })
    for variant, question in (("FINAL_RELIABILITY_OFF", "Reliability"), ("FINAL_EQUAL_BA", "Adaptive BA"), ("FINAL_BOND_ONLY", "Bond/Angle complementarity"), ("FINAL_ANGLE_ONLY", "Bond/Angle complementarity"), ("FINAL_FIXED_MOVEMENT", "Learned movement")):
        rows.append({"Variant": variant, "Question": question, "Evidence_scope": "NOT_FOUND_AS_EXACT_FINAL_FORMULATION_CONTROL", "Current_final_model_compatible": False, "Status": "NOT_RUN__NO_NEW_ABLATION_INVENTED"})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "FINAL_ABLATION_TABLE.csv", index=False)
    restricted = table[table.Variant == "FINAL_RESTRICTED_THREE_SEED_MEAN"].iloc[0]
    unrestricted = table[table.Variant == "FINAL_UNRESTRICTED_THREE_SEED_MEAN"].iloc[0]
    conclusion = f"""# Final ablation conclusion

Only already-completed artifacts were consolidated; no new ablation was invented or trained. Historical controls are retained with explicit scope and are not relabeled as exact-final controls.

```text
ABLATION_STATUS = PASS_REUSED_EXISTING_ARTIFACTS_WITH_SCOPE_GUARDS
RELIABILITY_EFFECT = POSITIVE_IN_HISTORICAL_J1_FACTORIAL__EXACT_FINAL_ISOLATION_NOT_AVAILABLE
ADAPTIVE_BA_EFFECT = POSITIVE_IN_CURRENT_RESTRICTED_SEED307_LINEAGE__THREE_SEED_EXACT_CONTROL_NOT_AVAILABLE
LEARNED_MOVEMENT_EFFECT = POSITIVE_IN_HISTORICAL_J1_R1_EQUAL_BA_NESTED_CONTROL__EXACT_FINAL_ISOLATION_NOT_AVAILABLE
BOND_ANGLE_COMPLEMENTARITY = HISTORICAL_SUPPORT__EXACT_FINAL_BOND_ONLY_ANGLE_ONLY_NOT_AVAILABLE
RESTRICTED_VS_UNRESTRICTED_EFFECT = V3D_DELTA_{unrestricted.V3D - restricted.V3D:+.6f}__PB_DELTA_{unrestricted.PB - restricted.PB:+.6f}__XTB_MEDIAN_DELTA_{unrestricted.xTB_median_DeltaE - restricted.xTB_median_DeltaE:+.6f}
NEW_TRAINING = NO
```
"""
    (OUT / "FINAL_ABLATION_CONCLUSION.md").write_text(conclusion, encoding="utf-8")
    (OUT / "FINAL_STATUS.json").write_text(json.dumps({"status": "PASS_REUSED_EXISTING_ARTIFACTS_WITH_SCOPE_GUARDS", "new_training": False, "exact_final_missing_controls": ["Reliability OFF", "Equal BA", "Bond-only", "Angle-only", "Fixed movement"], "table": str(OUT / "FINAL_ABLATION_TABLE.csv")}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
