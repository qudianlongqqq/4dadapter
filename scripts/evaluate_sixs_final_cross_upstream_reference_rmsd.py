#!/usr/bin/env python
"""Reference RMSD for the frozen final SIXS-U cross-upstream coordinates.

This adapter deliberately reuses the already-audited AvgFlow/DiTMC reference
correspondence reconstruction.  It changes only the evaluated method set to
the current frozen SIXS-U checkpoints/coordinates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap


ROOT = bootstrap()
OLD_ROOT = Path(r"E:\3dconformergenerationcode\4dadapter-lsgoba-v2-softplus-multiseed")
OLD_EVALUATOR = OLD_ROOT / "scripts/evaluate_cross_upstream_reference_rmsd.py"
OLD_CORRESPONDENCE = (
    OLD_ROOT
    / "reports/ecir_mvr/lsgoba_softplus_v2_final_cross_upstream/correspondence_reconstruction"
)
REPORT = ROOT / "reports/ecir_mvr/sixs_final_cross_upstream_unrestricted"
ASSET = Path(r"E:\3dconformergenerationcode\dataset\sixs_final_cross_upstream_unrestricted")
SEEDS = (307, 331, 353)
EXPECTED_CHECKPOINTS = {
    "307": "f63e9d796cc82297f2f2d5fd732c35aa80421ce2f604f2ac80d823a9f825b704",
    "331": "b9d655d14c7bc97dc6f54d1dd00e8bf84e40187cdf5d1518ffaaa517b69886af",
    "353": "e3be9e2ecf4d633cb2e5f7e3fff4ffd94b38fda13a1412013afd5f4f07450bc6",
}


def methods(upstream: str) -> tuple[str, ...]:
    prefix = upstream.upper()
    return (f"{prefix}_RAW", *(f"{prefix}_SIXS_U_SEED{seed}" for seed in SEEDS))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def load_audited_evaluator():
    spec = importlib.util.spec_from_file_location("audited_cross_upstream_reference", OLD_EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.CORRESPONDENCE = OLD_CORRESPONDENCE
    module.ARTIFACTS = ASSET
    module.SEEDS = SEEDS
    module.EXPECTED_CHECKPOINTS = EXPECTED_CHECKPOINTS
    module.methods = methods

    def seed_summary(summaries, upstream: str, _variant: str):
        per_seed = {
            str(seed): float(summaries[f"{upstream.upper()}_SIXS_U_SEED{seed}"]["mean"])
            for seed in SEEDS
        }
        values = pd.Series(per_seed, dtype=float)
        return {
            "per_seed_mean": per_seed,
            "mean": float(values.mean()),
            "std_sample": float(values.std(ddof=1)),
        }

    module.seed_summary = seed_summary
    return module


def evaluate(upstream: str) -> None:
    module = load_audited_evaluator()
    frame, summary, failures = module.avgflow() if upstream == "avgflow" else module.ditmc()
    if failures:
        raise RuntimeError(f"{upstream} audited Reference RMSD has {len(failures)} failures")
    expected_methods = methods(upstream)
    if tuple(summary["methods"].keys()) != expected_methods:
        raise RuntimeError(f"{upstream} evaluated method identity/order changed")

    artifact = ASSET / upstream
    output = frame.rename(
        columns={"sample_id": "record_id", "reference_rmsd": "reference_rmsd_angstrom"}
    )
    output["source_rmsd_angstrom"] = 0.0
    diagnostics = pd.read_parquet(artifact / "COORDINATE_DIAGNOSTICS.parquet")
    for seed in SEEDS:
        method = f"{upstream.upper()}_SIXS_U_SEED{seed}"
        lookup = diagnostics.loc[diagnostics.seed == seed].set_index("record_id")["source_rmsd_raw"]
        selected = output.method == method
        values = output.loc[selected, "record_id"].map(lookup)
        if values.isna().any():
            raise RuntimeError(f"{upstream}/seed{seed} source-RMSD identity join failed")
        output.loc[selected, "source_rmsd_angstrom"] = values.to_numpy()

    record_path = artifact / "FIDELITY_PER_RECORD.parquet"
    summary_path = artifact / "FIDELITY_SUMMARY.csv"
    completion_path = artifact / "FIDELITY_COMPLETION.json"
    atomic_frame(record_path, output)
    rows = []
    for method in expected_methods:
        part = output[output.method == method]
        rows.append(
            {
                "method": method,
                "reference_records": int(len(part)),
                "source_rmsd_mean": float(part.source_rmsd_angstrom.mean()),
                "source_rmsd_median": float(part.source_rmsd_angstrom.median()),
                "reference_rmsd_mean": float(part.reference_rmsd_angstrom.mean()),
                "reference_rmsd_median": float(part.reference_rmsd_angstrom.median()),
                "COV_P": "NOT_EVALUATED",
                "COV_R": "NOT_EVALUATED",
                "AMR_P": "NOT_EVALUATED",
                "AMR_R": "NOT_EVALUATED",
            }
        )
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    atomic_json(
        completion_path,
        {
            "schema_version": "sixs-final-cross-upstream-audited-reference-rmsd-v1",
            "status": "COMPLETE",
            "upstream": upstream,
            "reference_records_per_method": int(len(output) // len(expected_methods)),
            "methods": list(expected_methods),
            "reference_role": "evaluation_only",
            "reference_used_for_inference": False,
            "protocol": summary["protocol"],
            "correspondence_manifest": str(
                OLD_CORRESPONDENCE / f"{upstream.upper()}_REFERENCE_CORRESPONDENCE_V1.jsonl"
            ),
            "correspondence_manifest_sha256": module.sha256(
                OLD_CORRESPONDENCE / f"{upstream.upper()}_REFERENCE_CORRESPONDENCE_V1.jsonl"
            ),
            "audited_mapping_n": summary["audited_mapping_n"],
            "same_record_set_all_methods": True,
            "reference_summary": summary,
            "ensemble_cov_amr": "NOT_APPLICABLE" if upstream == "ditmc" else "NOT_EVALUATED",
            "model_changed": False,
            "coordinates_changed": False,
            "training_started": False,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, choices=("avgflow", "ditmc"))
    args = parser.parse_args()
    evaluate(args.upstream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
