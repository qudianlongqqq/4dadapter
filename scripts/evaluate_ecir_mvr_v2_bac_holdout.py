#!/usr/bin/env python3
"""Evaluate at most two frozen BAC candidates once on validation holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etflow.ecir.bac_evaluation import (  # noqa: E402
    evaluate_bac_candidate,
    summary_json,
)
from etflow.ecir.chemical_validity import ChemicalValidity  # noqa: E402
from etflow.ecir.mvr_v2_bac import MCVRBACModel  # noqa: E402
from etflow.ecir.run_a_evaluation import build_items  # noqa: E402


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v2_bac_overnight"),
    )
    parser.add_argument("--candidates", nargs="+", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= len(args.candidates) <= 2:
        raise ValueError("holdout permits one or two frozen candidates")
    formal_root = args.formal_root.expanduser().resolve()
    source_cache_root = args.source_cache_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    lock_path = output / "validation_holdout_summary.json"
    previous = json.loads(lock_path.read_text(encoding="utf-8"))
    if previous.get("status") != "NOT_RUN":
        raise RuntimeError("validation holdout has already been evaluated")
    cohorts = json.loads(
        (output / "validation_cohorts.json").read_text(encoding="utf-8")
    )
    holdout_molecules = set(cohorts["validation_holdout"]["molecule_ids"])
    source_metadata = json.loads(
        (formal_root / "real_sources" / "metadata.json").read_text(encoding="utf-8")
    )
    selected = []
    for path in args.candidates:
        checkpoint_path = path.expanduser().resolve()
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        config = checkpoint["config"]
        if int(checkpoint.get("step", -1)) != 2000:
            raise RuntimeError(f"candidate is not frozen at step 2000: {path}")
        selected.append(
            {
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": _sha(checkpoint_path),
                "checkpoint": checkpoint,
                "config": config,
                "experiment_id": config["experiment_id"],
                "mode": config["mode"],
            }
        )
    _append_jsonl(
        output / "decision_log.jsonl",
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_id": "validation_holdout_once",
            "parent_experiment": "matched_2k_tune_selection",
            "hypothesis": "A preserves the strongest Bond baseline; D tests unified BAC generalization",
            "selected_candidates": [
                {
                    "experiment_id": value["experiment_id"],
                    "mode": value["mode"],
                    "checkpoint_sha256": value["checkpoint_sha256"],
                }
                for value in selected
            ],
            "why_changed": "No model change; one-time holdout evaluation after frozen tune selection",
            "expected_outcome": "hard constraints pass without rollback-only apparent gain",
            "stop_criteria": ["failure", "identity", "chirality", "ring", "nonfinite"],
            "data_identities": {
                "formal_source": source_metadata["formal_source_identity_sha256"],
                "validation_cohort": cohorts["identity_sha256"],
            },
            "test_records_read": 0,
            "test_assets_opened": False,
            "validation_only": True,
        },
    )
    sources = pd.read_parquet(formal_root / "real_sources" / "val.parquet")
    targets = pd.read_parquet(formal_root / "minimal_targets" / "val.parquet")
    sources = sources[sources.molecule_id.astype(str).isin(holdout_molecules)]
    targets = targets[targets.sample_id.isin(sources.sample_id)]
    if sources.molecule_id.nunique() != 1000 or len(sources) != 2000:
        raise RuntimeError("holdout cohort identity/count mismatch")
    runtime = output / "runtime_manifests"
    source_path = runtime / "validation_holdout_sources.parquet"
    target_path = runtime / "validation_holdout_targets.parquet"
    sources.to_parquet(source_path, index=False)
    targets.to_parquet(target_path, index=False)
    validity = ChemicalValidity("data/ecir_mvr/validity_reference_stats.json")
    items = build_items(
        source_path,
        target_path,
        validity,
        source_cache_root=source_cache_root,
        target_cache_root=formal_root / "minimal_targets",
    )
    device = torch.device(args.device)
    results = []
    for value in selected:
        model = MCVRBACModel(**value["config"]["model"]).to(device)
        incompatible = model.load_state_dict(
            value["checkpoint"]["model_state_dict"], strict=True
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("holdout checkpoint strict-load mismatch")
        evaluation = evaluate_bac_candidate(
            model,
            items,
            validity,
            device=device,
            inference=value["config"]["inference"],
            source_identity_sha256=source_metadata["formal_source_identity_sha256"],
            bootstrap_draws=500,
        )
        candidate_dir = output / "holdout" / value["experiment_id"]
        candidate_dir.mkdir(parents=True, exist_ok=False)
        evaluation["records"].to_csv(
            candidate_dir / "validation_per_record.csv", index=False
        )
        evaluation["molecules"].to_csv(
            candidate_dir / "validation_per_molecule.csv", index=False
        )
        evaluation["summary"].to_csv(
            candidate_dir / "validation_summary.csv", index=False
        )
        _write_json(candidate_dir / "validation_summary.json", summary_json(evaluation))
        results.append(
            {
                "experiment_id": value["experiment_id"],
                "mode": value["mode"],
                "checkpoint_sha256": value["checkpoint_sha256"],
                "metrics": evaluation["metrics"],
                "evaluation_count": 1,
            }
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {
        "schema_version": "mcvr-v2-bac-validation-holdout-v1",
        "status": "COMPLETED_FROZEN_NO_FURTHER_TUNING",
        "cohort_identity_sha256": cohorts["identity_sha256"],
        "records": len(sources),
        "molecules": int(sources.molecule_id.nunique()),
        "candidates": results,
        "maximum_candidates": 2,
        "evaluation_count_per_candidate": 1,
        "further_tuning_permitted": False,
        "test_records_read": 0,
        "test_assets_opened": False,
        "validation_only": True,
    }
    _write_json(lock_path, summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
