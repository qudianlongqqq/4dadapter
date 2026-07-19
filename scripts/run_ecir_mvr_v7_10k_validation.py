#!/usr/bin/env python3
"""Run one frozen D1/V5-B/V7 method on the V7 10K development cohort."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etflow.ecir.bac_evaluation import evaluate_bac_candidate  # noqa: E402
from etflow.ecir.chemical_validity import ChemicalValidity  # noqa: E402
from etflow.ecir.mvr_v2_bac import MCVRBACModel  # noqa: E402
from etflow.ecir.mvr_v5_constraint_hybrid import (  # noqa: E402
    MCVRNeuralJacobianHybrid,
)
from etflow.ecir.run_a_evaluation import (  # noqa: E402
    _load_source_coordinates,
    graph_data,
    nearest_rmsd,
)
from scripts.run_ecir_mvr_v2_bac_pilots import _seed  # noqa: E402
from scripts.run_ecir_mvr_v7_constraint_specific import _build_model  # noqa: E402


SEED = 43018
D1_CHECKPOINT_SHA256 = (
    "9348744817ef7eec6d9d682dd95a35f0be86f0565b6dd060e8d5fe54e609e426"
)
EXPECTED_SELECTION = {
    "molecules": 10_000,
    "records": 30_000,
    "ordered_molecule_ids_sha256": (
        "17f19269598d7985b16bd0beb82f8e00f0401b2a44ba91c42b631bdc8489bf78"
    ),
    "ordered_sample_ids_sha256": (
        "880c68ced3e8f3e74b9aa44a207ea1abbc0715776e542298d06514840695c0a3"
    ),
}
FROZEN_FILES = {
    "etflow/ecir/mvr_v7_constraint_specific.py": (
        "74bde53ee2ab1ac22137f90c66bfa3d25a5c8fe97141731b99c4f37656fc711f"
    ),
    "scripts/run_ecir_mvr_v7_constraint_specific.py": (
        "3d515b337c67e1089c9b58ecd2dadc698ed1fb1ef31e6663dffa90eb0f3d6887"
    ),
    "scripts/report_ecir_mvr_v7_constraint_specific.py": (
        "401c688883e4c603386e066b4b478e2c3df68490812256014bd29412a118af51"
    ),
    "etflow/ecir/bac_evaluation.py": (
        "19c58781dbde09df7c29a9d4856436b192540be58381e545a13366a67dff2f63"
    ),
    "etflow/ecir/run_a_evaluation.py": (
        "b890459edc7244047a0d2c7547681523315f1ccc95778f753625aba05670576d"
    ),
    "etflow/ecir/bac_jacobian.py": (
        "a405ebbf0ab99128abe93fc2ecb1d5ec432409f5beedbd78c0f528bcd0603a00"
    ),
    "etflow/ecir/bac_constraints.py": (
        "cf084afd4c23d00faac66b0194d2bb4ec125e7cfaa0d51c44100da480caa83cf"
    ),
    "etflow/ecir/bac_safety.py": (
        "51a5972f9e3d2032baaee31e574b1683ece0170460bec5e521104aecfc6b3c37"
    ),
    "etflow/ecir/chemical_validity.py": (
        "03a6c64ba1e275198d955fc83a683a704d7c63c8c659e5deb7ca9c5eac9f63d6"
    ),
    "data/ecir_mvr/validity_reference_stats.json": (
        "ae5afaa8d3fce1b5418295309bf2c3197997180298e1781b4efc5c265258852e"
    ),
    "etflow/ecir/mvr_v2_bac.py": (
        "b0441c9852b38a7bd603a480fab7d5c56851f08224d3eed4e440fabdbd3c1439"
    ),
    "etflow/ecir/mvr_v5_constraint_hybrid.py": (
        "46b9c0810f8e490039ea3c86ea5d09439a0287753dba28b89863405723b954d3"
    ),
}
CONFIG_SHA = {
    "V5-B": "d1e70583f77d98e95194fe7ee06eac797da4cc268ef875d54a267295eef92a41",
    "V7": "cea71c6d9e5c12565707a127c7b7390f4caed7d22611fadaebb1fab321cfa645",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("D1", "V5-B", "V7"), required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v7_10k/manifests"),
    )
    parser.add_argument(
        "--d1-checkpoint",
        type=Path,
        default=Path(
            "diagnostics/ecir_mvr/v2_bac_recovery/runs/"
            "d1_pilot_1000step_seed43018/checkpoint_final.ckpt"
        ),
    )
    parser.add_argument(
        "--v5-config",
        type=Path,
        default=Path(
            "diagnostics/ecir_mvr/v5_constraint_hybrid/runs/"
            "v5_b_pilot_seed43018/config.resolved.yaml"
        ),
    )
    parser.add_argument(
        "--v7-config",
        type=Path,
        default=Path(
            "diagnostics/ecir_mvr/v7_constraint_specific/runs/"
            "v7_constraint_specific_pilot_seed43018/config.resolved.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v7_10k/runs"),
    )
    parser.add_argument("--molecules-per-chunk", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _verify_frozen_files() -> dict[str, str]:
    identities = {}
    for relative, expected in FROZEN_FILES.items():
        actual = _sha(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"frozen V7/evaluator SHA mismatch: {relative}: {actual}")
        identities[relative] = actual
    return identities


def _verify_manifest(manifest_dir: Path) -> dict[str, Any]:
    manifest = json.loads((manifest_dir / "manifest.json").read_text(encoding="utf-8"))
    stable = {
        key: value
        for key, value in manifest.items()
        if key not in {"identity_sha256", "created_at", "source_manifest", "target_manifest"}
    }
    if _canonical_sha(stable) != manifest["identity_sha256"]:
        raise RuntimeError("V7 10K manifest canonical identity mismatch")
    for key, expected in EXPECTED_SELECTION.items():
        if manifest[key] != expected:
            raise RuntimeError(f"V7 10K manifest field changed: {key}")
    flags = {
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "formal_large_run": False,
        "training_performed": False,
        "target_rematerialization": False,
        "validation_only": True,
    }
    for key, expected in flags.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"V7 10K isolation field changed: {key}")
    source = manifest_dir / "development_sources.parquet"
    target = manifest_dir / "development_targets.parquet"
    if _sha(source) != manifest["source_manifest_sha256"]:
        raise RuntimeError("V7 10K derived source manifest SHA mismatch")
    if _sha(target) != manifest["target_manifest_sha256"]:
        raise RuntimeError("V7 10K derived target manifest SHA mismatch")
    return manifest


def _build_items(
    sources: pd.DataFrame,
    targets: pd.DataFrame,
    validity: ChemicalValidity,
    *,
    source_cache_root: Path,
    target_cache_root: Path,
) -> list[dict[str, Any]]:
    source = sources.sort_values(["molecule_id", "sample_id"]).copy()
    target = targets.copy()
    source["source_path"] = [
        str(source_cache_root / str(split) / Path(str(value)).name)
        for split, value in zip(source.split, source.source_path, strict=True)
    ]
    target["target_cache_path"] = [
        str(target_cache_root / str(split) / Path(str(value)).name)
        for split, value in zip(target.split, target.target_cache_path, strict=True)
    ]
    target = target.set_index("sample_id")
    items = []
    for row in source.itertuples(index=False):
        record, coordinates = _load_source_coordinates(row)
        target_row = target.loc[row.sample_id]
        payload = torch.load(
            Path(target_row.target_cache_path), map_location="cpu", weights_only=False
        )
        minimal = torch.as_tensor(payload["x_target"], dtype=torch.float32)
        references = torch.as_tensor(
            record.get("x_ref_candidates", record.get("x_ref_aligned")),
            dtype=torch.float32,
        )
        if references.ndim == 2:
            references = references.unsqueeze(0)
        values = validity.evaluate(coordinates, record, baseline_coordinates=coordinates)
        rotatable = int(record.get("num_rotatable_bonds", 0))
        has_ring = bool(torch.as_tensor(record.get("bond_is_in_ring", [])).any())
        clean_metrics = (
            "bond_outlier_rate",
            "angle_outlier_rate",
            "ring_bond_outlier_rate",
            "ring_planarity_outlier_rate",
            "clash_penetration",
            "severe_clash_rate",
            "stereocenter_degenerate_rate",
        )
        clean = all(values[name] <= 0.0 for name in clean_metrics)
        active = torch.tensor(
            [
                float(values["bond_outlier_rate"] > 0),
                float(values["angle_outlier_rate"] > 0),
                float(
                    values["ring_bond_outlier_rate"] > 0
                    or values["ring_planarity_outlier_rate"] > 0
                ),
                float(
                    values["clash_penetration"] > 0
                    or values["severe_clash_rate"] > 0
                ),
                float(values["torsion_prior_outlier_score"] > 4.0),
                float(clean),
            ]
        )
        groups = ["all", "ETFlow_normal"]
        groups.append(
            "rotatable_le_2"
            if rotatable <= 2
            else ("rotatable_3_5" if rotatable <= 5 else "rotatable_ge_6")
        )
        groups.append("ring" if has_ring else "non_ring")
        if clean:
            groups.append("clean_valid")
        items.append(
            {
                "row": row,
                "record": record,
                "input": coordinates,
                "minimal_target": minimal,
                "references": references,
                "input_validity": values,
                "input_rmsd": nearest_rmsd(coordinates, references),
                "data": graph_data(record, coordinates, row, active_mode_mask=active),
                "groups": groups,
                "rotatable": rotatable,
                "has_ring": has_ring,
                "clean": clean,
            }
        )
    return items


def _load_model(
    method: str,
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    if _sha(args.v7_config) != CONFIG_SHA["V7"]:
        raise RuntimeError("V7 10K frozen inference config SHA mismatch")
    prior = MCVRBACModel(**checkpoint["config"]["model"])
    incompatible = prior.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("V7 10K D1 strict load failed")
    if method == "D1":
        config = yaml.safe_load(args.v7_config.read_text(encoding="utf-8"))
        return prior.to(device), config, _sha(args.v7_config)
    if method == "V5-B":
        if _sha(args.v5_config) != CONFIG_SHA[method]:
            raise RuntimeError("V7 10K V5-B config SHA mismatch")
        config = yaml.safe_load(args.v5_config.read_text(encoding="utf-8"))
        settings = dict(config["prototype_b"])
        jacobian = settings.pop("jacobian")
        model = MCVRNeuralJacobianHybrid(
            prior, jacobian_config=jacobian, **settings
        ).to(device)
        return model, config, CONFIG_SHA[method]
    config = yaml.safe_load(args.v7_config.read_text(encoding="utf-8"))
    return _build_model(checkpoint, config, device), config, CONFIG_SHA[method]


def _merge_solver(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    for summary in summaries:
        statuses.update(summary["status_counts"])
    solved = statuses.get("SOLVED", 0)
    calls = sum(int(summary["calls"]) for summary in summaries)
    failures = sum(int(summary["solver_failure_count"]) for summary in summaries)
    return {
        "calls": calls,
        "status_counts": dict(statuses),
        "inactive_constraint_calls": statuses.get("NO_ACTIVE_CONSTRAINT", 0),
        "solver_failure_count": failures,
        "solver_failure_rate": failures / max(calls, 1),
        "effective_rank_mean": (
            sum(
                float(summary["effective_rank_mean"])
                * int(summary["status_counts"].get("SOLVED", 0))
                for summary in summaries
            )
            / max(solved, 1)
        ),
        "condition_number_mean": (
            sum(
                float(summary["condition_number_mean"])
                * int(summary["status_counts"].get("SOLVED", 0))
                for summary in summaries
            )
            / max(solved, 1)
        ),
        "condition_number_max": max(
            (float(summary["condition_number_max"]) for summary in summaries),
            default=0.0,
        ),
        "singular_value_max": max(
            (float(summary["singular_value_max"]) for summary in summaries),
            default=0.0,
        ),
        "singular_value_min_retained": min(
            (
                float(summary["singular_value_min_retained"])
                for summary in summaries
                if float(summary["singular_value_min_retained"]) > 0.0
            ),
            default=0.0,
        ),
        "truncated_direction_count": sum(
            int(summary["truncated_direction_count"]) for summary in summaries
        ),
    }


def _merge_components(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    calls = sum(int(summary["calls"]) for summary in summaries)
    names = set(summaries[0]) - {"calls"} if summaries else set()
    return {
        "calls": calls,
        **{
            name: sum(
                float(summary[name]) * int(summary["calls"]) for summary in summaries
            )
            / max(calls, 1)
            for name in sorted(names)
        },
    }


def _metrics(records: pd.DataFrame, molecules: pd.DataFrame) -> dict[str, float]:
    all_rows = molecules[molecules.group == "all"].groupby("method").mean(numeric_only=True)
    candidate = all_rows.loc["v2_bac_accepted"]
    upstream = all_rows.loc["upstream"]
    return {
        "bond_delta": float(candidate.bond_outlier_rate - upstream.bond_outlier_rate),
        "angle_delta": float(candidate.angle_outlier_rate - upstream.angle_outlier_rate),
        "clash_delta": float(candidate.clash_penetration - upstream.clash_penetration),
        "ring_delta": float(
            candidate.ring_bond_outlier_rate - upstream.ring_bond_outlier_rate
        ),
        "rmsd_delta": float(candidate.aligned_RMSD - upstream.aligned_RMSD),
        "mat_p_delta": float(candidate.MAT_P - upstream.MAT_P),
        "mat_r_delta": float(candidate.MAT_R - upstream.MAT_R),
        "cov_p_delta": float(candidate.COV_P - upstream.COV_P),
        "cov_r_delta": float(candidate.COV_R - upstream.COV_R),
        "accepted_fraction": float(candidate.accepted),
        "rollback_fraction": 1.0 - float(candidate.accepted),
        "mean_displacement": float(candidate.molecule_rms_displacement),
        "failure_rate": 0.0,
        "records": float(records[records.method == "v2_bac_accepted"].shape[0]),
    }


def main() -> None:
    args = parse_args()
    if args.molecules_per_chunk < 1 or args.batch_size < 1:
        raise ValueError("V7 10K chunk and batch sizes must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested V7 10K CUDA device is unavailable")
    for name in (
        "formal_root",
        "source_cache_root",
        "manifest_dir",
        "d1_checkpoint",
        "v5_config",
        "v7_config",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    frozen = _verify_frozen_files()
    if _sha(args.d1_checkpoint) != D1_CHECKPOINT_SHA256:
        raise RuntimeError("V7 10K D1 checkpoint SHA mismatch")
    manifest = _verify_manifest(args.manifest_dir)
    run_dir = args.output_dir / args.method.lower().replace("-", "_")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite V7 10K run: {run_dir}")
    run_dir.mkdir(parents=True)
    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir()
    launch = {
        "schema_version": "mcvr-v7-10k-launch-v1",
        "method": args.method,
        "pid": os.getpid(),
        "command": " ".join(sys.argv),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "manifest_identity_sha256": manifest["identity_sha256"],
        "frozen_file_identities": frozen,
        "checkpoint_sha256": D1_CHECKPOINT_SHA256,
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "formal_large_run": False,
        "training_performed": False,
        "target_rematerialization": False,
        "validation_only": True,
    }
    _write_json(run_dir / "launch.json", launch)

    sources = pd.read_parquet(args.manifest_dir / "development_sources.parquet")
    targets = pd.read_parquet(args.manifest_dir / "development_targets.parquet")
    if len(sources) != 30_000 or sources.molecule_id.nunique() != 10_000:
        raise RuntimeError("V7 10K runtime cohort size changed")
    if bool(sources.test_record.astype(bool).any()) or int(targets.test_records_read.max()) != 0:
        raise RuntimeError("V7 10K runtime manifest violates test isolation")
    targets_by_sample = targets.set_index("sample_id", drop=False)
    checkpoint = torch.load(args.d1_checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    model, method_config, method_config_sha = _load_model(
        args.method, checkpoint, args, device
    )
    inference = dict(method_config["inference"])
    inference["batch_size"] = int(args.batch_size)
    frozen_v7 = yaml.safe_load(args.v7_config.read_text(encoding="utf-8"))["inference"]
    if {**inference, "batch_size": frozen_v7["batch_size"]} != frozen_v7:
        raise RuntimeError("V7 10K inference settings differ from frozen V7")
    validity = ChemicalValidity("data/ecir_mvr/validity_reference_stats.json")
    _seed(SEED)
    model.eval()
    molecule_ids = sorted(map(str, sources.molecule_id.unique()))
    record_paths = []
    molecule_paths = []
    solver_summaries = []
    component_summaries = []
    started = time.monotonic()
    total_chunks = math.ceil(len(molecule_ids) / args.molecules_per_chunk)
    for chunk_index, start in enumerate(
        range(0, len(molecule_ids), args.molecules_per_chunk), start=1
    ):
        selected_ids = set(molecule_ids[start : start + args.molecules_per_chunk])
        chunk_sources = sources[sources.molecule_id.astype(str).isin(selected_ids)]
        chunk_targets = targets_by_sample.loc[chunk_sources.sample_id].reset_index(drop=True)
        items = _build_items(
            chunk_sources,
            chunk_targets,
            validity,
            source_cache_root=args.source_cache_root,
            target_cache_root=args.formal_root / "minimal_targets",
        )
        if args.method == "V7":
            model.reset_statistics()
        elif args.method == "V5-B":
            model.reset_solver_statistics()
        evaluation = evaluate_bac_candidate(
            model,
            items,
            validity,
            device=device,
            inference=inference,
            source_identity_sha256=manifest["formal_source_identity_sha256"],
            bootstrap_draws=1,
        )
        record_path = chunks_dir / f"records_{chunk_index:04d}.csv"
        molecule_path = chunks_dir / f"molecules_{chunk_index:04d}.csv"
        evaluation["records"].to_csv(record_path, index=False)
        evaluation["molecules"].to_csv(molecule_path, index=False)
        record_paths.append(record_path)
        molecule_paths.append(molecule_path)
        chunk_summary: dict[str, Any] = {
            "chunk": chunk_index,
            "total_chunks": total_chunks,
            "molecules": len(selected_ids),
            "records": len(items),
            "elapsed_seconds": time.monotonic() - started,
        }
        if args.method == "V7":
            solver = model.angle_solver_summary()
            components = model.component_summary()
            solver_summaries.append(solver)
            component_summaries.append(components)
            chunk_summary["angle_solver"] = solver
            chunk_summary["components"] = components
        _write_json(chunks_dir / f"summary_{chunk_index:04d}.json", chunk_summary)
        print(json.dumps(chunk_summary, sort_keys=True), flush=True)
        del items, evaluation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    records = pd.concat([pd.read_csv(path) for path in record_paths], ignore_index=True)
    molecules = pd.concat([pd.read_csv(path) for path in molecule_paths], ignore_index=True)
    records.to_csv(run_dir / "development_per_record.csv", index=False)
    molecules.to_csv(run_dir / "development_per_molecule.csv", index=False)
    metrics = _metrics(records, molecules)
    angle_solver = _merge_solver(solver_summaries) if solver_summaries else None
    components = _merge_components(component_summaries) if component_summaries else None
    runtime = time.monotonic() - started
    metadata = {
        "schema_version": "mcvr-v7-10k-method-run-v1",
        "status": "COMPLETED",
        "method": args.method,
        "seed": SEED,
        "molecules": 10_000,
        "records": 30_000,
        "chunks": total_chunks,
        "molecules_per_chunk": args.molecules_per_chunk,
        "evaluation_seconds": runtime,
        "metrics": metrics,
        "angle_solver": angle_solver,
        "components": components,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "method_config_sha256": method_config_sha,
        "checkpoint_sha256": D1_CHECKPOINT_SHA256,
        "checkpoint_strict_load": True,
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "formal_large_run": False,
        "training_performed": False,
        "target_rematerialization": False,
        "validation_only": True,
    }
    _write_json(run_dir / "run_metadata.json", metadata)
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
