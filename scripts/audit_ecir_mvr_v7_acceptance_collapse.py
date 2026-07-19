#!/usr/bin/env python3
"""Run fixed-cohort trajectory and rollback audits without test access."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etflow.ecir.acceptance import displacement_metrics  # noqa: E402
from etflow.ecir.bac_evaluation import evaluate_bac_candidate  # noqa: E402
from etflow.ecir.chemical_validity import ChemicalValidity  # noqa: E402
from etflow.ecir.mvr_v7_formal import (  # noqa: E402
    file_sha256,
    load_v7_formal_config,
)
from etflow.ecir.run_a_evaluation import infer_mvr, method_rows  # noqa: E402
from scripts.run_ecir_mvr_v2_bac_pilots import _seed  # noqa: E402
from scripts.run_ecir_mvr_v7_10k_validation import (  # noqa: E402
    D1_CHECKPOINT_SHA256,
    _build_items,
    _load_model,
    _verify_manifest,
)
from scripts.run_ecir_mvr_v7_formal_validation import (  # noqa: E402
    FROZEN_SEEDS,
    SOURCE_IDENTITY_SHA256,
    SOURCE_MANIFEST_SHA256,
    TARGET_MANIFEST_SHA256,
    V5_CONFIG_SHA256,
    V7_CONFIG_SHA256,
    _load_models,
    _validate_cohort_frames,
)


METHODS = ("D1", "V5-B", "V7")
SEMANTICS = ("legacy_bac", "formal_d1b")
ISOLATION = {
    "test_records_read": 0,
    "test_assets_opened": False,
    "frozen_holdout_records_opened": 0,
    "formal_test_run": False,
    "training_performed": False,
}


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _metric(values: Mapping[str, Any], name: str) -> float:
    return float(values.get(name, 0.0))


def _norm(values: Mapping[str, Any], name: str) -> float:
    return float(values.get(name, 0.0))


def _audit_row(
    *,
    cohort: str,
    semantics: str,
    method: str,
    item: Mapping[str, Any],
    target_row: Any,
    metadata: Mapping[str, Any],
    checkpoint_sha: str,
) -> dict[str, Any]:
    source_metrics = dict(metadata.get("source_metrics", item["input_validity"]))
    proposal_metrics = dict(metadata.get("proposal_metrics", {}))
    final_metrics = dict(metadata.get("final_metrics", {}))
    failed = set(filter(None, str(metadata.get("all_failed_checks", "")).split(";")))
    proposal_displacement = dict(metadata.get("proposal_displacement", {}))
    accepted_displacement = dict(metadata.get("accepted_displacement", {}))
    data = item["data"]
    bonds = torch.as_tensor(
        getattr(data, "active_bond_constraint_index", torch.empty(2, 0))
    ).reshape(2, -1)
    angles = torch.as_tensor(
        getattr(data, "active_angle_constraint_index", torch.empty(0, 3))
    ).reshape(-1, 3)
    source_target = displacement_metrics(item["input"], item["minimal_target"])
    neural = dict(metadata.get("neural_delta", {}))
    angle = dict(metadata.get("angle_delta", {}))
    clash = dict(metadata.get("clash_delta", {}))
    fused = dict(metadata.get("fused_delta", {}))
    return {
        "cohort": cohort,
        "semantics": semantics,
        "seed": int(metadata.get("seed", 0)),
        "sample_id": str(item["row"].sample_id),
        "molecule_id": str(item["row"].molecule_id),
        "record_id": str(item["row"].sample_id),
        "method": method,
        "checkpoint_sha256": checkpoint_sha,
        "source_file": str(item["row"].source_path),
        "target_file": str(target_row.target_cache_path),
        "atom_count": int(item["input"].shape[0]),
        "active_bond_count": int(bonds.shape[1]),
        "active_angle_count": int(angles.shape[0]),
        "active_clash_count": int(
            _metric(source_metrics, "clash_penetration") > 0.0
            or _metric(source_metrics, "severe_clash_rate") > 0.0
        ),
        "source_bond": _metric(source_metrics, "bond_outlier_rate"),
        "source_angle": _metric(source_metrics, "angle_outlier_rate"),
        "source_active_angle": _metric(source_metrics, "angle_outlier_rate"),
        "source_clash": _metric(source_metrics, "clash_penetration"),
        "source_ring": _metric(source_metrics, "ring_bond_outlier_rate"),
        "proposal_bond": _metric(proposal_metrics, "bond_outlier_rate"),
        "proposal_angle": _metric(proposal_metrics, "angle_outlier_rate"),
        "proposal_active_angle": _metric(proposal_metrics, "angle_outlier_rate"),
        "proposal_clash": _metric(proposal_metrics, "clash_penetration"),
        "proposal_ring": _metric(proposal_metrics, "ring_bond_outlier_rate"),
        "final_bond": _metric(final_metrics, "bond_outlier_rate"),
        "final_angle": _metric(final_metrics, "angle_outlier_rate"),
        "final_active_angle": _metric(final_metrics, "angle_outlier_rate"),
        "final_clash": _metric(final_metrics, "clash_penetration"),
        "final_ring": _metric(final_metrics, "ring_bond_outlier_rate"),
        "neural_delta_rms": _norm(neural, "rms"),
        "neural_delta_max": _norm(neural, "max"),
        "angle_delta_rms": _norm(angle, "rms"),
        "angle_delta_max": _norm(angle, "max"),
        "clash_delta_rms": _norm(clash, "rms"),
        "clash_delta_max": _norm(clash, "max"),
        "fused_proposal_rms": _norm(fused, "rms"),
        "fused_proposal_max": _norm(fused, "max"),
        "proposal_displacement_rms": _norm(
            proposal_displacement, "aligned_rms_displacement"
        ),
        "proposal_displacement_max": _norm(
            proposal_displacement, "max_atom_displacement"
        ),
        "accepted_delta_rms": _norm(
            accepted_displacement, "aligned_rms_displacement"
        ),
        "accepted_delta_max": _norm(
            accepted_displacement, "max_atom_displacement"
        ),
        "accepted_scale": float(metadata.get("selected_scale", 0.0)),
        "backtracking_attempts": int(metadata.get("backtracking_attempts", 0)),
        "accepted": bool(metadata.get("accepted", False)),
        "rollback_reason": str(metadata.get("primary_reject_reason", "")),
        "all_failed_checks": ";".join(sorted(failed)),
        "minimum_bac_gain_passed": "no_bac_gain" not in failed,
        "bond_safety_passed": "new_bond_violation" not in failed,
        "angle_safety_passed": "new_angle_violation" not in failed,
        "ring_safety_passed": "new_ring_violation" not in failed,
        "chirality_safety_passed": not bool(
            {"chirality_changed", "stereocenter_degenerated"} & failed
        ),
        "graph_rms_safety_passed": "molecule_trust_radius" not in failed,
        "max_atom_safety_passed": "atom_trust_radius" not in failed,
        "final_coordinate_equals_source": bool(
            metadata.get("final_coordinate_equals_source", False)
        ),
        "final_coordinate_equals_proposal": bool(
            metadata.get("final_coordinate_equals_proposal", False)
        ),
        "source_target_rms": source_target["aligned_rms_displacement"],
        "source_target_max": source_target["max_atom_displacement"],
    }


def _native_rows(
    model: torch.nn.Module,
    items: list[dict[str, Any]],
    validity: ChemicalValidity,
    *,
    device: torch.device,
    inference: Mapping[str, Any],
    cohort: str,
    checkpoint_sha: str,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw, accepted, metadata = infer_mvr(
        model,
        items,
        validity,
        device=device,
        steps=int(inference["teacher_steps"]),
        step_size=float(inference["step_size"]),
        batch_size=int(inference["batch_size"]),
        acceptance_mode="best_of_trajectory",
    )
    final = method_rows(
        items,
        {"native_d1b": accepted},
        validity,
        method_metadata={"native_d1b": metadata},
    )
    targets_by_sample = targets.set_index("sample_id")
    audit = []
    for item, raw_coordinates, final_coordinates, extra in zip(
        items, raw, accepted, metadata, strict=True
    ):
        source = item["input"]
        proposal_metrics = validity.evaluate(
            raw_coordinates, item["record"], baseline_coordinates=source
        )
        final_metrics = validity.evaluate(
            final_coordinates, item["record"], baseline_coordinates=source
        )
        proposal_displacement = displacement_metrics(source, raw_coordinates)
        accepted_displacement = displacement_metrics(source, final_coordinates)
        augmented = {
            **extra,
            "seed": 43 if cohort == "formal" else 43018,
            "source_metrics": item["input_validity"],
            "proposal_metrics": proposal_metrics,
            "final_metrics": final_metrics,
            "proposal_displacement": proposal_displacement,
            "accepted_displacement": accepted_displacement,
            "primary_reject_reason": str(extra.get("reject_reasons", "")).split(";")[0],
            "all_failed_checks": str(extra.get("reject_reasons", "")),
            "final_coordinate_equals_source": bool(
                torch.equal(final_coordinates, source)
            ),
            "final_coordinate_equals_proposal": bool(
                torch.equal(final_coordinates, raw_coordinates)
            ),
        }
        target = targets_by_sample.loc[str(item["row"].sample_id)]
        audit.append(
            _audit_row(
                cohort=cohort,
                semantics="native_d1b",
                method="D1",
                item=item,
                target_row=target,
                metadata=augmented,
                checkpoint_sha=checkpoint_sha,
            )
        )
    return final, {"audit": audit}


def _summary(records: pd.DataFrame, audit: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (cohort, semantics, method), subset in audit.groupby(
        ["cohort", "semantics", "method"], sort=True
    ):
        rows.append(
            {
                "cohort": cohort,
                "semantics": semantics,
                "method": method,
                "records": len(subset),
                "acceptance": float(subset.accepted.mean()),
                "proposal_displacement_rms": float(
                    subset.proposal_displacement_rms.mean()
                ),
                "accepted_displacement_rms": float(subset.accepted_delta_rms.mean()),
                "bond_delta": float((subset.final_bond - subset.source_bond).mean()),
                "angle_delta": float((subset.final_angle - subset.source_angle).mean()),
            }
        )
    del records
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("formal", "development"), required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--formal-checkpoint", type=Path, required=True)
    parser.add_argument("--formal-v7-config", type=Path, required=True)
    parser.add_argument("--v5-config", type=Path, required=True)
    parser.add_argument("--formal-sources", type=Path, required=True)
    parser.add_argument("--formal-targets", type=Path, required=True)
    parser.add_argument("--development-manifest-dir", type=Path, required=True)
    parser.add_argument("--development-checkpoint", type=Path, required=True)
    parser.add_argument("--development-v7-config", type=Path, required=True)
    parser.add_argument("--validity-statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--molecules", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    if args.molecules != 100:
        raise RuntimeError("acceptance audit cohort is frozen at 100 molecules")
    args.output_dir.mkdir(parents=True)
    validity = ChemicalValidity(args.validity_statistics)
    device = torch.device(args.device)
    started = time.monotonic()

    if args.cohort == "formal":
        sources = pd.read_parquet(args.formal_sources)
        targets = pd.read_parquet(args.formal_targets)
        _validate_cohort_frames(sources, targets)
        if file_sha256(args.formal_sources) != SOURCE_MANIFEST_SHA256:
            raise RuntimeError("formal audit source SHA mismatch")
        if file_sha256(args.formal_targets) != TARGET_MANIFEST_SHA256:
            raise RuntimeError("formal audit target SHA mismatch")
        checkpoint_sha = FROZEN_SEEDS[43]["checkpoint_sha256"]
        if file_sha256(args.formal_checkpoint) != checkpoint_sha:
            raise RuntimeError("formal audit checkpoint SHA mismatch")
        checkpoint = torch.load(
            args.formal_checkpoint, map_location="cpu", weights_only=False
        )
        wrapper = load_v7_formal_config(args.formal_v7_config)
        if file_sha256(args.formal_v7_config) != V7_CONFIG_SHA256:
            raise RuntimeError("formal audit V7 config SHA mismatch")
        if file_sha256(args.v5_config) != V5_CONFIG_SHA256:
            raise RuntimeError("formal audit V5 config SHA mismatch")
        v5_config = yaml.safe_load(args.v5_config.read_text(encoding="utf-8"))
        models = _load_models(checkpoint, wrapper, v5_config, device)
        inference = dict(wrapper["inference"])
        seed = 43
    else:
        manifest = _verify_manifest(args.development_manifest_dir)
        sources = pd.read_parquet(
            args.development_manifest_dir / "development_sources.parquet"
        )
        targets = pd.read_parquet(
            args.development_manifest_dir / "development_targets.parquet"
        )
        checkpoint_sha = D1_CHECKPOINT_SHA256
        if file_sha256(args.development_checkpoint) != checkpoint_sha:
            raise RuntimeError("development audit checkpoint SHA mismatch")
        checkpoint = torch.load(
            args.development_checkpoint, map_location="cpu", weights_only=False
        )
        load_args = argparse.Namespace(
            v5_config=args.v5_config, v7_config=args.development_v7_config
        )
        models = {}
        configs = {}
        for method in METHODS:
            models[method], configs[method], _ = _load_model(
                method, checkpoint, load_args, device
            )
        inference = dict(configs["V7"]["inference"])
        if manifest.get("test_records_read") != 0:
            raise RuntimeError("development audit isolation changed")
        seed = 43018

    inference["batch_size"] = args.batch_size
    selected_molecules = sorted(map(str, sources.molecule_id.unique()))[: args.molecules]
    selected_set = set(selected_molecules)
    sources = sources.loc[sources.molecule_id.astype(str).isin(selected_set)].copy()
    targets = targets.loc[targets.sample_id.isin(set(sources.sample_id))].copy()
    sources = sources.sort_values(["molecule_id", "sample_id"])
    targets_by_sample = targets.set_index("sample_id")
    items = _build_items(
        sources,
        targets,
        validity,
        source_cache_root=args.source_cache_root,
        target_cache_root=args.formal_root / "minimal_targets",
    )
    _seed(seed)
    manifest_rows = sources[["sample_id", "molecule_id"]].sort_values("sample_id")
    selection_identity = _canonical_sha(manifest_rows.astype(str).values.tolist())
    manifest_payload = {
        "schema_version": "mcvr-v7-acceptance-audit-cohort-v1",
        "cohort": args.cohort,
        "molecules": len(selected_molecules),
        "records": len(sources),
        "selection_identity_sha256": selection_identity,
        "sample_ids": manifest_rows.to_dict(orient="records"),
        "checkpoint_sha256": checkpoint_sha,
        **ISOLATION,
    }
    _write_json(args.output_dir / "manifest.json", manifest_payload)

    record_frames = []
    audit_rows = []
    for semantics in SEMANTICS:
        for method in METHODS:
            model = models[method]
            if method == "V7":
                model.reset_statistics()
            elif method == "V5-B":
                model.reset_solver_statistics()
            result = evaluate_bac_candidate(
                model,
                items,
                validity,
                device=device,
                inference=inference,
                source_identity_sha256=SOURCE_IDENTITY_SHA256,
                bootstrap_draws=1,
                trajectory_semantics=semantics,
                safety_objective_mode=(
                    "weighted_thresholded_validity"
                    if semantics == "formal_d1b"
                    else "legacy_rate_sum"
                ),
            )
            candidate = result["records"].loc[
                result["records"].method == "v2_bac_accepted"
            ].copy()
            candidate["cohort"] = args.cohort
            candidate["semantics"] = semantics
            candidate["method"] = method
            record_frames.append(candidate)
            metadata = result["metadata"]
            for item, extra in zip(items, metadata, strict=True):
                extra = {**extra, "seed": seed}
                target = targets_by_sample.loc[str(item["row"].sample_id)]
                audit_rows.append(
                    _audit_row(
                        cohort=args.cohort,
                        semantics=semantics,
                        method=method,
                        item=item,
                        target_row=target,
                        metadata=extra,
                        checkpoint_sha=checkpoint_sha,
                    )
                )

    native_frame, native = _native_rows(
        models["D1"],
        items,
        validity,
        device=device,
        inference=inference,
        cohort=args.cohort,
        checkpoint_sha=checkpoint_sha,
        targets=targets,
    )
    native_frame["cohort"] = args.cohort
    native_frame["semantics"] = "native_d1b"
    native_frame["method"] = "D1"
    record_frames.append(native_frame)
    audit_rows.extend(native["audit"])
    records = pd.concat(record_frames, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    records.to_csv(args.output_dir / "per_record_metrics.csv", index=False)
    audit.to_csv(args.output_dir / "per_record_audit.csv", index=False)
    rollback = (
        audit.assign(
            rollback_reason=audit.rollback_reason.replace("", "accepted_or_no_reason")
        )
        .groupby(["cohort", "semantics", "method", "rollback_reason"])
        .size()
        .rename("count")
        .reset_index()
    )
    rollback["fraction"] = rollback["count"] / rollback.groupby(
        ["cohort", "semantics", "method"]
    )["count"].transform("sum")
    rollback.to_csv(args.output_dir / "rollback_reasons.csv", index=False)
    summary = {
        "schema_version": "mcvr-v7-acceptance-audit-summary-v1",
        "cohort": args.cohort,
        "selection_identity_sha256": selection_identity,
        "comparison": _summary(records, audit),
        "wall_clock_seconds": time.monotonic() - started,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
            "device": args.device,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "batch_size": args.batch_size,
        },
        **ISOLATION,
    }
    _write_json(args.output_dir / "summary.json", summary)
    files = (
        "manifest.json",
        "per_record_metrics.csv",
        "per_record_audit.csv",
        "rollback_reasons.csv",
        "summary.json",
    )
    (args.output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{file_sha256(args.output_dir / name)}  {name}\n" for name in files),
        encoding="ascii",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
