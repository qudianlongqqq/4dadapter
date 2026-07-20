#!/usr/bin/env python
"""Train MCVR V8 using real upstream train records and validation-only selection."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Mapping

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import numpy as np
import pandas as pd
import psutil
import torch
import yaml
from torch_geometric.loader import DataLoader
from torch.utils.data import Sampler, Subset

from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.mcvr_v8_full import MCVRV8FullRefiner
from etflow.ecir.mvr_dataset import MCVRMixedDataset
from etflow.ecir.v8_constraint_normalization import FrozenResidualScales
from etflow.ecir.v8_diagnostics import parameter_group_diagnostics, per_type_gradient_norms
from etflow.ecir.v8_losses import MCVRV8Loss
from etflow.ecir.v8_sampler import sampler_from_payload


ISOLATION = {
    "formal_test_records_read": 0,
    "formal_test_assets_opened": False,
    "minimal_validity_target_test_used": False,
    "frozen_holdout_records_read": 0,
    "parameter_selection_from_formal_test": False,
}


def _resource_snapshot(process: psutil.Process, device: torch.device) -> dict[str, Any]:
    memory = process.memory_info()
    snapshot: dict[str, Any] = {
        "process_cpu_percent": float(process.cpu_percent(interval=None)),
        "system_cpu_percent": float(psutil.cpu_percent(interval=None)),
        "process_ram_bytes": int(memory.rss),
        "system_ram_used_bytes": int(psutil.virtual_memory().used),
        "gpu_utilization_percent": None,
        "gpu_memory_used_bytes": None,
        "gpu_memory_total_bytes": None,
    }
    if device.type != "cuda":
        return snapshot
    try:
        index = int(device.index or 0)
        query = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        utilization, used_mib, total_mib = [
            float(value.strip()) for value in query.stdout.strip().split(",")
        ]
        snapshot.update(
            {
                "gpu_utilization_percent": utilization,
                "gpu_memory_used_bytes": int(used_mib * 1024 * 1024),
                "gpu_memory_total_bytes": int(total_mib * 1024 * 1024),
            }
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return snapshot


class _SkipSampler(Sampler[int]):
    def __init__(self, base: Sampler[int], skip: int) -> None:
        self.base = base
        self.skip = int(skip)

    def __iter__(self):
        return islice(iter(self.base), self.skip, None)

    def __len__(self) -> int:
        return max(len(self.base) - self.skip, 0)


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    base = payload.get("base_config")
    if base is None:
        return payload
    base_path = Path(base)
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    return _deep_merge(load_config(base_path), payload)


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _resume_scientific_identity(config: Mapping[str, Any]) -> str:
    """Bind all scientific settings while permitting only run-horizon changes."""
    payload = copy.deepcopy(dict(config))
    for key in ("experiment_name", "steps_total", "validation_protocol", "long_run"):
        payload.pop(key, None)
    payload.pop("deployment_validation", None)
    training = payload.get("training", {})
    for key in (
        "optimizer_steps",
        "validation_interval",
        "checkpoint_interval",
        "validation_steps",
        "checkpoint_steps",
    ):
        training.pop(key, None)
    return _canonical_sha(payload)


def _restore_rng_states(checkpoint: Mapping[str, Any]) -> None:
    states = checkpoint.get("rng_states")
    if not isinstance(states, Mapping):
        raise RuntimeError("V8 resume checkpoint has no RNG state")
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if torch.cuda.is_available():
        if states.get("cuda") is None:
            raise RuntimeError("V8 CUDA resume checkpoint has no CUDA RNG state")
        torch.cuda.set_rng_state_all(states["cuda"])


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_run_status(output_dir: Path, payload: Mapping[str, Any]) -> None:
    """Keep the canonical live status and legacy progress mirror in sync."""
    _atomic_json(output_dir / "status.json", payload)
    _atomic_json(output_dir / "progress.json", payload)


def _git_value(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _assert_manifest(path: Path, split: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"V8 required {split} manifest is missing: {path}")
    lowered = str(path.resolve()).lower().replace("\\", "/")
    if any(token in lowered for token in ("formal_test", "/test/", "holdout")):
        raise RuntimeError(f"V8 refuses forbidden data path: {path}")
    frame = pd.read_parquet(path)
    if "split" not in frame or set(frame.split.astype(str)) != {split}:
        raise RuntimeError(f"V8 {split} manifest split binding changed")
    for column in frame:
        name = str(column).lower()
        if ("test" in name or "holdout" in name) and bool(
            frame[column].fillna(False).astype(bool).any()
        ):
            raise RuntimeError(f"V8 forbidden record flag is active: {column}")
    return frame


def _real_dataset(
    source: Path,
    target: Path,
    validity: ChemicalValidity,
    *,
    source_cache_root: Path | None,
    target_cache_root: Path | None,
    source_identity: str,
) -> MCVRMixedDataset:
    dataset = MCVRMixedDataset(
        source,
        target,
        validity,
        length=len(pd.read_parquet(source)),
        ratios={"real_error": 1.0, "synthetic_error": 0.0, "clean_identity": 0.0},
        seed=43,
        source_cache_root=source_cache_root,
        target_cache_root=target_cache_root,
        canonical_constraints=True,
        constraint_source_identity_sha256=source_identity,
    )
    # Preserve source-manifest order so the train-only stratified weights bind exactly.
    dataset.plan = [
        {
            "row_index": int(index),
            "sample_type": "real_error",
            "corruption_type": "real",
            "source": str(row.generator_name),
            "severity": str(row.source_severity),
        }
        for index, row in dataset.sources.iterrows()
    ]
    return dataset


def _checkpoint(
    path: Path,
    *,
    model: MCVRV8FullRefiner,
    optimizer: torch.optim.Optimizer,
    step: int,
    resolved_config: Mapping[str, Any],
    scales: FrozenResidualScales,
) -> None:
    payload = {
        "schema_version": "mcvr-v8-full-v1-checkpoint-v1",
        "model_type": "MCVRV8FullRefiner",
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "resolved_config": dict(resolved_config),
        "resolved_config_sha256": _canonical_sha(resolved_config),
        "residual_scales": scales.__dict__,
        "residual_scales_identity_sha256": scales.identity_sha256,
        "unroll_steps": model.unroll_steps,
        "rng_states": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        **ISOLATION,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _move(batch: Any, device: torch.device) -> Any:
    return batch.to(device)


@torch.no_grad()
def _validate(
    model: MCVRV8FullRefiner,
    loss_fn: MCVRV8Loss,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    rows = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        batch = _move(batch, device)
        t = batch.x_input.new_full((batch.num_graphs,), 0.5)
        output = model(batch, batch.x_input, t)
        losses = loss_fn(output, batch)
        rows.append({key: float(value) for key, value in losses.items() if value.numel() == 1})
    model.train()
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]} if rows else {}


def _run_cached_validation(
    *,
    config: Mapping[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
    progress_path: Path,
    step: int,
    device: str,
) -> dict[str, Any] | None:
    protocol = config.get("validation_protocol")
    if not protocol:
        return None
    full_steps = {int(value) for value in protocol.get("full_steps", [])}
    fast_steps = {int(value) for value in protocol.get("fast_steps", [])}
    mode = "FULL" if step in full_steps else "FAST" if step in fast_steps else None
    if mode is None:
        return None
    validation_dir = output_dir / "validation_cache" / f"step{step:06d}" / mode.lower()
    prediction_dir = validation_dir / "prediction"
    source_manifest = Path(protocol["source_cache_manifest"]).resolve()
    if not source_manifest.is_file():
        raise RuntimeError("V8 validation source cache manifest is missing")
    prediction_command = [
        sys.executable,
        str(ROOT / "scripts/predict_ecir_mvr_v8_validation.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--source-cache-manifest",
        str(source_manifest),
        "--validity-statistics",
        str(Path(config["data"]["validity_statistics"]).resolve()),
        "--output-dir",
        str(prediction_dir),
        "--status-file",
        str(progress_path),
        "--batch-size",
        str(int(protocol.get("prediction_batch_size", 1))),
        "--chunk-size",
        str(int(protocol.get("prediction_chunk_size", 250))),
        "--device",
        str(device),
    ]
    if mode == "FAST":
        fast_manifest = Path(protocol["fast_manifest"]).resolve()
        if not fast_manifest.is_file():
            raise RuntimeError("V8 FAST validation manifest is missing")
        prediction_command.extend(["--fast-manifest", str(fast_manifest)])
    subprocess.run(prediction_command, cwd=ROOT, check=True)
    report_path = validation_dir / "evaluation.json"
    evaluation_command = [
        sys.executable,
        str(ROOT / "scripts/evaluate_ecir_mvr_v8_prediction_cache.py"),
        "--prediction-manifest",
        str(prediction_dir / "prediction_manifest.json"),
        "--source-cache-manifest",
        str(source_manifest),
        "--validity-statistics",
        str(Path(config["data"]["validity_statistics"]).resolve()),
        "--output",
        str(report_path),
        "--mode",
        mode,
        "--training-step",
        str(step),
        "--status-file",
        str(progress_path),
    ]
    if mode == "FULL":
        evaluation_command.extend(
            ["--bootstrap-draws", str(int(protocol.get("bootstrap_draws_full", 10_000)))]
        )
    subprocess.run(evaluation_command, cwd=ROOT, check=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if mode == "FULL":
        baseline_root = Path(protocol["baseline_cache_root"]).resolve()
        baseline_reports = {
            "d1": baseline_root / "d1" / "evaluation.json",
            "v5_b": baseline_root / "v5_b" / "evaluation.json",
            "v7": baseline_root / "v7" / "evaluation.json",
        }
        missing = [str(path) for path in baseline_reports.values() if not path.is_file()]
        if missing:
            raise RuntimeError(f"V8 frozen baseline evaluation cache is incomplete: {missing}")
        comparison_path = validation_dir / "paired_baseline_comparison.json"
        comparison_command = [
            sys.executable,
            str(ROOT / "scripts/compare_ecir_mvr_v8_cached_evaluations.py"),
            "--v8",
            str(report_path),
            "--d1",
            str(baseline_reports["d1"]),
            "--v5-b",
            str(baseline_reports["v5_b"]),
            "--v7",
            str(baseline_reports["v7"]),
            "--output",
            str(comparison_path),
            "--bootstrap-draws",
            str(int(protocol.get("bootstrap_draws_full", 10_000))),
        ]
        subprocess.run(comparison_command, cwd=ROOT, check=True)
        report["paired_baseline_comparison"] = json.loads(
            comparison_path.read_text(encoding="utf-8")
        )
    with (output_dir / "validation_protocol.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": step, "mode": mode, **report}) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-sources", type=Path)
    parser.add_argument("--train-targets", type=Path)
    parser.add_argument("--val-sources", type=Path)
    parser.add_argument("--val-targets", type=Path)
    parser.add_argument("--source-cache-root", type=Path)
    parser.add_argument("--target-cache-root", type=Path)
    parser.add_argument("--scales", type=Path)
    parser.add_argument("--sampler-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--tiny-records", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--fixed-tiny-batch", action="store_true")
    parser.add_argument("--tiny-hardest", action="store_true")
    args = parser.parse_args()
    resume_step_hint = 0
    if args.resume is not None:
        resume_header = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_step_hint = int(resume_header.get("step", 0))
    config = load_config(args.config)
    if args.batch_size is not None:
        config["training"]["batch_size"] = int(args.batch_size)
    if args.gradient_accumulation_steps is not None:
        config["training"]["gradient_accumulation_steps"] = int(args.gradient_accumulation_steps)
    if args.num_workers is not None:
        config["training"]["num_workers"] = int(args.num_workers)
    if config.get("isolation") != ISOLATION:
        raise RuntimeError("V8 resolved config isolation contract changed")
    data = config["data"]
    train_sources = (args.train_sources or Path(data["train_sources"])).resolve()
    train_targets = (args.train_targets or Path(data["train_targets"])).resolve()
    val_sources = (args.val_sources or Path(data["val_sources"])).resolve()
    val_targets = (args.val_targets or Path(data["val_targets"])).resolve()
    data.update(
        {
            "train_sources": str(train_sources),
            "train_targets": str(train_targets),
            "val_sources": str(val_sources),
            "val_targets": str(val_targets),
            "train_sources_sha256": _file_sha(train_sources),
            "train_targets_sha256": _file_sha(train_targets),
            "val_sources_sha256": _file_sha(val_sources),
            "val_targets_sha256": _file_sha(val_targets),
        }
    )
    train_frame = _assert_manifest(train_sources, "train")
    train_target_frame = _assert_manifest(train_targets, "train")
    val_frame = _assert_manifest(val_sources, "val")
    val_target_frame = _assert_manifest(val_targets, "val")
    formal_binding = config.get("formal_large_binding")
    if formal_binding:
        if len(train_frame) != int(formal_binding["expected_train_records"]):
            raise RuntimeError("V8 formal-large train record count changed")
        if len(train_target_frame) != len(train_frame):
            raise RuntimeError("V8 formal-large train source-target count changed")
        if len(val_frame) != int(formal_binding["expected_validation_records"]):
            raise RuntimeError("V8 formal-large validation record count changed")
        if len(val_target_frame) != len(val_frame):
            raise RuntimeError("V8 formal-large validation source-target count changed")
        preflight_path = Path(formal_binding["preflight_report"]).resolve()
        if _file_sha(preflight_path) != str(formal_binding["preflight_report_sha256"]):
            raise RuntimeError("V8 formal-large preflight report SHA256 changed")
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if preflight.get("status") != "MCVR_V8_FORMAL_LARGE_PREFLIGHT_READY":
            raise RuntimeError("V8 formal-large preflight is not ready")
        if any(preflight.get(key) != value for key, value in ISOLATION.items()):
            raise RuntimeError("V8 formal-large preflight isolation changed")
    if set(train_frame.molecule_id.astype(str)) & set(
        pd.read_parquet(val_sources).molecule_id.astype(str)
    ):
        raise RuntimeError("V8 molecule identity crosses train/validation splits")
    source_root = args.source_cache_root or (
        Path(data["source_cache_root"]) if data.get("source_cache_root") else None
    )
    target_root = args.target_cache_root or (
        Path(data["target_cache_root"]) if data.get("target_cache_root") else None
    )
    if source_root is not None:
        source_root = source_root.resolve()
        data["source_cache_root"] = str(source_root)
    if target_root is not None:
        target_root = target_root.resolve()
        data["target_cache_root"] = str(target_root)
    scales_path = (args.scales or Path(config["constraint_layer"]["frozen_scales"])).resolve()
    scales_file_sha256 = _file_sha(scales_path)
    if args.scales is not None:
        config["constraint_layer"]["frozen_scales"] = str(scales_path)
        config["constraint_layer"]["frozen_scales_sha256"] = scales_file_sha256
    scales = FrozenResidualScales.load(
        scales_path, expected_sha256=config["constraint_layer"].get("frozen_scales_sha256")
    )
    scales_payload = json.loads(scales_path.read_text(encoding="utf-8"))
    if scales_payload.get("train_source_manifest_sha256") != data["train_sources_sha256"]:
        raise RuntimeError("V8 residual scales source-manifest binding changed")
    if scales_payload.get("train_target_manifest_sha256") != data["train_targets_sha256"]:
        raise RuntimeError("V8 residual scales target-manifest binding changed")
    config["constraint_layer"]["frozen_scales_identity_sha256"] = scales.identity_sha256
    config["constraint_layer"]["resolved_frozen_scales"] = {
        "bond": scales.bond,
        "angle": scales.angle,
        "clash": scales.clash,
        "ring": scales.ring,
        "chirality": scales.chirality,
    }
    sampler_path = (args.sampler_manifest or Path(config["sampler"]["manifest"])).resolve()
    sampler_raw = sampler_path.read_bytes()
    sampler_file_sha256 = hashlib.sha256(sampler_raw).hexdigest()
    sampler_payload = json.loads(sampler_raw.decode("utf-8"))
    if args.sampler_manifest is not None:
        config["sampler"]["manifest"] = str(sampler_path)
        config["sampler"]["manifest_sha256"] = sampler_file_sha256
        config["sampler"]["manifest_identity_sha256"] = sampler_payload["identity_sha256"]
    if config["sampler"].get("manifest_sha256") not in (None, sampler_file_sha256):
        raise RuntimeError("V8 stratified sampler file SHA256 changed")
    if config["sampler"].get("manifest_identity_sha256") not in (
        None,
        sampler_payload["identity_sha256"],
    ):
        raise RuntimeError("V8 stratified sampler canonical identity changed")
    config["sampler"]["resolved_cohort_counts"] = sampler_payload["cohort_counts"]
    config["sampler"]["resolved_cohort_weights"] = sampler_payload["cohort_weights"]
    config["sampler"]["molecule_exposure_cap"] = sampler_payload.get("molecule_exposure_cap")
    if sampler_payload["source_manifest_sha256"] != data["train_sources_sha256"]:
        raise RuntimeError("V8 stratified sampler is not bound to the train source manifest")
    if sampler_payload.get("target_manifest_sha256") not in (
        None,
        data["train_targets_sha256"],
    ):
        raise RuntimeError("V8 stratified sampler is not bound to the train target manifest")
    if int(sampler_payload["record_count"]) != len(train_frame):
        raise RuntimeError("V8 stratified sampler record count changed")
    _seed(int(config["seed"]))
    validity = ChemicalValidity(data["validity_statistics"])
    train_dataset = _real_dataset(
        train_sources,
        train_targets,
        validity,
        source_cache_root=source_root,
        target_cache_root=target_root,
        source_identity=sampler_payload["source_manifest_sha256"],
    )
    val_dataset = _real_dataset(
        val_sources,
        val_targets,
        validity,
        source_cache_root=source_root,
        target_cache_root=target_root,
        source_identity=hashlib.sha256(val_sources.read_bytes()).hexdigest(),
    )
    training = config["training"]
    if args.tiny_records is not None:
        tiny_count = int(args.tiny_records)
        if not 1 <= tiny_count <= len(train_dataset):
            raise ValueError("tiny-records must be within the train dataset")
        if args.tiny_hardest:
            if "initial_to_target_rmsd" not in train_target_frame:
                raise RuntimeError("tiny-hardest requires train target RMSD metadata")
            hardest_ids = set(
                train_target_frame.nlargest(tiny_count, "initial_to_target_rmsd").sample_id.astype(
                    str
                )
            )
            tiny_indices = [
                int(index)
                for index, sample_id in enumerate(train_dataset.sources.sample_id.astype(str))
                if sample_id in hardest_ids
            ]
            if len(tiny_indices) != tiny_count:
                raise RuntimeError("tiny-hardest source/target identity binding failed")
            config["training"]["tiny_selection"] = "largest_train_target_rmsd"
        else:
            tiny_indices = list(range(tiny_count))
            config["training"]["tiny_selection"] = "first_train_records"
        train_dataset = Subset(train_dataset, tiny_indices)
        sampler_payload = dict(sampler_payload)
        sampler_payload["records"] = [sampler_payload["records"][index] for index in tiny_indices]
        config["training"]["tiny_train_record_count"] = tiny_count
    total_steps = int(args.steps or training["optimizer_steps"])
    training["optimizer_steps"] = total_steps
    batch_size = int(training["batch_size"])
    fixed_batch = None
    if args.fixed_tiny_batch:
        if args.tiny_records is None:
            raise ValueError("fixed-tiny-batch requires tiny-records")
        fixed_loader = DataLoader(
            train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=0
        )
        fixed_batch = next(iter(fixed_loader))
        train_loader = fixed_loader
        config["training"]["fixed_tiny_batch"] = True
    else:
        samples = total_steps * batch_size * int(training["gradient_accumulation_steps"])
        sampler = sampler_from_payload(
            sampler_payload, num_samples=samples, seed=int(config["seed"])
        )
        sampler = _SkipSampler(
            sampler,
            resume_step_hint * batch_size * int(training["gradient_accumulation_steps"]),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=int(training.get("num_workers", 0)),
        )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    constraint = dict(config["constraint_layer"])
    constraint.pop("frozen_scales", None)
    constraint.pop("frozen_scales_sha256", None)
    constraint.pop("use_frozen_scales", None)
    unroll_steps = int(constraint.pop("unroll_steps"))
    model_settings = config["model"]
    model = MCVRV8FullRefiner.from_d1_checkpoint(
        model_settings["d1_checkpoint"],
        expected_sha256=model_settings["d1_checkpoint_sha256"],
        error_state=config["error_state"],
        constraint_layer=constraint,
        residual_scales=scales,
        unroll_steps=unroll_steps,
        step_embedding_enabled=bool(model_settings["step_embedding_enabled"]),
        error_state_enabled=bool(config["error_state"]["enabled"]),
        train_d1_backbone=bool(model_settings["train_d1_backbone"]),
        train_d1_head=bool(model_settings["train_d1_head"]),
        max_cumulative_atom_displacement=float(config["safety"]["max_atom_displacement"]),
        max_cumulative_graph_rms=float(config["safety"]["graph_rms_limit"]),
    ).to(args.device)
    loss_fn = MCVRV8Loss(
        config["loss"],
        confidence_min=config["error_state"]["confidence_min"],
        confidence_max=config["error_state"]["confidence_max"],
        clash_settings=config.get("clash"),
        residual_scales=scales,
    )
    optimizer = torch.optim.AdamW(
        model.parameter_groups(
            new_head_lr=training["new_head_learning_rate"],
            d1_head_lr=training["d1_head_learning_rate"],
            d1_backbone_lr=training["d1_backbone_learning_rate"],
            weight_decay=training["weight_decay"],
        )
    )
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("schema_version") != "mcvr-v8-full-v1-checkpoint-v1":
            raise RuntimeError("V8 resume checkpoint schema changed")
        if _resume_scientific_identity(checkpoint["resolved_config"]) != _resume_scientific_identity(config):
            raise RuntimeError("V8 resume scientific identity changed")
        if checkpoint["residual_scales_identity_sha256"] != scales.identity_sha256:
            raise RuntimeError("V8 resume residual scales changed")
        if int(checkpoint["unroll_steps"]) != model.unroll_steps:
            raise RuntimeError("V8 resume unroll configuration changed")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["step"])
        if total_steps <= start_step:
            raise RuntimeError("V8 resume target must extend beyond the checkpoint step")
        _restore_rng_states(checkpoint)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if args.resume is None:
            raise RuntimeError(f"V8 output directory is nonempty and not a resume: {args.output_dir}")
        existing_config_path = args.output_dir / "config.resolved.json"
        if not existing_config_path.is_file():
            raise RuntimeError("V8 nonempty resume directory has no resolved config identity")
        existing_config = json.loads(existing_config_path.read_text(encoding="utf-8"))
        if _canonical_sha(existing_config) != _canonical_sha(config):
            raise RuntimeError("V8 nonempty resume directory belongs to a different run identity")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_sha256 = _canonical_sha(config)
    device_object = torch.device(args.device)
    gpu_info = (
        {
            "gpu_name": torch.cuda.get_device_name(device_object),
            "gpu_total_vram_bytes": torch.cuda.get_device_properties(device_object).total_memory,
        }
        if device_object.type == "cuda"
        else {"gpu_name": None, "gpu_total_vram_bytes": 0}
    )
    _atomic_json(args.output_dir / "config.resolved.json", config)
    _atomic_json(
        args.output_dir / "asset_hashes.json",
        {
            "schema_version": "mcvr-v8-formal-large-run-assets-v1",
            "actual_command": [sys.executable, *sys.argv],
            "git_branch": _git_value("branch", "--show-current"),
            "git_head": _git_value("rev-parse", "HEAD"),
            "resolved_config_sha256": resolved_config_sha256,
            "train_sources_sha256": data["train_sources_sha256"],
            "train_targets_sha256": data["train_targets_sha256"],
            "val_sources_sha256": data["val_sources_sha256"],
            "val_targets_sha256": data["val_targets_sha256"],
            "residual_scales_file_sha256": scales_file_sha256,
            "residual_scales_identity_sha256": scales.identity_sha256,
            "stratified_manifest_file_sha256": sampler_file_sha256,
            "stratified_manifest_identity_sha256": sampler_payload["identity_sha256"],
            "d1_checkpoint_sha256": model.d1_checkpoint_identity["sha256"],
            **gpu_info,
            **ISOLATION,
        },
    )
    effective_batch = batch_size * int(training["gradient_accumulation_steps"])
    _write_run_status(
        args.output_dir,
        {
            "status": "TRAINING",
            "phase": "TRAINING",
            "pid": os.getpid(),
            "step": start_step,
            "training_step": start_step,
            "validation_mode": None,
            "current_validation_record": 0,
            "prediction_chunks_completed": 0,
            "evaluation_chunks_completed": 0,
            "records_per_second": 0.0,
            "elapsed_seconds": 0.0,
            "estimated_remaining_seconds": None,
            "last_update_time": datetime.now(timezone.utc).isoformat(),
            "error": None,
            "optimizer_steps": total_steps,
            "effective_batch": effective_batch,
            "resolved_config_sha256": resolved_config_sha256,
            **ISOLATION,
        },
    )
    iterator = iter(train_loader)
    accumulation = int(training["gradient_accumulation_steps"])
    validation_steps = {int(value) for value in training.get("validation_steps", [])}
    checkpoint_steps = {int(value) for value in training.get("checkpoint_steps", [])}
    if validation_steps:
        validation_steps.add(total_steps)
    if checkpoint_steps:
        checkpoint_steps.add(total_steps)
    latest: dict[str, float] = {}
    started_at = time.perf_counter()
    process_monitor = psutil.Process(os.getpid())
    process_monitor.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None)
    exposed_molecules: set[str] = set()
    finite_loss_spike_count = 0
    recent_losses: list[float] = []
    consecutive_solver_failure_steps = 0
    model.train()
    for step in range(start_step + 1, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        rows = []
        type_gradient_rows = []
        step_sample_ids: list[str] = []
        step_molecule_ids: list[str] = []
        capture_type_gradients = bool(config["diagnostics"]["per_type_gradients"]) and (
            step % int(training["log_interval"]) == 0 or step == 1
        )
        for _ in range(accumulation):
            batch = _move(
                fixed_batch if fixed_batch is not None else next(iterator),
                torch.device(args.device),
            )
            step_sample_ids.extend(str(value) for value in batch.sample_id)
            step_molecule_ids.extend(str(value) for value in batch.molecule_id)
            exposed_molecules.update(step_molecule_ids)
            t = batch.x_input.new_full((batch.num_graphs,), 0.5)
            output = model(batch, batch.x_input, t)
            losses = loss_fn(output, batch)
            if not all(bool(torch.isfinite(value)) for value in losses.values()):
                raise FloatingPointError("V8 training produced NaN/Inf")
            if capture_type_gradients:
                type_gradient_rows.append(per_type_gradient_norms(losses, model.parameters()))
            (losses["loss"] / accumulation).backward()
            rows.append({key: float(value.detach()) for key, value in losses.items()})
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
        )
        optimizer.step()
        latest = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
        if type_gradient_rows:
            latest.update(
                {
                    key: sum(row[key] for row in type_gradient_rows) / len(type_gradient_rows)
                    for key in type_gradient_rows[0]
                }
            )
        latest.update({"step": step, "gradient_norm": gradient_norm})
        latest["BOND_DOMINANCE_WARNING"] = bool(
            latest["solver_angle_contribution"] > 0.0
            and latest["solver_contribution_ratio"]
            > float(config["diagnostics"]["bond_dominance_warning_ratio"])
        )
        latest["confidence_saturation_warning"] = bool(
            max(
                latest["confidence_lower_saturation_fraction"],
                latest["confidence_upper_saturation_fraction"],
            )
            > 0.25
        )
        latest["displacement_limit_warning"] = bool(
            latest["displacement_max"] > 0.95 * float(config["safety"]["max_atom_displacement"])
            or latest["graph_displacement_rms_max"]
            > 0.95 * float(config["safety"]["graph_rms_limit"])
        )
        if latest["displacement_max"] > float(config["safety"]["max_atom_displacement"]) + 1e-5:
            raise RuntimeError("V8 cumulative atom displacement projection failed")
        if latest["graph_displacement_rms_max"] > float(config["safety"]["graph_rms_limit"]) + 1e-5:
            raise RuntimeError("V8 cumulative graph RMS projection failed")
        if latest["solver_failure_count"] > 0:
            consecutive_solver_failure_steps += 1
        else:
            consecutive_solver_failure_steps = 0
        if consecutive_solver_failure_steps >= 3:
            raise RuntimeError("V8 solver failure persisted for three optimizer steps")
        spike = False
        if len(recent_losses) >= 10:
            baseline = float(np.median(recent_losses[-20:]))
            spike = latest["loss"] > max(10.0, 20.0 * max(baseline, 1.0e-8))
        if spike:
            finite_loss_spike_count += 1
            _atomic_json(
                args.output_dir / "finite_spikes" / f"step{step:06d}.json",
                {
                    "step": step,
                    "sample_ids": step_sample_ids,
                    "molecule_ids": step_molecule_ids,
                    "metrics": latest,
                    **ISOLATION,
                },
            )
        recent_losses.append(latest["loss"])
        if step % int(training["log_interval"]) == 0 or step == 1:
            elapsed = time.perf_counter() - started_at
            completed_here = max(step - start_step, 1)
            seconds_per_step = elapsed / completed_here
            remaining_seconds = seconds_per_step * max(total_steps - step, 0)
            group_diagnostics = parameter_group_diagnostics(optimizer)
            runtime = {
                "optimizer_steps": total_steps,
                "records_exposed": step * effective_batch,
                "molecule_draws_exposed": step * effective_batch,
                "epoch_equivalent_exposure": (step * effective_batch) / max(len(train_dataset), 1),
                "unique_molecules_exposed_this_process": len(exposed_molecules),
                "effective_batch": effective_batch,
                "elapsed_seconds": elapsed,
                "seconds_per_step": seconds_per_step,
                "records_per_second": completed_here * effective_batch / max(elapsed, 1.0e-9),
                "estimated_remaining_seconds": remaining_seconds,
                "finite_loss_spike_count": finite_loss_spike_count,
                "consecutive_solver_failure_steps": consecutive_solver_failure_steps,
                "gpu_memory_allocated_bytes": torch.cuda.memory_allocated(device_object)
                if device_object.type == "cuda"
                else 0,
                "gpu_memory_reserved_bytes": torch.cuda.memory_reserved(device_object)
                if device_object.type == "cuda"
                else 0,
                "gpu_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device_object)
                if device_object.type == "cuda"
                else 0,
                **_resource_snapshot(process_monitor, device_object),
            }
            with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            **latest,
                            **runtime,
                            "parameter_groups": group_diagnostics,
                            **ISOLATION,
                        }
                    )
                    + "\n"
                )
            print(
                f"train_progress={step}/{total_steps} loss={latest['loss']:.8g} "
                f"target={latest['target_loss']:.8g} failures={latest['solver_failure_count']:.0f}",
                flush=True,
            )
            _write_run_status(
                args.output_dir,
                {
                    "status": "TRAINING",
                    "phase": "TRAINING",
                    "pid": os.getpid(),
                    "step": step,
                    "training_step": step,
                    "validation_mode": None,
                    "current_validation_record": 0,
                    "prediction_chunks_completed": 0,
                    "evaluation_chunks_completed": 0,
                    "last_update_time": datetime.now(timezone.utc).isoformat(),
                    "error": None,
                    **runtime,
                    "latest": latest,
                    "parameter_groups": group_diagnostics,
                    "resolved_config_sha256": resolved_config_sha256,
                    **ISOLATION,
                },
            )
        run_validation = (
            step in validation_steps
            if validation_steps
            else step % int(training["validation_interval"]) == 0 or step == total_steps
        )
        if run_validation:
            validation = _validate(
                model, loss_fn, val_loader, torch.device(args.device), args.validation_batches
            )
            with (args.output_dir / "validation.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, **validation, **ISOLATION}) + "\n")
        save_checkpoint = (
            step in checkpoint_steps
            if checkpoint_steps
            else step % int(training["checkpoint_interval"]) == 0 or step == total_steps
        )
        if save_checkpoint:
            checkpoint_path = args.output_dir / "checkpoints" / f"step{step:06d}.ckpt"
            _checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=step,
                resolved_config=config,
                scales=scales,
            )
            if step == total_steps:
                _checkpoint(
                    args.output_dir / "checkpoints" / "last.ckpt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    resolved_config=config,
                    scales=scales,
                )
            deployment = config.get("deployment_validation", {})
            deployment_steps = {int(value) for value in deployment.get("steps", [])}
            if bool(deployment.get("enabled", False)) and step in deployment_steps:
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/evaluate_ecir_mvr_v8_validation.py"),
                        "--checkpoint",
                        str(checkpoint_path),
                        "--val-sources",
                        str(val_sources),
                        "--val-targets",
                        str(val_targets),
                        "--source-cache-root",
                        str(source_root),
                        "--target-cache-root",
                        str(target_root),
                        "--validity-statistics",
                        str(Path(data["validity_statistics"]).resolve()),
                        "--output",
                        str(args.output_dir / f"deployment_validation_step{step:06d}.json"),
                        "--max-records",
                        str(int(deployment.get("max_records", len(val_dataset)))),
                        "--device",
                        str(args.device),
                    ],
                    cwd=ROOT,
                    check=True,
                )
            cached_report = _run_cached_validation(
                config=config,
                checkpoint_path=checkpoint_path,
                output_dir=args.output_dir,
                progress_path=args.output_dir / "status.json",
                step=step,
                device=str(args.device),
            )
            if cached_report is not None and step < total_steps:
                _write_run_status(
                    args.output_dir,
                    {
                        "status": "TRAINING",
                        "phase": "TRAINING",
                        "pid": os.getpid(),
                        "step": step,
                        "training_step": step,
                        "validation_mode": None,
                        "current_validation_record": 0,
                        "prediction_chunks_completed": 0,
                        "evaluation_chunks_completed": 0,
                        "last_update_time": datetime.now(timezone.utc).isoformat(),
                        "error": None,
                        "optimizer_steps": total_steps,
                        "effective_batch": effective_batch,
                        "latest_validation": {
                            "mode": cached_report["mode"],
                            "metrics": cached_report["metrics"],
                        },
                        "resolved_config_sha256": resolved_config_sha256,
                        **ISOLATION,
                    },
                )
    _write_run_status(
        args.output_dir,
        {
            "status": "FINALIZING",
            "phase": "FINALIZING",
            "pid": os.getpid(),
            "step": total_steps,
            "training_step": total_steps,
            "validation_mode": None,
            "current_validation_record": 0,
            "prediction_chunks_completed": 0,
            "evaluation_chunks_completed": 0,
            "last_update_time": datetime.now(timezone.utc).isoformat(),
            "error": None,
            **ISOLATION,
        },
    )
    completed_elapsed = time.perf_counter() - started_at
    status = {
        "status": "COMPLETED",
        "phase": "COMPLETED",
        "steps": total_steps,
        "training_step": total_steps,
        "validation_mode": None,
        "current_validation_record": 0,
        "prediction_chunks_completed": 0,
        "evaluation_chunks_completed": 0,
        "records_per_second": (total_steps - start_step) * effective_batch
        / max(completed_elapsed, 1.0e-9),
        "elapsed_seconds": completed_elapsed,
        "estimated_remaining_seconds": 0.0,
        "last_update_time": datetime.now(timezone.utc).isoformat(),
        "error": None,
        "latest": latest,
        "device": str(args.device),
        "parameter_groups": parameter_group_diagnostics(optimizer),
        **ISOLATION,
    }
    _write_run_status(
        args.output_dir,
        {
            **status,
            "pid": os.getpid(),
            "step": total_steps,
            "optimizer_steps": total_steps,
            "records_exposed": total_steps * effective_batch,
            "effective_batch": effective_batch,
            "elapsed_seconds": completed_elapsed,
            "finite_loss_spike_count": finite_loss_spike_count,
        },
    )
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        try:
            if "--output-dir" in sys.argv:
                index = sys.argv.index("--output-dir") + 1
                failed_output = Path(sys.argv[index]).resolve()
                progress_path = failed_output / "status.json"
                previous = (
                    json.loads(progress_path.read_text(encoding="utf-8"))
                    if progress_path.is_file()
                    else {}
                )
                _write_run_status(
                    failed_output,
                    {
                        **previous,
                        "status": "FAILED_CLOSED",
                        "phase": "FAILED_CLOSED",
                        "error": str(error),
                        "last_update_time": datetime.now(timezone.utc).isoformat(),
                        **ISOLATION,
                    },
                )
        except BaseException:
            pass
        raise
