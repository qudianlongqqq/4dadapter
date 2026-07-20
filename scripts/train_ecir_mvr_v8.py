#!/usr/bin/env python
"""Train MCVR V8 using real upstream train records and validation-only selection."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
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
    train_frame = _assert_manifest(train_sources, "train")
    train_target_frame = _assert_manifest(train_targets, "train")
    _assert_manifest(val_sources, "val")
    _assert_manifest(val_targets, "val")
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
    scales_path = (args.scales or Path(config["constraint_layer"]["frozen_scales"])).resolve()
    scales = FrozenResidualScales.load(
        scales_path, expected_sha256=config["constraint_layer"].get("frozen_scales_sha256")
    )
    sampler_path = (args.sampler_manifest or Path(config["sampler"]["manifest"])).resolve()
    sampler_payload = json.loads(sampler_path.read_text(encoding="utf-8"))
    if (
        sampler_payload["source_manifest_sha256"]
        != hashlib.sha256(train_sources.read_bytes()).hexdigest()
    ):
        raise RuntimeError("V8 stratified sampler is not bound to the train source manifest")
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
        if checkpoint["resolved_config_sha256"] != _canonical_sha(config):
            raise RuntimeError("V8 resume resolved config changed")
        if checkpoint["residual_scales_identity_sha256"] != scales.identity_sha256:
            raise RuntimeError("V8 resume residual scales changed")
        if int(checkpoint["unroll_steps"]) != model.unroll_steps:
            raise RuntimeError("V8 resume unroll configuration changed")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["step"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.resolved.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    iterator = iter(train_loader)
    accumulation = int(training["gradient_accumulation_steps"])
    latest: dict[str, float] = {}
    model.train()
    for step in range(start_step + 1, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        rows = []
        type_gradient_rows = []
        capture_type_gradients = bool(config["diagnostics"]["per_type_gradients"]) and (
            step % int(training["log_interval"]) == 0 or step == 1
        )
        for _ in range(accumulation):
            batch = _move(
                fixed_batch if fixed_batch is not None else next(iterator),
                torch.device(args.device),
            )
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
        if step % int(training["log_interval"]) == 0 or step == 1:
            with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            **latest,
                            "parameter_groups": parameter_group_diagnostics(optimizer),
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
        if step % int(training["validation_interval"]) == 0 or step == total_steps:
            validation = _validate(
                model, loss_fn, val_loader, torch.device(args.device), args.validation_batches
            )
            with (args.output_dir / "validation.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, **validation, **ISOLATION}) + "\n")
        if step % int(training["checkpoint_interval"]) == 0 or step == total_steps:
            _checkpoint(
                args.output_dir / "checkpoints" / f"step{step:06d}.ckpt",
                model=model,
                optimizer=optimizer,
                step=step,
                resolved_config=config,
                scales=scales,
            )
    status = {
        "status": "COMPLETED",
        "steps": total_steps,
        "latest": latest,
        "device": str(args.device),
        "parameter_groups": parameter_group_diagnostics(optimizer),
        **ISOLATION,
    }
    (args.output_dir / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
