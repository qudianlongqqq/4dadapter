#!/usr/bin/env python
"""Discover and cross-check the old formal FlexBond-4D training budget.

Actual resolved runs and checkpoint metadata outrank launch-script declarations.
When the repository does not contain enough evidence, the report is still
written but confidence is low and formal training must not start.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _nested(data: dict, *paths, default=None):
    for path in paths:
        value: Any = data
        try:
            for key in path.split("."):
                value = value[key]
            return value
        except (KeyError, TypeError):
            continue
    return default


def _load_yaml(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return {}


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"step[=_-]?(\d+)", path.name, re.I)
    filename_step = int(match.group(1)) if match else 0
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        return int(payload.get("global_step", filename_step)) if isinstance(payload, dict) else filename_step
    except Exception:
        return filename_step


@dataclass
class Candidate:
    path: str
    kind: str
    score: int
    max_steps: int
    checkpoint_global_step: int
    config_path: str
    checkpoint_path: str
    evidence: list[str]


def _run_candidates(root: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    for config_path in root.glob("logs*/**/config.resolved.yaml"):
        name = str(config_path.parent).lower()
        if "jacobian" not in name and "flexbond" not in name and "4d" not in name:
            continue
        if any(token in name for token in ("smoke", "diagnostic", "2k")):
            continue
        config = _load_yaml(config_path)
        max_steps = int(_nested(config, "trainer.max_steps", "trainer_args.max_steps", default=0) or 0)
        checkpoints = list((config_path.parent / "checkpoints").glob("*.ckpt"))
        checkpoint = max(checkpoints, key=_checkpoint_step) if checkpoints else None
        step = _checkpoint_step(checkpoint) if checkpoint else 0
        score = 20 + (20 if "formal" in name else 0) + (15 if "long" in name else 0)
        score += 10 if "multiseed" in name else 0
        score += 20 if max(max_steps, step) >= 100000 else 0
        score += 10 if checkpoint else 0
        candidates.append(Candidate(str(config_path.parent), "resolved_run", score, max_steps, step,
                                    str(config_path), str(checkpoint or ""),
                                    ["resolved config", "checkpoint" if checkpoint else "checkpoint missing"]))
    return candidates


def _script_candidates(root: Path) -> list[Candidate]:
    candidates = []
    for path in root.glob("scripts/*4d*.sh"):
        text = path.read_text(encoding="utf-8", errors="replace")
        lowered = path.name.lower()
        if "global_coupled" in lowered:
            continue
        if "formal" not in lowered and "long" not in lowered:
            continue
        match = re.search(r"^MAX_STEPS=(\d+)", text, re.M)
        max_steps = int(match.group(1)) if match else 0
        score = 5 + (5 if "formal" in lowered else 0) + (3 if "multiseed" in lowered else 0)
        config_match = re.search(r'^CONFIG="([^"]+)"', text, re.M)
        config_path = root / config_match.group(1) if config_match else None
        candidates.append(Candidate(str(path), "launch_declaration", score, max_steps, 0,
                                    str(config_path) if config_path and config_path.is_file() else "", "",
                                    ["launch script declaration only; not proof of a completed run"]))
    return candidates


def _budget_from_candidate(candidate: Candidate) -> dict:
    config = _load_yaml(Path(candidate.config_path)) if candidate.config_path else {}
    script_text = Path(candidate.path).read_text(encoding="utf-8", errors="replace") if candidate.kind == "launch_declaration" else ""

    def script_int(name, default=0):
        match = re.search(rf"^{name}=(\d+)", script_text, re.M)
        return int(match.group(1)) if match else default

    batch = int(_nested(config, "data.batch_size", "datamodule_args.dataloader_args.batch_size", default=0) or script_int("BATCH_SIZE"))
    accumulate = int(_nested(config, "trainer.accumulate_grad_batches", "trainer_args.accumulate_grad_batches", default=0) or script_int("ACCUMULATE", 1))
    learning_rate = float(_nested(config, "optimizer.lr", "model_args.lr", default=0.0) or 0.0)
    max_steps = max(candidate.max_steps, candidate.checkpoint_global_step)
    confidence = "high" if candidate.kind == "resolved_run" and candidate.checkpoint_global_step and candidate.max_steps else "low"
    return {
        "reference_run": candidate.path,
        "config_path": candidate.config_path,
        "checkpoint_path": candidate.checkpoint_path,
        "max_steps": max_steps,
        "checkpoint_global_step": candidate.checkpoint_global_step,
        "batch_size": batch,
        "accumulate_grad_batches": accumulate,
        "effective_batch_size": batch * accumulate,
        "learning_rate": learning_rate,
        "scheduler": str(_nested(config, "optimizer.scheduler", "model_args.lr_scheduler_type", default="unknown")),
        "optimizer": str(_nested(config, "optimizer.type", "model_args.optimizer_type", default="unknown")),
        "t_min": float(_nested(config, "time_sampling.t_min", "eval_args.sampler_args.t_min", default=0.0) or 0.0),
        "t_max": float(_nested(config, "time_sampling.t_max", "eval_args.sampler_args.t_max", default=0.0) or 0.0),
        "seed": int(_nested(config, "seed", default=0) or 0),
        "precision": str(_nested(config, "trainer.precision", "trainer_args.precision", default="unknown")),
        "gpu_count": int(_nested(config, "trainer.devices", "trainer_args.devices", default=1) or 1),
        "train_split": str(_nested(config, "data.train_split", default="train")),
        "val_split": str(_nested(config, "data.val_split", default="val")),
        "train_num_molecules": int(_nested(config, "data.train_num_molecules", default=0) or 0),
        "val_num_molecules": int(_nested(config, "data.val_num_molecules", default=0) or 0),
        "validation_frequency": int(_nested(config, "trainer.val_check_interval", "trainer_args.val_check_interval", default=0) or script_int("VAL_CHECK_INTERVAL")),
        "checkpoint_interval": 0,
        "start_time": "unknown",
        "end_time": "unknown",
        "git_commit": "unknown",
        "confidence": confidence,
        "evidence": candidate.evidence,
    }


def _write_report(output_md: Path, budget: dict, candidates: list[Candidate], ambiguous: bool) -> None:
    lines = ["# Reference FlexBond-4D training budget", "",
             f"Confidence: **{budget.get('confidence', 'none')}**", "",
             f"Ambiguous: **{ambiguous}**", "",
             "## Selected evidence", ""]
    for key, value in budget.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Candidates", "", "| score | kind | max steps | checkpoint step | path |", "|---:|---|---:|---:|---|"])
    for row in sorted(candidates, key=lambda item: item.score, reverse=True):
        lines.append(f"| {row.score} | {row.kind} | {row.max_steps} | {row.checkpoint_global_step} | `{row.path}` |")
    if budget.get("confidence") != "high":
        lines.extend(["", "> Formal training is blocked: a launch script is not proof of a completed reference run."])
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output_json", type=Path, default=ROOT / "reports/reference_4d_training_budget.json")
    parser.add_argument("--output_md", type=Path, default=ROOT / "reports/reference_4d_training_budget.md")
    args = parser.parse_args()
    candidates = _run_candidates(args.root) + _script_candidates(args.root)
    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    ambiguous = len(ranked) > 1 and ranked[0].score == ranked[1].score
    if ranked and not ambiguous:
        budget = _budget_from_candidate(ranked[0])
    else:
        budget = {
            "reference_run": "", "config_path": "", "checkpoint_path": "",
            "max_steps": 0, "checkpoint_global_step": 0, "batch_size": 0,
            "accumulate_grad_batches": 0, "effective_batch_size": 0,
            "learning_rate": 0.0, "scheduler": "unknown", "optimizer": "unknown",
            "t_min": 0.0, "t_max": 0.0, "seed": 0, "precision": "unknown",
            "train_split": "unknown", "val_split": "unknown",
            "train_num_molecules": 0, "val_num_molecules": 0,
            "git_commit": "unknown", "confidence": "none",
            "reason": "no unique completed formal run candidate",
        }
    payload = {**budget, "ambiguous": ambiguous, "candidates": [asdict(row) for row in ranked]}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(args.output_md, budget, ranked, ambiguous)
    print(json.dumps({"confidence": budget.get("confidence"), "reference_run": budget.get("reference_run"), "max_steps": budget.get("max_steps"), "ambiguous": ambiguous}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
