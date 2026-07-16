from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
import yaml

from etflow.ecir.mvr_model import trust_clip_velocity
from etflow.ecir.mvr_safety import (
    evaluate_validation_safety,
    evaluate_velocity_safety,
    trust_clip_with_diagnostics,
)


ROOT = Path(__file__).resolve().parents[1]
V3_CONFIG = ROOT / "configs/ecir_mvr_medium_5k_500_run_a_seed42_20k_rescue_v3.yaml"


def _metrics(*, raw_graph=0.08, raw_atom=0.08, clipped_graph=0.06, clipped_atom=0.06, fraction=1.0):
    return {
        "raw_velocity_atom_mean": raw_atom,
        "raw_velocity_atom_p95": raw_atom,
        "raw_velocity_atom_max": raw_atom,
        "raw_velocity_graph_rms": raw_graph,
        "clipped_velocity_atom_mean": clipped_atom,
        "clipped_velocity_atom_p95": clipped_atom,
        "clipped_velocity_atom_max": clipped_atom,
        "clipped_velocity_graph_rms": clipped_graph,
        "graph_clipped_fraction": fraction,
        "atom_clipped_fraction": fraction,
    }


def _velocity_decision(metrics, history=()):
    return evaluate_velocity_safety(
        metrics, max_velocity_graph_rms_after_clip=0.06,
        max_velocity_atom_norm_after_clip=0.12, recent_raw_metrics=history,
    )


def _validation(step, validity, displacement, accuracy=0.001, *, clash=0.0, chirality=0.0, identity=1.0):
    return {
        "step": step, "validity_delta": validity,
        "mean_displacement": displacement, "severe_clash_delta": clash,
        "chirality_delta": chirality, "identity_fraction": identity,
        "bootstrap": {
            name: {"mean": accuracy} for name in ("aligned_RMSD", "MAT_P", "MAT_R")
        },
    }


def test_raw_graph_008_clipped_006_is_not_immediate_stop():
    velocity = torch.tensor([[0.08, 0.0, 0.0]])
    clipped, diagnostics = trust_clip_with_diagnostics(
        velocity, torch.zeros(1, dtype=torch.long), max_atom_norm=0.12, max_graph_rms=0.06
    )
    metrics = _metrics(
        raw_graph=diagnostics["raw"]["graph_rms"],
        raw_atom=diagnostics["raw"]["atom_max"],
        clipped_graph=diagnostics["clipped"]["graph_rms"],
        clipped_atom=diagnostics["clipped"]["atom_max"],
        fraction=diagnostics["graph_clipped_fraction"],
    )
    assert _velocity_decision(metrics)["status"] == "WARNING"
    assert float(torch.linalg.vector_norm(clipped, dim=-1).max()) <= 0.06000001


def test_clipped_float_roundoff_at_00600000024_does_not_stop():
    assert _velocity_decision(_metrics(clipped_graph=0.0600000024))["status"] == "WARNING"


def test_clipped_true_tolerance_excess_stops_immediately():
    decision = _velocity_decision(_metrics(clipped_graph=0.060002))
    assert decision == {"status": "HARD_STOP", "reason": "graph_trust_clipping_failed"}


def test_occasional_raw_trust_excess_is_warning_only():
    assert _velocity_decision(_metrics(raw_graph=0.09, fraction=0.25))["status"] == "WARNING"


def test_sustained_high_raw_and_clipping_with_safe_output_is_warning():
    row = _metrics(raw_graph=0.10, raw_atom=0.20, fraction=0.90)
    decision = _velocity_decision(row, [row] * 4)
    assert decision["status"] == "WARNING"


def test_two_composite_validation_transitions_are_required_for_stop():
    history = [
        _validation(3000, -0.1, 0.001),
        _validation(5000, 0.01, 0.002),
    ]
    assert evaluate_validation_safety(history)["status"] == "INFO"
    history.append(_validation(10000, 0.02, 0.003))
    assert evaluate_validation_safety(history)["reason"] == "two_validations_joint_validity_displacement_accuracy_worsening"


def test_nan_or_inf_velocity_stops_immediately():
    for value in (math.nan, math.inf):
        assert _velocity_decision(_metrics(raw_graph=value))["reason"] == "velocity_nan_or_inf"


def test_severe_clash_chirality_and_identity_rules_are_unchanged():
    assert evaluate_validation_safety([_validation(3000, -0.1, 0.001, clash=0.01)])["reason"] == "severe_clash_increased"
    assert evaluate_validation_safety([_validation(3000, -0.1, 0.001, chirality=0.01)])["reason"] == "chirality_worsened"
    assert evaluate_validation_safety([_validation(3000, -0.1, 0.001, identity=0.89)])["reason"] == "clean_identity_below_90pct"


def test_trust_clipping_mathematics_is_bitwise_unchanged():
    generator = torch.Generator().manual_seed(42)
    raw = torch.randn((40, 3), generator=generator)
    atom_batch = torch.arange(40) // 10
    legacy = trust_clip_velocity(raw, atom_batch, max_atom_norm=0.12, max_graph_rms=0.06)
    audited, _ = trust_clip_with_diagnostics(raw, atom_batch, max_atom_norm=0.12, max_graph_rms=0.06)
    assert torch.equal(legacy, audited)


def test_v2_checkpoint_and_frozen_identities_are_unchanged():
    config = yaml.safe_load(V3_CONFIG.read_text(encoding="utf-8"))
    checkpoint = ROOT / config["resume_checkpoint"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == "736cbe38a44396ed6d4c0da0af017b7f7cd622d333b02a07d225d9d0bc2e7b1e"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["global_step"] == 2450
    assert payload["frozen_identities"] == config["frozen_identities"]
    audit = json.loads((ROOT / config["provenance"]["raw_vs_clipped_audit"]).read_text(encoding="utf-8"))
    assert audit["decision"] == "POST_CLIP_THRESHOLD_SELF_TRIGGER"
