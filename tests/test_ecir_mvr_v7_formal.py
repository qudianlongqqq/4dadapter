from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from etflow.ecir.mvr_v2_bac import MCVRBACModel
from etflow.ecir.mvr_v7_formal import (
    build_v7_formal_model,
    load_v7_formal_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ecir_mvr_v7_formal_large.yaml"


def _checkpoint() -> dict[str, object]:
    model_config = {
        "hidden_dim": 16,
        "edge_hidden_dim": 16,
        "time_embedding_dim": 8,
        "num_layers": 2,
        "encoder_num_layers": 2,
        "error_embedding_dim": 8,
        "bond_head_enabled": True,
        "bond_explicit_alpha": 1.0,
        "torsion_gate_fixed_zero": True,
    }
    model = MCVRBACModel(**model_config)
    return {
        "schema_version": "ecir-mvr-medium-rescue-formal-large-d1b-checkpoint-v1",
        "model_type": "MCVRModel",
        "step": 25_000,
        "config": {"model": model_config, "seed": 42},
        "model_state_dict": model.state_dict(),
    }


def test_v7_formal_config_matches_frozen_development_values() -> None:
    config = load_v7_formal_config(CONFIG)
    assert config["v7"] == {
        "integration_step_size": 0.25,
        "angle_max_graph_rms": 0.01,
        "angle_max_atom": 0.02,
        "clash_max_graph_rms": 0.01,
        "clash_max_atom": 0.02,
        "clash_cutoff": 2.0,
        "clash_allowed_contact": 1.0,
        "clash_exclude_topology_distance": 2,
        "max_clash_edges_per_graph": 128,
        "jacobian_config": {
            "damping_lambda": 1.0e-3,
            "rank_tol": 1.0e-6,
            "max_condition_number": 1.0e8,
            "near_linear_sine_threshold": 1.0e-3,
            "near_linear_weight": 0.1,
        },
    }
    assert config["inference"]["teacher_steps"] == 4
    assert config["inference"]["step_size"] == 0.25
    assert config["checkpoint_or_config_selected_from_test"] is False
    assert config["test_records_read"] == 0


def test_v7_formal_factory_strict_loads_parameter_free_wrapper() -> None:
    model = build_v7_formal_model(
        _checkpoint(), load_v7_formal_config(CONFIG), device="cpu"
    )
    assert not model.prior.has_bac_modules
    assert all(not parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "wrong", "schema changed"),
        ("model_type", "WrongModel", "model type changed"),
        ("step", 18_750, "not the completed checkpoint"),
    ],
)
def test_v7_formal_factory_rejects_wrong_checkpoint_contract(
    field: str, value: object, message: str
) -> None:
    checkpoint = _checkpoint()
    checkpoint[field] = value
    with pytest.raises(RuntimeError, match=message):
        build_v7_formal_model(
            checkpoint, load_v7_formal_config(CONFIG), device="cpu"
        )


def test_v7_formal_factory_rejects_learned_bac_prior() -> None:
    checkpoint = deepcopy(_checkpoint())
    checkpoint["config"]["model"]["bac_mode"] = "V2_D_BOND_ANGLE_CLASH"
    with pytest.raises(RuntimeError, match="learned BAC modules"):
        build_v7_formal_model(
            checkpoint, load_v7_formal_config(CONFIG), device="cpu"
        )
