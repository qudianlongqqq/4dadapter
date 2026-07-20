import copy

import pytest

from scripts.train_ecir_mvr_v8 import _resume_scientific_identity


def _config():
    return {
        "experiment_name": "5k",
        "steps_total": 5000,
        "model": {"version": "mcvr-v8-full-v1"},
        "loss": {"target_weight": 100.0},
        "constraint_layer": {"solver_lambda_bond": 1.0},
        "training": {
            "optimizer_steps": 5000,
            "batch_size": 16,
            "gradient_accumulation_steps": 4,
            "validation_steps": [1000, 5000],
            "checkpoint_steps": [1000, 5000],
        },
        "deployment_validation": {"steps": [1000, 5000]},
    }


def test_resume_identity_allows_only_horizon_and_validation_schedule_changes():
    pilot = _config()
    long = copy.deepcopy(pilot)
    long["experiment_name"] = "200k"
    long["steps_total"] = 200000
    long["training"]["optimizer_steps"] = 200000
    long["training"]["validation_steps"] = list(range(10000, 200001, 10000))
    long["training"]["checkpoint_steps"] = long["training"]["validation_steps"]
    long["deployment_validation"] = {"enabled": False}
    long["validation_protocol"] = {"fast_every": 10000}
    assert _resume_scientific_identity(pilot) == _resume_scientific_identity(long)


@pytest.mark.parametrize(
    "path,value",
    [
        (("loss", "target_weight"), 99.0),
        (("constraint_layer", "solver_lambda_bond"), 2.0),
        (("training", "batch_size"), 32),
    ],
)
def test_resume_identity_rejects_scientific_or_exposure_changes(path, value):
    pilot = _config()
    changed = copy.deepcopy(pilot)
    changed[path[0]][path[1]] = value
    assert _resume_scientific_identity(pilot) != _resume_scientific_identity(changed)
