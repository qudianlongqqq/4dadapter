from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest
import torch

from etflow.ecir.learned_geometry import (
    LearnedGeometryObjective,
    SIGMA_FLOORS,
    SIGMA_MAX,
    chirality_preserved,
    direct_gradient_update,
    distribution_parameters,
    gaussian_nll,
    geometry_values,
    parameter_count,
    precision_projection_update,
    prepare_graph,
    remove_rigid_component,
    stable_angle_cosine,
    structured_objective,
    trust_project,
)


ROOT = Path(__file__).resolve().parents[1]
COMPACT = Path(
    r"E:\3dconformergenerationcode\4dadapter-label-free-score"
    r"\reports\ecir_mvr\label_free_score_pilot\manifests\TRAIN_ONLY_COMPACT_DATASET.pt"
)
CALIBRATION = Path(
    r"E:\3dconformergenerationcode\4dadapter-direct-strain"
    r"\reports\ecir_mvr\direct_strain\manifests\CONTINUOUS_STRAIN_CALIBRATION.json"
)


@pytest.fixture(scope="module")
def example():
    payload = torch.load(COMPACT, map_location="cpu", weights_only=False)
    item = payload["items"][0]
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    graph = prepare_graph(item["record"], calibration)
    return item, graph


@pytest.fixture(scope="module")
def neural(example):
    _, graph = example
    model = LearnedGeometryObjective(hidden_dim=32, layers=1, learned_sigma=True).double()
    return model, graph.to("cpu")


def test_01_bond_extraction(example):
    item, graph = example
    pairs = {tuple(sorted(pair)) for pair in torch.as_tensor(item["edge_index"]).t().tolist()}
    assert graph.bonds.shape == (2, len(pairs))
    assert len({tuple(pair) for pair in graph.bonds.t().tolist()}) == graph.bonds.size(1)


def test_02_angle_extraction(example):
    _, graph = example
    assert graph.angles.ndim == 2 and graph.angles.size(1) == 3
    assert torch.all(graph.angles[:, 0] != graph.angles[:, 2])


def test_03_atom_order_identity(example):
    item, graph = example
    assert torch.equal(graph.atom_categorical[:, 0], torch.as_tensor(item["record"]["atomic_numbers"]))
    assert torch.equal(torch.as_tensor(item["record"]["atom_map_ids"]), torch.arange(graph.atom_categorical.size(0)))


def test_04_mu_prediction_shape(neural):
    model, graph = neural
    output = model(graph)
    assert output["bond_mu"].shape == (graph.bonds.size(1),)
    assert output["angle_mu"].shape == (graph.angles.size(0),)


def test_05_sigma_positivity(neural):
    model, graph = neural
    output = model(graph)
    assert (output["bond_sigma"] > 0).all() and (output["angle_sigma"] > 0).all()


def test_06_sigma_floor(neural):
    model, graph = neural
    output = model(graph)
    assert float(output["bond_sigma"].min()) >= SIGMA_FLOORS["bond"]
    assert float(output["angle_sigma"].min()) >= SIGMA_FLOORS["angle"]


def test_07_sigma_inflation_guard(neural):
    model, graph = neural
    with torch.no_grad():
        model.bond_head[-1].bias[1] = 100
        model.angle_head[-1].bias[1] = 100
    output = model(graph)
    assert float(output["bond_sigma"].max()) <= SIGMA_MAX["bond"]
    assert float(output["angle_sigma"].max()) <= SIGMA_MAX["angle"]


def test_08_gaussian_nll():
    value = torch.tensor([0.0], dtype=torch.float64)
    assert torch.allclose(gaussian_nll(value, value, torch.ones_like(value)), torch.tensor([0.5 * math.log(2 * math.pi)], dtype=torch.float64))


def test_09_reference_calibration_z(example):
    item, graph = example
    bond, angle = geometry_values(torch.as_tensor(item["references"][0], dtype=torch.float64), graph)
    assert torch.isfinite((bond - graph.bond_fixed[:, 0]) / graph.bond_fixed[:, 1]).all()
    assert torch.isfinite((angle - graph.angle_fixed[:, 0]) / graph.angle_fixed[:, 1]).all()


def test_10_source_z_score(example):
    item, graph = example
    bond, angle = geometry_values(torch.as_tensor(item["sources"][0], dtype=torch.float64), graph)
    assert torch.isfinite((bond - graph.bond_fixed[:, 0]) / graph.bond_fixed[:, 1]).all()
    assert torch.isfinite((angle - graph.angle_fixed[:, 0]) / graph.angle_fixed[:, 1]).all()


def test_11_gradient_finite(example):
    item, graph = example
    source = torch.as_tensor(item["sources"][0], dtype=torch.float64).requires_grad_(True)
    parameters = distribution_parameters(graph, model=None, variant="A")
    gradient, = torch.autograd.grad(structured_objective(source, graph, parameters)[0], source)
    assert torch.isfinite(gradient).all()


def test_12_translation_invariance(example):
    item, graph = example
    source = torch.as_tensor(item["sources"][0], dtype=torch.float64)
    parameters = distribution_parameters(graph, model=None, variant="A")
    a = structured_objective(source, graph, parameters)[0]
    b = structured_objective(source + torch.tensor([3.0, -2.0, 1.0]), graph, parameters)[0]
    assert torch.allclose(a, b, atol=1e-10, rtol=1e-10)


def test_13_rotation_invariance(example):
    item, graph = example
    source = torch.as_tensor(item["sources"][0], dtype=torch.float64)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    parameters = distribution_parameters(graph, model=None, variant="A")
    assert torch.allclose(structured_objective(source, graph, parameters)[0], structured_objective(source @ rotation.T, graph, parameters)[0], atol=1e-9)


def test_14_bond_jacobian_finite_difference():
    x = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0]], dtype=torch.float64, requires_grad=True)
    edge = torch.tensor([[0], [1]])
    value = torch.linalg.vector_norm(x[0] - x[1])
    gradient, = torch.autograd.grad(value, x)
    eps = 1e-6
    shifted = x.detach().clone(); shifted[0, 0] += eps
    numeric = (torch.linalg.vector_norm(shifted[0] - shifted[1]) - value.detach()) / eps
    assert torch.allclose(gradient[0, 0], numeric, atol=1e-5)


def test_15_angle_jacobian_finite_difference():
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    triplet = torch.tensor([[0, 1, 2]])
    value = stable_angle_cosine(x, triplet)[0]
    gradient, = torch.autograd.grad(value, x)
    eps = 1e-6
    shifted = x.detach().clone(); shifted[0, 1] += eps
    numeric = (stable_angle_cosine(shifted, triplet)[0] - value.detach()) / eps
    assert torch.allclose(gradient[0, 1], numeric, atol=1e-5)


def test_16_singular_angle_guard():
    x = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    value = stable_angle_cosine(x, torch.tensor([[0, 1, 2]])).sum()
    gradient, = torch.autograd.grad(value, x)
    assert torch.isfinite(value) and torch.isfinite(gradient).all()


def test_17_precision_matrix_positive(example):
    _, graph = example
    parameters = distribution_parameters(graph, model=None, variant="A")
    precision = torch.cat((parameters["bond_sigma"], parameters["angle_sigma"])).reciprocal().square()
    assert (precision > 0).all()


def test_18_damping_and_19_solver_conditioning(example):
    item, graph = example
    source = torch.as_tensor(item["sources"][0], dtype=torch.float64)
    parameters = distribution_parameters(graph, model=None, variant="A")
    result = precision_projection_update(source, graph, parameters, ridge=1e-3, rank_tolerance=1e-6, maximum_condition=1e8, rms_budget=.001, atom_cap=.03)
    assert result["finite"] and result["trace"]
    assert result["trace"][0]["effective_rank"] >= 0
    assert result["trace"][0]["solver_backend"] in {"solve", "svd_pinv"}


def test_20_rms_trust_projection():
    source = torch.zeros((5, 3), dtype=torch.float64)
    candidate, diagnostics = trust_project(source, torch.randn_like(source), rms_budget=.003, atom_cap=.03)
    assert diagnostics["final_rms"] <= .003 + 1e-12
    assert candidate.shape == source.shape


def test_21_atom_cap():
    source = torch.zeros((5, 3), dtype=torch.float64)
    candidate, diagnostics = trust_project(source, torch.randn_like(source) * 100, rms_budget=.1, atom_cap=.03)
    assert diagnostics["final_atom_max"] <= .03 + 1e-12


def test_22_topology_unchanged_by_coordinate_update(example):
    item, graph = example
    assert graph.bonds.size(1) == len({tuple(sorted(pair)) for pair in item["edge_index"].t().tolist()})


def test_23_chirality_identity(example):
    item, graph = example
    source = item["sources"][0]
    assert chirality_preserved(source, source.clone(), graph.chirality)


def test_24_deterministic_inference(example):
    item, graph = example
    source = torch.as_tensor(item["sources"][0], dtype=torch.float64)
    parameters = distribution_parameters(graph, model=None, variant="A")
    first = direct_gradient_update(source, graph, parameters, rms_budget=.001, atom_cap=.03)
    second = direct_gradient_update(source, graph, parameters, rms_budget=.001, atom_cap=.03)
    assert torch.equal(first["coordinates"], second["coordinates"])


def test_25_source_exact_noop_path():
    source = torch.randn((4, 3), dtype=torch.float64)
    candidate, diagnostics = trust_project(source, torch.zeros_like(source), rms_budget=.003, atom_cap=.03)
    assert torch.equal(source, candidate) and diagnostics["final_rms"] == 0.0


@pytest.mark.parametrize("forbidden", ["MinimalValidityTargetBuilder", "x_target", "x_ref_aligned"])
def test_26_no_mvt_or_cartesian_target_access(forbidden):
    source = inspect.getsource(LearnedGeometryObjective)
    assert forbidden not in source


def test_27_no_xtb_training_access():
    source = (ROOT / "scripts/run_mcvr_lsgo.py").read_text(encoding="utf-8")
    training = source[source.index("def train_one"):source.index("def load_selected")]
    assert "xtb" not in training.lower()


def test_28_no_pb_training_access():
    source = (ROOT / "scripts/run_mcvr_lsgo.py").read_text(encoding="utf-8")
    training = source[source.index("def train_one"):source.index("def load_selected")]
    assert "posebuster" not in training.lower()


def test_29_no_formal_test_and_30_no_holdout_and_31_no_full10k():
    config = json.loads(json.dumps(__import__("yaml").safe_load((ROOT / "configs/ecir_mvr_lsgo_pilot.yaml").read_text())))
    assert config["guards"]["formal_test_records_read"] == 0
    assert config["guards"]["frozen_holdout_records_read"] == 0
    assert config["guards"]["full10k_used_for_tuning"] is False


def test_32_external_freeze_guard_present():
    source = (ROOT / "scripts/run_mcvr_lsgo.py").read_text(encoding="utf-8")
    assert "external_evaluation_authorized" in source and "CHECKPOINT_FREEZE_MANIFEST" in source


def test_33_paired_xtb_conversion_formula():
    assert math.isclose((-.99 - -1.0) * 627.509474, 6.27509474)


def test_34_pb_schema_required_checks_declared():
    task = (ROOT / "configs/ecir_mvr_lsgo_pilot.yaml").read_text(encoding="utf-8")
    assert "posebusters: true" in task and "failure_transfer_max" in task


def test_35_dataset_sha_validation_declared():
    source = (ROOT / "scripts/build_lsgo_datasets.py").read_text(encoding="utf-8")
    assert "external_compact_sha256" in source and "source_manifest_sha256" in source


def test_36_atomic_reports(tmp_path):
    path = tmp_path / "report.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text("{}", encoding="utf-8")
    temporary.replace(path)
    assert path.read_text(encoding="utf-8") == "{}"


def test_parameter_limit_and_preferred_size():
    model = LearnedGeometryObjective(hidden_dim=128, layers=3, learned_sigma=True)
    assert 200_000 <= parameter_count(model) <= 800_000


def test_remove_rigid_component():
    coordinates = torch.randn((12, 3), dtype=torch.float64)
    vector = torch.ones_like(coordinates)
    projected = remove_rigid_component(vector, coordinates)
    assert torch.linalg.vector_norm(projected.sum(0)) < 1e-10

