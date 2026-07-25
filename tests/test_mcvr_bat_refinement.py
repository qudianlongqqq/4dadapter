from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from etflow.ecir.bat_refinement import (
    BATGraph,
    TorsionHead,
    _topology_distances,
    canonical_rotatable_torsions,
    circular_difference,
    combined_gradient_update,
    dihedral_angles,
    frozen_ba_update,
    mixture_log_prob,
    mixture_responsibilities,
    prepare_bat_graph,
    steric_barrier,
    steric_metrics,
    steric_pairs,
    torsion_pathology,
    von_mises_log_prob,
    wrap_periodic,
)
from etflow.ecir.learned_geometry import (
    GraphGeometry,
    direct_gradient_update,
    distribution_parameters,
    prepare_graph,
    safety_accept,
)
from etflow.ecir.lsgo_io import file_sha256


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/ecir_mvr/bat_refinement"
CONFIG = yaml.safe_load((ROOT / "configs/ecir_mvr_bat_refinement.yaml").read_text(encoding="utf-8"))
COMPACT = Path(CONFIG["dataset"]["training_compact"])
CALIBRATION = Path(CONFIG["dataset"]["drcsr_calibration"])


@pytest.fixture(scope="module")
def example():
    payload = torch.load(COMPACT, map_location="cpu", weights_only=False)
    item = payload["items"][0]
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    graph = prepare_graph(item["record"], calibration)
    return item, graph, prepare_bat_graph(graph, item["record"])


def minimal_graph(atom_numbers=(6, 6, 6, 6), bonds=((0, 1), (1, 2), (2, 3))):
    atom_count = len(atom_numbers)
    edge = []
    for left, right in bonds:
        edge.extend(((left, right), (right, left)))
    edge_index = torch.tensor(edge, dtype=torch.long).T if edge else torch.empty((2, 0), dtype=torch.long)
    unique = torch.tensor(bonds, dtype=torch.long).T if bonds else torch.empty((2, 0), dtype=torch.long)
    return GraphGeometry(
        atom_categorical=torch.tensor([[z, 5, 4, 0, 0, 2] for z in atom_numbers]),
        edge_index=edge_index,
        edge_categorical=torch.ones((len(edge), 3), dtype=torch.long),
        bonds=unique,
        bond_categorical=torch.ones((len(bonds), 3), dtype=torch.long),
        angles=torch.empty((0, 3), dtype=torch.long),
        angle_edge_categorical=torch.empty((0, 6), dtype=torch.long),
        rings=(), chirality=(),
    )


def minimal_bat(pair=(0, 3), safe=2.0, catastrophic=1.2):
    base = minimal_graph()
    return BATGraph(
        base=base,
        torsions=torch.empty((0, 4), dtype=torch.long),
        torsion_categorical=torch.empty((0, 5), dtype=torch.long),
        nonbonded_pairs=torch.tensor([pair], dtype=torch.long),
        safe_distances=torch.tensor([safe], dtype=torch.float64),
        catastrophic_distances=torch.tensor([catastrophic], dtype=torch.float64),
        topology_distances=torch.tensor([3], dtype=torch.long),
        canonical_metadata=(),
    )


@pytest.mark.parametrize("value", [-9 * math.pi, -3.2, -math.pi, -0.1, 0.0, 0.1, math.pi, 3.2, 9 * math.pi])
def test_wrap_periodic_range_and_periodicity(value):
    result = wrap_periodic(torch.tensor(value, dtype=torch.float64))
    shifted = wrap_periodic(torch.tensor(value + 8 * math.pi, dtype=torch.float64))
    assert -math.pi <= float(result) < math.pi
    assert torch.allclose(result, shifted, atol=1e-12)


@pytest.mark.parametrize("offset", [-4, -2, 0, 2, 4])
def test_circular_difference_periodic(offset):
    value = torch.tensor(math.pi - 0.05 + offset * math.pi)
    target = torch.tensor(-math.pi + 0.05)
    assert abs(abs(float(circular_difference(value, target))) - 0.1) < 1e-5


def test_dihedral_known_right_angle_and_reverse():
    xyz = torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 1., 0.], [0., 1., 1.]], dtype=torch.float64)
    q = torch.tensor([[0, 1, 2, 3]])
    forward = dihedral_angles(xyz, q)
    reverse = dihedral_angles(xyz, torch.flip(q, dims=[1]))
    assert torch.allclose(forward, reverse, atol=1e-12)
    assert math.isclose(abs(float(forward)), math.pi / 2, abs_tol=1e-12)


def test_dihedral_degenerate_is_finite_zero():
    xyz = torch.zeros((4, 3), dtype=torch.float64)
    assert torch.equal(dihedral_angles(xyz, torch.tensor([[0, 1, 2, 3]])), torch.zeros(1, dtype=torch.float64))


def test_canonical_rotors_are_deterministic_unique_and_nonring(example):
    item, _, bat = example
    second = canonical_rotatable_torsions(item["record"])
    assert torch.equal(bat.torsions, second[0])
    central = [tuple(sorted(q[1:3])) for q in bat.torsions.tolist()]
    assert len(central) == len(set(central))
    assert all(not row["ring_bond"] and row["bond_type"] == "SINGLE" for row in bat.canonical_metadata)


def test_canonical_rotor_shapes_and_atom_bounds(example):
    _, graph, bat = example
    assert bat.torsions.shape[1:] == (4,)
    assert bat.torsion_categorical.shape == (bat.torsions.size(0), 5)
    assert int(bat.torsions.max()) < graph.atom_categorical.size(0)


@pytest.mark.parametrize("kappa", [0.0, 0.25, 1.0, 8.0, 32.0])
def test_von_mises_is_finite_and_peak_is_at_mean(kappa):
    center = von_mises_log_prob(torch.tensor(0.2), torch.tensor(0.2), torch.tensor(kappa))
    opposite = von_mises_log_prob(torch.tensor(0.2 + math.pi), torch.tensor(0.2), torch.tensor(kappa))
    assert torch.isfinite(center)
    assert center >= opposite


def test_von_mises_kappa_zero_is_uniform():
    actual = von_mises_log_prob(torch.tensor([0., 1., 2.]), torch.zeros(3), torch.zeros(3))
    assert torch.allclose(actual, torch.full((3,), -math.log(2 * math.pi)), atol=1e-6)


def test_mixture_logsumexp_and_responsibilities():
    values = torch.tensor([0.1, -2.0])
    logits = torch.tensor([[0., 1., -1.], [2., 0., -2.]])
    means = torch.tensor([[0., 2., -2.], [0., 2., -2.]])
    kappa = torch.full_like(means, 8.)
    log_prob = mixture_log_prob(values, logits, means, kappa)
    responsibilities = mixture_responsibilities(values, logits, means, kappa)
    assert log_prob.shape == values.shape and torch.isfinite(log_prob).all()
    assert torch.allclose(responsibilities.sum(-1), torch.ones(2), atol=1e-6)


@pytest.mark.parametrize("components,learned", [(1, False), (3, False), (3, True)])
def test_torsion_head_shapes_and_kappa_bounds(components, learned):
    head = TorsionHead(hidden_dim=16, components=components, learned_kappa=learned)
    prediction = head(torch.randn((8, 16)), torch.tensor([[0, 1, 2, 3], [3, 4, 5, 6]]), fixed_kappa=8.)
    assert prediction["means"].shape == (2, components)
    assert torch.allclose(prediction["weights"].sum(-1), torch.ones(2), atol=1e-6)
    assert float(prediction["kappa"].detach().min()) >= .25 and float(prediction["kappa"].detach().max()) <= 32.
    if not learned:
        assert torch.equal(prediction["kappa"], torch.full_like(prediction["kappa"], 8.))


def test_torsion_head_is_below_parameter_limit():
    assert sum(p.numel() for p in TorsionHead().parameters()) < 500_000


def test_torsion_pathology_empty_and_duplicates():
    empty = {key: torch.empty((0, 3)) for key in ("weights", "means", "kappa")}
    assert torsion_pathology(empty)["empty"]
    prediction = {
        "weights": torch.full((10, 3), 1 / 3),
        "means": torch.zeros((10, 3)),
        "kappa": torch.full((10, 3), 8.),
    }
    assert torsion_pathology(prediction)["mode_duplication_fraction"] == 1.0


@pytest.mark.parametrize("pair,distance", [((0, 1), 1), ((0, 2), 2), ((0, 3), 3)])
def test_topology_distances_on_chain(pair, distance):
    assert _topology_distances(4, torch.tensor([[0, 1, 2], [1, 2, 3]]))[pair] == distance


def test_steric_pairs_exclude_12_13_and_keep_14():
    pairs, safe, catastrophic, relation = steric_pairs(minimal_graph())
    assert pairs.tolist() == [[0, 3]] and relation.tolist() == [3]
    assert torch.all(safe > catastrophic)


def test_steric_pairs_exclude_hydrogen_by_default():
    pairs, *_ = steric_pairs(minimal_graph(atom_numbers=(6, 6, 6, 1)))
    assert pairs.numel() == 0


def test_steric_14_threshold_is_less_than_nonbonded():
    base = minimal_graph(atom_numbers=(6, 6, 6, 6, 6), bonds=((0, 1), (1, 2), (2, 3)))
    pairs, safe, _, relation = steric_pairs(base)
    fourteen = safe[relation == 3][0]
    remote = safe[(pairs == torch.tensor([0, 4])).all(-1)][0]
    assert fourteen < remote


def test_steric_barrier_exact_zero_in_safe_region():
    xyz = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [3., 0., 0.]], dtype=torch.float64, requires_grad=True)
    energy, payload = steric_barrier(xyz, minimal_bat())
    gradient, = torch.autograd.grad(energy, xyz)
    assert energy == 0 and not payload["active"].any() and torch.equal(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize("distance", [1.9, 1.5, 1.0, 0.5])
def test_steric_barrier_increases_with_penetration(distance):
    xyz = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [distance, 0., 0.]], dtype=torch.float64)
    energy, _ = steric_barrier(xyz, minimal_bat())
    shallower = max(distance, 1.9)
    baseline, _ = steric_barrier(torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [shallower, 0., 0.]], dtype=torch.float64), minimal_bat())
    assert energy >= baseline


def test_steric_gradient_pushes_pair_apart():
    xyz = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [1.5, 0., 0.]], dtype=torch.float64, requires_grad=True)
    energy, _ = steric_barrier(xyz, minimal_bat())
    gradient, = torch.autograd.grad(energy, xyz)
    assert gradient[0, 0] > 0 and gradient[3, 0] < 0


@pytest.mark.parametrize("distance,violations,catastrophic", [(3., 0, 0), (1.5, 1, 0), (1.0, 1, 1)])
def test_steric_metrics_thresholds(distance, violations, catastrophic):
    xyz = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [distance, 0., 0.]], dtype=torch.float64)
    metrics = steric_metrics(xyz, minimal_bat())
    assert metrics["violation_count"] == violations and metrics["catastrophic_count"] == catastrophic


def test_bat_graph_to_preserves_metadata(example):
    _, _, bat = example
    moved = bat.to("cpu")
    assert moved.canonical_metadata == bat.canonical_metadata
    assert torch.equal(moved.nonbonded_pairs, bat.nonbonded_pairs)


def test_exact_frozen_ba_inheritance(example):
    item, graph, bat = example
    source = torch.as_tensor(item["sources"][0], dtype=torch.float64)
    parameters = distribution_parameters(graph, model=None, variant="A")
    historical = direct_gradient_update(source, graph, parameters, rms_budget=.003, atom_cap=.03, steps=1)
    expected = safety_accept(source, historical["coordinates"], graph)[0]
    actual = frozen_ba_update(source, bat, parameters, rms_budget=.003, atom_cap=.03)["coordinates"]
    assert torch.equal(actual, expected)


def test_combined_update_respects_trust_and_safety(example):
    item, graph, bat = example
    source = torch.as_tensor(item["sources"][0], dtype=torch.float64)
    parameters = distribution_parameters(graph, model=None, variant="A")
    result = combined_gradient_update(source, bat, parameters, rms_budget=.003, atom_cap=.03, tau=.05)
    delta = result["coordinates"] - source
    assert float(torch.sqrt(delta.square().sum(-1).mean())) <= .003 + 1e-12
    assert float(torch.linalg.vector_norm(delta, dim=-1).max()) <= .03 + 1e-12
    assert result["steric_after"]["catastrophic_count"] <= result["steric_before"]["catastrophic_count"]


def test_checkpoint_manifest_shas_and_no_learned_sigma():
    manifest = json.loads(Path(CONFIG["ba_anchor"]["manifest"]).read_text(encoding="utf-8"))
    rows = [row for row in manifest["checkpoints"] if row["variant"] == "B" and row["seed"] in CONFIG["ba_seeds"]]
    assert len(rows) == 3
    assert all(file_sha256(Path(row["path"])) == row["sha256"] for row in rows)
    assert CONFIG["ba_anchor"]["learned_sigma"] is False and CONFIG["guards"]["learned_sigma_rescue"] is False


def test_dataset_identity_and_protected_split_guards():
    identity = json.loads((OUT / "DATASET_IDENTITY.json").read_text(encoding="utf-8"))
    exclusion = json.loads((OUT / "OLD_EXTERNAL_EXCLUSION_AUDIT.json").read_text(encoding="utf-8"))
    assert identity["formal_test_records_read"] == identity["frozen_holdout_records_read"] == 0
    assert exclusion["overlap_with_exposed_union"] == 0 and exclusion["status"] == "PASS"
    assert identity["cohorts"]["BAT_EXTERNAL_CONFIRM"]["molecule_count"] == 200


@pytest.mark.parametrize("forbidden", ["MinimalValidityTargetBuilder", "PoseBusters", "xtb", "learned_sigma", "LPWGP"])
def test_bat_primitive_has_no_forbidden_training_interface(forbidden):
    source = inspect.getsource(__import__("etflow.ecir.bat_refinement", fromlist=["*"]))
    if forbidden == "learned_sigma":
        assert "learned_sigma" not in source
    else:
        assert forbidden.lower() not in source.lower()
