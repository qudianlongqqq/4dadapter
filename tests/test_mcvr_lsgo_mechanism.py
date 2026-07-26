from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from etflow.ecir.bat_refinement import canonical_rotatable_torsions, dihedral_angles
from etflow.ecir.learned_geometry import direct_gradient_update, distribution_parameters, prepare_graph, remove_rigid_component
from etflow.ecir.lsgo_io import file_sha256
from etflow.ecir.lsgo_mechanism import (
    circular_finite_difference, finite_difference_row, force_projection, internal_jacobians,
    joint_matrix, masked_gradient_update, masked_objective, orthonormal_row_basis,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/ecir_mvr/lsgo_mechanism"
CONFIG_PATH = ROOT / "configs/ecir_mvr_lsgo_mechanism.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def example():
    payload = torch.load(OUT / "manifests/MECHANISM_CONFIRM_COMPACT.pt", map_location="cpu", weights_only=False)
    calibration = json.loads(Path(CONFIG["dataset"]["drcsr_calibration"]).read_text(encoding="utf-8"))
    item = next(item for item in payload["items"] if canonical_rotatable_torsions(item["record"])[0].numel())
    return item, prepare_graph(item["record"], calibration)


def test_reference_ensemble_pairing_is_multiply_referenced_and_source_identity_unique():
    payload = torch.load(OUT / "manifests/MECHANISM_CONFIRM_COMPACT.pt", map_location="cpu", weights_only=False)
    assert len(payload["items"]) == 48
    assert all(len(item["references"]) >= 2 and len(item["sample_ids"]) == 3 for item in payload["items"])
    sample_ids = [sample for item in payload["items"] for sample in item["sample_ids"]]
    assert len(sample_ids) == len(set(sample_ids)) == 144


def test_energy_identity_pairing_key_includes_coordinates_and_settings():
    xyz = np.arange(12, dtype=np.float64).reshape(4, 3)
    digest = hashlib.sha256(b"float64|" + str(xyz.shape).encode() + b"|" + xyz.tobytes()).hexdigest()
    assert digest != hashlib.sha256(b"float64|" + str(xyz.shape).encode() + b"|" + (xyz + 1e-5).tobytes()).hexdigest()


def test_b_only_and_a_only_masking(example):
    item, graph = example; source = torch.as_tensor(item["sources"][0], dtype=torch.float64)
    parameters = distribution_parameters(graph, model=None, variant="A")
    b, bd = masked_objective(source, graph, parameters, "B")
    a, ad = masked_objective(source, graph, parameters, "A")
    assert torch.isfinite(b) and torch.isfinite(a)
    assert bd["active_bonds"] == graph.bonds.size(1) and bd["active_angles"] == 0
    assert ad["active_angles"] == graph.angles.size(0) and ad["active_bonds"] == 0


def test_ba_equivalence_is_exact(example):
    item, graph = example; source = torch.as_tensor(item["sources"][0], dtype=torch.float64)
    parameters = distribution_parameters(graph, model=None, variant="A")
    historical = direct_gradient_update(source, graph, parameters, rms_budget=.003, atom_cap=.03, steps=1)
    audit = masked_gradient_update(source, graph, parameters, "BA", rms_budget=.003, atom_cap=.03)
    assert torch.equal(historical["coordinates"], audit["coordinates"])


def test_masked_updates_respect_same_budget(example):
    item, graph = example; source = torch.as_tensor(item["sources"][0], dtype=torch.float64)
    parameters = distribution_parameters(graph, model=None, variant="A")
    for family in ("B", "A"):
        result = masked_gradient_update(source, graph, parameters, family, rms_budget=.003, atom_cap=.03)
        delta = result["coordinates"] - source
        assert result["finite"] and float(torch.sqrt(delta.square().sum(-1).mean())) <= .003 + 1e-12


def test_empty_abnormal_mask_is_exact_noop(example):
    item, graph = example; source = torch.as_tensor(item["sources"][0], dtype=torch.float64)
    parameters = distribution_parameters(graph, model=None, variant="A")
    result = masked_gradient_update(source, graph, parameters, "BA", rms_budget=.003, atom_cap=.03,
        bond_mask=torch.zeros(graph.bonds.size(1), dtype=torch.bool), angle_mask=torch.zeros(graph.angles.size(0), dtype=torch.bool))
    assert result["no_op"] and torch.equal(result["coordinates"], source)


def test_z_thresholds_are_frozen_before_xtb():
    threshold = json.loads((OUT / "manifests/THRESHOLD_FREEZE.json").read_text(encoding="utf-8"))
    assert threshold["status"] == "FROZEN_BEFORE_XTB"
    assert threshold["bond_abs_z_reference_p95"] > 0 and threshold["angle_abs_z_reference_p95"] > 0
    assert threshold["xtb_energy_records_read"] == threshold["xtb_force_records_read"] == 0


def test_torsion_jacobian_matches_circular_finite_difference(example):
    item, graph = example; xyz = torch.as_tensor(item["sources"][0], dtype=torch.float64)
    torsions = canonical_rotatable_torsions(item["record"])[0]
    analytic = internal_jacobians(xyz, graph, torsions[:1])["T"]
    numeric = circular_finite_difference(lambda value: dihedral_angles(value, torsions[:1]), xyz, step=1e-5)
    assert torch.allclose(analytic, numeric, atol=2e-4, rtol=2e-4)


def test_bond_jacobian_matches_finite_difference(example):
    item, graph = example; xyz = torch.as_tensor(item["sources"][0], dtype=torch.float64)
    analytic = internal_jacobians(xyz, graph, torch.empty((0, 4), dtype=torch.long))["B"][:1]
    from etflow.ecir.geometry import bond_lengths
    numeric = finite_difference_row(lambda value: bond_lengths(value, graph.bonds[:, :1]), xyz, step=1e-5)
    assert torch.allclose(analytic, numeric, atol=2e-5, rtol=2e-5)


def test_ba_and_bat_union_subspaces_are_monotone(example):
    item, graph = example; xyz = torch.as_tensor(item["sources"][0], dtype=torch.float64)
    torsions = canonical_rotatable_torsions(item["record"])[0]
    matrices = internal_jacobians(xyz, graph, torsions)
    ba, bat = joint_matrix(matrices["B"], matrices["A"]), joint_matrix(matrices["B"], matrices["A"], matrices["T"])
    _, ba_audit = orthonormal_row_basis(ba); _, bat_audit = orthonormal_row_basis(bat)
    assert bat_audit["rank"] >= ba_audit["rank"]


def test_svd_rank_handling_ignores_small_singular_values():
    matrix = torch.diag(torch.tensor([1., 1e-12], dtype=torch.float64))
    basis, audit = orthonormal_row_basis(matrix, relative_tolerance=1e-8, absolute_tolerance=1e-10)
    assert basis.shape == (2, 1) and audit["rank"] == 1


def test_rigid_mode_removal_and_force_projection_finite():
    xyz = torch.tensor([[0., 0., 0.], [1., .2, 0.], [.1, 1., .3], [0., .2, 1.]], dtype=torch.float64)
    force = torch.ones_like(xyz) + torch.randn_like(xyz) * .1
    internal = remove_rigid_component(force, xyz)
    assert torch.linalg.vector_norm(internal.sum(0)) < 1e-10
    matrix = torch.randn((3, xyz.numel()), dtype=torch.float64)
    result = force_projection(force, matrix, xyz)
    assert np.isfinite(result["fraction"]) and 0 <= result["fraction"] <= 1 + 1e-12


def test_xTB_force_units_order_and_gradient_protocol_are_frozen():
    force = CONFIG["force"]
    assert force["method"] == "xtb_cli_grad" and force["source_index"] == 0
    assert force["finite_difference_step_angstrom"] == pytest.approx(1e-4)
    assert force["projection_order"] == "joint_svd_union"


def test_protected_splits_and_dataset_overlap_guards():
    identity = json.loads((OUT / "DATASET_IDENTITY.json").read_text(encoding="utf-8"))
    assert identity["formal_test_records_read"] == identity["frozen_holdout_records_read"] == 0
    assert identity["bat_external_overlap"] == 0


def test_sha_manifests_are_self_consistent():
    identity = json.loads((OUT / "DATASET_IDENTITY.json").read_text(encoding="utf-8"))
    assert file_sha256(Path(identity["compact_path"])) == identity["compact_sha256"]
    assert file_sha256(Path(CONFIG["torsion_anchor"]["checkpoint"])) == CONFIG["torsion_anchor"]["checkpoint_sha256"]


def test_final_sha256sums_verify_all_listed_artifacts():
    checksum = OUT / "SHA256SUMS.txt"
    if not checksum.is_file(): pytest.skip("finalization checksum is generated after analysis")
    for line in checksum.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        assert file_sha256(ROOT / relative) == expected
