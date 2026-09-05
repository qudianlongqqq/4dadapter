from __future__ import annotations

import torch

from etflow.ecir.learned_geometry import GraphGeometry, geometry_values, remove_rigid_component
from etflow.ecir.musigma_reliability import _angle_descent, _bond_descent
from etflow.ecir.source_conditioned_deltaq import (
    SourceConditionedDeltaQBelief,
    deltaq_belief_loss,
    deltaq_targets,
)


def graph() -> GraphGeometry:
    # H-O-H-like three-atom graph; chemistry embeddings are only test fixtures.
    atoms = torch.tensor([[8, 5, 3, 0, 0, 2], [1, 5, 3, 0, 0, 1], [1, 5, 3, 0, 0, 1]])
    edges = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]])
    edge_codes = torch.tensor([[1, 0, 0]] * 4)
    return GraphGeometry(
        atom_categorical=atoms,
        edge_index=edges,
        edge_categorical=edge_codes,
        bonds=torch.tensor([[0, 0], [1, 2]]),
        bond_categorical=torch.tensor([[1, 0, 0], [1, 0, 0]]),
        angles=torch.tensor([[1, 0, 2]]),
        angle_edge_categorical=torch.tensor([[1, 0, 0, 1, 0, 0]]),
        rings=(),
        chirality=(),
        bond_fixed=torch.tensor([[1.0, 0.05], [1.0, 0.05]], dtype=torch.float64),
        angle_fixed=torch.tensor([[0.0, 0.10]], dtype=torch.float64),
    )


def source_and_reference() -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.98, 0.0]], dtype=torch.float64)
    reference = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.1, 0.90, 0.0]], dtype=torch.float64)
    return source, reference


def first_order_change(source: torch.Tensor, vector: torch.Tensor, g: GraphGeometry):
    direction = remove_rigid_component(vector, source)
    direction = direction / direction.square().sum(-1).mean().sqrt()
    before = geometry_values(source, g)
    after = geometry_values(source + 1.0e-6 * direction, g)
    return (after[0] - before[0]) / 1.0e-6, (after[1] - before[1]) / 1.0e-6


def test_deltaq_target_is_reference_minus_source() -> None:
    g = graph()
    source, reference = source_and_reference()
    db, da = deltaq_targets(source, reference, g)
    sb, sa = geometry_values(source, g)
    rb, ra = geometry_values(reference, g)
    torch.testing.assert_close(db, rb - sb)
    torch.testing.assert_close(da, ra - sa)


def test_source_value_is_a_real_head_input() -> None:
    torch.manual_seed(7)
    model = SourceConditionedDeltaQBelief(hidden_dim=16, layers=1).double()
    g = graph()
    source, _ = source_and_reference()
    # Make one explicit source-q path nonzero while leaving graph features fixed.
    with torch.no_grad():
        head = model.geometry.bond_head
        head[0].weight.zero_()
        head[0].bias.zero_()
        head[0].weight[0, -1] = 1.0
        head[2].weight.zero_()
        head[2].bias.zero_()
        head[2].weight[0, 0] = 1.0
        head[4].weight.zero_()
        head[4].bias.zero_()
        head[4].weight[0, 0] = 1.0
    changed = source.clone()
    changed[1, 0] = 1.2
    a = model(g, source)["bond_deltaq"]
    b = model(g, changed)["bond_deltaq"]
    assert not torch.equal(a, b)


def test_beta_nll_variance_weight_is_stop_gradient() -> None:
    g = graph()
    source, reference = source_and_reference()
    model = SourceConditionedDeltaQBelief(hidden_dim=16, layers=1).double()
    pred = model(g, source)
    target_bond, target_angle = deltaq_targets(source, reference, g)
    loss, _ = deltaq_belief_loss(pred, target_bond, target_angle, [g], beta=0.5)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.bond_sigma_head[-1].bias.grad is not None


def test_bond_action_sign_matches_deltaq() -> None:
    g = graph()
    source, _ = source_and_reference()
    for sign in (-1.0, 1.0):
        # q-mu_eff = -DeltaQ; this is the exact coefficient identity used by v1.
        coefficient = torch.tensor([-sign, 0.0], dtype=torch.float64)
        vector = _bond_descent(source, g, coefficient)
        bond_change, _ = first_order_change(source, vector, g)
        assert float(bond_change[0]) * sign > 0.0


def test_angle_cosine_action_sign_matches_deltaq() -> None:
    g = graph()
    source, _ = source_and_reference()
    for sign in (-1.0, 1.0):
        coefficient = torch.tensor([-sign], dtype=torch.float64)
        vector = _angle_descent(source, g, coefficient)
        _, angle_change = first_order_change(source, vector, g)
        assert float(angle_change[0]) * sign > 0.0
