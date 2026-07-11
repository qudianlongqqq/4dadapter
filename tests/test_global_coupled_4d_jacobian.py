import torch

from etflow.commons.global_coupled_4d_jacobian import (
    apply_global_coupled_4d_jacobian,
    build_global_coupled_4d_jacobian,
    first_order_joint_update,
)
from etflow.commons.global_coupled_4d_projection import svd_oracle
from etflow.commons.global_coupled_4d_topology import build_global_coupled_4d_topology


def directed(pairs):
    values = []
    for left, right in pairs:
        values.extend(((left, right), (right, left)))
    return torch.tensor(values, dtype=torch.long).t()


def single_joint():
    pos = torch.tensor([[0., 0, 0], [1., 0, 0], [2., 0, 0], [2., 1, 0]])
    topology = build_global_coupled_4d_topology(
        4, directed([(0, 1), (1, 2), (2, 3)]), torch.tensor([[1], [2]])
    )
    return pos, topology


def chain():
    pos = torch.tensor([[0., 0, 0], [1., 0, 0], [2., .2, 0], [3., 1, .2], [4., 1, 1.]])
    topology = build_global_coupled_4d_topology(
        5, directed([(0, 1), (1, 2), (2, 3), (3, 4)]),
        torch.tensor([[1, 2], [2, 3]])
    )
    return pos, topology


def test_shape_and_stretch_mapping():
    pos, topology = single_joint()
    jacobian, geometry = build_global_coupled_4d_jacobian(pos, topology)
    assert jacobian.shape == (12, 4)
    velocity, _ = apply_global_coupled_4d_jacobian(pos, torch.tensor([[2., 0, 0, 0]]), topology)
    torch.testing.assert_close(velocity[:2], torch.zeros_like(velocity[:2]))
    torch.testing.assert_close(velocity[2:], 2 * geometry.axis[0].expand(2, 3))


def test_torsion_and_bending_have_expected_cross_products():
    pos, topology = single_joint()
    torsion, geometry = apply_global_coupled_4d_jacobian(
        pos, torch.tensor([[0., 1, 0, 0]]), topology
    )
    assert torch.linalg.norm(torsion[2]) < 1e-7
    torch.testing.assert_close(torsion[3], torch.tensor([0., 0, 1.]))
    bending, _ = apply_global_coupled_4d_jacobian(
        pos, torch.tensor([[0., 0, 0, 1.]]), topology
    )
    assert abs(float(bending[2, 1])) > 0.9
    assert abs(float((geometry.axis[0] * torch.tensor([0., 0, 1.])).sum())) < 1e-7


def test_global_coupling_adds_terminal_motion_and_gram_cross_blocks():
    pos, topology = chain()
    jacobian, _ = build_global_coupled_4d_jacobian(pos, topology)
    assert jacobian.shape == (15, 8)
    gram = jacobian.T @ jacobian
    assert torch.linalg.norm(gram[:4, 4:]) > 1e-4
    q = torch.tensor([[.1, .2, -.1, .3], [-.2, .1, .2, -.2]])
    velocity, _ = apply_global_coupled_4d_jacobian(pos, q, topology)
    torch.testing.assert_close(velocity.reshape(-1), jacobian @ q.reshape(-1))
    first_only, _ = apply_global_coupled_4d_jacobian(pos, q * torch.tensor([[1.], [0.]]), topology)
    second_only, _ = apply_global_coupled_4d_jacobian(pos, q * torch.tensor([[0.], [1.]]), topology)
    torch.testing.assert_close(velocity[-1], first_only[-1] + second_only[-1])


def test_pseudoinverse_reconstructs_and_residual_is_orthogonal():
    torch.manual_seed(4)
    pos, topology = chain()
    jacobian, _ = build_global_coupled_4d_jacobian(pos, topology)
    q = torch.randn(jacobian.size(1))
    exact = (jacobian @ q).reshape_as(pos)
    reconstruction = svd_oracle(jacobian, exact)
    torch.testing.assert_close(reconstruction.projected, exact, atol=2e-5, rtol=2e-5)
    random = torch.randn_like(pos)
    projection = svd_oracle(jacobian, random)
    assert torch.linalg.norm(jacobian.T @ projection.residual.reshape(-1)) < 2e-5


def test_finite_difference_matches_jacobian():
    pos, topology = single_joint()
    q = torch.tensor([[.2, .1, -.3, .4]])
    epsilon = 1e-4
    advanced = first_order_joint_update(pos, q, topology, epsilon)
    finite = (advanced - pos) / epsilon
    velocity, _ = apply_global_coupled_4d_jacobian(pos, q, topology)
    torch.testing.assert_close(finite, velocity, atol=1e-3, rtol=1e-3)


def test_translation_invariance_and_rotation_equivariance():
    pos, topology = chain()
    q = torch.tensor([[.2, .1, -.3, .4], [-.1, .2, .1, -.2]])
    velocity, geometry = apply_global_coupled_4d_jacobian(pos, q, topology)
    translated, _ = apply_global_coupled_4d_jacobian(pos + torch.tensor([3., -2, 1.]), q, topology)
    torch.testing.assert_close(translated, velocity)
    rotation = torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1.]])
    rotated_q = q.clone(); rotated_q[:, 1:] = q[:, 1:] @ rotation.T
    rotated, _ = apply_global_coupled_4d_jacobian(pos @ rotation.T, rotated_q, topology)
    torch.testing.assert_close(rotated, velocity @ rotation.T, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(geometry.axis @ rotation.T,
                               build_global_coupled_4d_jacobian(pos @ rotation.T, topology)[1].axis)


def test_rank_deficient_short_axis_and_empty_joint_are_finite():
    pos = torch.tensor([[0., 0, 0], [0., 0, 0], [1., 0, 0]])
    topology = build_global_coupled_4d_topology(
        3, directed([(0, 1), (1, 2)]), torch.tensor([[0], [1]])
    )
    jacobian, geometry = build_global_coupled_4d_jacobian(pos, topology)
    assert not geometry.valid.any() and torch.isfinite(jacobian).all()
    result = svd_oracle(jacobian, torch.randn_like(pos))
    assert torch.isfinite(result.residual).all()
    empty = build_global_coupled_4d_topology(
        3, directed([(0, 1), (1, 2)]), torch.empty((2, 0), dtype=torch.long)
    )
    empty_j, _ = build_global_coupled_4d_jacobian(pos, empty)
    assert empty_j.shape == (9, 0)

