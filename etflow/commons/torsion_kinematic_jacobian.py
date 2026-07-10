"""Global torsion kinematic Jacobian with dense and matrix-free operations."""

from __future__ import annotations

import torch
from torch import Tensor

from .molecular_kinematics import MolecularKinematicTopology


def _joint_geometry(
    pos: Tensor, topology: MolecularKinematicTopology, eps: float = 1.0e-8
) -> tuple[Tensor, Tensor]:
    if topology.num_joints == 0:
        return pos.new_empty((0, 3)), torch.empty(0, dtype=torch.bool, device=pos.device)
    axis_vector = pos[topology.child_atom] - pos[topology.parent_atom]
    norm = torch.linalg.norm(axis_vector, dim=-1)
    valid = torch.isfinite(axis_vector).all(dim=-1) & (norm > eps)
    axis = axis_vector / norm.clamp_min(eps).unsqueeze(-1)
    axis = torch.where(valid[:, None], axis, torch.zeros_like(axis))
    return axis, valid


def build_dense_jacobian(
    pos: Tensor, topology: MolecularKinematicTopology, eps: float = 1.0e-8
) -> tuple[Tensor, Tensor]:
    """Build ``J_tau`` with shape ``[3*N, M]`` for tests/targets only."""

    joints = topology.num_joints
    dense = pos.new_zeros((pos.size(0), 3, joints))
    if joints == 0:
        return dense.reshape(3 * pos.size(0), 0), torch.empty(0, dtype=torch.bool, device=pos.device)
    axis, valid = _joint_geometry(pos, topology, eps)
    atoms = topology.affected_atom_index
    joint_ids = topology.affected_joint_index
    lever = pos[atoms] - pos[topology.parent_atom[joint_ids]]
    columns = torch.cross(axis[joint_ids], lever, dim=-1)
    columns = columns * valid[joint_ids, None]
    # Each atom/joint pair is unique by topology construction; accumulate is
    # still used to make the no-averaging semantics explicit.
    flat_index = atoms * joints + joint_ids
    flat = pos.new_zeros((pos.size(0) * joints, 3))
    flat.index_add_(0, flat_index, columns)
    dense = flat.reshape(pos.size(0), joints, 3).permute(0, 2, 1)
    return dense.reshape(3 * pos.size(0), joints), valid


def apply_jacobian(
    pos: Tensor,
    rates: Tensor,
    topology: MolecularKinematicTopology,
    eps: float = 1.0e-8,
) -> tuple[Tensor, Tensor]:
    """Matrix-free ``J_tau @ rates``; overlapping joints are summed."""

    if rates.shape != (topology.num_joints,):
        raise ValueError(f"rates must have shape [{topology.num_joints}], got {tuple(rates.shape)}")
    velocity = torch.zeros_like(pos)
    if topology.num_joints == 0:
        return velocity, torch.empty(0, dtype=torch.bool, device=pos.device)
    axis, valid = _joint_geometry(pos, topology, eps)
    atoms, joints = topology.affected_atom_index, topology.affected_joint_index
    lever = pos[atoms] - pos[topology.parent_atom[joints]]
    contribution = torch.cross(axis[joints], lever, dim=-1)
    contribution = contribution * rates[joints, None] * valid[joints, None]
    velocity.index_add_(0, atoms, contribution)
    return velocity, valid


def apply_jacobian_transpose(
    pos: Tensor,
    atom_velocity: Tensor,
    topology: MolecularKinematicTopology,
    eps: float = 1.0e-8,
) -> tuple[Tensor, Tensor]:
    """Matrix-free ``J_tau.T @ atom_velocity``."""

    if atom_velocity.shape != pos.shape:
        raise ValueError("atom_velocity and pos must have the same [N, 3] shape")
    output = pos.new_zeros((topology.num_joints,))
    if topology.num_joints == 0:
        return output, torch.empty(0, dtype=torch.bool, device=pos.device)
    axis, valid = _joint_geometry(pos, topology, eps)
    atoms, joints = topology.affected_atom_index, topology.affected_joint_index
    lever = pos[atoms] - pos[topology.parent_atom[joints]]
    basis = torch.cross(axis[joints], lever, dim=-1) * valid[joints, None]
    output.index_add_(0, joints, (basis * atom_velocity[atoms]).sum(dim=-1))
    return output, valid


def project_matrix_free(
    pos: Tensor,
    vector: Tensor,
    topology: MolecularKinematicTopology,
    *,
    iterations: int = 24,
    tolerance: float = 1.0e-7,
) -> tuple[Tensor, Tensor]:
    """Project with conjugate gradients on normal equations, without dense J."""

    if topology.num_joints == 0:
        return torch.zeros_like(vector), vector.new_zeros((0,))
    rhs, _ = apply_jacobian_transpose(pos, vector, topology)
    solution = torch.zeros_like(rhs); residual = rhs.clone(); direction = residual.clone()
    residual_sq = torch.dot(residual, residual)
    for _ in range(iterations):
        jd, _ = apply_jacobian(pos, direction, topology)
        normal_d, _ = apply_jacobian_transpose(pos, jd, topology)
        denominator = torch.dot(direction, normal_d)
        if float(denominator.detach().abs()) <= 1.0e-20:
            break
        alpha = residual_sq / denominator
        solution = solution + alpha * direction
        next_residual = residual - alpha * normal_d
        next_sq = torch.dot(next_residual, next_residual)
        if float(next_sq.detach().sqrt()) <= tolerance:
            residual = next_residual; break
        direction = next_residual + (next_sq / residual_sq.clamp_min(1.0e-20)) * direction
        residual, residual_sq = next_residual, next_sq
    projected, _ = apply_jacobian(pos, solution, topology)
    return projected, solution
