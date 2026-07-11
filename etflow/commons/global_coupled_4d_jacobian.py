"""Complete molecular ``3N x 4M`` Jacobian for articulated joint deformation.

Each joint is parameterized computationally by ``[stretch, omega_xyz]``.  The
semantic degrees of freedom are one axial stretch, two axis-direction bending
rates, and one torsion rate.  Overlapping ancestor contributions are summed;
there is deliberately no per-atom averaging.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .global_coupled_4d_topology import MolecularKinematicTopology


def skew_matrix(vector: Tensor) -> Tensor:
    if vector.ndim < 1 or vector.size(-1) != 3:
        raise ValueError("vector must end in dimension 3")
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(*vector.shape[:-1], 3, 3)


@dataclass
class JointGeometry:
    axis: Tensor
    pivot: Tensor
    valid: Tensor


def joint_geometry(
    pos: Tensor,
    topology: MolecularKinematicTopology,
    eps: float = 1.0e-8,
) -> JointGeometry:
    """Compute current axes and parent-side pivots for every joint."""

    if pos.ndim != 2 or pos.size(-1) != 3:
        raise ValueError("pos must have shape [N, 3]")
    if pos.size(0) != topology.num_atoms:
        raise ValueError("pos atom count does not match topology")
    if topology.num_joints == 0:
        empty = pos.new_empty((0, 3))
        return JointGeometry(
            empty,
            empty,
            torch.empty(0, dtype=torch.bool, device=pos.device),
        )
    pivot = pos[topology.parent_atom]
    vector = pos[topology.child_atom] - pivot
    norm = torch.linalg.norm(vector, dim=-1)
    valid = torch.isfinite(vector).all(-1) & torch.isfinite(pivot).all(-1) & (norm > eps)
    axis = vector / norm.clamp_min(eps).unsqueeze(-1)
    axis = torch.where(valid[:, None], axis, torch.zeros_like(axis))
    return JointGeometry(axis, pivot, valid)


def build_global_coupled_4d_jacobian(
    pos: Tensor,
    topology: MolecularKinematicTopology,
    eps: float = 1.0e-8,
) -> tuple[Tensor, JointGeometry]:
    """Build the complete dense Jacobian with shape ``[3*N, 4*M]``."""

    num_atoms, num_joints = pos.size(0), topology.num_joints
    geometry = joint_geometry(pos, topology, eps)
    if num_joints == 0:
        return pos.new_zeros((3 * num_atoms, 0)), geometry

    atoms = topology.affected_atom_index
    joints = topology.affected_joint_index
    lever = pos[atoms] - geometry.pivot[joints]
    stretch = geometry.axis[joints].unsqueeze(-1)
    rotation = -skew_matrix(lever)
    blocks = torch.cat((stretch, rotation), dim=-1)
    blocks = blocks * geometry.valid[joints, None, None]

    # One row block per atom/joint pair. index_add makes summation semantics
    # explicit and remains safe if a future topology contains duplicate pairs.
    flat = pos.new_zeros((num_atoms * num_joints, 3, 4))
    flat.index_add_(0, atoms * num_joints + joints, blocks)
    dense = flat.reshape(num_atoms, num_joints, 3, 4).permute(0, 2, 1, 3)
    return dense.reshape(3 * num_atoms, 4 * num_joints), geometry


def apply_global_coupled_4d_jacobian(
    pos: Tensor,
    q: Tensor,
    topology: MolecularKinematicTopology,
    eps: float = 1.0e-8,
) -> tuple[Tensor, JointGeometry]:
    """Apply ``J q`` matrix-free while summing all overlapping joints."""

    if q.shape == (4 * topology.num_joints,):
        q = q.reshape(topology.num_joints, 4)
    if q.shape != (topology.num_joints, 4):
        raise ValueError(
            f"q must have shape [{topology.num_joints}, 4] or "
            f"[{4 * topology.num_joints}], got {tuple(q.shape)}"
        )
    velocity = torch.zeros_like(pos)
    geometry = joint_geometry(pos, topology, eps)
    if topology.num_joints == 0:
        return velocity, geometry
    atoms = topology.affected_atom_index
    joints = topology.affected_joint_index
    lever = pos[atoms] - geometry.pivot[joints]
    stretch = q[joints, :1] * geometry.axis[joints]
    angular = torch.cross(q[joints, 1:], lever, dim=-1)
    contribution = (stretch + angular) * geometry.valid[joints, None]
    velocity.index_add_(0, atoms, contribution)
    return velocity, geometry


def decompose_joint_rates(q: Tensor, axis: Tensor) -> dict[str, Tensor]:
    """Split computational ``[s, omega]`` into stretch/bending/torsion."""

    if q.ndim == 1:
        q = q.reshape(-1, 4)
    if q.shape != (axis.size(0), 4) or axis.shape[-1] != 3:
        raise ValueError("q and axis shapes must be [M,4] and [M,3]")
    stretch = q[:, 0]
    omega = q[:, 1:]
    torsion = (omega * axis).sum(-1)
    torsion_vector = torsion[:, None] * axis
    bending_vector = omega - torsion_vector
    axis_rate = torch.cross(omega, axis, dim=-1)
    return {
        "stretch": stretch,
        "omega": omega,
        "torsion": torsion,
        "torsion_vector": torsion_vector,
        "bending_vector": bending_vector,
        "bending_norm": torch.linalg.norm(bending_vector, dim=-1),
        "axis_rate": axis_rate,
    }


def apply_joint_rate_mode(q: Tensor, axis: Tensor, mode: str) -> Tensor:
    """Apply inference-only component ablations without changing the model."""

    values = q.reshape(-1, 4).clone()
    parts = decompose_joint_rates(values, axis)
    if mode in ("full_4d", "none", None):
        return values
    if mode == "torsion_only":
        values[:, 0] = 0
        values[:, 1:] = parts["torsion_vector"]
    elif mode in ("angular_only", "bending_torsion"):
        values[:, 0] = 0
    elif mode == "stretch_only":
        values[:, 1:] = 0
    elif mode == "internal_zero":
        values.zero_()
    else:
        raise ValueError(f"unknown joint-rate mode: {mode}")
    return values


def first_order_joint_update(
    pos: Tensor,
    q: Tensor,
    topology: MolecularKinematicTopology,
    step_size: float,
) -> Tensor:
    """First-order articulated update used to validate the velocity mapping."""

    velocity, _ = apply_global_coupled_4d_jacobian(pos, q, topology)
    return pos + float(step_size) * velocity

