"""Independent structured-residual model for Serial Global4D Stage 2."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from etflow.commons.global_coupled_4d_jacobian import (
    apply_joint_rate_mode,
    build_global_coupled_4d_jacobian,
    joint_geometry,
)
from etflow.commons.global_coupled_4d_topology import (
    GlobalCoupled4DTopologyCache,
    PreparedGlobalCoupled4DTopology,
)
from etflow.models.components.light_egnn_refiner import (
    LightEGNNLayer,
    SinusoidalTimeEmbedding,
    _mlp,
)


def _field(batch: Any, name: str):
    return batch[name] if isinstance(batch, Mapping) else getattr(batch, name)


def _optional_field(batch: Any, name: str, default=None):
    if isinstance(batch, Mapping):
        return batch.get(name, default)
    return getattr(batch, name, default)


class SerialGlobal4DBackbone(nn.Module):
    """EGNN invariant trunk with only a Global4D coefficient head."""

    def __init__(
        self,
        atom_feature_dim: int = 10,
        edge_attr_dim: int = 1,
        hidden_dim: int = 128,
        edge_hidden_dim: int = 128,
        time_embedding_dim: int = 64,
        num_layers: int = 6,
        dropout: float = 0.0,
        cutoff: float = 10.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.cutoff = float(cutoff)
        self.edge_attr_dim = int(edge_attr_dim)
        self.atom_embedding = nn.Linear(atom_feature_dim, hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.layers = nn.ModuleList(
            [
                LightEGNNLayer(
                    hidden_dim,
                    edge_hidden_dim,
                    edge_attr_dim,
                    time_embedding_dim,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        # parent/child nodes, parent/child fragment pools, time, and 3 invariants
        self.q_head = _mlp(
            4 * hidden_dim + time_embedding_dim + 3,
            edge_hidden_dim,
            4,
            dropout,
        )
        nn.init.zeros_(self.q_head[-1].weight)
        nn.init.zeros_(self.q_head[-1].bias)

    def forward(
        self,
        node_attr: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
        atom_time: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if edge_attr is None:
            edge_attr = pos.new_zeros((edge_index.size(1), self.edge_attr_dim))
        if edge_attr.ndim == 1:
            edge_attr = edge_attr[:, None]
        edge_attr = edge_attr.to(dtype=pos.dtype)
        time_embedding = self.time_embedding(atom_time)
        h = self.atom_embedding(node_attr.to(dtype=pos.dtype))
        for layer in self.layers:
            h, _ = layer(
                h,
                pos,
                edge_index,
                edge_attr,
                time_embedding,
                self.cutoff,
            )
        return h, time_embedding


class SerialGlobal4DResidualRefiner(nn.Module):
    """Predict ``q`` and a graph gate; every Cartesian correction is exactly ``Jq``.

    This module intentionally has no Cartesian velocity or Cartesian residual
    head. Phase A fixes the gate to one. Phase B freezes the backbone and q
    head and calibrates only the graph-level gate.
    """

    def __init__(
        self,
        atom_feature_dim: int = 10,
        edge_attr_dim: int = 1,
        hidden_dim: int = 128,
        edge_hidden_dim: int = 128,
        time_embedding_dim: int = 64,
        num_layers: int = 6,
        dropout: float = 0.0,
        cutoff: float = 10.0,
        stretch_scale: float = 1.0,
        angular_scale: float = 1.0,
        internal_beta: float = 1.0,
        gate_hidden_dim: int = 64,
        gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if stretch_scale <= 0 or angular_scale <= 0 or internal_beta <= 0:
            raise ValueError("Global4D scales and internal_beta must be positive")
        self.hidden_dim = int(hidden_dim)
        self.time_embedding_dim = int(time_embedding_dim)
        self.stretch_scale = float(stretch_scale)
        self.angular_scale = float(angular_scale)
        self.internal_beta = float(internal_beta)
        self.backbone = SerialGlobal4DBackbone(
            atom_feature_dim,
            edge_attr_dim,
            hidden_dim,
            edge_hidden_dim,
            time_embedding_dim,
            num_layers,
            dropout,
            cutoff,
        )
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim + time_embedding_dim + 2, gate_hidden_dim),
            nn.SiLU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, float(gate_init_bias))
        self.topology_cache = GlobalCoupled4DTopologyCache()

    @staticmethod
    def _atom_batch(batch: Any, pos: Tensor) -> Tensor:
        value = _optional_field(batch, "batch")
        if value is None:
            value = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
        return torch.as_tensor(value, device=pos.device, dtype=torch.long)

    def _topologies(self, batch: Any, atom_batch: Tensor):
        edge_index = _field(batch, "edge_index")
        rotatable = _field(batch, "rotatable_bond_index")
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        result = []
        for graph in range(graphs):
            atoms = torch.nonzero(atom_batch == graph, as_tuple=False).reshape(-1)
            if atoms.numel() == 0:
                continue
            start = int(atoms.min())
            edge_mask = (atom_batch[edge_index[0]] == graph) & (
                atom_batch[edge_index[1]] == graph
            )
            rotatable_mask = (
                atom_batch[rotatable[0]] == graph
                if rotatable.numel()
                else torch.zeros(0, dtype=torch.bool, device=pos_device(edge_index))
            )
            prepared = self.topology_cache.get_prepared(
                int(atoms.numel()),
                edge_index[:, edge_mask] - start,
                rotatable[:, rotatable_mask] - start,
            )
            result.append((graph, start, int(atoms.numel()), prepared))
        return result

    @staticmethod
    def _joint_basis(
        pos: Tensor,
        prepared: PreparedGlobalCoupled4DTopology,
        geometry,
    ) -> tuple[Tensor, Tensor, Tensor]:
        topology = prepared.topology
        downstream_sum = pos.new_zeros((topology.num_joints, 3))
        downstream_sum.index_add_(
            0,
            topology.affected_joint_index,
            pos[topology.affected_atom_index],
        )
        counts = (topology.affected_ptr[1:] - topology.affected_ptr[:-1]).clamp_min(1)
        reference = downstream_sum / counts[:, None].to(pos.dtype) - geometry.pivot
        perpendicular = (
            reference
            - (reference * geometry.axis).sum(-1, keepdim=True) * geometry.axis
        )
        norm = torch.linalg.vector_norm(perpendicular, dim=-1, keepdim=True)
        first = perpendicular / norm.clamp_min(1.0e-8)
        first = torch.where(norm > 1.0e-8, first, torch.zeros_like(first))
        second = torch.cross(geometry.axis, first, dim=-1)
        return geometry.axis, first, second

    @staticmethod
    def _pool(values: Tensor, atom_batch: Tensor, graphs: int) -> Tensor:
        pooled = values.new_zeros((graphs, values.size(-1)))
        pooled.index_add_(0, atom_batch, values)
        counts = torch.bincount(atom_batch, minlength=graphs).clamp_min(1)
        return pooled / counts[:, None].to(values.dtype)

    def _gate(
        self,
        batch: Any,
        h: Tensor,
        time_embedding: Tensor,
        atom_batch: Tensor,
        graphs: int,
        override: float | None,
    ) -> Tensor:
        if override is not None:
            if not 0.0 <= float(override) <= 1.0:
                raise ValueError("gate_override must be in [0, 1]")
            return h.new_full((graphs, 1), float(override))
        pooled_h = self._pool(h, atom_batch, graphs)
        pooled_time = self._pool(time_embedding, atom_batch, graphs)
        num_atoms = torch.bincount(atom_batch, minlength=graphs).to(h.dtype)
        rotatable = _field(batch, "rotatable_bond_index")
        if rotatable.numel():
            rotatable_graph = atom_batch[rotatable[0]]
            if not torch.equal(rotatable_graph, atom_batch[rotatable[1]]):
                raise ValueError("rotatable_bond_index crosses graph boundaries")
            joint_count = torch.bincount(rotatable_graph, minlength=graphs).to(h.dtype)
        else:
            joint_count = h.new_zeros(graphs)
        flexibility = torch.stack(
            (joint_count, joint_count / num_atoms.clamp_min(1)), dim=-1
        )
        return torch.sigmoid(
            self.gate_head(torch.cat((pooled_h, pooled_time, flexibility), dim=-1))
        )

    def forward(
        self,
        batch: Any,
        pos: Tensor | None = None,
        t: Tensor | None = None,
        *,
        gate_override: float | None = None,
        joint_mode: str = "full_4d",
    ) -> dict[str, Any]:
        pos = _field(batch, "x_cart") if pos is None else pos
        atom_batch = self._atom_batch(batch, pos)
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        if t is None:
            t = pos.new_zeros(graphs)
        t = torch.as_tensor(t, device=pos.device, dtype=pos.dtype).reshape(-1)
        if t.numel() == 1 and graphs > 1:
            t = t.expand(graphs)
        if t.numel() != graphs:
            raise ValueError(f"Expected {graphs} Stage 2 times, got {t.numel()}")
        h, time_embedding = self.backbone(
            _field(batch, "node_attr"),
            pos,
            _field(batch, "edge_index"),
            _optional_field(batch, "edge_attr"),
            t[atom_batch],
        )
        graph_gate = self._gate(
            batch, h, time_embedding, atom_batch, graphs, gate_override
        )
        v_internal = torch.zeros_like(pos) + 0.0 * h.sum()
        q_values: list[Tensor] = []
        q_graph_index: list[Tensor] = []
        graph_details = []
        for graph, start, count, prepared in self._topologies(batch, atom_batch):
            topology = prepared.topology
            local_pos = pos[start : start + count]
            if topology.num_joints == 0:
                graph_details.append(
                    {
                        "graph": graph,
                        "status": topology.status,
                        "num_joints": 0,
                        "jacobian": local_pos.new_zeros((3 * count, 0)),
                    }
                )
                continue
            geometry = joint_geometry(local_pos, topology)
            axis, bend_one, bend_two = self._joint_basis(local_pos, prepared, geometry)
            local_h = h[start : start + count]
            pools = local_h.new_zeros((len(topology.fragments), local_h.size(-1)))
            pools.index_add_(
                0,
                prepared.fragment_index,
                local_h[prepared.fragment_atom_index],
            )
            pools = pools / prepared.fragment_counts[:, None].to(local_h.dtype)
            distance_sq = (
                (local_pos[topology.child_atom] - local_pos[topology.parent_atom])
                .square()
                .sum(-1, keepdim=True)
            )
            bend_lever_sq = bend_one.square().sum(-1, keepdim=True)
            affected_fraction = (
                topology.affected_ptr[1:] - topology.affected_ptr[:-1]
            ).to(local_pos.dtype).unsqueeze(-1) / max(count, 1)
            feature = torch.cat(
                (
                    h[start + topology.parent_atom],
                    h[start + topology.child_atom],
                    pools[topology.parent_fragment],
                    pools[topology.child_fragment],
                    time_embedding[start + topology.parent_atom],
                    distance_sq,
                    bend_lever_sq,
                    affected_fraction,
                ),
                dim=-1,
            )
            raw = self.backbone.q_head(feature)
            q = torch.cat(
                (
                    self.stretch_scale * torch.tanh(raw[:, :1]),
                    self.angular_scale * torch.tanh(raw[:, 1:]),
                ),
                dim=-1,
            )
            q = apply_joint_rate_mode(q, axis, joint_mode)
            jacobian, _ = build_global_coupled_4d_jacobian(
                local_pos,
                topology,
                flat_index=prepared.jacobian_flat_index,
            )
            local_internal = (jacobian @ q.reshape(-1)).reshape_as(local_pos)
            v_internal[start : start + count] = local_internal
            q_values.append(q)
            q_graph_index.append(
                torch.full((q.size(0),), graph, dtype=torch.long, device=q.device)
            )
            graph_details.append(
                {
                    "graph": graph,
                    "status": topology.status,
                    "num_joints": topology.num_joints,
                    "jacobian": jacobian,
                }
            )
        q_pred = (
            torch.cat(q_values, dim=0)
            if q_values
            else pos.new_empty((0, 4)) + 0.0 * h.sum()
        )
        q_batch = (
            torch.cat(q_graph_index)
            if q_graph_index
            else torch.empty(0, dtype=torch.long, device=pos.device)
        )
        delta = self.internal_beta * graph_gate[atom_batch] * v_internal
        return {
            "q_pred": q_pred,
            "q_batch": q_batch,
            "v_internal": v_internal,
            "gate": graph_gate.reshape(-1),
            "delta": delta,
            "atom_batch": atom_batch,
            "graph_details": graph_details,
        }

    def stage2_positions(self, batch: Any) -> tuple[Tensor, Tensor]:
        x_cart = _field(batch, "x_cart")
        x_ref = _field(batch, "x_ref_aligned")
        atom_batch = self._atom_batch(batch, x_cart)
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        t = torch.as_tensor(
            _field(batch, "target_time"), device=x_cart.device, dtype=x_cart.dtype
        ).reshape(-1)
        if t.numel() != graphs:
            raise ValueError("target_time must contain one value per graph")
        atom_t = t[atom_batch, None]
        return (1.0 - atom_t) * x_cart + atom_t * x_ref, t

    def phase_a_loss(
        self,
        batch: Any,
        *,
        coefficient_weight: float = 1.0,
        internal_weight: float = 1.0,
    ) -> dict[str, Tensor]:
        pos, t = self.stage2_positions(batch)
        output = self(batch, pos, t, gate_override=1.0)
        q_target = _field(batch, "q_res_star").to(output["q_pred"])
        r_target = _field(batch, "r_J_star").to(output["v_internal"])
        if output["q_pred"].shape != q_target.shape:
            raise ValueError(
                f"q target shape mismatch: {output['q_pred'].shape} vs {q_target.shape}"
            )
        if q_target.numel():
            # Robust per-joint loss, then equal graph weighting. This prevents
            # a few very flexible or extreme-coordinate records from silently
            # dominating coefficient learning.
            per_joint = F.smooth_l1_loss(
                output["q_pred"], q_target, reduction="none"
            ).mean(-1)
            graphs = int(output["gate"].numel())
            per_graph = per_joint.new_zeros(graphs)
            per_graph.index_add_(0, output["q_batch"], per_joint)
            counts = torch.bincount(output["q_batch"], minlength=graphs).clamp_min(1)
            q_loss = (per_graph / counts.to(per_graph.dtype)).mean()
        else:
            q_loss = output["v_internal"].sum() * 0.0
        internal_loss = F.mse_loss(output["v_internal"], r_target)
        loss = (
            float(coefficient_weight) * q_loss + float(internal_weight) * internal_loss
        )
        dot = (output["v_internal"] * r_target).sum()
        cosine = dot / (
            torch.linalg.vector_norm(output["v_internal"])
            * torch.linalg.vector_norm(r_target)
        ).clamp_min(1.0e-12)
        return {
            **output,
            "loss": loss,
            "q_loss": q_loss,
            "internal_loss": internal_loss,
            "internal_cosine": cosine,
        }

    def phase_b_loss(self, batch: Any) -> dict[str, Tensor]:
        pos, t = self.stage2_positions(batch)
        output = self(batch, pos, t)
        atom_batch = output["atom_batch"]
        graphs = output["gate"].numel()
        residual = _field(batch, "u_stage2").to(output["v_internal"])
        prediction = output["v_internal"].detach()
        dot = prediction.new_zeros(graphs)
        energy = prediction.new_zeros(graphs)
        residual_energy = prediction.new_zeros(graphs)
        dot.index_add_(0, atom_batch, (residual * prediction).sum(-1))
        energy.index_add_(0, atom_batch, prediction.square().sum(-1))
        residual_energy.index_add_(0, atom_batch, residual.square().sum(-1))
        gate_target = (dot / (self.internal_beta * energy + 1.0e-12)).clamp(0.0, 1.0)
        corrected_energy = prediction.new_zeros(graphs)
        corrected = residual - (
            self.internal_beta * gate_target[atom_batch, None] * prediction
        )
        corrected_energy.index_add_(0, atom_batch, corrected.square().sum(-1))
        gain = residual_energy - corrected_energy
        gate_target = torch.where(
            (dot > 0) & (gain > 0), gate_target, torch.zeros_like(gate_target)
        )
        gate_loss = F.smooth_l1_loss(output["gate"], gate_target)
        return {
            **output,
            "loss": gate_loss,
            "gate_loss": gate_loss,
            "gate_target": gate_target,
            "gate_gain_target": gain,
        }

    def freeze_for_phase_b(self) -> None:
        self.requires_grad_(False)
        self.gate_head.requires_grad_(True)


def pos_device(reference: Tensor) -> torch.device:
    """Small helper that keeps empty topology masks on the graph device."""

    return reference.device
