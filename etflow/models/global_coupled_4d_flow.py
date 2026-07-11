"""Global Coupled 4D Joint-Deformation Refiner.

The model predicts one invariant stretch and one SO(3)-equivariant angular
velocity per oriented joint.  A complete molecular Jacobian maps these rates to
Cartesian velocity, while the raw Cartesian head is projected onto the exact
orthogonal complement of the same complete joint subspace.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch import Tensor, nn

from etflow.commons.global_coupled_4d_jacobian import (
    apply_global_coupled_4d_jacobian,
    apply_joint_rate_mode,
    build_global_coupled_4d_jacobian,
    decompose_joint_rates,
    joint_geometry,
)
from etflow.commons.global_coupled_4d_projection import (
    project_orthogonal_residual,
    svd_oracle,
)
from etflow.commons.global_coupled_4d_topology import GlobalCoupled4DTopologyCache
from etflow.commons.refinement_utils import clip_atom_displacement
from etflow.models.components.light_egnn_refiner import (
    LightEGNNLayer,
    SinusoidalTimeEmbedding,
    _mlp,
)


MOTION_MODE = "global_coupled_4d_joint_deformation"
ABLATION_MODES = (
    "full_4d",
    "torsion_only",
    "bending_torsion",
    "angular_only",
    "stretch_only",
    "internal_zero",
)


def _field(batch: Any, name: str):
    return batch[name] if isinstance(batch, Mapping) else getattr(batch, name)


def _optional_field(batch: Any, name: str, default=None):
    return batch.get(name, default) if isinstance(batch, Mapping) else getattr(batch, name, default)


class GlobalCoupled4DBackbone(nn.Module):
    """Fair EGNN trunk plus an invariant four-scalar joint readout."""

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
        self.cartesian_layer_weights = nn.Parameter(torch.zeros(num_layers))
        # parent/child nodes, parent/child fragment pools, and three invariants
        self.joint_head = _mlp(
            4 * hidden_dim + time_embedding_dim + 3,
            edge_hidden_dim,
            4,
            dropout,
        )

    def encode(
        self,
        node_attr: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        atom_time: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if edge_attr is None:
            edge_attr = pos.new_zeros((edge_index.size(1), self.edge_attr_dim))
        if edge_attr.ndim == 1:
            edge_attr = edge_attr[:, None]
        edge_attr = edge_attr.to(dtype=pos.dtype)
        time_embedding = self.time_embedding(atom_time)
        h = self.atom_embedding(node_attr.to(dtype=pos.dtype))
        vectors = []
        for layer in self.layers:
            h, vector = layer(
                h, pos, edge_index, edge_attr, time_embedding, self.cutoff
            )
            vectors.append(vector)
        weights = torch.softmax(self.cartesian_layer_weights, dim=0)
        cartesian = sum(weight * vector for weight, vector in zip(weights, vectors))
        return h, cartesian, time_embedding


class GlobalCoupled4DFlowLightningModule(LightningModule):
    def __init__(
        self,
        motion_mode: str = MOTION_MODE,
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
        projection_rank_tol: float = 1.0e-6,
        target_rank_tol: float = 1.0e-6,
        final_weight: float = 1.0,
        internal_weight: float = 1.0,
        residual_weight: float = 1.0,
        coefficient_weight: float = 0.05,
        lr: float = 2.0e-4,
        weight_decay: float = 1.0e-6,
        grad_clip: float = 1.0,
        t_min: float = 0.0,
        t_max: float = 0.25,
    ) -> None:
        super().__init__()
        if motion_mode != MOTION_MODE:
            raise ValueError(f"motion_mode must be {MOTION_MODE!r}")
        if stretch_scale <= 0 or angular_scale <= 0:
            raise ValueError("joint output scales must be positive")
        self.save_hyperparameters()
        self.motion_mode = motion_mode
        self.backbone = GlobalCoupled4DBackbone(
            atom_feature_dim,
            edge_attr_dim,
            hidden_dim,
            edge_hidden_dim,
            time_embedding_dim,
            num_layers,
            dropout,
            cutoff,
        )
        self.topology_cache = GlobalCoupled4DTopologyCache()

    def _atom_batch(self, batch: Any, pos: Tensor) -> Tensor:
        value = _optional_field(batch, "batch")
        if value is None:
            value = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
        return value

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
            if rotatable.numel():
                rotatable_mask = atom_batch[rotatable[0]] == graph
            else:
                rotatable_mask = torch.zeros(
                    0, dtype=torch.bool, device=edge_index.device
                )
            local_edge = edge_index[:, edge_mask] - start
            local_rotatable = rotatable[:, rotatable_mask] - start
            topology = self.topology_cache.get(
                atoms.numel(), local_edge, local_rotatable
            )
            result.append((graph, start, int(atoms.numel()), topology))
        return result

    @staticmethod
    def _joint_basis(pos: Tensor, topology, geometry) -> tuple[Tensor, Tensor, Tensor]:
        """Smooth geometry-derived equivariant basis, never used as a label."""

        downstream_centroids = []
        for joint in range(topology.num_joints):
            atoms = topology.affected_atom_index[
                topology.affected_joint_index == joint
            ]
            downstream_centroids.append(pos[atoms].mean(0))
        reference = torch.stack(downstream_centroids) - geometry.pivot
        perpendicular = reference - (
            reference * geometry.axis
        ).sum(-1, keepdim=True) * geometry.axis
        norm = torch.linalg.norm(perpendicular, dim=-1, keepdim=True)
        first = perpendicular / norm.clamp_min(1.0e-8)
        first = torch.where(norm > 1.0e-8, first, torch.zeros_like(first))
        second = torch.cross(geometry.axis, first, dim=-1)
        return geometry.axis, first, second

    def forward(
        self,
        batch: Any,
        pos: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
        joint_mode: str = "full_4d",
        disable_orthogonalization: bool = False,
    ) -> dict[str, Any]:
        if joint_mode not in ABLATION_MODES:
            raise ValueError(f"joint_mode must be one of {ABLATION_MODES}")
        pos = _field(batch, "x_init") if pos is None else pos
        atom_batch = self._atom_batch(batch, pos)
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        if t is None:
            t = pos.new_zeros(graphs)
        t = torch.as_tensor(t, device=pos.device, dtype=pos.dtype).reshape(-1)
        if t.numel() == 1 and graphs > 1:
            t = t.expand(graphs)
        atom_time = t[atom_batch]

        started = time.perf_counter()
        h, v_cart_raw, time_embedding = self.backbone.encode(
            _field(batch, "node_attr"),
            pos,
            _field(batch, "edge_index"),
            _optional_field(batch, "edge_attr"),
            atom_time,
        )
        backbone_time = time.perf_counter() - started
        topology_started = time.perf_counter()
        topologies = self._topologies(batch, atom_batch)
        topology_time = time.perf_counter() - topology_started

        v_internal = torch.zeros_like(pos)
        v_projection = torch.zeros_like(pos)
        q_values = []
        q_unablated = []
        axes = []
        statuses = []
        graph_details = []
        jacobian_time = solve_time = head_time = 0.0
        for graph, start, count, topology in topologies:
            local_pos = pos[start : start + count]
            statuses.append(topology.status)
            if topology.num_joints == 0:
                graph_details.append(
                    {"graph": graph, "start": start, "count": count, "topology": topology,
                     "jacobian": local_pos.new_zeros((3 * count, 0)), "q": local_pos.new_empty((0, 4)),
                     "projection": None}
                )
                continue
            geometry = joint_geometry(local_pos, topology)
            pools = torch.stack(
                [h[start : start + count][list(fragment)].mean(0) for fragment in topology.fragments]
            )
            axis, bend_one, bend_two = self._joint_basis(local_pos, topology, geometry)
            distance_sq = (
                local_pos[topology.child_atom] - local_pos[topology.parent_atom]
            ).square().sum(-1, keepdim=True)
            bend_lever_sq = (
                bend_one.square().sum(-1, keepdim=True)
            )
            affected_count = (
                topology.affected_ptr[1:] - topology.affected_ptr[:-1]
            ).to(local_pos.dtype).unsqueeze(-1) / max(count, 1)
            feature = torch.cat(
                [
                    h[start + topology.parent_atom],
                    h[start + topology.child_atom],
                    pools[topology.parent_fragment],
                    pools[topology.child_fragment],
                    time_embedding[start + topology.parent_atom],
                    distance_sq,
                    bend_lever_sq,
                    affected_count,
                ],
                dim=-1,
            )
            head_started = time.perf_counter()
            raw = self.backbone.joint_head(feature)
            stretch = float(self.hparams.stretch_scale) * torch.tanh(raw[:, :1])
            coefficients = float(self.hparams.angular_scale) * torch.tanh(raw[:, 1:])
            omega = (
                coefficients[:, :1] * axis
                + coefficients[:, 1:2] * bend_one
                + coefficients[:, 2:3] * bend_two
            )
            q_raw = torch.cat((stretch, omega), dim=-1)
            q = apply_joint_rate_mode(q_raw, axis, joint_mode)
            head_time += time.perf_counter() - head_started

            jacobian_started = time.perf_counter()
            jacobian, _ = build_global_coupled_4d_jacobian(local_pos, topology)
            local_internal, _ = apply_global_coupled_4d_jacobian(local_pos, q, topology)
            jacobian_time += time.perf_counter() - jacobian_started
            v_internal[start : start + count] = local_internal

            projection = None
            if not disable_orthogonalization:
                solve_started = time.perf_counter()
                projection = project_orthogonal_residual(
                    jacobian,
                    v_cart_raw[start : start + count],
                    rank_tol=float(self.hparams.projection_rank_tol),
                )
                solve_time += time.perf_counter() - solve_started
                v_projection[start : start + count] = projection.projected
            q_values.append(q)
            q_unablated.append(q_raw)
            axes.append(axis)
            graph_details.append(
                {"graph": graph, "start": start, "count": count, "topology": topology,
                 "jacobian": jacobian, "q": q, "projection": projection}
            )

        cat_q = torch.cat(q_values) if q_values else pos.new_empty((0, 4))
        cat_raw_q = torch.cat(q_unablated) if q_unablated else pos.new_empty((0, 4))
        cat_axis = torch.cat(axes) if axes else pos.new_empty((0, 3))
        v_residual = v_cart_raw - v_projection
        v_final = v_internal + v_residual
        fallback_count = sum(
            int(detail["projection"].solver_fallback_count > 0)
            for detail in graph_details
            if detail["projection"] is not None
        )
        solved_count = sum(detail["projection"] is not None for detail in graph_details)
        return {
            "v_cart_raw": v_cart_raw,
            "v_cart_projection": v_projection,
            "v_residual": v_residual,
            "v_internal": v_internal,
            "v_final": v_final,
            "q": cat_q,
            "q_unablated": cat_raw_q,
            "axis": cat_axis,
            "joint_mode": joint_mode,
            "topology_status": statuses,
            "solver_fallback_rate": pos.new_tensor(fallback_count / solved_count if solved_count else 0.0),
            "_graph_details": graph_details,
            "timing": {
                "backbone_time": backbone_time,
                "topology_time": topology_time,
                "jacobian_construction_time": jacobian_time,
                "solve_projection_time": solve_time,
                "joint_head_time": head_time,
                "peak_gpu_memory": torch.cuda.max_memory_allocated(pos.device) if pos.is_cuda else 0,
            },
        }

    @staticmethod
    def _mean(values: list[Tensor], reference: Tensor) -> Tensor:
        return torch.stack(values).mean() if values else reference.new_zeros(())

    def _shared_step(self, batch: Any, stage: str) -> Tensor:
        x_init = _field(batch, "x_init")
        x_ref = _field(batch, "x_ref_aligned")
        atom_batch = self._atom_batch(batch, x_init)
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        t = x_init.new_empty(graphs).uniform_(self.hparams.t_min, self.hparams.t_max)
        x_t = (1 - t[atom_batch, None]) * x_init + t[atom_batch, None] * x_ref
        target = x_ref - x_init
        output = self(batch, x_t, t)
        target_internal = torch.zeros_like(target)
        target_residual = target.clone()
        coefficient_losses = []
        oracle_ratios = []
        ranks = []
        conditions = []
        orthogonalities = []
        reconstructions = []
        stretch_energy = bending_energy = torsion_energy = target.new_zeros(())
        joint_offset = 0
        for detail in output["_graph_details"]:
            start, count = detail["start"], detail["count"]
            jacobian = detail["jacobian"]
            local_target = target[start : start + count]
            with torch.no_grad():
                oracle = svd_oracle(
                    jacobian,
                    local_target,
                    rank_tol=float(self.hparams.target_rank_tol),
                )
            target_internal[start : start + count] = oracle.projected.detach()
            target_residual[start : start + count] = oracle.residual.detach()
            oracle_ratios.append(oracle.explained_ratio.detach())
            ranks.append(target.new_tensor(float(oracle.effective_rank)))
            conditions.append(target.new_tensor(oracle.condition_number))
            projection = detail["projection"]
            if projection is not None:
                orthogonalities.append(projection.orthogonality_error.detach())
                raw = output["v_cart_raw"][start : start + count]
                reconstructed = projection.projected + projection.residual
                reconstructions.append(
                    torch.linalg.norm(reconstructed - raw)
                    / torch.linalg.norm(raw).clamp_min(1.0e-20)
                )
            joints = detail["topology"].num_joints
            if joints:
                predicted = output["q"][joint_offset : joint_offset + joints]
                q_star = oracle.coefficients.reshape(joints, 4).detach()
                # Column-norm weighting converts heterogeneous coefficient units
                # into their Cartesian velocity scale without equal raw MSE.
                column_norm = torch.linalg.norm(jacobian.detach(), dim=0)
                coefficient_losses.append(
                    ((predicted.reshape(-1) - q_star.reshape(-1)) * column_norm).square().mean()
                )
                parts = decompose_joint_rates(q_star, output["axis"][joint_offset : joint_offset + joints])
                modes = {
                    "stretch": torch.cat((parts["stretch"][:, None], torch.zeros_like(parts["omega"])), -1),
                    "bending": torch.cat((torch.zeros_like(parts["stretch"][:, None]), parts["bending_vector"]), -1),
                    "torsion": torch.cat((torch.zeros_like(parts["stretch"][:, None]), parts["torsion_vector"]), -1),
                }
                energies = {}
                for name, mode_q in modes.items():
                    velocity, _ = apply_global_coupled_4d_jacobian(
                        x_t[start : start + count], mode_q, detail["topology"]
                    )
                    energies[name] = velocity.square().sum()
                stretch_energy = stretch_energy + energies["stretch"]
                bending_energy = bending_energy + energies["bending"]
                torsion_energy = torsion_energy + energies["torsion"]
            joint_offset += joints

        final_loss = F.mse_loss(output["v_final"], target)
        internal_loss = F.mse_loss(output["v_internal"], target_internal)
        residual_loss = F.mse_loss(output["v_residual"], target_residual)
        coefficient_loss = self._mean(coefficient_losses, target)
        loss = (
            self.hparams.final_weight * final_loss
            + self.hparams.internal_weight * internal_loss
            + self.hparams.residual_weight * residual_loss
            + self.hparams.coefficient_weight * coefficient_loss
        )
        total_energy = (stretch_energy + bending_energy + torsion_energy).clamp_min(1.0e-20)
        target_energy = target.square().sum().clamp_min(1.0e-20)
        pred_error = (target - output["v_internal"]).square().sum()
        internal_norm = torch.linalg.norm(output["v_internal"])
        residual_norm = torch.linalg.norm(output["v_residual"])
        metrics = {
            f"{stage}/loss": loss,
            f"{stage}/final_loss": final_loss,
            f"{stage}/internal_loss": internal_loss,
            f"{stage}/residual_loss": residual_loss,
            f"{stage}/coefficient_loss": coefficient_loss,
            f"{stage}/oracle_internal_explained_ratio": self._mean(oracle_ratios, target),
            f"{stage}/pred_internal_explained_ratio": 1 - pred_error / target_energy,
            f"{stage}/stretch_energy_fraction": stretch_energy / total_energy,
            f"{stage}/bending_energy_fraction": bending_energy / total_energy,
            f"{stage}/torsion_energy_fraction": torsion_energy / total_energy,
            f"{stage}/v_internal_norm": internal_norm,
            f"{stage}/v_residual_norm": residual_norm,
            f"{stage}/internal_velocity_fraction": internal_norm / (internal_norm + residual_norm).clamp_min(1.0e-20),
            f"{stage}/jacobian_effective_rank": self._mean(ranks, target),
            f"{stage}/jacobian_condition_number": self._mean(conditions, target),
            f"{stage}/projection_orthogonality_error": self._mean(orthogonalities, target),
            f"{stage}/projection_reconstruction_error": self._mean(reconstructions, target),
            f"{stage}/solver_fallback_rate": output["solver_fallback_rate"],
        }
        for name, value in output["timing"].items():
            metrics[f"{stage}/performance/{name}"] = target.new_tensor(float(value))
        metrics[f"{stage}/performance/parameter_count"] = target.new_tensor(
            float(sum(parameter.numel() for parameter in self.parameters()))
        )
        self.log_dict(
            metrics,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=graphs,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams.grad_clip)

    @torch.no_grad()
    def refine(
        self,
        batch: Any,
        refinement_steps: int = 10,
        update_scale: float = 0.5,
        max_displacement: Optional[float] = 0.1,
        max_coordinate_norm: float = 1000.0,
        joint_mode: str = "full_4d",
        save_trajectory_metrics: bool = False,
    ):
        x = _field(batch, "x_init").clone()
        trajectory = []
        stable, reason = True, ""
        fallback_rates = []
        timings = []
        for step in range(refinement_steps):
            t = x.new_tensor(step / max(refinement_steps - 1, 1))
            output = self(batch, x, t, joint_mode=joint_mode)
            raw_update = float(update_scale) / refinement_steps * output["v_final"]
            update, clipped = clip_atom_displacement(
                raw_update, max_displacement=max_displacement
            )
            candidate = x + update
            finite = bool(torch.isfinite(candidate).all())
            bounded = finite and bool(
                torch.linalg.norm(candidate, dim=-1).max() < max_coordinate_norm
            )
            fallback_rates.append(float(output["solver_fallback_rate"]))
            timings.append(output["timing"])
            if save_trajectory_metrics:
                trajectory.append(
                    {
                        "rollout_step": step,
                        "update_norm": float(torch.linalg.norm(update, dim=-1).mean()),
                        "internal_norm": float(torch.linalg.norm(output["v_internal"], dim=-1).mean()),
                        "residual_norm": float(torch.linalg.norm(output["v_residual"], dim=-1).mean()),
                        "orthogonality_error": float(max(
                            [d["projection"].orthogonality_error for d in output["_graph_details"] if d["projection"] is not None]
                            or [0.0]
                        )),
                        "solver_fallback_rate": float(output["solver_fallback_rate"]),
                        "coordinate_finite": finite,
                        "clipping_fraction": float(clipped.float().mean()),
                    }
                )
            if not bounded:
                stable = False
                reason = "nonfinite_coordinate" if not finite else "coordinate_norm"
                break
            x = candidate
        mean_timing = {
            key: sum(float(row[key]) for row in timings) / len(timings)
            for key in timings[0]
        } if timings else {}
        return x, {
            "stable": stable,
            "failure_reason": reason,
            "trajectory": trajectory,
            "update_scale": update_scale,
            "joint_mode": joint_mode,
            "solver_fallback_rate": sum(fallback_rates) / len(fallback_rates) if fallback_rates else 0.0,
            "mean_timing": mean_timing,
        }
