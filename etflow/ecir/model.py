"""Error encoder and error-conditioned Cartesian ECIR refiner."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from etflow.models.components.light_egnn_refiner import LightEGNNRefinerBackbone, _mlp
from etflow.serial_global4d.safety import trust_region_clip
from .time_schedule import inference_time_schedule

from .geometry import (
    angle_triplets,
    bond_angles,
    bond_lengths,
    clash_score,
    dihedral_angles,
    internal_mode_velocities,
    torsion_quads,
    unique_bonds,
)


def _field(batch: Any, name: str, default=None):
    if isinstance(batch, Mapping):
        return batch.get(name, default)
    return getattr(batch, name, default)


def _atom_batch(batch: Any, pos: Tensor) -> Tensor:
    value = _field(batch, "batch")
    if value is None:
        return torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
    return torch.as_tensor(value, device=pos.device, dtype=torch.long)


def _pool(values: Tensor, assignment: Tensor, count: int) -> Tensor:
    result = values.new_zeros((count, values.size(-1)))
    result.index_add_(0, assignment, values)
    counts = torch.bincount(assignment, minlength=count).clamp_min(1)
    return result / counts[:, None].to(values.dtype)


def _graph_geometry_features(pos: Tensor, batch: Any, atom_batch: Tensor, graphs: int) -> Tensor:
    """Eight source-agnostic geometry summaries, computed per molecule."""

    edge_index = torch.as_tensor(_field(batch, "edge_index"), device=pos.device)
    rotatable = torch.as_tensor(
        _field(batch, "rotatable_bond_index", torch.empty(2, 0)), device=pos.device
    )
    ring_flags = torch.as_tensor(
        _field(batch, "bond_is_in_ring", torch.zeros(edge_index.size(1))),
        device=pos.device,
        dtype=torch.bool,
    )
    rows = []
    for graph in range(graphs):
        atoms = torch.nonzero(atom_batch == graph, as_tuple=False).reshape(-1)
        start = int(atoms.min()) if atoms.numel() else 0
        edge_mask = (atom_batch[edge_index[0]] == graph) & (atom_batch[edge_index[1]] == graph)
        local_edges = edge_index[:, edge_mask] - start
        local_pos = pos[atoms]
        rot_mask = atom_batch[rotatable[0]] == graph if rotatable.numel() else torch.zeros(0, dtype=torch.bool, device=pos.device)
        local_rot = rotatable[:, rot_mask] - start if rotatable.numel() else rotatable
        bonds = unique_bonds(local_edges).to(pos.device)
        angles = angle_triplets(local_edges.cpu(), local_pos.size(0)).to(pos.device)
        torsions = torsion_quads(local_edges.cpu(), local_rot.cpu(), local_pos.size(0)).to(pos.device)
        lengths = bond_lengths(local_pos, bonds)
        angle_values = bond_angles(local_pos, angles)
        torsion_values = dihedral_angles(local_pos, torsions)
        local_ring_flags = ring_flags[edge_mask]
        ring_bonds = local_edges[:, (local_edges[0] < local_edges[1]) & local_ring_flags]
        ring_lengths = bond_lengths(local_pos, ring_bonds)

        def mean_std(values: Tensor) -> tuple[Tensor, Tensor]:
            if values.numel() == 0:
                return pos.new_zeros(()), pos.new_zeros(())
            return values.mean(), values.std(unbiased=False)

        length_mean, length_std = mean_std(lengths)
        angle_mean, angle_std = mean_std(angle_values)
        torsion_abs = torsion_values.abs().mean() if torsion_values.numel() else pos.new_zeros(())
        ring_std = ring_lengths.std(unbiased=False) if ring_lengths.numel() else pos.new_zeros(())
        rows.append(
            torch.stack(
                [
                    length_mean,
                    length_std,
                    angle_mean,
                    angle_std,
                    torsion_abs,
                    ring_std,
                    clash_score(local_pos, local_edges),
                    pos.new_tensor(float(local_rot.size(1)) / 10.0),
                ]
            )
        )
    return torch.stack(rows)


class ECIRErrorEncoder(nn.Module):
    """Predict six error modes, uncertainty, and graph/atom/bond repair gates.

    Output order is ``bond, angle, torsion, ring, clash, chirality``. Metadata
    has four normalized values and is dropped as a whole with probability 0.5
    by default. Geometry and graph features are always present.
    """

    def __init__(
        self,
        atom_feature_dim: int = 10,
        edge_attr_dim: int = 1,
        hidden_dim: int = 128,
        edge_hidden_dim: int = 128,
        time_embedding_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.0,
        cutoff: float = 10.0,
        metadata_dim: int = 4,
        metadata_dropout: float = 0.5,
        error_embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.metadata_dim = int(metadata_dim)
        self.metadata_dropout = float(metadata_dropout)
        self.backbone = LightEGNNRefinerBackbone(
            atom_feature_dim,
            edge_attr_dim,
            hidden_dim,
            edge_hidden_dim,
            time_embedding_dim,
            num_layers,
            dropout,
            cutoff,
        )
        graph_dim = hidden_dim + 8 + metadata_dim
        self.error_head = _mlp(graph_dim, hidden_dim, 12, dropout)
        self.repair_gate_head = _mlp(graph_dim, hidden_dim, 1, dropout)
        self.atom_gate_head = _mlp(hidden_dim + graph_dim, hidden_dim, 1, dropout)
        self.bond_gate_head = _mlp(2 * hidden_dim + graph_dim, hidden_dim, 1, dropout)
        self.error_embedding = _mlp(13, hidden_dim, error_embedding_dim, dropout)
        nn.init.constant_(self.error_head[-1].bias[:6], -4.0)
        nn.init.zeros_(self.repair_gate_head[-1].weight)
        nn.init.constant_(self.repair_gate_head[-1].bias, -2.0)

    def forward(
        self,
        batch: Any,
        pos: Tensor,
        t: Tensor,
        *,
        upstream_metadata: Tensor | None = None,
        apply_metadata_dropout: bool | None = None,
    ) -> dict[str, Tensor]:
        atom_batch = _atom_batch(batch, pos)
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        t = torch.as_tensor(t, device=pos.device, dtype=pos.dtype).reshape(-1)
        if t.numel() == 1:
            t = t.expand(graphs)
        h, _, _ = self.backbone.encode(
            _field(batch, "node_attr"),
            pos,
            _field(batch, "edge_index"),
            _field(batch, "edge_attr"),
            t[atom_batch],
        )
        pooled = _pool(h, atom_batch, graphs)
        geometry = _graph_geometry_features(pos, batch, atom_batch, graphs)
        metadata = upstream_metadata
        if metadata is None:
            metadata = _field(batch, "upstream_metadata")
        if metadata is None:
            metadata = pos.new_zeros((graphs, self.metadata_dim))
        metadata = torch.as_tensor(metadata, device=pos.device, dtype=pos.dtype).reshape(graphs, self.metadata_dim)
        drop = self.training if apply_metadata_dropout is None else bool(apply_metadata_dropout)
        if drop and self.metadata_dropout > 0:
            keep = (torch.rand((graphs, 1), device=pos.device) >= self.metadata_dropout).to(pos.dtype)
            metadata = metadata * keep
        graph_features = torch.cat([pooled, geometry, metadata], dim=-1)
        distribution = self.error_head(graph_features)
        error_mean = F.softplus(distribution[:, :6])
        error_logvar = distribution[:, 6:].clamp(-8.0, 6.0)
        repair_gate = torch.sigmoid(self.repair_gate_head(graph_features))
        atom_gate = torch.sigmoid(
            self.atom_gate_head(torch.cat([h, graph_features[atom_batch]], dim=-1))
        )
        edge_index = torch.as_tensor(_field(batch, "edge_index"), device=pos.device)
        edge_graph = atom_batch[edge_index[0]]
        bond_gate = torch.sigmoid(
            self.bond_gate_head(
                torch.cat([h[edge_index[0]], h[edge_index[1]], graph_features[edge_graph]], dim=-1)
            )
        )
        embedding = self.error_embedding(
            torch.cat([error_mean, error_logvar, repair_gate], dim=-1)
        )
        return {
            "error_mean": error_mean,
            "error_logvar": error_logvar,
            "repair_gate": repair_gate,
            "atom_gate": atom_gate,
            "bond_gate": bond_gate,
            "error_embedding": embedding,
            "geometry_features": geometry,
            "metadata_used": metadata,
        }


class ECIRFlowRefiner(nn.Module):
    """Complete equivariant Cartesian velocity head; no 4D coefficient output."""

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
        error_embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        self.backbone = LightEGNNRefinerBackbone(
            atom_feature_dim,
            edge_attr_dim,
            hidden_dim,
            edge_hidden_dim,
            time_embedding_dim,
            num_layers,
            dropout,
            cutoff,
        )
        self.edge_velocity = _mlp(
            2 * hidden_dim + error_embedding_dim + 1,
            edge_hidden_dim,
            1,
            dropout,
        )
        self.base_scale = _mlp(hidden_dim + error_embedding_dim, hidden_dim, 1, dropout)
        nn.init.zeros_(self.edge_velocity[-1].weight)
        nn.init.zeros_(self.edge_velocity[-1].bias)
        nn.init.zeros_(self.base_scale[-1].weight)
        nn.init.zeros_(self.base_scale[-1].bias)

    def forward(
        self,
        batch: Any,
        pos: Tensor,
        t: Tensor,
        predicted_error_embedding: Tensor,
    ) -> Tensor:
        atom_batch = _atom_batch(batch, pos)
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        t = torch.as_tensor(t, device=pos.device, dtype=pos.dtype).reshape(-1)
        if t.numel() == 1:
            t = t.expand(graphs)
        h, base_velocity, _ = self.backbone.encode(
            _field(batch, "node_attr"),
            pos,
            _field(batch, "edge_index"),
            _field(batch, "edge_attr"),
            t[atom_batch],
        )
        embedding = predicted_error_embedding[atom_batch]
        base = self.base_scale(torch.cat([h, embedding], dim=-1)) * base_velocity
        edge_index = torch.as_tensor(_field(batch, "edge_index"), device=pos.device)
        src, dst = edge_index
        relative = pos[src] - pos[dst]
        distance_sq = relative.square().sum(-1, keepdim=True)
        scalar = self.edge_velocity(
            torch.cat([h[src], h[dst], predicted_error_embedding[atom_batch[src]], distance_sq], dim=-1)
        )
        correction = torch.zeros_like(pos)
        correction.index_add_(0, dst, scalar * relative)
        return base + correction


class ECIRFlowSystem(nn.Module):
    """Joint ECIR error model, flow-matching refiner and four-step teacher."""

    def __init__(
        self,
        *,
        max_atom_displacement: float = 0.12,
        max_molecule_rms_displacement: float = 0.06,
        uncertainty_gate: bool = True,
        identity_gate: bool = True,
        training_t_min: float = 0.0,
        training_t_max: float = 1.0,
        **model_kwargs,
    ) -> None:
        super().__init__()
        encoder_keys = {
            "atom_feature_dim", "edge_attr_dim", "hidden_dim", "edge_hidden_dim",
            "time_embedding_dim", "dropout", "cutoff", "error_embedding_dim"
        }
        encoder_kwargs = {k: v for k, v in model_kwargs.items() if k in encoder_keys}
        if "encoder_num_layers" in model_kwargs:
            encoder_kwargs["num_layers"] = model_kwargs["encoder_num_layers"]
        refiner_kwargs = {k: v for k, v in model_kwargs.items() if k in encoder_keys}
        if "num_layers" in model_kwargs:
            refiner_kwargs["num_layers"] = model_kwargs["num_layers"]
        self.error_encoder = ECIRErrorEncoder(**encoder_kwargs)
        self.refiner = ECIRFlowRefiner(**refiner_kwargs)
        self.max_atom_displacement = float(max_atom_displacement)
        self.max_molecule_rms_displacement = float(max_molecule_rms_displacement)
        self.use_uncertainty_gate = bool(uncertainty_gate)
        self.use_identity_gate = bool(identity_gate)
        if float(training_t_min) > float(training_t_max):
            raise ValueError("training_t_min must not exceed training_t_max")
        self.training_t_min = float(training_t_min)
        self.training_t_max = float(training_t_max)

    def forward(
        self,
        batch: Any,
        pos: Tensor,
        t: Tensor,
        *,
        gate_override: float | Tensor | None = None,
        upstream_metadata: Tensor | None = None,
    ) -> dict[str, Tensor]:
        atom_batch = _atom_batch(batch, pos)
        encoded = self.error_encoder(
            batch, pos, t, upstream_metadata=upstream_metadata
        )
        velocity = self.refiner(batch, pos, t, encoded["error_embedding"])
        gate = encoded["repair_gate"]
        if self.use_uncertainty_gate:
            gate = gate * torch.exp(-0.5 * encoded["error_logvar"].mean(-1, keepdim=True)).clamp(0.0, 1.0)
        if self.use_identity_gate:
            gate = gate * (1.0 - torch.exp(-encoded["error_mean"].sum(-1, keepdim=True)))
        if gate_override is not None:
            override = torch.as_tensor(gate_override, device=pos.device, dtype=pos.dtype)
            gate = torch.ones_like(gate) * override
        gated_velocity = velocity * gate[atom_batch]
        return {**encoded, "velocity": velocity, "gate": gate, "gated_velocity": gated_velocity}

    def loss(
        self,
        batch: Any,
        *,
        lambda_mode: float = 0.25,
        lambda_error: float = 0.25,
        lambda_identity: float = 0.5,
        lambda_trust: float = 0.1,
    ) -> dict[str, Tensor]:
        x_input = _field(batch, "x_input", _field(batch, "x_init"))
        x_target = _field(batch, "x_target")
        atom_batch = _atom_batch(batch, x_input)
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        t = torch.empty(graphs, device=x_input.device, dtype=x_input.dtype).uniform_(
            self.training_t_min, self.training_t_max
        )
        atom_t = t[atom_batch, None]
        x_t = (1.0 - atom_t) * x_input + atom_t * x_target
        target_velocity = x_target - x_input
        output = self(batch, x_t, t)
        predicted = output["gated_velocity"]
        flow = F.smooth_l1_loss(predicted, target_velocity)
        predicted_modes = internal_mode_velocities(x_t, predicted, batch)
        target_modes = internal_mode_velocities(x_t, target_velocity, batch)
        mode_terms = [
            F.smooth_l1_loss(predicted_modes[name], target_modes[name])
            for name in ("bond", "angle", "torsion")
            if predicted_modes[name].numel()
        ]
        internal = torch.stack(mode_terms).mean() if mode_terms else flow.new_zeros(())
        error_target = torch.as_tensor(_field(batch, "error_label"), device=x_input.device, dtype=x_input.dtype).reshape(graphs, 6)
        error_delta = error_target - output["error_mean"]
        error_nll = 0.5 * (
            output["error_logvar"] + error_delta.square() * torch.exp(-output["error_logvar"])
        ).mean()
        clean = torch.as_tensor(_field(batch, "is_clean"), device=x_input.device, dtype=torch.bool).reshape(graphs)
        identity = predicted[clean[atom_batch]].square().mean() if bool(clean.any()) else flow.new_zeros(())
        atom_norm = torch.linalg.vector_norm(predicted, dim=-1)
        atom_excess = (atom_norm - self.max_atom_displacement).clamp_min(0.0).square().mean()
        energy = predicted.new_zeros(graphs)
        energy.index_add_(0, atom_batch, predicted.square().sum(-1))
        counts = torch.bincount(atom_batch, minlength=graphs).clamp_min(1)
        graph_rms = (energy / counts.to(predicted.dtype) + 1.0e-12).sqrt()
        graph_excess = (graph_rms - self.max_molecule_rms_displacement).clamp_min(0.0).square().mean()
        trust = atom_excess + graph_excess
        total = flow + float(lambda_mode) * internal + float(lambda_error) * error_nll + float(lambda_identity) * identity + float(lambda_trust) * trust
        return {
            "loss": total,
            "flow_matching_loss": flow,
            "internal_mode_loss": internal,
            "error_prediction_loss": error_nll,
            "identity_loss": identity,
            "trust_loss": trust,
            "gate_mean": output["gate"].mean(),
        }

    @torch.inference_mode()
    def refine(
        self,
        batch: Any,
        *,
        coordinates: Tensor | None = None,
        steps: int = 4,
        step_size: float = 0.25,
        gate_override: float | Tensor | None = None,
        update_scale: float = 1.0,
        trust_radius_scale: float = 1.0,
        gate_threshold: float = 0.0,
        time_schedule_mode: str = "train_range",
        inference_t_min: float | None = None,
        inference_t_max: float | None = None,
        fixed_t: float | None = None,
        explicit_time_schedule: list[float] | Tensor | None = None,
        strict_training_range: bool = False,
        return_trajectory: bool = False,
    ) -> tuple[Tensor, list[dict[str, Any]]]:
        current = torch.as_tensor(
            coordinates if coordinates is not None else _field(batch, "x_init")
        ).clone()
        atom_batch = _atom_batch(batch, current)
        diagnostics = []
        schedule = inference_time_schedule(
            current,
            int(steps),
            mode=time_schedule_mode,
            training_t_min=self.training_t_min,
            training_t_max=self.training_t_max,
            inference_t_min=inference_t_min,
            inference_t_max=inference_t_max,
            fixed_t=fixed_t,
            explicit_time_schedule=explicit_time_schedule,
            strict_training_range=strict_training_range,
        )
        for step, time_value in enumerate(schedule):
            t = current.new_full(
                (int(atom_batch.max()) + 1 if atom_batch.numel() else 1,),
                float(time_value),
            )
            output = self(batch, current, t, gate_override=gate_override)
            active_gate = output["gate"] * (output["gate"] >= float(gate_threshold))
            gated_velocity = output["velocity"] * active_gate[atom_batch]
            raw = float(step_size) * float(update_scale) * gated_velocity
            clipped, clip = trust_region_clip(
                raw,
                atom_batch,
                max_atom_displacement=self.max_atom_displacement * float(trust_radius_scale),
                max_graph_rms_displacement=self.max_molecule_rms_displacement * float(trust_radius_scale),
                max_internal_velocity_norm=None,
            )
            current = current + clipped
            diagnostics.append(
                {
                    "step": step + 1,
                    "time": float(time_value),
                    "gate_mean": float(output["gate"].mean()),
                    "active_gate_fraction": float((output["gate"] >= float(gate_threshold)).float().mean()),
                    "raw_rms": float(raw.square().sum(-1).mean().sqrt()),
                    "accepted_rms": float(clipped.square().sum(-1).mean().sqrt()),
                    **clip,
                    **(
                        {
                            "coordinates": current.detach().clone(),
                            "graph_gate": output["gate"].detach().reshape(-1).clone(),
                            "graph_uncertainty": torch.exp(
                                0.5 * output["error_logvar"].mean(-1)
                            ).detach().clone(),
                        }
                        if return_trajectory else {}
                    ),
                }
            )
        return current, diagnostics
