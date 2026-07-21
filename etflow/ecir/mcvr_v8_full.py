"""MCVR V8 Full end-to-end neural--analytic conformer refiner."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .audit import field
from .model import _atom_batch, _pool
from .mvr_model import MCVRModel
from .v8_constraint_layer import (
    ConstraintLayerConfig,
    DifferentiableMolecularConstraintLayer,
)
from .v8_constraint_normalization import FrozenResidualScales
from .v8_error_state import V8ErrorStateHead


V8_SCHEMA_VERSION = "mcvr-v8-full-v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MCVRV8FullRefiner(nn.Module):
    """D1-compatible prior plus shared-step differentiable constraint refinement."""

    def __init__(
        self,
        d1_prior: MCVRModel,
        *,
        error_state: Mapping[str, Any] | None = None,
        constraint_layer: ConstraintLayerConfig | Mapping[str, Any] | None = None,
        residual_scales: FrozenResidualScales | Mapping[str, Any] | None = None,
        unroll_steps: int = 2,
        step_embedding_enabled: bool = False,
        error_state_enabled: bool = True,
        train_d1_backbone: bool = True,
        train_d1_head: bool = True,
        max_cumulative_atom_displacement: float | None = None,
        max_cumulative_graph_rms: float | None = None,
    ) -> None:
        super().__init__()
        if int(unroll_steps) not in {1, 2}:
            raise ValueError("MCVR V8 Full v1 supports one or two shared refinement steps")
        self.version = V8_SCHEMA_VERSION
        self.prior = d1_prior
        self.unroll_steps = int(unroll_steps)
        self.step_embedding_enabled = bool(step_embedding_enabled)
        self.error_state_enabled = bool(error_state_enabled)
        self.max_cumulative_atom_displacement = (
            float(max_cumulative_atom_displacement)
            if max_cumulative_atom_displacement is not None
            else None
        )
        self.max_cumulative_graph_rms = (
            float(max_cumulative_graph_rms) if max_cumulative_graph_rms is not None else None
        )
        if (self.max_cumulative_atom_displacement is None) != (
            self.max_cumulative_graph_rms is None
        ):
            raise ValueError("both cumulative trust limits must be set together")
        if self.max_cumulative_atom_displacement is not None and (
            self.max_cumulative_atom_displacement <= 0
            or self.max_cumulative_graph_rms is None
            or self.max_cumulative_graph_rms <= 0
        ):
            raise ValueError("cumulative trust limits must be positive")
        settings = dict(error_state or {})
        hidden_dim = int(self.prior.backbone.atom_embedding.out_features)
        self.error_state_head = V8ErrorStateHead(
            hidden_dim,
            confidence_min=float(settings.get("confidence_min", 0.25)),
            confidence_max=float(settings.get("confidence_max", 4.0)),
            confidence_mode=str(settings.get("confidence_mode", "learned_bounded")),
            fixed_confidence=float(settings.get("fixed_confidence", 1.0)),
        )
        self.constraint_layer = DifferentiableMolecularConstraintLayer(
            constraint_layer, scales=residual_scales
        )
        self.step_embedding = nn.Embedding(self.unroll_steps, 10)
        nn.init.zeros_(self.step_embedding.weight)
        if not self.error_state_enabled:
            self.error_state_head.requires_grad_(False)
        if not self.step_embedding_enabled:
            self.step_embedding.requires_grad_(False)
        self.set_d1_trainability(train_backbone=train_d1_backbone, train_head=train_d1_head)

    @classmethod
    def from_d1_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        expected_sha256: str | None = None,
        map_location: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> "MCVRV8FullRefiner":
        if expected_sha256 is not None and file_sha256(checkpoint_path) != expected_sha256:
            raise RuntimeError("frozen D1 checkpoint SHA256 changed")
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        if checkpoint.get("model_type") != "MCVRModel":
            raise RuntimeError("V8 requires a frozen MCVRModel D1 checkpoint")
        if int(checkpoint.get("step", -1)) <= 0:
            raise RuntimeError("V8 D1 checkpoint is not a completed training checkpoint")
        prior = MCVRModel(**checkpoint["config"]["model"])
        incompatible = prior.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("strict D1 checkpoint load unexpectedly returned incompatible keys")
        model = cls(prior, **kwargs)
        model.d1_checkpoint_identity = {
            "path": str(Path(checkpoint_path).resolve()),
            "sha256": file_sha256(checkpoint_path),
            "schema_version": str(checkpoint.get("schema_version")),
            "step": int(checkpoint["step"]),
            "strict_load": True,
        }
        return model

    def set_d1_trainability(self, *, train_backbone: bool, train_head: bool) -> None:
        backbone_prefixes = ("error_encoder.", "backbone.", "deterministic_embedding.")
        for name, parameter in self.prior.named_parameters():
            is_backbone = name.startswith(backbone_prefixes)
            parameter.requires_grad_(bool(train_backbone if is_backbone else train_head))

    def parameter_groups(
        self,
        *,
        new_head_lr: float,
        d1_head_lr: float,
        d1_backbone_lr: float,
        weight_decay: float = 0.0,
    ) -> list[dict[str, Any]]:
        backbone_prefixes = ("error_encoder.", "backbone.", "deterministic_embedding.")
        backbone, head = [], []
        for name, parameter in self.prior.named_parameters():
            if not parameter.requires_grad:
                continue
            (backbone if name.startswith(backbone_prefixes) else head).append(parameter)
        new = [
            parameter
            for parameter in list(self.error_state_head.parameters())
            + list(self.step_embedding.parameters())
            if parameter.requires_grad
        ]
        groups = []
        for name, parameters, lr in (
            ("v8_new_heads", new, new_head_lr),
            ("d1_correction_head", head, d1_head_lr),
            ("d1_backbone", backbone, d1_backbone_lr),
        ):
            if parameters:
                groups.append(
                    {
                        "name": name,
                        "params": parameters,
                        "lr": float(lr),
                        "weight_decay": float(weight_decay),
                    }
                )
        return groups

    def _step_deterministic_features(
        self, batch: Any, step: int, coordinates: Tensor, explicit: Tensor | None
    ) -> Tensor | None:
        base = explicit if explicit is not None else field(batch, "deterministic_error_features")
        if not self.step_embedding_enabled:
            return base
        atom_batch = _atom_batch(batch, coordinates)
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        if base is None:
            base = coordinates.new_zeros((graphs, 10))
        base = torch.as_tensor(base, device=coordinates.device, dtype=coordinates.dtype).reshape(
            graphs, 10
        )
        ids = torch.full((graphs,), int(step), device=coordinates.device, dtype=torch.long)
        return base + self.step_embedding(ids).to(coordinates.dtype)

    def _error_state(
        self, prior_output: Mapping[str, Tensor], coordinates: Tensor
    ) -> dict[str, Tensor]:
        node = prior_output["node_embedding"]
        atom_batch = prior_output["atom_batch"]
        if self.error_state_enabled:
            return self.error_state_head(node, atom_batch)
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        return {
            "predicted_error_magnitude": coordinates.new_zeros(coordinates.size(0)),
            "prior_confidence_logit": coordinates.new_zeros(coordinates.size(0)),
            "bounded_prior_confidence": coordinates.new_ones(coordinates.size(0)),
            "predicted_graph_correction_rms": coordinates.new_zeros(graphs),
            "predicted_max_atom_correction": coordinates.new_zeros(graphs),
            "upstream_error_embedding": coordinates.new_zeros((graphs, 1)),
            "graph_features": _pool(node, atom_batch, graphs),
        }

    def _trust_cumulative(
        self, cumulative: Tensor, atom_batch: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.max_cumulative_atom_displacement is None:
            one = cumulative.new_ones(1)
            return cumulative, {"atom_trust_scale": one, "graph_trust_scale": one}
        norms = torch.linalg.vector_norm(cumulative, dim=-1)
        atom_scale = torch.clamp(
            float(self.max_cumulative_atom_displacement) / norms.clamp_min(1.0e-12), max=1.0
        )
        atom_clipped = cumulative * atom_scale[:, None]
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        energy = cumulative.new_zeros(graphs)
        energy.index_add_(0, atom_batch, atom_clipped.square().sum(-1))
        counts = torch.bincount(atom_batch, minlength=graphs).clamp_min(1).to(cumulative.dtype)
        rms = torch.sqrt(energy / counts + 1.0e-12)
        graph_scale = torch.clamp(
            float(self.max_cumulative_graph_rms) / rms.clamp_min(1.0e-12), max=1.0
        )
        return atom_clipped * graph_scale[atom_batch, None], {
            "atom_trust_scale": atom_scale,
            "graph_trust_scale": graph_scale,
        }

    def forward(
        self,
        batch: Any,
        pos: Tensor,
        t: Tensor,
        *,
        deterministic_features: Tensor | None = None,
        upstream_metadata: Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        source = pos
        current = pos
        step_deltas: list[Tensor] = []
        step_outputs: list[dict[str, Any]] = []
        first_prior: Tensor | None = None
        last_prior_output: Mapping[str, Tensor] | None = None
        last_error: dict[str, Tensor] | None = None
        for step in range(self.unroll_steps):
            features = self._step_deterministic_features(
                batch, step, current, deterministic_features
            )
            prior_output = self.prior(
                batch,
                current,
                t,
                deterministic_features=features,
                upstream_metadata=upstream_metadata,
                **kwargs,
            )
            delta_prior = prior_output["v_final"]
            if first_prior is None:
                first_prior = delta_prior
            error = self._error_state(prior_output, current)
            constrained = self.constraint_layer(
                current,
                delta_prior,
                error["bounded_prior_confidence"],
                batch,
            )
            delta_solver = constrained["delta_final"]
            atom_batch = prior_output["atom_batch"]
            if self.constraint_layer.config.enabled:
                trusted_cumulative, trust_diagnostics = self._trust_cumulative(
                    current + delta_solver - source, atom_batch
                )
                delta = trusted_cumulative - (current - source)
            else:
                one = current.new_ones(1)
                trust_diagnostics = {"atom_trust_scale": one, "graph_trust_scale": one}
                delta = delta_solver
            current = current + delta
            step_deltas.append(delta)
            step_outputs.append(
                {
                    **constrained,
                    "delta_prior": delta_prior,
                    "delta_solver": delta_solver,
                    "delta_final": delta,
                    "x_output": current,
                    **trust_diagnostics,
                }
            )
            last_prior_output = prior_output
            last_error = error
        assert first_prior is not None and last_prior_output is not None and last_error is not None
        return {
            **last_prior_output,
            **last_error,
            "version": self.version,
            "delta_prior": first_prior,
            "node_features": last_prior_output["node_embedding"],
            "delta_final": step_deltas[-1],
            "cumulative_delta": current - source,
            "x_final": current,
            "step_deltas": tuple(step_deltas),
            "step_outputs": tuple(step_outputs),
            "unroll_steps": self.unroll_steps,
            "d1_parity_mode": bool(
                self.unroll_steps == 1
                and not self.constraint_layer.config.enabled
                and not self.error_state_enabled
            ),
        }
