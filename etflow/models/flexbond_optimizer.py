"""Generator-agnostic FlexBond-4D conformer refinement module."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from lightning.pytorch import LightningModule
from torch import Tensor

from etflow.commons.flexbond_jacobian import (
    apply_bond_jacobian,
    identify_target_bonds,
    solve_q_star_least_squares,
)
from etflow.commons.refinement_utils import clip_atom_displacement
from etflow.models.components.light_egnn_refiner import LightEGNNRefinerBackbone


OPTIMIZER_MODES = (
    "cartesian_optimizer",
    "flexbond4d_only_optimizer",
    "flexbond4d_hybrid_optimizer",
)


def sample_uniform_times(
    reference: Tensor, num_graphs: int, t_min: float = 0.0, t_max: float = 1.0
) -> Tensor:
    """Sample adapter refinement times while preserving old [0, 1] behavior."""

    if num_graphs < 1:
        raise ValueError("num_graphs must be positive.")
    if not 0.0 <= float(t_min) <= float(t_max) <= 1.0:
        raise ValueError("Require 0 <= t_min <= t_max <= 1.")
    return reference.new_empty(num_graphs).uniform_(float(t_min), float(t_max))


def _field(batch: Any, name: str):
    if isinstance(batch, Mapping):
        return batch[name]
    return getattr(batch, name)


class FlexBondOptimizerLightningModule(LightningModule):
    """Flow-matching secondary optimizer with three controlled ablations."""

    def __init__(
        self,
        mode: str = "flexbond4d_hybrid_optimizer",
        atom_feature_dim: int = 10,
        edge_attr_dim: int = 1,
        hidden_dim: int = 128,
        edge_hidden_dim: int = 128,
        time_embedding_dim: int = 64,
        num_layers: int = 6,
        dropout: float = 0.0,
        cutoff: float = 10.0,
        correction_scale: float = 0.01,
        q_loss_weight: float = 0.001,
        corr_reg_weight: float = 0.0001,
        min_affected_atoms: int = 2,
        max_bonds_per_mol: int = 16,
        ridge_eps: float = 1.0e-5,
        max_q_norm: float = 10.0,
        max_condition: float = 1.0e6,
        lr: float = 2.0e-4,
        weight_decay: float = 1.0e-6,
        grad_clip: float = 1.0,
        t_min: float = 0.0,
        t_max: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in OPTIMIZER_MODES:
            raise ValueError(f"Unknown mode {mode!r}; choose from {OPTIMIZER_MODES}.")
        if not 0.0 <= float(t_min) <= float(t_max) <= 1.0:
            raise ValueError("Require 0 <= t_min <= t_max <= 1.")
        self.save_hyperparameters()
        self.optimizer_mode = mode
        self.backbone = LightEGNNRefinerBackbone(
            atom_feature_dim=atom_feature_dim,
            edge_attr_dim=edge_attr_dim,
            hidden_dim=hidden_dim,
            edge_hidden_dim=edge_hidden_dim,
            time_embedding_dim=time_embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
            cutoff=cutoff,
        )

    def _atom_batch(self, batch: Any, num_atoms: int, device) -> Tensor:
        atom_batch = getattr(batch, "batch", None)
        if atom_batch is None and isinstance(batch, Mapping):
            atom_batch = batch.get("batch")
        if atom_batch is None:
            atom_batch = torch.zeros(num_atoms, dtype=torch.long, device=device)
        return atom_batch

    def _target_bonds(self, batch: Any, atom_batch: Tensor) -> dict[str, Tensor]:
        return identify_target_bonds(
            _field(batch, "rotatable_bond_index"),
            _field(batch, "atom_bond_influence_index"),
            atom_batch,
            min_affected_atoms=self.hparams.min_affected_atoms,
            max_bonds_per_mol=self.hparams.max_bonds_per_mol,
        )

    def forward(
        self,
        batch: Any,
        pos: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
    ) -> dict[str, Any]:
        pos = _field(batch, "x_init") if pos is None else pos
        atom_batch = self._atom_batch(batch, pos.size(0), pos.device)
        num_graphs = int(atom_batch.max().item()) + 1 if atom_batch.numel() else 1
        if t is None:
            t = pos.new_zeros((num_graphs,))
        t = torch.as_tensor(t, device=pos.device, dtype=pos.dtype).reshape(-1)
        if t.numel() == 1 and num_graphs > 1:
            t = t.expand(num_graphs)
        if t.numel() != num_graphs:
            raise ValueError(f"Expected {num_graphs} times, received {t.numel()}.")
        atom_time = t[atom_batch]
        targets = self._target_bonds(batch, atom_batch)
        v_cart_raw, q_b = self.backbone(
            _field(batch, "node_attr"),
            pos,
            _field(batch, "edge_index"),
            getattr(batch, "edge_attr", None)
            if not isinstance(batch, Mapping)
            else batch.get("edge_attr"),
            atom_time,
            targets["anchor_index"],
            targets["moving_index"],
        )
        v_4d, jacobian_diagnostics = apply_bond_jacobian(pos, q_b, targets)
        # Preserve a zero-gradient autograd path for 4D-only batches that have
        # no eligible rotatable bond; FM loss must still backward cleanly.
        v_4d = v_4d + 0.0 * v_cart_raw
        zero = torch.zeros_like(pos)
        if self.optimizer_mode == "cartesian_optimizer":
            v_cart, v_4d_used, v_final = v_cart_raw, zero, v_cart_raw
        elif self.optimizer_mode == "flexbond4d_only_optimizer":
            v_cart, v_4d_used, v_final = zero, v_4d, v_4d
        else:
            v_cart, v_4d_used = v_cart_raw, v_4d
            v_final = v_cart + self.hparams.correction_scale * v_4d
        return {
            "v_cart": v_cart,
            "q_b": q_b,
            "v_4d": v_4d_used,
            "v_final": v_final,
            "target_bonds": targets,
            "jacobian_diagnostics": jacobian_diagnostics,
        }

    def _shared_step(self, batch: Any, stage: str) -> Tensor:
        x_init = _field(batch, "x_init")
        x_ref = _field(batch, "x_ref_aligned")
        atom_batch = self._atom_batch(batch, x_init.size(0), x_init.device)
        num_graphs = int(atom_batch.max().item()) + 1 if atom_batch.numel() else 1
        t = sample_uniform_times(
            x_init, num_graphs, self.hparams.t_min, self.hparams.t_max
        )
        atom_t = t[atom_batch, None]
        x_t = (1 - atom_t) * x_init + atom_t * x_ref
        target_velocity = x_ref - x_init
        output = self(batch, x_t, t)
        cart_loss = (output["v_cart"] - target_velocity).square().mean()
        final_loss = (output["v_final"] - target_velocity).square().mean()
        # The primary hybrid objective is deliberately L_final. L_cart is a
        # diagnostic that makes the contribution of the 4D path observable.
        flow_loss = final_loss
        q_loss = final_loss.new_zeros(())
        q_star_nan_count = 0
        num_skipped_too_small = int(
            output["target_bonds"]["num_skipped_too_small"].item()
        )
        num_skipped_rank_deficient = 0
        num_valid_bonds = int(
            output["jacobian_diagnostics"]["valid_geometry_mask"].sum().item()
        )

        # q_b_star uses the true residual and is strictly a training-time
        # pseudo-label. Inference calls only forward(), never this LS solve.
        if self.optimizer_mode == "flexbond4d_hybrid_optimizer":
            residual = target_velocity - output["v_cart"].detach()
            q_star, q_valid, q_stats = solve_q_star_least_squares(
                x_t,
                residual,
                output["target_bonds"],
                ridge_eps=self.hparams.ridge_eps,
                max_q_norm=self.hparams.max_q_norm,
                max_condition=self.hparams.max_condition,
            )
            if q_valid.any():
                q_loss = (
                    output["q_b"][q_valid] - q_star[q_valid]
                ).square().mean()
            q_star_nan_count = int(q_stats["q_star_nan_count"])
            num_valid_bonds = int(q_stats["num_valid_bonds"])
            num_skipped_too_small = int(q_stats["num_skipped_too_small"])
            num_skipped_rank_deficient = int(
                q_stats["num_skipped_rank_deficient"]
            )

        corr_reg_loss = output["v_4d"].square().mean()
        loss = flow_loss
        if self.optimizer_mode == "flexbond4d_only_optimizer":
            loss = loss + self.hparams.corr_reg_weight * corr_reg_loss
        elif self.optimizer_mode == "flexbond4d_hybrid_optimizer":
            loss = (
                loss
                + self.hparams.q_loss_weight * q_loss
                + self.hparams.corr_reg_weight * corr_reg_loss
            )

        target_norm = torch.linalg.norm(target_velocity, dim=-1).mean().clamp_min(1e-8)
        residual_norm = torch.linalg.norm(
            target_velocity - output["v_cart"].detach(), dim=-1
        ).mean().clamp_min(1e-8)
        metrics = {
            f"{stage}/refinement_time_mean": t.mean(),
            f"{stage}/refinement_time_min": t.min(),
            f"{stage}/refinement_time_max": t.max(),
            f"{stage}/flow_matching_loss": flow_loss,
            f"{stage}/cartesian_loss": cart_loss,
            f"{stage}/final_loss": final_loss,
            f"{stage}/loss": loss,
            f"{stage}/cartesian/corr_norm": torch.linalg.norm(
                output["v_cart"], dim=-1
            ).mean(),
            f"{stage}/cartesian/corr_to_target_ratio": torch.linalg.norm(
                output["v_cart"], dim=-1
            ).mean()
            / target_norm,
            f"{stage}/flexbond/q_loss": q_loss,
            f"{stage}/flexbond/corr_reg_loss": corr_reg_loss,
            f"{stage}/target_velocity_norm": target_norm,
            f"{stage}/flexbond/corr_norm": torch.linalg.norm(
                output["v_4d"], dim=-1
            ).mean(),
            f"{stage}/flexbond/corr_to_residual_ratio": torch.linalg.norm(
                output["v_4d"], dim=-1
            ).mean()
            / residual_norm,
            f"{stage}/flexbond/num_valid_bonds": loss.new_tensor(num_valid_bonds),
            f"{stage}/flexbond/q_star_nan_count": loss.new_tensor(q_star_nan_count),
            f"{stage}/flexbond/num_skipped_too_small": loss.new_tensor(
                num_skipped_too_small
            ),
            f"{stage}/flexbond/num_skipped_rank_deficient": loss.new_tensor(
                num_skipped_rank_deficient
            ),
            f"{stage}/flexbond/num_skipped_by_cap": output["target_bonds"][
                "num_skipped_by_cap"
            ].to(device=loss.device, dtype=loss.dtype),
        }
        self.log_dict(
            metrics,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=num_graphs,
            prog_bar=False,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        optimizer = self.optimizers(use_pl_optimizer=False)
        self.log(
            "learning_rate",
            optimizer.param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
        )

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ) -> None:
        norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), self.hparams.grad_clip
        )
        self.log("grad_norm", norm, on_step=True, on_epoch=False)

    @torch.no_grad()
    def refine(
        self,
        batch: Any,
        refinement_steps: int = 10,
        step_size: Optional[float] = None,
        update_scale: float = 1.0,
        max_displacement: Optional[float] = None,
        adaptive_alpha_by_update_norm: bool = False,
        target_update_norm: Optional[float] = None,
        max_coordinate_norm: float = 1.0e3,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Euler rollout with step clipping, followed by label-free alpha scaling."""

        if refinement_steps < 1:
            raise ValueError("refinement_steps must be positive.")
        if update_scale < 0:
            raise ValueError("update_scale must be non-negative.")
        if adaptive_alpha_by_update_norm and (
            target_update_norm is None or float(target_update_norm) <= 0
        ):
            raise ValueError(
                "target_update_norm must be positive when adaptive alpha is enabled."
            )
        x_init = _field(batch, "x_init")
        x = x_init.clone()
        base_dt = 1.0 / refinement_steps if step_size is None else float(step_size)
        if base_dt <= 0:
            raise ValueError("step_size must be positive when provided.")
        # First compute one common full rollout, then scale its total update.
        # This makes alpha sweeps directly comparable:
        # x_refined = x_init + update_scale * (x_rollout - x_init).
        dt = base_dt
        stable = True
        failed_step = None
        clipped_atom_steps = 0
        total_atom_steps = 0
        raw_step_norms = []
        applied_step_norms = []
        for step in range(refinement_steps):
            t = x.new_tensor(step / max(refinement_steps - 1, 1))
            raw_update = dt * self(batch, x, t)["v_final"]
            applied_step, clipped = clip_atom_displacement(
                raw_update, max_displacement=max_displacement
            )
            raw_step_norms.append(torch.linalg.norm(raw_update, dim=-1))
            applied_step_norms.append(torch.linalg.norm(applied_step, dim=-1))
            clipped_atom_steps += int(clipped.sum().item())
            total_atom_steps += int(clipped.numel())
            candidate = x + applied_step
            finite = bool(torch.isfinite(candidate).all())
            bounded = bool(torch.linalg.norm(candidate, dim=-1).max() < max_coordinate_norm)
            if not finite or not bounded:
                stable, failed_step = False, step
                break
            x = candidate
        rollout_update = x - x_init
        mean_rollout_update_norm = float(
            torch.linalg.norm(rollout_update, dim=-1).mean()
        )
        alpha_eff = float(update_scale)
        if adaptive_alpha_by_update_norm:
            alpha_eff = min(
                alpha_eff,
                float(target_update_norm) / (mean_rollout_update_norm + 1.0e-8),
            )
        applied_update = alpha_eff * rollout_update
        x = x_init + applied_update
        final_finite = bool(torch.isfinite(x).all())
        final_bounded = bool(
            torch.linalg.norm(x, dim=-1).max() < max_coordinate_norm
        )
        stable = stable and final_finite and final_bounded
        if not stable and failed_step is None:
            failed_step = refinement_steps
        total_norm = torch.linalg.norm(applied_update, dim=-1)
        raw_norm = torch.cat(raw_step_norms) if raw_step_norms else total_norm.new_empty(0)
        applied_step_norm = (
            torch.cat(applied_step_norms)
            if applied_step_norms
            else total_norm.new_empty(0)
        )
        return x, {
            "stable": stable,
            "failed_step": failed_step,
            "update_scale": float(update_scale),
            "alpha": float(update_scale),
            "alpha_eff": alpha_eff,
            "adaptive_alpha_by_update_norm": bool(adaptive_alpha_by_update_norm),
            "target_update_norm": target_update_norm,
            "max_displacement": max_displacement,
            "mean_rollout_update_norm": mean_rollout_update_norm,
            "mean_update_norm": float(total_norm.mean()),
            "median_update_norm": float(total_norm.median()),
            "max_update_norm": float(total_norm.max()),
            "mean_step_update_norm_raw": (
                float(raw_norm.mean()) if raw_norm.numel() else 0.0
            ),
            "mean_step_update_norm_applied": (
                float(applied_step_norm.mean()) if applied_step_norm.numel() else 0.0
            ),
            "clipping_fraction": (
                clipped_atom_steps / total_atom_steps if total_atom_steps else 0.0
            ),
            # Backward-compatible aliases.
            "mean_raw_step_update_norm": (
                float(raw_norm.mean()) if raw_norm.numel() else 0.0
            ),
            "mean_applied_step_update_norm": (
                float(applied_step_norm.mean()) if applied_step_norm.numel() else 0.0
            ),
            "fraction_clipped_atoms": (
                clipped_atom_steps / total_atom_steps if total_atom_steps else 0.0
            ),
            "fraction_clipped_atom_steps": (
                clipped_atom_steps / total_atom_steps if total_atom_steps else 0.0
            ),
        }


class CartesianOptimizer(FlexBondOptimizerLightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__(mode="cartesian_optimizer", **kwargs)


class FlexBond4DOnlyOptimizer(FlexBondOptimizerLightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__(mode="flexbond4d_only_optimizer", **kwargs)


class FlexBond4DHybridOptimizer(FlexBondOptimizerLightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__(mode="flexbond4d_hybrid_optimizer", **kwargs)
