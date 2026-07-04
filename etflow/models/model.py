import warnings
from typing import Any, Dict, List, Optional, TypeVar

import numpy as np
import torch
from lightning.pytorch import seed_everything
from torch import Tensor
from torch_geometric.data import Batch

from etflow.commons.angular_loss import compute_target_dot_tau
from etflow.commons.bond_local_velocity import bond_local_velocity_loss
from etflow.commons.configs import CONFIG_DICT
from etflow.commons.covmat import set_multiple_rdmol_positions
from etflow.commons.featurization import MoleculeFeaturizer, get_mol_from_smiles
from etflow.commons.jacobian_4d_velocity import solve_q_targets
from etflow.commons.utils import signed_volume
from etflow.models.base import BaseModel
from etflow.models.loss import batchwise_l2_loss
from etflow.models.utils import (
    HarmonicSampler,
    center_of_mass,
    extend_bond_index,
    rmsd_align,
    unsqueeze_like,
)
from etflow.networks.torchmd_net import TorchMDDynamics

__all__ = ["BaseFlow"]

Config = TypeVar("Config", str, Dict[str, Any])


class BaseFlow(BaseModel):
    """LightningModule for Flow Matching"""

    __prior_types__ = ["gaussian", "harmonic"]
    __interpolation_types__ = ["linear", "gvp", "gvp_w_sigma", "gvp_squared"]

    def __init__(
        self,
        # flow matching network args
        network_type: str = "TorchMDDynamics",
        hidden_channels: int = 128,
        num_layers: int = 8,
        num_rbf: int = 64,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = False,
        activation: str = "silu",
        neighbor_embedding: int = True,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 10.0,
        max_z: int = 100,
        node_attr_dim: int = 0,
        edge_attr_dim: int = 0,
        attn_activation: str = "silu",
        num_heads: int = 8,
        distance_influence: str = "both",
        reduce_op: str = "sum",
        qk_norm: bool = False,
        output_layer_norm: bool = False,
        clip_during_norm: bool = False,
        max_num_neighbors: int = 32,
        so3_equivariant: bool = False,
        use_angular_head: bool = False,
        angular_mu: float = 1.0,
        angular_head_hidden_channels: Optional[int] = None,
        angular_mu_schedule: str = "constant",
        angular_mu_max: float = 0.3,
        angular_mu_sigmoid_k: float = 10.0,
        angular_mu_sigmoid_t0: float = 0.5,
        use_angular_loss: bool = False,
        angular_loss_weight: float = 0.01,
        # flow matching args
        sigma: float = 0.1,
        prior_type: str = "gaussian",
        sample_time_dist: str = "uniform",
        harmonic_alpha: float = 1.0,
        parity_switch: Optional[str] = None,
        # make edge_type one_hot
        edge_one_hot: bool = False,
        edge_one_hot_types: int = 5,
        use_bond_local_velocity_loss: bool = False,
        bond_velocity_loss_weight: float = 0.003,
        bond_velocity_on_rotatable_only: bool = False,
        use_jacobian_4d_correction: bool = False,
        jacobian_4d_on_rotatable_only: bool = True,
        jacobian_4d_affect_smaller_side: bool = True,
        jacobian_4d_min_affected_atoms: int = 2,
        jacobian_4d_max_bonds_per_mol: int = 16,
        jacobian_4d_correction_scale: float = 0.03,
        jacobian_4d_warmup_steps: int = 500,
        jacobian_4d_q_loss_weight: float = 0.001,
        jacobian_4d_corr_reg_weight: float = 0.0001,
        jacobian_4d_use_q_target: bool = True,
        jacobian_4d_ridge_eps: float = 1.0e-4,
        jacobian_4d_max_q_norm: float = 10.0,
        jacobian_4d_max_condition: float = 1.0e6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # setup network
        if network_type == "TorchMDDynamics":
            self.network = TorchMDDynamics(
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                num_rbf=num_rbf,
                rbf_type=rbf_type,
                trainable_rbf=trainable_rbf,
                activation=activation,
                neighbor_embedding=neighbor_embedding,
                cutoff_lower=cutoff_lower,
                cutoff_upper=cutoff_upper,
                max_z=max_z,
                node_attr_dim=node_attr_dim,
                edge_attr_dim=edge_attr_dim,
                attn_activation=attn_activation,
                num_heads=num_heads,
                distance_influence=distance_influence,
                reduce_op=reduce_op,
                qk_norm=qk_norm,
                output_layer_norm=output_layer_norm,
                clip_during_norm=clip_during_norm,
                so3_equivariant=so3_equivariant,
                use_angular_head=use_angular_head,
                angular_mu=angular_mu,
                angular_mu_schedule=angular_mu_schedule,
                angular_mu_max=angular_mu_max,
                angular_mu_sigmoid_k=angular_mu_sigmoid_k,
                angular_mu_sigmoid_t0=angular_mu_sigmoid_t0,
                angular_head_hidden_channels=angular_head_hidden_channels,
                use_jacobian_4d_correction=use_jacobian_4d_correction,
                jacobian_4d_min_affected_atoms=jacobian_4d_min_affected_atoms,
                jacobian_4d_max_bonds_per_mol=jacobian_4d_max_bonds_per_mol,
            )
        else:
            raise NotImplementedError(f"Network {network_type} not implemented.")

        self.sigma = sigma
        self.cutoff = cutoff_upper
        self.parity_switch = parity_switch
        self.prior_type = prior_type
        self.sample_time_dist = sample_time_dist
        self.edge_one_hot = edge_one_hot
        self.edge_one_hot_types = edge_one_hot_types
        self.max_num_neighbors = max_num_neighbors
        self.use_angular_head = use_angular_head
        self.use_angular_loss = bool(use_angular_loss)
        self.angular_loss_weight = float(angular_loss_weight)
        if self.use_angular_loss and not self.use_angular_head:
            raise ValueError("Angular loss requires use_angular_head=true.")
        if self.angular_loss_weight < 0:
            raise ValueError(
                f"angular_loss_weight must be non-negative, got {angular_loss_weight}."
            )
        self.use_bond_local_velocity_loss = bool(use_bond_local_velocity_loss)
        self.bond_velocity_loss_weight = float(bond_velocity_loss_weight)
        self.bond_velocity_on_rotatable_only = bool(
            bond_velocity_on_rotatable_only
        )
        if (
            not np.isfinite(self.bond_velocity_loss_weight)
            or self.bond_velocity_loss_weight < 0
        ):
            raise ValueError(
                "bond_velocity_loss_weight must be finite and non-negative, got "
                f"{bond_velocity_loss_weight}."
            )
        self.use_jacobian_4d_correction = bool(use_jacobian_4d_correction)
        self.jacobian_4d_on_rotatable_only = bool(
            jacobian_4d_on_rotatable_only
        )
        self.jacobian_4d_affect_smaller_side = bool(
            jacobian_4d_affect_smaller_side
        )
        self.jacobian_4d_min_affected_atoms = int(
            jacobian_4d_min_affected_atoms
        )
        self.jacobian_4d_max_bonds_per_mol = int(
            jacobian_4d_max_bonds_per_mol
        )
        self.jacobian_4d_correction_scale = float(
            jacobian_4d_correction_scale
        )
        self.jacobian_4d_warmup_steps = int(jacobian_4d_warmup_steps)
        self.jacobian_4d_q_loss_weight = float(jacobian_4d_q_loss_weight)
        self.jacobian_4d_corr_reg_weight = float(
            jacobian_4d_corr_reg_weight
        )
        self.jacobian_4d_use_q_target = bool(jacobian_4d_use_q_target)
        self.jacobian_4d_ridge_eps = float(jacobian_4d_ridge_eps)
        self.jacobian_4d_max_q_norm = float(jacobian_4d_max_q_norm)
        self.jacobian_4d_max_condition = float(jacobian_4d_max_condition)
        if self.use_jacobian_4d_correction:
            if not self.jacobian_4d_on_rotatable_only:
                raise ValueError(
                    "The prototype currently supports cached rotatable bonds only."
                )
            if not self.jacobian_4d_affect_smaller_side:
                raise ValueError(
                    "The cached bond orientation currently supports the smaller "
                    "affected side only."
                )
        if self.jacobian_4d_min_affected_atoms < 1:
            raise ValueError("jacobian_4d_min_affected_atoms must be positive.")
        if self.jacobian_4d_max_bonds_per_mol < 1:
            raise ValueError("jacobian_4d_max_bonds_per_mol must be positive.")
        if self.jacobian_4d_warmup_steps < 0:
            raise ValueError("jacobian_4d_warmup_steps must be non-negative.")
        for name, value, strictly_positive in (
            ("jacobian_4d_correction_scale", self.jacobian_4d_correction_scale, False),
            ("jacobian_4d_q_loss_weight", self.jacobian_4d_q_loss_weight, False),
            ("jacobian_4d_corr_reg_weight", self.jacobian_4d_corr_reg_weight, False),
            ("jacobian_4d_ridge_eps", self.jacobian_4d_ridge_eps, True),
            ("jacobian_4d_max_q_norm", self.jacobian_4d_max_q_norm, True),
            ("jacobian_4d_max_condition", self.jacobian_4d_max_condition, True),
        ):
            invalid = not np.isfinite(value) or (
                value <= 0 if strictly_positive else value < 0
            )
            if invalid:
                relation = "positive" if strictly_positive else "non-negative"
                raise ValueError(f"{name} must be finite and {relation}, got {value}.")

        if parity_switch is not None:
            assert (
                parity_switch == "post_hoc"
            ), f"Parity switch {parity_switch} not implemented"

        assert (
            self.prior_type in self.__prior_types__
        ), f"""\nPrior type {prior_type} not available.
            This is the list of implemented prior types {self.__prior_types__}.\n"""

        if prior_type == "harmonic":
            self.harmonic_sampler = HarmonicSampler(alpha=harmonic_alpha)

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        """Load legacy checkpoints while strictly checking the existing model."""

        optional_head_prefixes = []
        if self.use_angular_head:
            optional_head_prefixes.append("network.bond_angular_head.")
        if self.use_jacobian_4d_correction:
            optional_head_prefixes.append("network.jacobian_4d_head.")
        if not strict or not optional_head_prefixes:
            return super().load_state_dict(state_dict, strict=strict, **kwargs)

        incompatible = super().load_state_dict(state_dict, strict=False, **kwargs)
        missing_keys = set(incompatible.missing_keys)
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not any(key.startswith(prefix) for prefix in optional_head_prefixes)
        ]
        partially_missing_head = False
        for prefix in optional_head_prefixes:
            expected = {
                key for key in self.state_dict() if key.startswith(prefix)
            }
            missing_for_head = {key for key in missing_keys if key.startswith(prefix)}
            if missing_for_head and missing_for_head != expected:
                partially_missing_head = True
        if (
            invalid_missing
            or partially_missing_head
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                "Checkpoint is incompatible beyond an optional residual head. "
                f"Missing keys: {incompatible.missing_keys}; "
                f"unexpected keys: {incompatible.unexpected_keys}."
            )
        if incompatible.missing_keys:
            warnings.warn(
                "Loaded a legacy checkpoint without optional residual-head "
                "weights; zero-initialized output layers preserve the original "
                "velocity field at initialization.",
                stacklevel=2,
            )
        return incompatible

    @classmethod
    def from_config(cls, cfg: Config):
        import yaml

        if isinstance(cfg, str):
            cfg = yaml.safe_load(open(cfg))
        if isinstance(cfg, dict):
            return cls(**cfg["model_args"])
        else:
            raise ValueError("cfg should be a dictionary or a path to a yaml file")

    @classmethod
    def from_default(
        cls,
        model: str = "drugs-o3",
        device: str | torch.DeviceObjType = "cuda",
        cache: Optional[str] = None,
    ):
        model = model.lower()
        if model not in CONFIG_DICT:
            raise ValueError(
                f"Model config {model} not found. Available checkpoints are {CONFIG_DICT.keys()}"
            )
        else:
            config = CONFIG_DICT.get(model, None)()
            print(f"Loading {model} from config")
            config.checkpoint_config.set_cache(cache)
            checkpoint_path = config.checkpoint_config.fetch_checkpoint().local_path

        found_device = get_device()
        if isinstance(device, str):
            device = torch.device(device)
        if device != found_device and device != torch.device("cpu"):
            print(f"Device {device} not found. Using {found_device} instead")
            device = found_device

        etflow_model = cls.from_config(config.model_dict())
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                # Standard Lightning checkpoint
                etflow_model.load_state_dict(checkpoint["state_dict"])
            else:
                # Plain state dict
                etflow_model.load_state_dict(checkpoint)
        etflow_model.eval()
        return etflow_model

    def sigma_t(self, t):
        return self.sigma * torch.sqrt(t * (1 - t))

    def sigma_dot_t(self, t):
        return self.sigma * 0.5 * (1 - 2 * t) / torch.sqrt(t * (1 - t))

    def sample_conditional_pt(self, x0: Tensor, x1: Tensor, t: Tensor, batch: Tensor):
        # Have this here in case sample_conditional_pt
        # is used outside of compute_conditional_vector_field
        # center both x0 and pos (x1: data distribution)
        x0 = center_of_mass(x0, batch=batch)
        x1 = center_of_mass(x1, batch=batch)

        # unsqueeze t and then reshape to number of atoms
        t = t[batch] if batch is not None else t
        t = unsqueeze_like(t, target=x0)

        # linear interpolation between x0 and x1
        # mu_t = self.interpolation_fn(x0, x1, t)
        eps = torch.randn_like(x1)

        # center each around center of mass
        eps = center_of_mass(eps, batch=batch)
        mu_t = t * x1 + (1 - t) * x0

        # no noise at t = 0 or t = 1
        x_t = mu_t + self.sigma_t(t) * eps

        return x_t, eps

    def compute_conditional_vector_field(self, x0, x1, t, batch=None):
        if batch is None:
            batch = torch.zeros((x1.size(0),)).to(self.device)

        # sample a gaussian centered around the interpolation of x1, x0
        x_t, eps = self.sample_conditional_pt(x0, x1, t, batch=batch)
        t = unsqueeze_like(t[batch], x1)

        # derivative of interpolate plus derivative of sigma function * noise
        u_t = x1 - x0 + self.sigma_dot_t(t) * eps

        return x_t, u_t

    def switch_parity_of_pos(
        self, pos, chiral_index, chiral_nbr_index, chiral_tag, batch
    ):
        assert all(
            [
                key is not None
                for key in [chiral_index, chiral_nbr_index, chiral_tag, batch]
            ]
        )
        num_graphs = batch.max().item() + 1
        sv = signed_volume(
            pos[chiral_nbr_index.view(chiral_index.shape[1], 4)].unsqueeze(2)
        ).squeeze()
        ct = chiral_tag
        z_flip = sv * ct

        graph_diag = torch.ones(num_graphs, device=self.device)
        graph_diag[batch[chiral_index][:, (z_flip == -1.0)].squeeze()] = -1.0
        node_factor = graph_diag[batch].unsqueeze(1)

        return pos * node_factor

    def sample_base_dist(
        self,
        size: torch.Size,
        edge_index: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        smiles: Optional[str] = None,
    ) -> Tensor:
        """Sample from prior distribution (Either harmonic or gaussian)"""
        if self.prior_type == "harmonic":
            assert (edge_index is not None) and (batch is not None)
            x0 = self.harmonic_sampler.sample(
                size=size, edge_index=edge_index, batch=batch, smiles=smiles
            ).to(self.device)

            # check if x0 is nan
            if torch.isnan(x0).any():
                raise ValueError("x0 is NaN. Check edge_index for disconnected graphs!")

            return x0

        # gaussian prior if not harmonic
        return torch.randn(size=size, device=self.device)

    def sample_time(
        self,
        num_samples: int,
        low: float = 1e-4,
        high: float = 0.9999,
        stage: str = "train",
    ):
        """Sample flow-matching time steps for training or validation"""
        if self.sample_time_dist == "uniform" or stage == "val":
            return torch.zeros(size=(num_samples, 1), device=self.device).uniform_(
                low, high
            )
        elif self.sample_time_dist == "logit_norm":
            return torch.sigmoid(torch.randn(size=(num_samples, 1), device=self.device))

        raise NotImplementedError(
            f"Time sampling with {self.sample_time_dist} not implemented"
        )

    def forward(
        self,
        z: Tensor,
        t: Tensor,
        pos: Tensor,
        bond_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        rotatable_bond_index: Optional[Tensor] = None,
        atom_bond_influence_index: Optional[Tensor] = None,
        jacobian_4d_warmup_scale=1.0,
        return_aux: bool = False,
    ):
        # center the positions at 0
        pos = center_of_mass(pos, batch=batch)

        # compute extended bond index
        edge_index, edge_type = extend_bond_index(
            pos=pos,
            bond_index=bond_index,
            batch=batch,
            bond_attr=edge_attr,
            device=self.device,
            one_hot=self.edge_one_hot,
            one_hot_types=self.edge_one_hot_types,
            cutoff=self.cutoff,
            max_num_neighbors=self.max_num_neighbors,
        )

        # compute energy and score from network
        v_t = self.network(
            z=z,
            t=t[batch],
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_type,
            node_attr=node_attr,
            batch=batch,
            rotatable_bond_index=rotatable_bond_index,
            atom_bond_influence_index=atom_bond_influence_index,
            jacobian_4d_correction_scale=self.jacobian_4d_correction_scale,
            jacobian_4d_warmup_scale=jacobian_4d_warmup_scale,
            return_aux=return_aux,
        )

        return v_t

    def _jacobian_4d_warmup_scale(self, reference: Tensor) -> Tensor:
        if self.jacobian_4d_warmup_steps == 0:
            return reference.new_ones(())
        trainer = getattr(self, "_trainer", None)
        global_step = trainer.global_step if trainer is not None else 0
        step = reference.new_tensor(float(global_step))
        return (step / float(self.jacobian_4d_warmup_steps)).clamp(0.0, 1.0)

    def generic_step(self, batched_data, batch_idx: int, stage: str):
        # atomic numbers
        z, pos, bond_index, node_attr, edge_attr, batch = (
            batched_data["atomic_numbers"],
            batched_data["pos"],
            batched_data["edge_index"],
            batched_data.get("node_attr", None),  # optional
            batched_data.get("edge_attr", None),  # optional
            batched_data.get("batch", None),  # optional
        )
        batch_size = batch.max().item() + 1 if batch is not None else 1

        # sample base distribution, either from harmonic or gaussian
        # x0 is sampling distribution and not data distribution
        x0 = self.sample_base_dist(
            pos.shape,
            edge_index=bond_index,
            batch=batch,
            smiles=batched_data.get("smiles", None),
        )

        # sample time steps equal to number of molecules in a batch
        t = self.sample_time(num_samples=batch_size, stage=stage)

        if self.prior_type == "harmonic":
            x0 = rmsd_align(pos=x0, ref_pos=pos, batch=batch)

        # sample conditional vector field for positions
        x_t, u_t = self.compute_conditional_vector_field(
            x0=x0, x1=pos, t=t, batch=batch
        )

        rotatable_bond_index = batched_data.get("rotatable_bond_index", None)
        atom_bond_influence_index = batched_data.get("atom_bond_influence_index", None)
        return_aux = self.use_angular_loss or self.use_jacobian_4d_correction
        jacobian_warmup_scale = (
            self._jacobian_4d_warmup_scale(x_t)
            if self.use_jacobian_4d_correction
            else x_t.new_ones(())
        )
        # run flow matching network
        network_output = self(
            z=z,
            t=t,
            pos=x_t,
            bond_index=bond_index,
            edge_attr=edge_attr,
            node_attr=node_attr,
            batch=batch,
            rotatable_bond_index=rotatable_bond_index,
            atom_bond_influence_index=atom_bond_influence_index,
            jacobian_4d_warmup_scale=jacobian_warmup_scale,
            return_aux=return_aux,
        )

        if return_aux:
            v_t, branch_aux = network_output
        else:
            v_t = network_output
            branch_aux = None

        # regress against vector field
        flow_matching_loss = batchwise_l2_loss(v_t, u_t, batch=batch, reduce="mean")
        loss = flow_matching_loss

        if self.use_angular_loss:
            dot_tau_pred = branch_aux["dot_tau_pred"]
            dot_tau_target, valid_bond_mask = compute_target_dot_tau(
                pos=branch_aux["pos"],
                target_velocity=u_t,
                rotatable_bond_index=branch_aux["rotatable_bond_index"],
                atom_bond_influence_index=branch_aux[
                    "atom_bond_influence_index"
                ],
                batch=batch,
            )
            if dot_tau_pred.shape != dot_tau_target.shape:
                raise ValueError(
                    "dot_tau prediction/target shape mismatch: "
                    f"{tuple(dot_tau_pred.shape)} vs {tuple(dot_tau_target.shape)}."
                )

            if valid_bond_mask.any():
                angular_dot_tau_loss = (
                    dot_tau_pred[valid_bond_mask] - dot_tau_target[valid_bond_mask]
                ).square().mean()
                mean_abs_dot_tau_pred = dot_tau_pred[valid_bond_mask].detach().abs().mean()
                mean_abs_dot_tau_target = dot_tau_target[valid_bond_mask].abs().mean()
            else:
                angular_dot_tau_loss = flow_matching_loss.new_zeros(())
                mean_abs_dot_tau_pred = flow_matching_loss.new_zeros(())
                mean_abs_dot_tau_target = flow_matching_loss.new_zeros(())

            loss = loss + self.angular_loss_weight * angular_dot_tau_loss
            self.log_helper(
                f"{stage}/angular/dot_tau_loss",
                angular_dot_tau_loss,
                batch_size=batch_size,
            )
            self.log_helper(
                f"{stage}/angular/num_valid_angular_bonds",
                valid_bond_mask.sum().to(dtype=flow_matching_loss.dtype),
                batch_size=batch_size,
            )
            self.log_helper(
                f"{stage}/angular/mean_abs_dot_tau_pred",
                mean_abs_dot_tau_pred,
                batch_size=batch_size,
            )
            self.log_helper(
                f"{stage}/angular/mean_abs_dot_tau_target",
                mean_abs_dot_tau_target,
                batch_size=batch_size,
            )

        if self.use_jacobian_4d_correction:
            v_atom = branch_aux["v_atom"]
            q_pred = branch_aux["q_pred"]
            flow_matching_loss_base = batchwise_l2_loss(
                v_atom, u_t, batch=batch, reduce="mean"
            )
            residual = (u_t - v_atom).detach()
            if self.jacobian_4d_use_q_target:
                q_target, q_valid_mask, conditions = solve_q_targets(
                    pos=branch_aux["pos"].detach(),
                    residual=residual,
                    anchor_index=branch_aux["anchor_index"],
                    moving_index=branch_aux["moving_index"],
                    affected_atom_index=branch_aux["affected_atom_index"],
                    affected_bond_index=branch_aux["affected_bond_index"],
                    ridge_eps=self.jacobian_4d_ridge_eps,
                    max_q_norm=self.jacobian_4d_max_q_norm,
                    max_condition=self.jacobian_4d_max_condition,
                )
                q_valid_mask = q_valid_mask & branch_aux["geometry_valid"]
            else:
                q_target = torch.zeros_like(q_pred)
                q_valid_mask = branch_aux["geometry_valid"]
                conditions = q_pred.new_zeros((q_pred.size(0),))

            supervision_mask = (
                q_valid_mask
                if self.jacobian_4d_use_q_target
                else torch.zeros_like(q_valid_mask)
            )
            supervision_weight = supervision_mask.to(dtype=q_pred.dtype)
            supervision_count = supervision_weight.sum().clamp_min(1.0)
            q_error = q_pred - q_target
            q_loss = (
                q_error.square() * supervision_weight[:, None]
            ).sum() / (4.0 * supervision_count)
            s_loss = (
                q_error[:, 0].square() * supervision_weight
            ).sum() / supervision_count
            omega_loss = (
                q_error[:, 1:].square() * supervision_weight[:, None]
            ).sum() / (3.0 * supervision_count)
            omega_parallel_loss = (
                q_error[:, 1].square() * supervision_weight
            ).sum() / supervision_count
            omega_perp_loss = (
                q_error[:, 2:].square() * supervision_weight[:, None]
            ).sum() / (2.0 * supervision_count)
            mean_abs_s_pred = (
                q_pred[:, 0].detach().abs() * supervision_weight
            ).sum() / supervision_count
            mean_abs_s_target = (
                q_target[:, 0].abs() * supervision_weight
            ).sum() / supervision_count
            mean_abs_omega_pred = (
                q_pred[:, 1:].detach().abs() * supervision_weight[:, None]
            ).sum() / (3.0 * supervision_count)
            mean_abs_omega_target = (
                q_target[:, 1:].abs() * supervision_weight[:, None]
            ).sum() / (3.0 * supervision_count)
            q_target_norm = (
                torch.linalg.norm(q_target, dim=-1) * supervision_weight
            ).sum() / supervision_count
            finite_condition = torch.where(
                supervision_mask, conditions, torch.zeros_like(conditions)
            )
            condition_mean = finite_condition.sum() / supervision_count

            scaled_v_corr = branch_aux["scaled_v_corr"]
            corr_reg_loss = scaled_v_corr.square().mean()
            loss = (
                loss
                + self.jacobian_4d_q_loss_weight * q_loss
                + self.jacobian_4d_corr_reg_weight * corr_reg_loss
            )

            selected_count = q_pred.new_tensor(float(q_pred.size(0)))
            valid_count = q_valid_mask.sum().to(dtype=q_pred.dtype)
            skip_rate = (selected_count - valid_count) / selected_count.clamp_min(1.0)
            corr_norm = scaled_v_corr.detach().square().mean().sqrt()
            atom_norm = v_atom.detach().square().mean().sqrt()
            residual_norm = residual.square().mean().sqrt()
            jacobian_metrics = {
                "num_selected_bonds": selected_count,
                "num_valid_bonds": valid_count,
                "skip_rate": skip_rate,
                "q_loss": q_loss,
                "s_loss": s_loss,
                "omega_loss": omega_loss,
                "omega_parallel_loss": omega_parallel_loss,
                "omega_perp_loss": omega_perp_loss,
                "mean_abs_s_pred": mean_abs_s_pred,
                "mean_abs_s_target": mean_abs_s_target,
                "mean_abs_omega_pred": mean_abs_omega_pred,
                "mean_abs_omega_target": mean_abs_omega_target,
                "corr_norm": corr_norm,
                "corr_to_atom_ratio": corr_norm / atom_norm.clamp_min(1.0e-8),
                "corr_to_residual_ratio": corr_norm
                / residual_norm.clamp_min(1.0e-8),
                "q_target_norm": q_target_norm,
                "condition_mean": condition_mean,
            }
            self.log_helper(
                f"{stage}/flow_matching_loss_base",
                flow_matching_loss_base,
                batch_size=batch_size,
            )
            for name, value in jacobian_metrics.items():
                self.log_helper(
                    f"{stage}/jacobian_4d/{name}",
                    value,
                    batch_size=batch_size,
                )

        if self.use_bond_local_velocity_loss:
            if self.bond_velocity_on_rotatable_only:
                if rotatable_bond_index is None:
                    raise ValueError(
                        "bond_velocity_on_rotatable_only requires "
                        "rotatable_bond_index in the batch."
                    )
                local_bond_index = rotatable_bond_index
            else:
                # Dataset chemical bonds are stored in both directions. Selecting
                # the increasing orientation keeps exactly one edge per bond.
                local_bond_index = bond_index[:, bond_index[0] < bond_index[1]]

            bond_velocity_loss, bond_velocity_stats = bond_local_velocity_loss(
                pos=x_t,
                pred_velocity=v_t,
                target_velocity=u_t,
                bond_index=local_bond_index,
            )
            loss = loss + self.bond_velocity_loss_weight * bond_velocity_loss
            for name, value in bond_velocity_stats.items():
                self.log_helper(
                    f"{stage}/bond_local/{name}",
                    value,
                    batch_size=batch_size,
                )

        if torch.isnan(loss):
            raise ValueError("Loss is NaN, fix bug")

        # log loss
        if not self.use_jacobian_4d_correction:
            self.log_helper(
                f"{stage}/flow_matching_loss_base",
                flow_matching_loss,
                batch_size=batch_size,
            )
        self.log_helper(
            f"{stage}/flow_matching_loss",
            flow_matching_loss,
            batch_size=batch_size,
        )
        self.log_helper(f"{stage}/loss", loss, batch_size=batch_size)
        for name, value in self.network.last_angular_stats.items():
            self.log_helper(f"{stage}/angular/{name}", value, batch_size=batch_size)

        return loss

    def _compute_delta_t(self, t_schedule: Tensor, t: Tensor):
        if t + 1 >= t_schedule.size(0):
            return 0.0

        t_curr, t_next = t_schedule[t : t + 2]
        return t_next - t_curr

    @torch.no_grad()
    def sample(
        self,
        z: Tensor,
        bond_index: Tensor,
        batch: Tensor,
        node_attr: Tensor = None,
        edge_attr: Tensor = None,
        chiral_index: Tensor = None,
        chiral_nbr_index: Tensor = None,
        chiral_tag: Tensor = None,
        rotatable_bond_index: Tensor = None,
        atom_bond_influence_index: Tensor = None,
        n_timesteps: int = 50,
        s_churn: float = 1.0,
        t_min: float = 1.0,
        t_max: float = 1.0,
        std: float = 1.0,
        sampler_type: str = "ode",
    ):
        """
        By default performs ODE (sampler_type="ode") sampling
        If sampler_type is set to "stochastic", then it performs stochastic sampling
        """
        t_schedule = torch.linspace(0, 1.0, steps=n_timesteps + 1, device=self.device)

        x = center_of_mass(
            self.sample_base_dist((z.size(0), 3), bond_index, batch), batch=batch
        )
        gamma = torch.tensor(s_churn / n_timesteps).to(self.device)

        n = t_schedule.size(0) - 1
        for i in range(n):
            t = t_schedule[i].repeat(x.size(0))
            t = unsqueeze_like(t, x)
            delta_t = self._compute_delta_t(t_schedule, t=i)

            # We do ODE when t is outside of [s_min, s_max]
            if (
                t_schedule[i] < t_min or t_schedule[i] >= t_max
            ) or sampler_type == "ode":
                v_t = self(
                    z=z,
                    t=t,
                    pos=x,
                    bond_index=bond_index,
                    edge_attr=edge_attr,
                    node_attr=node_attr,
                    batch=batch,
                    rotatable_bond_index=rotatable_bond_index,
                    atom_bond_influence_index=atom_bond_influence_index,
                )
                x = x + delta_t * v_t

            # Stochastic sampling
            else:
                # delta_hat = gamma*delta_t
                delta_hat = gamma * (1 - t_schedule[i])
                t_prev_int = t_schedule[i] - delta_hat
                t_prev = t_prev_int.repeat(x.size(0))
                t_prev = unsqueeze_like(t_prev, x)
                """linear noise"""
                sig_t_sq = t_schedule[i] ** 2
                sig_t_prev_sq = t_prev_int**2
                mean = torch.zeros_like(x)
                noise = torch.normal(mean=mean, std=std)
                noise = center_of_mass(noise, batch=batch)
                x_prev = (
                    x
                    + torch.sqrt(torch.abs(sig_t_sq - sig_t_prev_sq))
                    * noise
                    * delta_hat
                )  # quadratic + linear decay

                v_t_prev = self(
                    z=z,
                    t=t_prev,
                    pos=x_prev,
                    bond_index=bond_index,
                    edge_attr=edge_attr,
                    node_attr=node_attr,
                    batch=batch,
                    rotatable_bond_index=rotatable_bond_index,
                    atom_bond_influence_index=atom_bond_influence_index,
                )
                # update step
                x = x_prev + v_t_prev * (delta_t + delta_hat)

        if self.parity_switch == "post_hoc":
            x = self.switch_parity_of_pos(
                x, chiral_index, chiral_nbr_index, chiral_tag, batch
            )

        return x

    @torch.no_grad()
    def predict(
        self,
        smiles: List[str],
        max_batch_size: int = 1,
        num_samples: int = 1,
        n_timesteps: int = 50,
        seed: int = 42,
        device: str = "cpu",
        s_churn: float = 1.0,
        t_min: float = 1.0,
        t_max: float = 1.0,
        std: float = 1.0,
        sampler_type: str = "ode",
        as_mol: bool = False,
    ):
        if seed is not None:
            seed_everything(seed)

        def sample(
            data,
            max_batch_size,
            num_samples,
            n_timesteps,
            device,
            s_churn,
            t_min,
            t_max,
            std,
            sampler_type,
        ):
            sampled_pos = []

            for batch_start in range(0, num_samples, max_batch_size):
                # get batch_size
                batch_size = min(max_batch_size, num_samples - batch_start)
                # batch the data
                batched_data = Batch.from_data_list([data] * batch_size)

                # get one_hot, edge_index, batch
                (
                    z,
                    edge_index,
                    batch,
                    node_attr,
                    chiral_index,
                    chiral_nbr_index,
                    chiral_tag,
                    rotatable_bond_index,
                    atom_bond_influence_index,
                ) = (
                    batched_data["atomic_numbers"].to(device),
                    batched_data["edge_index"].to(device),
                    batched_data["batch"].to(device),
                    batched_data["node_attr"].to(device),
                    batched_data["chiral_index"].to(device),
                    batched_data["chiral_nbr_index"].to(device),
                    batched_data["chiral_tag"].to(device),
                    batched_data["rotatable_bond_index"].to(device),
                    batched_data["atom_bond_influence_index"].to(device),
                )

                with torch.no_grad():
                    pos = self.sample(
                        z=z,
                        bond_index=edge_index,
                        batch=batch,
                        node_attr=node_attr,
                        n_timesteps=n_timesteps,
                        chiral_index=chiral_index,
                        chiral_nbr_index=chiral_nbr_index,
                        chiral_tag=chiral_tag,
                        rotatable_bond_index=rotatable_bond_index,
                        atom_bond_influence_index=atom_bond_influence_index,
                        s_churn=s_churn,
                        t_min=t_min,
                        t_max=t_max,
                        std=std,
                        sampler_type=sampler_type,
                    )

                # reshape to (num_samples, num_atoms, 3) using batch
                pos = pos.view(batch_size, -1, 3).cpu().detach().numpy()

                # append to generated_positions
                sampled_pos.append(pos)

            # concatenate generated_positions
            sampled_pos = np.concatenate(
                sampled_pos, axis=0
            )  # (num_samples, num_atoms, 3)

            return sampled_pos

        feat = MoleculeFeaturizer()
        if not isinstance(smiles, list):
            smiles = [smiles]

        data = {}
        for smile in smiles:
            pos = sample(
                feat.get_data_from_smiles(
                    smile,
                ),
                max_batch_size,
                num_samples,
                n_timesteps,
                device,
                s_churn,
                t_min,
                t_max,
                std,
                sampler_type,
            )
            if as_mol:
                mol = get_mol_from_smiles(smile)
                data[smile] = set_multiple_rdmol_positions(mol, pos)
            else:
                data[smile] = pos
        return data


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
