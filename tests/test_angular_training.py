from types import MethodType

import torch
from torch import nn

from etflow.models.model import BaseFlow


def _make_batch():
    pos = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    return {
        "atomic_numbers": torch.tensor([6, 6, 1, 1]),
        "pos": pos,
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "batch": torch.zeros(4, dtype=torch.long),
        "rotatable_bond_index": torch.tensor([[0], [1]], dtype=torch.long),
        "atom_bond_influence_index": torch.tensor(
            [[2, 3], [0, 0]],
            dtype=torch.long,
        ),
    }


def _make_model(use_angular_loss):
    model = BaseFlow(
        hidden_channels=8,
        num_layers=1,
        num_rbf=4,
        num_heads=1,
        node_attr_dim=0,
        edge_attr_dim=0,
        output_layer_norm=False,
        so3_equivariant=True,
        use_angular_head=True,
        angular_mu=0.3,
        use_angular_loss=use_angular_loss,
        angular_loss_weight=0.01,
        prior_type="gaussian",
        sigma=0.0,
        lr_scheduler_type=None,
    )

    def sample_base_dist(self, size, **kwargs):
        return torch.zeros(size, device=self.device)

    def sample_time(self, num_samples, **kwargs):
        return torch.full((num_samples, 1), 0.5, device=self.device)

    def conditional_vector_field(self, x0, x1, t, batch=None):
        axis = x1.new_tensor([1.0, 0.0, 0.0])
        center = 0.5 * (x1[0] + x1[1])
        velocity = torch.zeros_like(x1)
        influenced_atoms = torch.tensor([2, 3], device=x1.device)
        basis = torch.cross(
            axis.expand(influenced_atoms.numel(), -1),
            x1[influenced_atoms] - center,
            dim=-1,
        )
        velocity[influenced_atoms] = 2.0 * basis
        return x1, velocity

    model.sample_base_dist = MethodType(sample_base_dist, model)
    model.sample_time = MethodType(sample_time, model)
    model.compute_conditional_vector_field = MethodType(
        conditional_vector_field,
        model,
    )
    model.log_helper = MethodType(lambda self, *args, **kwargs: None, model)
    return model


def test_disabled_angular_loss_preserves_flow_matching_only_behavior():
    model = _make_model(use_angular_loss=False)
    return_aux_values = []

    def fake_forward(self, pos, return_aux=False, **kwargs):
        return_aux_values.append(return_aux)
        return torch.zeros_like(pos)

    model.forward = MethodType(fake_forward, model)
    loss = model.training_step(_make_batch(), 0)

    assert return_aux_values == [False]
    torch.testing.assert_close(loss, torch.tensor(1.5))


def test_auxiliary_angular_loss_runs_one_optimization_step():
    model = _make_model(use_angular_loss=True)
    model.register_parameter("smoke_dot_tau", nn.Parameter(torch.zeros(1)))

    def fake_forward(
        self,
        pos,
        rotatable_bond_index,
        atom_bond_influence_index,
        return_aux=False,
        **kwargs,
    ):
        velocity = torch.zeros_like(pos)
        assert return_aux
        return velocity, {
            "dot_tau_pred": self.smoke_dot_tau,
            "rotatable_bond_index": rotatable_bond_index,
            "atom_bond_influence_index": atom_bond_influence_index,
            "pos": pos,
        }

    model.forward = MethodType(fake_forward, model)
    optimizer = torch.optim.SGD([model.smoke_dot_tau], lr=0.1)

    loss = model.training_step(_make_batch(), 0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert model.smoke_dot_tau.item() > 0.0
