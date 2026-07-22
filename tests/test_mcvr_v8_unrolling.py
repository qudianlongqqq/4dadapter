from types import SimpleNamespace

import torch
from torch import nn

from etflow.ecir.mcvr_v8_full import MCVRV8FullRefiner
from tests.v8_test_utils import batch


class DummyPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.atom_embedding = nn.Linear(10, 6)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, data, pos, t, **kwargs):
        atom_batch = data.batch
        return {
            "v_final": self.scale * pos,
            "node_embedding": self.backbone.atom_embedding(data.node_attr),
            "atom_batch": atom_batch,
        }


class RecordingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(enabled=True)
        self.coordinates = []

    def forward(self, coordinates, delta_prior, confidence, data):
        self.coordinates.append(coordinates.detach().clone())
        return {
            "delta_final": delta_prior,
            "solver_status": ("SOLVED",),
            "solver_failure": coordinates.new_zeros(1),
        }


def test_second_step_recomputes_from_updated_coordinates_with_shared_prior():
    model = MCVRV8FullRefiner(
        DummyPrior(), error_state_enabled=False, constraint_layer={"enabled": False}, unroll_steps=2
    )
    recorder = RecordingLayer()
    model.constraint_layer = recorder
    data = batch()
    output = model(data, data.x_input, torch.tensor([0.5]))
    assert len(recorder.coordinates) == 2
    assert not torch.equal(recorder.coordinates[0], recorder.coordinates[1])
    assert not torch.equal(output["step_deltas"][0], output["step_deltas"][1])


def test_cumulative_two_step_displacement_is_differentiably_bounded():
    model = MCVRV8FullRefiner(
        DummyPrior(),
        error_state_enabled=False,
        constraint_layer={"enabled": False},
        unroll_steps=2,
        max_cumulative_atom_displacement=0.12,
        max_cumulative_graph_rms=0.06,
    )
    recorder = RecordingLayer()
    model.constraint_layer = recorder
    data = batch()
    output = model(data, data.x_input, torch.tensor([0.5]))
    assert float(output["cumulative_delta"].norm(dim=-1).max()) <= 0.120001
    assert float(torch.sqrt(output["cumulative_delta"].square().sum(-1).mean())) <= 0.060001


def test_no_constraint_keeps_independent_cumulative_safety_projection():
    model = MCVRV8FullRefiner(
        DummyPrior(),
        error_state_enabled=False,
        constraint_layer={"enabled": False},
        unroll_steps=2,
        max_cumulative_atom_displacement=0.012,
        max_cumulative_graph_rms=0.006,
    )
    data = batch()
    output = model(data, data.x_input, torch.tensor([0.5]))
    assert output["step_outputs"][0]["solver_status"] == ("DISABLED",)
    assert float(output["cumulative_delta"].norm(dim=-1).max()) <= 0.012001
    atom_batch = output["atom_batch"]
    for graph in range(int(atom_batch.max()) + 1):
        local = output["cumulative_delta"][atom_batch == graph]
        assert float(torch.sqrt(local.square().sum(-1).mean())) <= 0.006001
