import torch

from etflow.ecir.v8_constraint_layer import DifferentiableMolecularConstraintLayer
from tests.v8_test_utils import batch, graph


def test_batched_and_per_graph_solves_match_without_cross_graph_coupling():
    combined = batch(two=True)
    layer = DifferentiableMolecularConstraintLayer({}, scales={"bond": 0.1, "angle": 0.1})
    prior = torch.randn_like(combined.x_input) * 0.01
    batched = layer(combined.x_input, prior, torch.ones(8), combined)["delta_final"]
    singles = []
    for index, item in enumerate((graph(), graph(10.0))):
        local = batch(two=False) if index == 0 else type(combined).from_data_list([item])
        singles.append(
            layer(local.x_input, prior[index * 4 : (index + 1) * 4], torch.ones(4), local)[
                "delta_final"
            ]
        )
    assert torch.allclose(batched, torch.cat(singles), atol=1e-6, rtol=1e-5)
