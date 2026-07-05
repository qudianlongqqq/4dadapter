import torch

from etflow.commons.geometry_diagnostics import (
    path_geometry_metrics,
    wrapped_angle_delta,
)


def test_identical_endpoint_path_has_zero_internal_geometry_error():
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 1.0]]
    )
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    rotatable = torch.tensor([[1], [2]], dtype=torch.long)
    metrics = path_geometry_metrics(pos, pos, pos, 0.5, edge_index, rotatable)
    assert metrics["bond_length_rel_error_max"] < 1.0e-6
    assert metrics["angle_error_deg_max"] < 1.0e-5
    assert metrics["torsion_error_deg_max"] < 1.0e-5


def test_wrapped_angle_delta_uses_shortest_periodic_difference():
    first = torch.deg2rad(torch.tensor([-179.0]))
    second = torch.deg2rad(torch.tensor([179.0]))
    delta = torch.rad2deg(wrapped_angle_delta(first, second))
    torch.testing.assert_close(delta, torch.tensor([2.0]), atol=1.0e-4, rtol=0.0)
