import torch

from etflow.networks.torchmd_net.model_dynamics import TorchMDDynamics


class _ScheduleHarness:
    _get_angular_mu_t = TorchMDDynamics._get_angular_mu_t
    _apply_angular_mu_schedule = TorchMDDynamics._apply_angular_mu_schedule

    def __init__(self, schedule, mu=0.7, mu_max=0.3, k=10.0, t0=0.5):
        self.angular_mu_schedule = schedule
        self.angular_mu = mu
        self.angular_mu_max = mu_max
        self.angular_mu_sigmoid_k = k
        self.angular_mu_sigmoid_t0 = t0


def test_constant_schedule_preserves_legacy_mu_without_max_clamp():
    harness = _ScheduleHarness("constant", mu=0.7, mu_max=0.3)
    v_ang = torch.zeros((4, 3), dtype=torch.float64)

    mu_t = harness._get_angular_mu_t(
        t=torch.tensor([0.0, 1.0]),
        batch=torch.tensor([0, 0, 1, 1]),
        v_ang=v_ang,
    )

    assert mu_t.shape == (4, 1)
    assert mu_t.dtype == v_ang.dtype
    torch.testing.assert_close(mu_t, torch.full((4, 1), 0.7, dtype=v_ang.dtype))


def test_quadratic_schedule_maps_graph_times_to_atoms():
    harness = _ScheduleHarness("quadratic", mu_max=0.4)
    v_ang = torch.zeros((5, 3))
    batch = torch.tensor([0, 0, 1, 1, 1])

    mu_t = harness._get_angular_mu_t(
        t=torch.tensor([0.5, 1.0]),
        batch=batch,
        v_ang=v_ang,
    )

    expected = torch.tensor([[0.1], [0.1], [0.4], [0.4], [0.4]])
    torch.testing.assert_close(mu_t, expected)


def test_quadratic_schedule_accepts_atom_column_and_clamps():
    harness = _ScheduleHarness("quadratic", mu_max=0.3)
    v_ang = torch.zeros((3, 3))

    mu_t = harness._get_angular_mu_t(
        t=torch.tensor([[0.0], [0.5], [2.0]]),
        batch=torch.zeros(3, dtype=torch.long),
        v_ang=v_ang,
    )

    expected = torch.tensor([[0.0], [0.075], [0.3]])
    torch.testing.assert_close(mu_t, expected)


def test_sigmoid_schedule_matches_formula_for_atom_times():
    harness = _ScheduleHarness("sigmoid", mu_max=0.3, k=10.0, t0=0.5)
    v_ang = torch.zeros((3, 3))
    t = torch.tensor([0.0, 0.5, 1.0])

    mu_t = harness._get_angular_mu_t(t=t, batch=None, v_ang=v_ang)

    expected = 0.3 * torch.sigmoid(10.0 * (t - 0.5)).unsqueeze(-1)
    torch.testing.assert_close(mu_t, expected)
