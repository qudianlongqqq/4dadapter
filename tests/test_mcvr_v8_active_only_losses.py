import torch

from etflow.ecir.v8_losses import active_only_mean


def test_inactive_rows_do_not_dilute_active_angle_or_clash_loss():
    active = active_only_mean(torch.tensor([4.0]), torch.tensor([1.0]))
    padded = active_only_mean(torch.tensor([4.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(active, padded)
