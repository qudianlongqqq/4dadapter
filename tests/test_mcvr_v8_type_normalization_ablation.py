import torch

from etflow.ecir.v8_constraint_normalization import (
    FrozenResidualScales,
    effective_residual_scales,
)
from etflow.ecir.v8_losses import active_only_mean, active_only_reduce


def test_default_type_normalization_path_is_exactly_unchanged():
    scales = FrozenResidualScales(
        bond=0.13,
        angle=0.27,
        clash=0.41,
        ring=0.59,
        chirality=0.73,
        identity_sha256="frozen",
    )
    effective = effective_residual_scales(scales, type_normalization_enabled=True)
    assert effective is scales
    values = torch.tensor([4.0, 9.0, 16.0])
    active = torch.tensor([1.0, 1.0, 0.0])
    assert torch.equal(
        active_only_mean(values, active),
        active_only_reduce(values, active, normalize_by_active_count=True),
    )


def test_no_type_normalization_uses_unit_scales_and_active_sum():
    scales = FrozenResidualScales(
        bond=0.13,
        angle=0.27,
        clash=0.41,
        ring=0.59,
        chirality=0.73,
        identity_sha256="frozen",
    )
    effective = effective_residual_scales(scales, type_normalization_enabled=False)
    assert effective.bond == effective.angle == effective.clash == 1.0
    assert effective.ring == effective.chirality == 1.0
    values = torch.tensor([4.0, 9.0, 16.0])
    active = torch.tensor([1.0, 1.0, 0.0])
    assert torch.equal(
        active_only_reduce(values, active, normalize_by_active_count=False),
        torch.tensor(13.0),
    )
    assert torch.equal(active_only_mean(values, active), torch.tensor(6.5))
