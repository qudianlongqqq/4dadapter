"""Window diagnostics and gradient attribution for MCVR V8."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn


def tensor_gradient_norm(parameters: Iterable[nn.Parameter]) -> Tensor:
    values = [
        parameter.grad.square().sum() for parameter in parameters if parameter.grad is not None
    ]
    if not values:
        return torch.zeros(())
    return torch.sqrt(torch.stack(values).sum())


def parameter_group_diagnostics(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    rows = []
    for index, group in enumerate(optimizer.param_groups):
        parameters = list(group["params"])
        rows.append(
            {
                "name": str(group.get("name", f"group_{index}")),
                "learning_rate": float(group["lr"]),
                "parameter_count": sum(parameter.numel() for parameter in parameters),
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in parameters if parameter.requires_grad
                ),
                "gradient_norm": float(tensor_gradient_norm(parameters)),
            }
        )
    return rows


def per_type_gradient_norms(
    losses: Mapping[str, Tensor], parameters: Iterable[nn.Parameter]
) -> dict[str, float]:
    selected = [parameter for parameter in parameters if parameter.requires_grad]
    result = {}
    for name in ("bond_loss", "angle_loss", "clash_loss"):
        value = losses.get(name)
        if value is None or not value.requires_grad:
            result[f"grad_norm_{name.removesuffix('_loss')}"] = 0.0
            continue
        gradients = torch.autograd.grad(value, selected, retain_graph=True, allow_unused=True)
        squares = [gradient.square().sum() for gradient in gradients if gradient is not None]
        result[f"grad_norm_{name.removesuffix('_loss')}"] = float(
            torch.sqrt(torch.stack(squares).sum()) if squares else 0.0
        )
    return result


class V8DiagnosticWindow:
    def __init__(self, *, bond_dominance_warning_ratio: float = 20.0) -> None:
        self.warning_ratio = float(bond_dominance_warning_ratio)
        self.rows: list[Mapping[str, float]] = []

    def add(self, values: Mapping[str, Any]) -> None:
        self.rows.append(
            {
                key: float(value.detach())
                if isinstance(value, Tensor) and value.numel() == 1
                else float(value)
                for key, value in values.items()
                if isinstance(value, (int, float))
                or (isinstance(value, Tensor) and value.numel() == 1)
            }
        )

    def summary(self) -> dict[str, Any]:
        sums: dict[str, float] = defaultdict(float)
        for row in self.rows:
            for key, value in row.items():
                sums[key] += value
        means = {key: value / max(len(self.rows), 1) for key, value in sums.items()}
        bond = means.get("solver_bond_contribution", 0.0)
        angle = means.get("solver_angle_contribution", 0.0)
        warning = bool(angle > 0.0 and bond / angle > self.warning_ratio)
        return {
            "windows": len(self.rows),
            **means,
            "warning": "BOND_DOMINANCE_WARNING" if warning else None,
        }
