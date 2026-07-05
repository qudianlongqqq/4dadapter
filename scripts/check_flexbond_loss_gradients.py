#!/usr/bin/env python
"""Check the hybrid loss gradient and detached pseudo-label contracts."""

import importlib.util
from pathlib import Path

import torch


def _load_geometry_module():
    path = Path(__file__).resolve().parents[1] / "etflow/commons/jacobian_4d_velocity.py"
    spec = importlib.util.spec_from_file_location("flexbond_geometry_check", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_checks() -> None:
    geometry = _load_geometry_module()
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 2.0]]
    )
    anchor = torch.tensor([0])
    moving = torch.tensor([1])
    affected_atom = torch.tensor([1, 2, 3])
    affected_bond = torch.tensor([0, 0, 0])
    target_velocity = torch.full_like(pos, 0.2)
    v_cart = torch.zeros_like(pos, requires_grad=True)
    residual = target_velocity - v_cart.detach()
    q_star, valid, _ = geometry.solve_q_targets(
        pos, residual, anchor, moving, affected_atom, affected_bond
    )
    if residual.requires_grad or q_star.requires_grad:
        raise AssertionError("q_b* construction retained a gradient path to v_cart")
    if not valid.any():
        raise AssertionError("synthetic pseudo-label system unexpectedly has no valid bond")

    q_head = torch.nn.Linear(3, 4)
    q_b = q_head(torch.ones(1, 3))
    v_4d, _, _ = geometry.apply_jacobian_4d_correction(
        pos, q_b, anchor, moving, affected_atom, affected_bond
    )
    v_final = v_cart + 0.01 * v_4d
    final_loss = (v_final - target_velocity).square().mean()
    final_loss.backward()
    gradient = sum(
        parameter.grad.abs().sum()
        for parameter in q_head.parameters()
        if parameter.grad is not None
    )
    if not bool(gradient > 0):
        raise AssertionError("L_final did not backpropagate to q_head")


def main() -> None:
    run_checks()
    print("PASS: q_b* is detached and L_final backpropagates to q_head")


if __name__ == "__main__":
    main()
