import torch

from etflow.ecir.v8_error_state import V8ErrorStateHead


def test_error_state_heads_receive_gradient():
    head = V8ErrorStateHead(8)
    features = torch.randn(6, 8, requires_grad=True)
    output = head(features, torch.tensor([0, 0, 0, 1, 1, 1]))
    sum(value.sum() for value in output.values()).backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert any(parameter.grad is not None for parameter in head.parameters())
