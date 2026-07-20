import torch

from etflow.ecir.v8_error_state import V8ErrorStateHead


def test_learned_confidence_is_bounded_and_neutral_at_initialization():
    head = V8ErrorStateHead(4, confidence_min=0.25, confidence_max=4.0)
    confidence = head(torch.randn(20, 4), torch.zeros(20, dtype=torch.long))[
        "bounded_prior_confidence"
    ]
    assert bool((confidence >= 0.25).all() and (confidence <= 4.0).all())
    assert torch.allclose(confidence, torch.ones_like(confidence), atol=1e-6)
