import torch


def test_fast_selection_is_seeded_and_source_ordered():
    first = sorted(torch.randperm(10_000, generator=torch.Generator().manual_seed(43))[:1000].tolist())
    second = sorted(torch.randperm(10_000, generator=torch.Generator().manual_seed(43))[:1000].tolist())
    assert first == second
    assert first == sorted(first)
    assert len(first) == len(set(first)) == 1000
