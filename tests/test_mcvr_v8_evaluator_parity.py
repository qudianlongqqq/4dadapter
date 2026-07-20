import numpy as np
import pytest
import torch

from etflow.ecir.v8_validation_cache import compare_metric_reports, tensor_sha256
from scripts.compare_ecir_mvr_v8_cached_evaluations import _paired_all


def test_coordinate_hash_binds_dtype_shape_and_values():
    value = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    assert tensor_sha256(value) == tensor_sha256(value.clone())
    assert tensor_sha256(value) != tensor_sha256(value.double())
    assert tensor_sha256(value) != tensor_sha256(value.reshape(3, 1))


def test_metric_parity_is_stricter_than_cuda_coordinate_repeat_tolerance():
    baseline = {"records": 1, "rejection_reasons": {}, "metrics": {"weighted": 1.0}}
    close = {"records": 1, "rejection_reasons": {}, "metrics": {"weighted": 1.0 + 5e-7}}
    assert compare_metric_reports(baseline, close)["status"] == "PARITY_OK"
    changed = {"records": 1, "rejection_reasons": {}, "metrics": {"weighted": 1.0 + 5e-4}}
    with pytest.raises(RuntimeError, match="weighted"):
        compare_metric_reports(baseline, changed)


def test_shared_paired_bootstrap_indices_match_frozen_per_metric_semantics():
    values = np.arange(2 * 3 * 7, dtype=np.float64).reshape(2, 3, 7)
    actual = _paired_all(values, draws=250, seed=43)
    expected = np.empty_like(actual)
    for baseline in range(values.shape[0]):
        for metric in range(values.shape[1]):
            rng = np.random.default_rng(43)
            for start in range(0, 250, 100):
                count = min(100, 250 - start)
                indices = rng.integers(0, 7, size=(count, 7))
                expected[baseline, metric, start : start + count] = values[
                    baseline, metric, indices
                ].mean(axis=1)
    np.testing.assert_array_equal(actual, expected)
