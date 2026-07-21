import torch

from etflow.ecir.external_refinement_baselines import fallback_to_source
from tests.external_refinement_test_utils import synthetic_record


def test_failure_is_retained_and_bitwise_falls_back():
    record, source = synthetic_record()
    result = fallback_to_source("MMFF94S", "test", source, "unsupported", started=0.0, mol=record["_formal_rdkit_mol"], unsupported=True)
    assert not result.success and result.fallback_to_source and result.unsupported
    assert torch.equal(source, result.refined_coordinates)
