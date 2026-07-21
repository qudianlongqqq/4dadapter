import torch

from etflow.ecir.external_refinement_baselines import fallback_to_source
from tests.external_refinement_test_utils import synthetic_record


def test_raw_fallback_coordinates_are_bitwise_source():
    record, source = synthetic_record()
    result = fallback_to_source("RAW", "test", source, "test", started=0.0, mol=record["_formal_rdkit_mol"])
    assert torch.equal(result.refined_coordinates, source)
