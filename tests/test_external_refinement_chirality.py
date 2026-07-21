from etflow.ecir.external_refinement_baselines import validate_chirality
from tests.external_refinement_test_utils import synthetic_record


def test_reflection_is_rejected_for_chiral_record():
    record, source = synthetic_record()
    reflected = source.clone(); reflected[:, 0] *= -1
    assert record["_formal_chiral_center_quads"]
    assert not validate_chirality(source, reflected, record)
