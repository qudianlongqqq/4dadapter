from etflow.ecir.external_refinement_baselines import canonical_sha256, coordinate_sha256
from tests.external_refinement_test_utils import synthetic_record


def test_cache_identity_changes_with_source_coordinates():
    _, source = synthetic_record()
    moved = source.clone(); moved[0, 0] += 0.1
    assert coordinate_sha256(source) != coordinate_sha256(moved)
    assert canonical_sha256({"source": coordinate_sha256(source)}) != canonical_sha256({"source": coordinate_sha256(moved)})
