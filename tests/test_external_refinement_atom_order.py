from etflow.ecir.external_refinement_baselines import mol_from_frozen_record, validate_atom_identity
from tests.external_refinement_test_utils import synthetic_record


def test_atom_order_is_cache_bound():
    record, source = synthetic_record()
    assert validate_atom_identity(mol_from_frozen_record(record, source), record)
