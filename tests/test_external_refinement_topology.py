from rdkit import Chem

from etflow.ecir.external_refinement_baselines import mol_from_frozen_record, validate_topology_identity
from tests.external_refinement_test_utils import synthetic_record


def test_topology_change_fails_closed():
    record, source = synthetic_record()
    before = mol_from_frozen_record(record, source)
    editable = Chem.RWMol(before)
    bond = before.GetBondWithIdx(0)
    editable.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    assert not validate_topology_identity(before, editable.GetMol(), record)
