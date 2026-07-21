from etflow.ecir.external_refinement_baselines import derive_total_charge, derive_unpaired_electrons
from tests.external_refinement_test_utils import synthetic_record


def test_charge_and_uhf_come_from_frozen_molecule():
    charged, _ = synthetic_record("C[NH3+]")
    radical, _ = synthetic_record("[CH3]")
    assert derive_total_charge(charged["_formal_rdkit_mol"]) == 1
    assert derive_unpaired_electrons(radical["_formal_rdkit_mol"]) > 0
