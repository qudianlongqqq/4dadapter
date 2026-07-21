from rdkit.Chem import AllChem

from tests.external_refinement_test_utils import synthetic_record


def test_mmff_parameter_coverage_is_checked():
    record, _ = synthetic_record("CCO")
    assert isinstance(bool(AllChem.MMFFHasAllMoleculeParams(record["_formal_rdkit_mol"])), bool)
