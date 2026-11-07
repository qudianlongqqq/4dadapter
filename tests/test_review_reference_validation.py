import pytest
from rdkit import Chem
from etflow.ecir.review_reference_validation import validate_reference_ensemble


def molecule():
    m=Chem.MolFromSmiles('CCCO');c=Chem.Conformer(m.GetNumAtoms())
    for i in range(m.GetNumAtoms()):c.SetAtomPosition(i,(float(i),0.,0.))
    m.AddConformer(c);return m


def test_detects_element_preserving_wrong_atom_order():
    a=molecule();b=Chem.RenumberAtoms(a,[1,0,2,3])
    assert [x.GetAtomicNum() for x in a.GetAtoms()]==[x.GetAtomicNum() for x in b.GetAtoms()]
    with pytest.raises(ValueError,match='topology'):
        validate_reference_ensemble([a,b],source_locator='raw/test.pkl',declared_kind='raw_geom')


def test_requires_correct_geometry_provenance():
    m=molecule()
    with pytest.raises(ValueError,match='cannot be labelled'):
        validate_reference_ensemble([m],source_locator='DRUGS/standardized_pickles/001.pickle',declared_kind='raw_geom')
    assert validate_reference_ensemble([m],source_locator='DRUGS/standardized_pickles/001.pickle',
                                      declared_kind='torsion_matched_rdkit')['reference_count']==1
