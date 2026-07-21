from rdkit.Chem import AllChem

from etflow.ecir.external_refinement_baselines import refine_with_mmff94s
from tests.external_refinement_test_utils import config, synthetic_record


def test_mmff94s_variant_is_explicit(monkeypatch):
    record, source = synthetic_record()
    variants = []
    original = AllChem.MMFFGetMoleculeProperties
    def wrapped(mol, *args, **kwargs):
        variants.append(kwargs.get("mmffVariant"))
        return original(mol, *args, **kwargs)
    monkeypatch.setattr(AllChem, "MMFFGetMoleculeProperties", wrapped)
    refine_with_mmff94s(record, source, config()["mmff94s"])
    assert variants and set(variants) == {"MMFF94s"}
