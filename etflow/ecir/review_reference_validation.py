"""Strict Reference provenance and atom-order validation for new experiments."""
import numpy as np


def ordered_topology_signature(molecule):
    atoms=tuple((a.GetAtomicNum(),a.GetFormalCharge(),a.GetIsAromatic()) for a in molecule.GetAtoms())
    bonds=tuple(sorted((min(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),max(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),
                        str(b.GetBondType()),b.GetIsAromatic()) for b in molecule.GetBonds()))
    return atoms,bonds


def validate_reference_ensemble(molecules, *, source_locator, declared_kind):
    """Require explicit geometry identity; matching atom elements is insufficient.

    Accepted kinds separate raw GEOM coordinates from torsion-matched RDKit
    geometry. This validates fixed ordering, not a symmetry-aware remapping.
    """
    if declared_kind not in {'raw_geom','torsion_matched_rdkit'}:
        raise ValueError('Reference geometry kind must be declared explicitly')
    standardized='standardized_pickles' in str(source_locator).replace('\\','/').split('/')
    if standardized and declared_kind=='raw_geom':
        raise ValueError('standardized_pickles cannot be labelled raw GEOM coordinates')
    molecules=list(molecules)
    if not molecules:raise ValueError('empty Reference ensemble')
    expected=ordered_topology_signature(molecules[0])
    for index,molecule in enumerate(molecules):
        if ordered_topology_signature(molecule)!=expected:
            raise ValueError(f'ordered topology mismatch at Reference {index}')
        if molecule.GetNumConformers()!=1:
            raise ValueError('one conformer per Reference molecule required')
        xyz=np.asarray(molecule.GetConformer().GetPositions())
        if xyz.shape!=(molecule.GetNumAtoms(),3) or not np.isfinite(xyz).all():
            raise ValueError(f'invalid coordinates at Reference {index}')
    return {'reference_kind':declared_kind,'reference_count':len(molecules),
            'atom_order_validation':'atomic attributes and indexed bonds','source_locator':str(source_locator)}
