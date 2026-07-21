from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def synthetic_record(smiles: str = "C[C@H](O)F"):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=43) == 0
    mapping = tuple(range(mol.GetNumAtoms()))
    xyz = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
    chiral = []
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            neighbors = [value.GetIdx() for value in atom.GetNeighbors()]
            if len(neighbors) >= 3:
                chiral.append((atom.GetIdx(), *neighbors[:3]))
    record = {
        "_formal_rdkit_mol": mol,
        "_formal_cache_to_rdkit": mapping,
        "_formal_chiral_center_quads": tuple(chiral),
        "num_atoms": mol.GetNumAtoms(),
        "atomic_numbers": torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()]),
        "topology_signature": "synthetic",
    }
    return record, xyz.clone()


def config():
    return json.loads((ROOT / "configs/ecir_external_refinement_baselines.json").read_text(encoding="utf-8"))
