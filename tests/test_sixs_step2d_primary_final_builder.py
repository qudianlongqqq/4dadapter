from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import pickle

import numpy as np
from rdkit import Chem


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_identity_removes_maps_and_explicit_h_but_preserves_stereo():
    source = load_script("build_sixs_step2d_source_universe.py")
    mapped = Chem.MolFromSmiles("[H][C@]([F])([CH3:7])[OH:9]")
    assert mapped is not None
    identity, normalized = source.canonical_identity(mapped)
    assert ":7" not in identity and ":9" not in identity
    assert "[H]" not in identity
    assert "@" in identity
    assert all(atom.GetAtomMapNum() == 0 for atom in normalized.GetAtoms())


def test_frozen_selection_key_is_domain_separated_and_deterministic():
    selector = load_script("freeze_sixs_prospective_final_cohort.py")
    values = ["CCO", "CCN", "C[C@H](O)F"]
    observed = sorted(values, key=lambda item: (selector.selection_key(item), item))
    assert observed == sorted(
        values, key=lambda item: (selector.selection_key(item), item)
    )
    assert selector.selection_key("CCO") != selector.identity_sha256("CCO")


def test_step2d_builder_does_not_treat_native_test_as_eligibility_gate():
    builder = load_script("build_sixs_step2d_primary_final_cohort.py")
    selector = load_script("freeze_sixs_prospective_final_cohort.py")
    molecule_id = "CCO"
    row = {
        "molecule_id": molecule_id,
        "molecule_identity_sha256": selector.identity_sha256(molecule_id),
        "identity_definition": selector.IDENTITY_DEFINITION,
        "source_dataset_identity": "fixture::CCO",
        "reference_identity": "fixture::CCO::reference",
        "history_status": "CONSERVATIVE_EXCLUSION_NATIVE_TRAIN_VAL_OR_UNKNOWN",
        "native_split": "unassigned",
        "reference_available": True,
        "etflow_compatible": True,
        "mmff94s_compatible": True,
        "xtb_compatible": True,
        "valid_graph": True,
        "single_component": True,
        "heavy_atom_count": 3,
        "topology_compatible": True,
        "atomic_numbers": [6, 6, 8],
        "formal_charge": 0,
        "rotatable_bond_count": 0,
        "ring_count": 0,
    }

    eligible, rejected, breakdown = builder.apply_frozen_eligibility(
        selector, [row], set()
    )

    assert eligible == [row]
    assert not rejected
    assert breakdown["N_AFTER_NATIVE_SPLIT_FILTER"] == 1
    assert breakdown["FINAL_ELIGIBLE_N"] == 1


def test_windows_safe_raw_member_mapping_is_lossless():
    extractor = load_script("extract_sixs_step2d_full_archive.py")
    relative, stem = extractor.destination_for(
        r"DRUGS/drugs/BrC(=C\c1ccccc1)_C=N_N.pickle", 17
    )
    assert relative.as_posix() == "DRUGS/drugs_windows_safe/000017.pickle"
    assert stem == r"BrC(=C\c1ccccc1)_C=N_N"


def test_synthetic_source_index_is_resumable_and_fail_closed(tmp_path: Path):
    source = load_script("build_sixs_step2d_source_universe.py")
    standardized = tmp_path / "standardized_pickles"
    standardized.mkdir()
    molecules = {
        "A": {"conformers": [{"rd_mol": Chem.MolFromSmiles("CCO")}]},
        "B": {"conformers": [{"rd_mol": Chem.MolFromSmiles("CCN")}]},
        "Z": {"conformers": [{"rd_mol": Chem.MolFromSmiles("C[C@H](O)F")}]},
    }
    with (standardized / "000.pickle").open("wb") as stream:
        pickle.dump(molecules, stream)
    split = np.empty(3, dtype=object)
    split[0] = np.array([0], dtype=np.int64)
    split[1] = np.array([1], dtype=np.int64)
    split[2] = np.array([2], dtype=np.int64)
    np.save(tmp_path / "split.npy", split, allow_pickle=True)
    raw_map = tmp_path / "raw_map.jsonl"
    raw_map.write_text(
        "".join(
            json.dumps({"source_stem": value}) + "\n" for value in ("A", "B", "Z")
        ),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        standardized_dir=standardized,
        split=tmp_path / "split.npy",
        raw_filename_index=None,
        raw_member_map=raw_map,
        raw_index_status="complete",
        cache_dir=tmp_path / "cache",
        csv_output=tmp_path / "universe.csv",
        jsonl_output=tmp_path / "universe.jsonl",
        failure_output=tmp_path / "failures.jsonl",
        duplicate_output=tmp_path / "duplicates.jsonl",
        summary_output=tmp_path / "summary.json",
    )
    first = source.build(args)
    second = source.build(args)
    rows = [
        json.loads(line)
        for line in args.jsonl_output.read_text(encoding="utf-8").splitlines()
    ]
    assert first["source_universe_n_molecules"] == 3
    assert first["proven_native_test"] == 1
    assert second["cache_hits"] == 1
    assert {row["native_split"] for row in rows} == {"train", "val", "test"}
    assert all(row["native_split_linkage"] == "EXACT_COMPLETE_RAW_INDEX" for row in rows)
