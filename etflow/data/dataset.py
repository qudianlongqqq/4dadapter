import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from loguru import logger as log
from torch_geometric.data import Dataset
from tqdm import tqdm

from etflow.commons.featurization import (
    MoleculeData,
    MoleculeFeaturizer,
    get_sample_field,
    mol_to_ordered_smiles,
    recover_mol_from_sample,
)
from etflow.commons.io import get_base_data_dir


class EuclideanDataset(Dataset):
    """Returns 3D Graph for different datasets

    Usage
    -----
    ```python
    from etflow.data import EuclideanDataset
    # pass path to processed data_dir
    dataset = EuclideanDataset(
        data_dir="processed",
        split="train",  # "train", "val", or "test"
        partition="drugs",  # "drugs" or "qm9"
    )
    ```
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        split: str = "train",
        partition: str = "drugs",
        skip_unrecoverable_mol: bool = True,
        bad_sample_csv: Path | str | None = None,
    ):
        super().__init__()
        self.mol_feat = MoleculeFeaturizer()

        # Set up paths
        if data_dir is None:
            self.data_dir = Path(get_base_data_dir()) / "processed"
        else:
            self.data_dir = Path(data_dir)

        # Set split and partition
        self.split = split
        self.partition = partition
        self.skip_unrecoverable_mol = skip_unrecoverable_mol
        self.bad_sample_csv = bad_sample_csv

        # Find all data files for the specified partition and split
        self.data_files = list((self.data_dir / partition.lower() / split).glob("*.pt"))

        if len(self.data_files) == 0:
            raise ValueError(
                f"No data files found for partition {partition} and split {split}"
            )

        # Sort files for reproducibility
        self.data_files.sort()
        self._filter_unrecoverable_files()

    @staticmethod
    def _csv_bool(value: object) -> Optional[bool]:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
        return None

    def _resolve_bad_sample_csv(self) -> Optional[Path]:
        if self.bad_sample_csv is not None:
            expanded = Path(
                os.path.expandvars(os.path.expanduser(str(self.bad_sample_csv)))
            )
            candidates = [expanded]
            if not expanded.is_absolute():
                project_root = Path(__file__).resolve().parents[2]
                candidates.extend(
                    [
                        project_root / expanded,
                        self.data_dir / expanded,
                        self.data_dir / self.partition.lower() / expanded,
                    ]
                )
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
            raise FileNotFoundError(
                "Configured bad_sample_csv does not exist. Checked: "
                f"{[str(path) for path in candidates]}"
            )

        for filename in ("unrecoverable_mol_samples.csv", "none_mol_samples.csv"):
            for candidate in (
                self.data_dir / filename,
                self.data_dir / self.partition.lower() / filename,
            ):
                if candidate.is_file():
                    return candidate
        return None

    def _bad_paths_from_csv(self, csv_path: Path) -> Tuple[Set[Path], Set[str]]:
        exact_paths: Set[Path] = set()
        filenames: Set[str] = set()
        with csv_path.open(newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                raise ValueError(f"Bad-sample CSV has no header: {csv_path}")
            path_column = next(
                (
                    column
                    for column in ("file_path", "path", "data_path", "sample_path")
                    if column in reader.fieldnames
                ),
                None,
            )
            if path_column is None:
                raise ValueError(
                    f"Bad-sample CSV {csv_path} needs one of the path columns "
                    "file_path/path/data_path/sample_path."
                )

            for row in reader:
                row_split = str(row.get("split", "")).strip().lower()
                row_partition = str(row.get("partition", "")).strip().lower()
                if row_split and row_split != self.split.lower():
                    continue
                if row_partition and row_partition != self.partition.lower():
                    continue

                # Scanner CSVs contain recoverable mol=None rows too. Only rows
                # explicitly marked unrecoverable should be removed.
                recoverable = self._csv_bool(row.get("recoverable"))
                if recoverable is True:
                    continue

                raw_path = str(row.get(path_column, "")).strip()
                if not raw_path:
                    continue
                expanded = Path(os.path.expandvars(os.path.expanduser(raw_path)))
                if not expanded.is_absolute():
                    expanded = (csv_path.parent / expanded).resolve()
                else:
                    expanded = expanded.resolve()
                exact_paths.add(expanded)
                filenames.add(Path(raw_path).name)
        return exact_paths, filenames

    def _filter_from_csv(self, csv_path: Path) -> Tuple[List[Path], Dict[Path, str]]:
        exact_paths, filenames = self._bad_paths_from_csv(csv_path)
        bad_files = []
        reasons: Dict[Path, str] = {}
        for file_path in self.data_files:
            if file_path.resolve() in exact_paths or file_path.name in filenames:
                bad_files.append(file_path)
                reasons[file_path] = f"listed in {csv_path}"
        return bad_files, reasons

    def _precheck_unrecoverable(self) -> Tuple[List[Path], Dict[Path, str]]:
        bad_files = []
        reasons: Dict[Path, str] = {}
        disable_progress = os.environ.get("LOCAL_RANK", "0") not in {"", "0"}
        for file_path in tqdm(
            self.data_files,
            desc=f"Checking molecules ({self.partition}/{self.split})",
            disable=disable_progress,
        ):
            try:
                sample = torch.load(file_path, map_location="cpu", weights_only=False)
                atomic_numbers = get_sample_field(sample, "atomic_numbers")
                if atomic_numbers is None:
                    raise ValueError("sample has no 'atomic_numbers' tensor")
                recover_mol_from_sample(
                    sample,
                    expected_atomic_numbers=torch.as_tensor(atomic_numbers).view(-1),
                )
            except Exception as exc:
                bad_files.append(file_path)
                reasons[file_path] = str(exc)
        return bad_files, reasons

    def _filter_unrecoverable_files(self) -> None:
        original_count = len(self.data_files)
        bad_files: List[Path] = []
        reasons: Dict[Path, str] = {}
        source = "strict"

        if self.skip_unrecoverable_mol:
            csv_path = self._resolve_bad_sample_csv()
            if csv_path is not None:
                source = f"csv:{csv_path}"
                bad_files, reasons = self._filter_from_csv(csv_path)
            else:
                source = "precheck"
                bad_files, reasons = self._precheck_unrecoverable()

            bad_set = set(bad_files)
            self.data_files = [path for path in self.data_files if path not in bad_set]

        filtered_count = original_count - len(self.data_files)
        log.info(
            "Molecule filtering: "
            f"split={self.split}, original_samples={original_count}, "
            f"filtered_samples={filtered_count}, remaining_samples={len(self.data_files)}, "
            f"source={source}"
        )
        for file_path in bad_files[:5]:
            log.warning(f"Filtered unrecoverable sample {file_path}: {reasons[file_path]}")
        if len(bad_files) > 5:
            log.warning(f"... and {len(bad_files) - 5} more filtered samples")
        if not self.data_files:
            raise ValueError(
                f"No usable samples remain for partition={self.partition}, split={self.split}."
            )

    def len(self):
        return len(self.data_files)

    def get(self, idx):
        data_path = self.data_files[idx]
        try:
            data = torch.load(data_path, weights_only=False)
            pos_confs = get_sample_field(data, "pos")
            atomic_numbers = get_sample_field(data, "atomic_numbers")
            if pos_confs is None:
                raise ValueError("sample has no 'pos' tensor")
            if atomic_numbers is None:
                raise ValueError("sample has no 'atomic_numbers' tensor")
            atomic_numbers = torch.as_tensor(atomic_numbers).view(-1)

            recovery = recover_mol_from_sample(
                data,
                expected_atomic_numbers=atomic_numbers,
            )
            topology_mol = recovery.mol
            smiles = mol_to_ordered_smiles(topology_mol)
            self.mol_feat.cache_recovered_mol(smiles, topology_mol)

            pos_confs = torch.as_tensor(pos_confs)
            if pos_confs.ndim == 2:
                pos = pos_confs
            elif pos_confs.ndim == 3 and pos_confs.size(0) > 0:
                conf_idx = np.random.randint(0, pos_confs.shape[0])
                pos = pos_confs[conf_idx]
            else:
                raise ValueError(
                    "sample positions must have shape [N, 3] or [C, N, 3], "
                    f"got {tuple(pos_confs.shape)}"
                )
            expected_pos_shape = (int(atomic_numbers.numel()), 3)
            if tuple(pos.shape) != expected_pos_shape:
                raise ValueError(
                    f"selected conformer must have shape {expected_pos_shape}, "
                    f"got {tuple(pos.shape)}"
                )

            node_attr = self.mol_feat.get_atom_features(smiles)
            chiral_index, chiral_nbr_index, chiral_tag = (
                self.mol_feat.get_chiral_centers(smiles)
            )
            edge_index, edge_attr = self.mol_feat.get_edge_index(smiles, False)
            rotatable_bond_index, atom_bond_influence_index = (
                self.mol_feat.get_rotatable_bond_features(smiles)
            )
            mol = self.mol_feat.get_mol_with_conformer_from_mol(topology_mol, pos)

            return MoleculeData(
                num_nodes=int(atomic_numbers.size(0)),
                pos=pos,
                atomic_numbers=atomic_numbers,
                smiles=smiles,
                edge_index=edge_index,
                chiral_index=chiral_index,
                chiral_nbr_index=chiral_nbr_index,
                chiral_tag=chiral_tag,
                mol=mol,
                node_attr=node_attr,
                edge_attr=edge_attr,
                rotatable_bond_index=rotatable_bond_index,
                atom_bond_influence_index=atom_bond_influence_index,
            )
        except Exception as exc:
            raise ValueError(
                "Failed to load or featurize processed molecule: "
                f"sample_idx={idx}, split={self.split!r}, "
                f"file_path={str(data_path)!r}. Cause: {exc}"
            ) from exc
