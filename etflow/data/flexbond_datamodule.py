"""Lightning data module for prebuilt FlexBond adapter caches."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
from torch_geometric.loader import DataLoader

from .flexbond_optimizer_dataset import FlexBondOptimizerDataset


class FlexBondOptimizerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cache_dir: str | Path,
        batch_size: int = 4,
        num_workers: int = 0,
        max_molecules: Optional[int] = None,
        pin_memory: bool = False,
        validate_cache: bool = False,
    ) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_molecules = max_molecules
        self.pin_memory = pin_memory
        self.validate_cache = validate_cache

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = FlexBondOptimizerDataset(
                self.cache_dir,
                "train",
                self.max_molecules,
                validate=self.validate_cache,
            )
            self.val_dataset = FlexBondOptimizerDataset(
                self.cache_dir,
                "val",
                self.max_molecules,
                validate=self.validate_cache,
            )
        if stage in (None, "test", "predict"):
            self.test_dataset = FlexBondOptimizerDataset(
                self.cache_dir,
                "test",
                self.max_molecules,
                validate=self.validate_cache,
            )

    def _loader(self, dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            exclude_keys=[
                "x_ref_candidates",
                "reference_conformer_ptr",
                "metadata",
            ],
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset)

    def test_dataloader(self):
        return self._loader(self.test_dataset)
