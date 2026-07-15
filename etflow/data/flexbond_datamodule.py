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
        persistent_workers: Optional[bool] = None,
        prefetch_factor: Optional[int] = 2,
        validate_cache: bool = False,
    ) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_molecules = max_molecules
        self.pin_memory = pin_memory
        if int(num_workers) < 0:
            raise ValueError("num_workers must be non-negative")
        if prefetch_factor is not None and int(prefetch_factor) < 1:
            raise ValueError("prefetch_factor must be positive when provided")
        requested_persistent = (
            num_workers > 0 if persistent_workers is None else bool(persistent_workers)
        )
        self.persistent_workers = bool(requested_persistent and num_workers > 0)
        self.prefetch_factor = (
            int(prefetch_factor)
            if num_workers > 0 and prefetch_factor is not None
            else None
        )
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
        kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "exclude_keys": [
                "x_ref_candidates",
                "reference_conformer_ptr",
                "metadata",
            ],
        }
        # PyTorch rejects prefetch_factor for a synchronous (worker-free)
        # DataLoader.  Omit the keyword entirely instead of passing a sentinel.
        if self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(
            **kwargs,
        )

    def resolved_loader_config(self) -> dict:
        return {
            "batch_size": int(self.batch_size),
            "num_workers": int(self.num_workers),
            "pin_memory": bool(self.pin_memory),
            "persistent_workers": bool(self.persistent_workers),
            "prefetch_factor": self.prefetch_factor,
        }

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset)

    def test_dataloader(self):
        return self._loader(self.test_dataset)
