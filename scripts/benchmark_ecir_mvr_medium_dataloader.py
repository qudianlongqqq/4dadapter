#!/usr/bin/env python
"""Short, order-preserving DataLoader benchmark for Medium Rescue V2."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch
import yaml
from torch_geometric.loader import DataLoader

from etflow.commons.global_coupled_4d_sampling import atomic_json_save
from etflow.ecir.chemical_validity import ChemicalValidity
from scripts.train_ecir_mvr_run_a import _dataset


def _flatten_sample_ids(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=32)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["training"]["batch_size"] != 8:
        raise RuntimeError("DataLoader benchmark requires frozen batch_size=8")
    validity = ChemicalValidity(config["data"]["validity_statistics"])
    started = time.monotonic()
    rows = []
    reference_order = None
    for workers in (0, 2, 4):
        dataset = _dataset(config, "train", validity)
        dataset.set_epoch(0)
        kwargs = {"num_workers": workers, "pin_memory": True}
        if workers > 0:
            kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
        candidate_started = time.monotonic()
        stable = True
        error = None
        sample_order: list[str] = []
        waits = []
        records = 0
        try:
            loader = DataLoader(dataset, batch_size=8, shuffle=False, **kwargs)
            iterator = iter(loader)
            for index in range(args.batches):
                wait_started = time.monotonic()
                batch = next(iterator)
                waits.append(time.monotonic() - wait_started)
                records += int(batch.num_graphs)
                if index < 8:
                    sample_order.extend(_flatten_sample_ids(batch.sample_id))
            del iterator, loader
        except BaseException as exc:
            stable = False
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - candidate_started
        if reference_order is None and stable:
            reference_order = sample_order
        order_preserved = bool(stable and sample_order == reference_order)
        rows.append({
            "num_workers": workers, "pin_memory": True,
            "persistent_workers": workers > 0, "prefetch_factor": 2 if workers > 0 else None,
            "stable": stable, "sample_order_preserved": order_preserved,
            "records": records, "batches": args.batches,
            "seconds": elapsed, "records_per_second": records / elapsed if elapsed > 0 else 0.0,
            "mean_dataloader_wait_seconds": sum(waits) / len(waits) if waits else None,
            "p95_dataloader_wait_seconds": float(torch.quantile(torch.tensor(waits), 0.95)) if waits else None,
            "error": error,
        })
    total = time.monotonic() - started
    eligible = [row for row in rows if row["stable"] and row["sample_order_preserved"]]
    selected = max(eligible, key=lambda row: row["records_per_second"]) if eligible else None
    status = "PASS" if selected is not None and total <= 600.0 else "FAIL"
    result = {
        "schema_version": "ecir-mvr-medium-dataloader-benchmark-v1",
        "status": status, "batch_size": 8, "candidates": rows,
        "selected": selected, "sample_order_preserved": bool(eligible and all(row["sample_order_preserved"] for row in eligible)),
        "total_seconds": total, "max_allowed_seconds": 600.0,
        "selection_basis": ["throughput", "dataloader_wait", "stability", "sample_order"],
        "validation_metrics_used": False, "test_records_read": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_save(result, args.output)
    print(json.dumps(result, indent=2))
    if status != "PASS":
        raise SystemExit("DATALOADER_BENCHMARK_FAIL")


if __name__ == "__main__":
    main()
