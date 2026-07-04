#!/usr/bin/env python
"""Print required atom-order and Kabsch diagnostics for cached pairs."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from etflow.data.flexbond_optimizer_dataset import validate_cache_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.cache_dir / args.split if (args.cache_dir / args.split).is_dir() else args.cache_dir
    files = sorted(root.glob("*.pt"))
    if not files:
        raise SystemExit(f"No cache files in {root}")
    rng = random.Random(args.seed)
    selected = rng.sample(files, k=min(args.num_samples, len(files)))
    for path in selected:
        record = torch.load(path, map_location="cpu", weights_only=False)
        check = validate_cache_record(record)
        print(
            f"mol_id={record['mol_id']} num_atoms={check['x_init'].size(0)} "
            f"atom_order_ok=True edge_count={torch.as_tensor(record['edge_index']).size(1)} "
            f"num_ref_conformers={check['x_ref_candidates'].size(0)} "
            f"num_rotatable_bonds={torch.as_tensor(record['rotatable_bond_index']).size(1)} "
            f"rmsd_before={check['rmsd_before']:.6f} "
            f"rmsd_after={check['rmsd_after']:.6f} "
            f"selected_reference_conformer_id={check['selected_reference_index']}"
        )
    print(f"PASS: validated {len(selected)} data pairs (atom order, graph, Kabsch).")


if __name__ == "__main__":
    main()
