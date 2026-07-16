#!/usr/bin/env python
"""Build the ECIR heterogeneous conformer-error atlas and offline targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import pandas as pd
import torch

from etflow.commons.global_coupled_4d_sampling import atomic_json_save, atomic_torch_save
from etflow.commons.kabsch_utils import select_best_reference_conformer
from etflow.ecir.geometry import geometry_error_vector, severe_clash
from etflow.ecir.target_building import build_real_error_target


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_sources(args) -> list[dict[str, Any]]:
    if args.sources_config is not None:
        payload = json.loads(args.sources_config.read_text(encoding="utf-8"))
        sources = payload.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("sources_config must contain a nonempty sources list")
        return [dict(source) for source in sources]
    if args.cache_dir is None:
        raise ValueError("--cache_dir or --sources_config is required")
    return [
        {
            "source_type": args.source_type,
            "cache_dir": str(args.cache_dir),
            "checkpoint": args.checkpoint,
            "NFE": args.nfe,
            "solver": args.solver,
            "seed": args.seed,
        }
    ]


def _source_files(source: Mapping[str, Any], split: str) -> list[Path]:
    split_paths = source.get("split_paths") or {}
    configured = split_paths.get(split, source.get("cache_dir"))
    if configured is None:
        return []
    root = Path(str(configured)).expanduser()
    if (root / split).is_dir():
        root = root / split
    return sorted(root.glob("*.pt"))


def _atlas_row(
    record: Mapping[str, Any], source: Mapping[str, Any], split: str,
    target: Mapping[str, Any], x_input: torch.Tensor, source_path: Path
) -> dict[str, Any]:
    references = torch.as_tensor(
        record.get("x_ref_candidates", record.get("x_ref_aligned")), dtype=torch.float32
    )
    _, nearest, nearest_index, rmsds = select_best_reference_conformer(x_input, references)
    order = torch.argsort(rmsds)
    second = int(order[1]) if order.numel() > 1 else int(order[0])
    errors = geometry_error_vector(x_input, nearest, record)
    relaxation = dict(target.get("relaxation") or {})
    soft = dict(target.get("soft_coupling") or {})
    heavy_atoms = int((torch.as_tensor(record["atomic_numbers"]) > 1).sum())
    checkpoint = source.get("checkpoint", record.get("generator_checkpoint", ""))
    seed = source.get("seed", record.get("sample_seed", 0))
    return {
        "split": split,
        "source_path": str(source_path.resolve()),
        "coordinate_key": str(source.get("coordinate_key", "x_init")),
        "molecule_id": str(record.get("source_mol_id", record.get("mol_id"))),
        "sample_id": str(record.get("sample_id", record.get("mol_id"))),
        "source_type": str(source.get("source_type", record.get("generator_name", "unknown"))),
        "checkpoint": str(checkpoint or ""),
        "NFE": int(source.get("NFE", 0) or 0),
        "solver": str(source.get("solver", "unknown")),
        "seed": int(seed or 0),
        "rotatable_bond_count": int(record.get("num_rotatable_bonds", torch.as_tensor(record["rotatable_bond_index"]).size(1))),
        "heavy_atom_count": heavy_atoms,
        "aligned_rmsd": float(rmsds[nearest_index]),
        "nearest_reference_id": str(
            record.get("selected_ref_id")
            or f"{record.get('source_mol_id', record.get('mol_id'))}__ref{nearest_index:04d}"
        ),
        "nearest_reference_cost": float(soft.get("nearest_reference_cost", rmsds[nearest_index])),
        "second_nearest_reference_cost": float(soft.get("second_nearest_reference_cost", rmsds[second])),
        "reference_conformer_count": int(references.size(0) if references.ndim == 3 else 1),
        "bond_error": float(errors[0]),
        "angle_error": float(errors[1]),
        "torsion_circular_error": float(errors[2]),
        "ring_geometry_score": float(errors[3]),
        "clash_score": float(errors[4]),
        "severe_clash": bool(severe_clash(x_input, record["edge_index"])),
        "chirality_status": "preserved" if float(errors[5]) == 0.0 else "inverted",
        "chirality_error": float(errors[5]),
        "MMFF_coverage": relaxation.get("method") == "MMFF94s" and bool(relaxation.get("supported")),
        "force_field_method": str(relaxation.get("method", "unsupported")),
        "MMFF_energy_per_heavy_atom": relaxation.get("energy_per_heavy_atom_before") if relaxation.get("method") == "MMFF94s" else None,
        "UFF_energy_per_heavy_atom": relaxation.get("energy_per_heavy_atom_before") if relaxation.get("method") == "UFF" else None,
        "relaxation_energy_drop": relaxation.get("energy_drop"),
        "relaxation_RMSD": relaxation.get("relaxation_rmsd"),
        "force_field_optimization_success": bool(relaxation.get("optimization_success", False)),
        "relaxation_target_accepted": bool(relaxation.get("accepted", False)),
        "target_source": str(target.get("target_source")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=Path)
    parser.add_argument("--sources_config", type=Path)
    parser.add_argument("--source_type", default="upstream")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--nfe", type=int, default=0)
    parser.add_argument("--solver", default="unknown")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--limits", default="500,100,100")
    parser.add_argument("--output_dir", type=Path, default=Path("data/ecir_error_atlas"))
    parser.add_argument("--report", type=Path, default=Path("docs/ECIR_ERROR_ATLAS_REPORT.md"))
    parser.add_argument("--relaxation_steps", type=int, default=50)
    parser.add_argument("--max_records_per_molecule", type=int, default=2)
    args = parser.parse_args()

    sources = _load_sources(args)
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    limits = [int(value) for value in args.limits.split(",")]
    if len(limits) != len(splits):
        raise ValueError("--limits must have one value per split")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "sources": sources,
        "splits": {},
        "force_field_energy_policy": "paired deltas and energy per heavy atom; MMFF94s and UFF never pooled",
        "test_used_for_selection": False,
    }
    report_rows = []
    started = time.perf_counter()
    for split, limit in zip(splits, limits):
        rows: list[dict[str, Any]] = []
        per_molecule: Counter[str] = Counter()
        target_dir = args.output_dir / "targets" / split
        target_dir.mkdir(parents=True, exist_ok=True)
        available = [(source, _source_files(source, split)) for source in sources]
        available = [(source, files) for source, files in available if files]
        if not available:
            raise ValueError(f"No configured source has records for split {split}")
        for source, source_files in available:
            source_quota = (limit + len(available) - 1) // len(available)
            source_rows = 0
            for path in source_files:
                if len(rows) >= limit or source_rows >= source_quota:
                    break
                record = torch.load(path, map_location="cpu", weights_only=False)
                if not isinstance(record, Mapping):
                    raise TypeError(f"Source record is not a mapping: {path}")
                molecule_id = str(record.get("source_mol_id", record.get("mol_id")))
                source_molecule = f"{source.get('source_type', 'unknown')}::{molecule_id}"
                if per_molecule[source_molecule] >= args.max_records_per_molecule:
                    continue
                coordinate_key = str(source.get("coordinate_key", "x_init"))
                if coordinate_key not in record:
                    raise ValueError(f"Source {path} lacks coordinate_key={coordinate_key}")
                x_input = torch.as_tensor(record[coordinate_key], dtype=torch.float32)
                generator = torch.Generator().manual_seed(args.seed + len(rows))
                target = build_real_error_target(
                    record,
                    coordinates=x_input,
                    generator=generator,
                    relaxation_kwargs={"max_steps": args.relaxation_steps},
                )
                row = _atlas_row(record, source, split, target, x_input, path)
                prefix = str(source.get("source_type", "unknown")).replace("/", "_")
                target_path = target_dir / (
                    path.name if len(available) == 1 else f"{prefix}__{path.name}"
                )
                payload = {
                    "schema_version": "1.0",
                    "sample_id": row["sample_id"],
                    "source_path": str(path.resolve()),
                    "x_target": target.pop("x_target"),
                    "target_metadata": target,
                }
                atomic_torch_save(payload, target_path)
                row["target_cache_path"] = str(target_path.resolve())
                rows.append(row)
                per_molecule[source_molecule] += 1
                source_rows += 1
            if len(rows) >= limit:
                break
        if not rows:
            raise ValueError(f"No atlas rows found for split {split}")
        frame = pd.DataFrame(rows)
        parquet = args.output_dir / f"{split}.parquet"
        frame.to_parquet(parquet, index=False)
        counts = Counter(frame["source_type"])
        metadata["splits"][split] = {
            "records": len(frame),
            "molecules": int(frame["molecule_id"].nunique()),
            "source_counts": dict(counts),
            "parquet": str(parquet.resolve()),
            "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
            "target_directory": str(target_dir.resolve()),
        }
        report_rows.append(
            (split, len(frame), int(frame["molecule_id"].nunique()), float(frame["aligned_rmsd"].mean()), float(frame["clash_score"].mean()), float(frame["MMFF_coverage"].mean()))
        )
    metadata["elapsed_seconds"] = time.perf_counter() - started
    metadata["identity_sha256"] = _canonical_sha(metadata)
    atomic_json_save(metadata, args.output_dir / "metadata.json")
    lines = [
        "# ECIR conformer error atlas",
        "",
        "Energies are reported only as paired deltas or per-heavy-atom values. MMFF94s and UFF are separate populations.",
        "",
        "| split | records | molecules | aligned RMSD | clash score | MMFF coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {split} | {records} | {molecules} | {rmsd:.6f} | {clash:.6f} | {coverage:.3f} |"
        for split, records, molecules, rmsd, clash, coverage in report_rows
    )
    lines.extend(["", f"Atlas identity: `{metadata['identity_sha256']}`", ""])
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
