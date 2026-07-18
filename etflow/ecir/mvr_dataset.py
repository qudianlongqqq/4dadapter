"""Source/severity-balanced mixed dataset for MCVR."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from collections import OrderedDict, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info
from torch_geometric.data import Data

from .audit import field
from .bac_constraints import canonical_constraint_fields
from .geometry import angle_triplets, torsion_quads, unique_bonds
from .structured_corruption import corrupt_conformer


SAMPLE_TYPES = ("real_error", "synthetic_error", "clean_identity")
SYNTHETIC_MODES = (
    "bond_only", "angle_only", "clash_only", "ring_only", "torsion_only", "mixed"
)
DEFAULT_RATIOS = {
    "real_error": 0.45,
    "synthetic_error": 0.30,
    "clean_identity": 0.25,
}
DEFAULT_SYNTHETIC_RATIOS = {
    "bond_only": 0.20,
    "angle_only": 0.20,
    "clash_only": 0.15,
    "ring_only": 0.15,
    "torsion_only": 0.15,
    "mixed": 0.15,
}
SEVERITY_SCORE = {
    "normal": 0.0, "mild": 0.25, "medium": 0.5, "severe": 0.75,
    "out_of_domain_extreme": 1.0,
}
MODE_INDEX = {name: index for index, name in enumerate((
    "bond", "angle", "ring", "clash", "torsion", "clean"
))}
FORMAL_ADAPTER_CACHE_SCHEMA = "ecir-mvr-formal-adapter-worker-lru-v1"
FORMAL_ADAPTER_FEATURE_VERSION = "formal-rdkit-static-v1"
CANONICAL_BATCH_SCHEMA_VERSION = "ecir-mvr-canonical-batch-v1"
STATIC_TOPOLOGY_SCHEMA_VERSION = "ecir-mvr-static-topology-cache-v1"
STATIC_TOPOLOGY_FEATURE_VERSION = "molecular-static-topology-v1"
RUNTIME_STATISTICS_SCHEMA_VERSION = "ecir-mvr-runtime-cache-statistics-v1"
RUNTIME_STATISTICS_FEATURE_VERSION = "formal-static-runtime-cache-v1"


def runtime_statistics_identity(
    formal_adapter_lru_size: int, precompute_training_topology: bool
) -> str:
    payload = {
        "schema_version": RUNTIME_STATISTICS_SCHEMA_VERSION,
        "feature_version": RUNTIME_STATISTICS_FEATURE_VERSION,
        "formal_adapter_lru_size": int(formal_adapter_lru_size),
        "precompute_training_topology": bool(precompute_training_topology),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class RuntimeCacheStatistics:
    """Low-overhead per-worker counters kept outside the canonical batch."""

    def __init__(self, max_workers: int, identity_sha256: str) -> None:
        self.max_workers = max(1, int(max_workers))
        self.identity_sha256 = str(identity_sha256)
        self._seen = multiprocessing.Array("q", self.max_workers, lock=False)
        self._pids = multiprocessing.Array("q", self.max_workers, lock=False)
        self._hits = multiprocessing.Array("q", self.max_workers, lock=False)
        self._misses = multiprocessing.Array("q", self.max_workers, lock=False)
        self._adapter_builds = multiprocessing.Array(
            "q", self.max_workers, lock=False
        )
        self._topology_builds = multiprocessing.Array(
            "q", self.max_workers, lock=False
        )

    def publish(
        self,
        *,
        worker_id: int,
        pid: int,
        identity_sha256: str,
        cache_hits: int,
        cache_misses: int,
        rdkit_adapter_build_count: int,
        topology_build_count: int,
    ) -> None:
        index = int(worker_id)
        if not 0 <= index < self.max_workers:
            raise RuntimeError("runtime cache worker id exceeds shared statistics")
        if str(identity_sha256) != self.identity_sha256:
            raise RuntimeError("runtime cache statistics identity changed")
        self._pids[index] = int(pid)
        self._hits[index] = int(cache_hits)
        self._misses[index] = int(cache_misses)
        self._adapter_builds[index] = int(rdkit_adapter_build_count)
        self._topology_builds[index] = int(topology_build_count)
        self._seen[index] = 1

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "schema_version": RUNTIME_STATISTICS_SCHEMA_VERSION,
                "feature_version": RUNTIME_STATISTICS_FEATURE_VERSION,
                "identity_sha256": self.identity_sha256,
                "worker_id": worker_id,
                "pid": int(self._pids[worker_id]),
                "cache_hits": int(self._hits[worker_id]),
                "cache_misses": int(self._misses[worker_id]),
                "rdkit_adapter_build_count": int(
                    self._adapter_builds[worker_id]
                ),
                "topology_build_count": int(
                    self._topology_builds[worker_id]
                ),
            }
            for worker_id in range(self.max_workers)
            if self._seen[worker_id]
        ]


@lru_cache(maxsize=1)
def _formal_adapter_sha256() -> str:
    from . import formal_rdkit_adapter

    return hashlib.sha256(Path(formal_rdkit_adapter.__file__).read_bytes()).hexdigest()


def formal_adapter_cache_key(row: Any, record: Mapping[str, Any]) -> str:
    atomic_numbers = torch.as_tensor(
        record.get("atomic_numbers", []), dtype=torch.long
    ).view(-1)
    payload = {
        "schema_version": FORMAL_ADAPTER_CACHE_SCHEMA,
        "adapter_sha256": _formal_adapter_sha256(),
        "sample_id": str(getattr(row, "sample_id", record.get("sample_id", ""))),
        "source_record_id": str(record.get("source_record_id", "")),
        "source_identity_sha256": str(
            getattr(row, "coordinate_sha256", "")
            or record.get("source_file_sha256", "")
        ),
        "ordered_smiles": str(record.get("ordered_smiles", record.get("smiles", ""))),
        "atomic_numbers": atomic_numbers.tolist(),
        "topology_signature": str(record.get("topology_signature", "")),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class FormalAdapterLRU:
    """Per-process cache of identity-bound runtime RDKit adapter fields."""

    def __init__(self, max_size: int) -> None:
        self.max_size = int(max_size)
        if self.max_size <= 0:
            raise ValueError("formal adapter LRU size must be positive")
        self._values: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def adapt(self, row: Any, record: Mapping[str, Any]) -> dict[str, Any]:
        from .formal_rdkit_adapter import adapt_formal_cache_record

        key = formal_adapter_cache_key(row, record)
        cached = self._values.get(key)
        if cached is not None:
            if (
                cached.get("schema_version") != FORMAL_ADAPTER_CACHE_SCHEMA
                or cached.get("feature_version")
                != FORMAL_ADAPTER_FEATURE_VERSION
                or cached.get("identity_sha256") != key
            ):
                raise RuntimeError("formal adapter LRU entry identity changed")
            self.hits += 1
            self._values.move_to_end(key)
            result = dict(record)
            result.update(cached["runtime_fields"])
            return result
        self.misses += 1
        adapted = adapt_formal_cache_record(record)
        runtime = {
            name: value
            for name, value in adapted.items()
            if str(name).startswith("_formal_")
        }
        self._values[key] = {
            "schema_version": FORMAL_ADAPTER_CACHE_SCHEMA,
            "feature_version": FORMAL_ADAPTER_FEATURE_VERSION,
            "identity_sha256": key,
            "runtime_fields": runtime,
        }
        self._values.move_to_end(key)
        while len(self._values) > self.max_size:
            self._values.popitem(last=False)
        return adapted


def canonical_static_topology_fields(
    record: Mapping[str, Any], edge_index: torch.Tensor, num_atoms: int
) -> dict[str, Any]:
    rotatable = torch.as_tensor(
        record.get("rotatable_bond_index", torch.empty(2, 0)),
        dtype=torch.long,
    )
    bonds = unique_bonds(edge_index)
    angles = angle_triplets(edge_index, num_atoms)
    torsions = torsion_quads(edge_index, rotatable, num_atoms)
    ring_flags = torch.as_tensor(
        record.get("bond_is_in_ring", torch.zeros(edge_index.size(1))),
        dtype=torch.bool,
    )
    ring_bonds = edge_index[:, (edge_index[0] < edge_index[1]) & ring_flags]
    identity_payload = {
        "schema_version": STATIC_TOPOLOGY_SCHEMA_VERSION,
        "feature_version": STATIC_TOPOLOGY_FEATURE_VERSION,
        "atomic_numbers": torch.as_tensor(
            record.get("atomic_numbers", []), dtype=torch.long
        ).view(-1).tolist(),
        "edge_index": edge_index.tolist(),
        "bond_index": bonds.tolist(),
        "angle_index": angles.tolist(),
        "torsion_index": torsions.tolist(),
        "ring_bond_index": ring_bonds.tolist(),
        "rotatable_bond_index": rotatable.tolist(),
    }
    encoded = json.dumps(
        identity_payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "canonical_bond_index": bonds,
        "canonical_angle_index": angles.t().contiguous(),
        "canonical_torsion_index": torsions.t().contiguous(),
        "canonical_ring_bond_index": ring_bonds,
        "canonical_static_topology_schema_version": (
            STATIC_TOPOLOGY_SCHEMA_VERSION
        ),
        "canonical_static_topology_feature_version": (
            STATIC_TOPOLOGY_FEATURE_VERSION
        ),
        "canonical_static_topology_identity_sha256": hashlib.sha256(
            encoded
        ).hexdigest(),
    }


def _validate_ratios(values: Mapping[str, float], expected: set[str]) -> dict[str, float]:
    values = {str(key): float(value) for key, value in values.items()}
    if set(values) != expected or any(value < 0 for value in values.values()):
        raise ValueError(f"ratios must contain nonnegative {sorted(expected)}")
    if abs(sum(values.values()) - 1.0) > 1.0e-8:
        raise ValueError("ratios must sum to one")
    return values


def balanced_sample_plan(
    records: pd.DataFrame,
    length: int,
    *,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    synthetic_ratios: Mapping[str, float] = DEFAULT_SYNTHETIC_RATIOS,
    seed: int = 42,
    out_of_domain_extreme_ratio: float = 0.0,
) -> list[dict[str, Any]]:
    """Freeze an epoch plan with exact type ratios and round-robin sources."""

    ratios = _validate_ratios(ratios, set(SAMPLE_TYPES))
    synthetic_ratios = _validate_ratios(synthetic_ratios, set(SYNTHETIC_MODES))
    if synthetic_ratios["mixed"] > 0.30 + 1.0e-12:
        raise ValueError("mixed may not exceed 30% of synthetic_error")
    if not 0.0 <= out_of_domain_extreme_ratio <= 0.05:
        raise ValueError("out_of_domain_extreme ratio must be in [0,0.05]")
    if length < 1:
        raise ValueError("length must be positive")
    rng = torch.Generator().manual_seed(int(seed))
    counts = {name: int(round(length * ratio)) for name, ratio in ratios.items()}
    counts["real_error"] += length - sum(counts.values())
    types = [name for name in SAMPLE_TYPES for _ in range(counts[name])]
    order = torch.randperm(len(types), generator=rng).tolist()
    types = [types[index] for index in order]

    eligible = records.copy()
    if out_of_domain_extreme_ratio == 0.0:
        eligible = eligible[eligible.source_severity != "out_of_domain_extreme"]
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in eligible.iterrows():
        groups[str(row.generator_name)][str(row.source_severity)].append(int(index))
    if not groups:
        raise ValueError("no eligible real-error records")
    for by_severity in groups.values():
        for severity, indices in by_severity.items():
            order = torch.randperm(len(indices), generator=rng).tolist()
            by_severity[severity] = [indices[index] for index in order]
    source_keys = sorted(groups)
    severity_cursor = defaultdict(int)
    group_offsets = defaultdict(int)
    real_cursor = 0
    synthetic_cursor = 0
    synthetic_counts = {
        name: int(round(counts["synthetic_error"] * ratio))
        for name, ratio in synthetic_ratios.items()
    }
    synthetic_counts["bond_only"] += counts["synthetic_error"] - sum(synthetic_counts.values())
    synthetic_modes = [name for name in SYNTHETIC_MODES for _ in range(synthetic_counts[name])]
    if synthetic_modes:
        synth_order = torch.randperm(len(synthetic_modes), generator=rng).tolist()
        synthetic_modes = [synthetic_modes[index] for index in synth_order]
    all_indices = list(eligible.index)
    plan = []
    for position, sample_type in enumerate(types):
        if sample_type == "real_error":
            source = source_keys[real_cursor % len(source_keys)]
            severities = sorted(groups[source])
            severity = severities[severity_cursor[source] % len(severities)]
            severity_cursor[source] += 1
            key = (source, severity)
            choices = groups[source][severity]
            row_index = choices[group_offsets[key] % len(choices)]
            group_offsets[key] += 1
            real_cursor += 1
            mode = "real"
        else:
            row_index = all_indices[position % len(all_indices)]
            mode = synthetic_modes[synthetic_cursor] if sample_type == "synthetic_error" else "clean"
            synthetic_cursor += int(sample_type == "synthetic_error")
        row = records.loc[row_index]
        plan.append({
            "row_index": int(row_index),
            "sample_type": sample_type,
            "corruption_type": mode,
            "source": str(row.generator_name) if sample_type == "real_error" else sample_type,
            "severity": str(row.source_severity) if sample_type == "real_error" else "normal",
        })
    return plan


def _load_record_and_coordinates(
    row,
    *,
    dataset_index: int,
    target_path: Path,
    formal_adapter_cache: FormalAdapterLRU | None = None,
):
    record = torch.load(Path(row.source_path), map_location="cpu", weights_only=False)
    if str(getattr(row, "schema_version", "")) == "ecir-mvr-formal-large-real-sources-v1":
        from .formal_rdkit_adapter import adapt_formal_cache_record

        try:
            record = (
                formal_adapter_cache.adapt(row, record)
                if formal_adapter_cache is not None
                else adapt_formal_cache_record(record)
            )
        except ValueError as error:
            atomic_numbers = torch.as_tensor(
                record.get("atomic_numbers", []), dtype=torch.long
            ).view(-1)
            raise ValueError(
                "MCVR formal dataset item load failed: "
                f"split={row.split}; dataset_index={dataset_index}; "
                f"sample_id={row.sample_id}; source_cache_path={row.source_path}; "
                f"target_path={target_path}; smiles={record.get('smiles', '')}; "
                f"cache_atomic_numbers={atomic_numbers.tolist()}; "
                f"adapter_error={error}"
            ) from error
    if row.coordinate_path is not None and not pd.isna(row.coordinate_path):
        payload = torch.load(Path(row.coordinate_path), map_location="cpu", weights_only=False)
        coordinates = torch.as_tensor(payload[row.coordinate_key], dtype=torch.float32)
    else:
        coordinates = torch.as_tensor(record[row.coordinate_key], dtype=torch.float32)
    return record, coordinates


def deterministic_error_features(validity: Mapping[str, float], record: Any, severity: str) -> torch.Tensor:
    rotatable = float(field(record, "num_rotatable_bonds", 0))
    return torch.tensor([
        validity["bond_outlier_magnitude"],
        validity["angle_outlier_magnitude"],
        validity["ring_bond_outlier_rate"] + validity["ring_planarity_outlier_rate"],
        validity["clash_penetration"],
        validity["severe_clash_rate"],
        1.0 - validity["chirality_preserved"],
        max(0.0, validity["torsion_prior_outlier_score"] - 4.0),
        min(rotatable / 10.0, 1.0),
        min(rotatable / 6.0, 1.0),
        SEVERITY_SCORE.get(str(severity), 0.0),
    ], dtype=torch.float32)


def _active_mask(validity: Mapping[str, float], *, clean: bool = False) -> torch.Tensor:
    mask = torch.zeros(6, dtype=torch.float32)
    mask[MODE_INDEX["bond"]] = float(validity["bond_outlier_rate"] > 0)
    mask[MODE_INDEX["angle"]] = float(validity["angle_outlier_rate"] > 0)
    mask[MODE_INDEX["ring"]] = float(
        validity["ring_bond_outlier_rate"] > 0 or validity["ring_planarity_outlier_rate"] > 0
    )
    mask[MODE_INDEX["clash"]] = float(
        validity["clash_penetration"] > 0 or validity["severe_clash_rate"] > 0
    )
    mask[MODE_INDEX["torsion"]] = float(validity["torsion_prior_outlier_score"] > 4.0)
    mask[MODE_INDEX["clean"]] = float(clean)
    return mask


class MCVRMixedDataset(Dataset):
    """Offline-target dataset; online minimal-target construction is forbidden."""

    def __init__(
        self,
        source_manifest: str | Path,
        target_manifest: str | Path,
        validity,
        *,
        length: int | None = None,
        ratios: Mapping[str, float] = DEFAULT_RATIOS,
        synthetic_ratios: Mapping[str, float] = DEFAULT_SYNTHETIC_RATIOS,
        seed: int = 42,
        out_of_domain_extreme_ratio: float = 0.0,
        formal_adapter_lru_size: int = 0,
        precompute_training_topology: bool = False,
        runtime_statistics: Any | None = None,
        source_cache_root: str | Path | None = None,
        target_cache_root: str | Path | None = None,
        canonical_constraints: bool = False,
        constraint_source_identity_sha256: str | None = None,
    ) -> None:
        self.sources = pd.read_parquet(source_manifest).reset_index(drop=True)
        targets = pd.read_parquet(target_manifest)
        if source_cache_root is not None:
            source_root = Path(source_cache_root)
            self.sources["source_path"] = [
                str(source_root / str(split) / Path(str(value)).name)
                for split, value in zip(
                    self.sources["split"], self.sources["source_path"], strict=True
                )
            ]
        if target_cache_root is not None:
            target_root = Path(target_cache_root)
            targets["target_cache_path"] = [
                str(target_root / str(split) / Path(str(value)).name)
                for split, value in zip(
                    targets["split"], targets["target_cache_path"], strict=True
                )
            ]
        if set(self.sources.split.unique()) != set(targets.split.unique()):
            raise ValueError("source and target split identities differ")
        self.targets = targets.set_index("sample_id")
        missing = set(self.sources.sample_id) - set(self.targets.index)
        if missing:
            raise ValueError(f"missing offline minimal targets: {len(missing)}")
        self.validity = validity
        self.seed = int(seed)
        self.length = int(length or len(self.sources))
        self.ratios = dict(ratios)
        self.synthetic_ratios = dict(synthetic_ratios)
        self.out_of_domain_extreme_ratio = float(out_of_domain_extreme_ratio)
        self.formal_adapter_cache = (
            FormalAdapterLRU(formal_adapter_lru_size)
            if int(formal_adapter_lru_size) > 0
            else None
        )
        self.precompute_training_topology = bool(precompute_training_topology)
        self.canonical_constraints = bool(canonical_constraints)
        self.constraint_source_identity_sha256 = (
            str(constraint_source_identity_sha256)
            if constraint_source_identity_sha256 is not None
            else None
        )
        if self.canonical_constraints and not self.constraint_source_identity_sha256:
            raise ValueError(
                "constraint_source_identity_sha256 is required for canonical constraints"
            )
        self.runtime_statistics = runtime_statistics
        self.adapter_build_count = 0
        self.topology_build_count = 0
        self.uncached_adapter_misses = 0
        self.runtime_statistics_identity_sha256 = runtime_statistics_identity(
            formal_adapter_lru_size, self.precompute_training_topology
        )
        self.epoch = 0
        self.plan = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.plan = balanced_sample_plan(
            self.sources, self.length, ratios=self.ratios,
            synthetic_ratios=self.synthetic_ratios,
            seed=self.seed + self.epoch,
            out_of_domain_extreme_ratio=self.out_of_domain_extreme_ratio,
        )

    def __len__(self):
        return self.length

    def _publish_runtime_statistics(self) -> None:
        if self.runtime_statistics is None:
            return
        worker = get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        hits = self.formal_adapter_cache.hits if self.formal_adapter_cache else 0
        cached_misses = (
            self.formal_adapter_cache.misses if self.formal_adapter_cache else 0
        )
        misses = cached_misses + self.uncached_adapter_misses
        self.runtime_statistics.publish(
            worker_id=worker_id,
            pid=os.getpid(),
            identity_sha256=self.runtime_statistics_identity_sha256,
            cache_hits=hits,
            cache_misses=misses,
            rdkit_adapter_build_count=self.adapter_build_count,
            topology_build_count=self.topology_build_count,
        )

    def __getitem__(self, index: int):
        spec = self.plan[int(index)]
        row = self.sources.loc[spec["row_index"]]
        target_row = self.targets.loc[row.sample_id]
        target_path = Path(target_row.target_cache_path)
        formal_record = (
            str(getattr(row, "schema_version", ""))
            == "ecir-mvr-formal-large-real-sources-v1"
        )
        prior_cache_misses = (
            self.formal_adapter_cache.misses if self.formal_adapter_cache else 0
        )
        record, real_coordinates = _load_record_and_coordinates(
            row,
            dataset_index=int(index),
            target_path=target_path,
            formal_adapter_cache=self.formal_adapter_cache,
        )
        if formal_record:
            if self.formal_adapter_cache is None:
                self.uncached_adapter_misses += 1
                self.adapter_build_count += 1
            else:
                self.adapter_build_count += (
                    self.formal_adapter_cache.misses - prior_cache_misses
                )
        reference = torch.as_tensor(
            record.get("x_ref_aligned", real_coordinates), dtype=torch.float32
        )
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003 + int(index))
        affected = torch.zeros(reference.size(0), dtype=torch.float32)
        metadata_availability = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        if spec["sample_type"] == "real_error":
            target_payload = torch.load(
                target_path,
                map_location="cpu", weights_only=False,
            )
            x_input = real_coordinates
            x_target = torch.as_tensor(target_payload["x_target"], dtype=torch.float32)
            target_status = str(target_payload["target_metadata"]["target_status"])
            validity = target_payload["target_metadata"]["initial_validity"]
            active = _active_mask(validity)
            # Real errors use a deterministic anomaly soft mask. Until atom-local
            # excesses are materialized, active molecules conservatively expose all atoms.
            if bool(active[:4].any()):
                affected.fill_(1.0)
        elif spec["sample_type"] == "synthetic_error":
            mode_map = {
                "bond_only": "bond_length", "angle_only": "bond_angle",
                "clash_only": "clash", "ring_only": "ring",
                "torsion_only": "torsion", "mixed": "mixed",
            }
            requested = mode_map[spec["corruption_type"]]
            if requested == "ring" and not bool(torch.as_tensor(record.get("bond_is_in_ring", [])).any()):
                requested = "bond_length"
            if requested in {"torsion", "bond_angle"} and torch.as_tensor(
                record.get("rotatable_bond_index", torch.empty(2, 0))
            ).size(1) == 0:
                requested = "bond_length"
            x_input, corruption = corrupt_conformer(
                record, coordinates=reference, mode=requested, generator=generator
            )
            x_target = reference.clone()
            affected[corruption["affected_atoms"]] = 1.0
            validity = self.validity.evaluate(x_input, record, baseline_coordinates=x_target)
            active = torch.zeros(6, dtype=torch.float32)
            synthetic_mode = spec["corruption_type"].split("_")[0]
            if synthetic_mode in MODE_INDEX:
                active[MODE_INDEX[synthetic_mode]] = 1.0
            elif spec["corruption_type"] == "mixed":
                active[:5] = 1.0
            target_status = "synthetic_clean_coordinate_target"
        else:
            valid_real = self.validity.evaluate(
                real_coordinates, record, baseline_coordinates=real_coordinates
            )
            real_is_clean = all(valid_real[name] <= 0.0 for name in (
                "bond_outlier_rate", "angle_outlier_rate", "ring_bond_outlier_rate",
                "ring_planarity_outlier_rate", "clash_penetration", "severe_clash_rate",
            ))
            x_input = real_coordinates.clone() if real_is_clean else reference.clone()
            x_target = x_input.clone()
            validity = self.validity.evaluate(x_input, record, baseline_coordinates=x_input)
            active = _active_mask(validity, clean=True)
            target_status = "clean_identity"

        features = deterministic_error_features(validity, record, spec["severity"])
        edge_index = torch.as_tensor(record["edge_index"], dtype=torch.long)
        topology_fields = {}
        if self.precompute_training_topology:
            topology_fields = canonical_static_topology_fields(
                record, edge_index, x_input.size(0)
            )
            self.topology_build_count += 1
        constraint_fields = {}
        if self.canonical_constraints:
            constraint_fields = canonical_constraint_fields(
                self.validity,
                record,
                source_identity_sha256=self.constraint_source_identity_sha256,
            )
        result = Data(
            num_nodes=x_input.size(0),
            node_attr=torch.as_tensor(record["node_attr"], dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=torch.as_tensor(record.get("edge_attr", torch.ones(edge_index.size(1), 1)), dtype=torch.float32),
            bond_is_in_ring=torch.as_tensor(record.get("bond_is_in_ring", torch.zeros(edge_index.size(1))), dtype=torch.bool),
            rotatable_bond_index=torch.as_tensor(record.get("rotatable_bond_index", torch.empty(2, 0)), dtype=torch.long),
            atom_bond_influence_index=torch.as_tensor(record.get("atom_bond_influence_index", torch.empty(2, 0)), dtype=torch.long),
            x_init=x_input, x_input=x_input, x_target=x_target,
            sample_type=spec["sample_type"], source=spec["source"], severity=spec["severity"],
            active_mode_mask=active.view(1, -1), affected_atom_mask=affected,
            deterministic_error_features=features.view(1, -1),
            metadata_availability=metadata_availability.view(1, -1),
            upstream_metadata=torch.tensor([[
                min(float(row.NFE) / 10.0, 1.0),
                float(row.update_scale),
                (float(row.seed) % 10_000.0) / 10_000.0,
                1.0,
            ]], dtype=torch.float32),
            difficulty_target=torch.tensor([
                float(target_status == "identity_fallback")
                + 0.5 * float(spec["severity"] in {"severe", "out_of_domain_extreme"})
            ], dtype=torch.float32).clamp(max=1.0),
            target_status=target_status,
            num_rotatable_bonds=torch.tensor([int(field(record, "num_rotatable_bonds", 0))]),
            sample_id=str(row.sample_id), molecule_id=str(row.molecule_id),
            canonical_batch_schema_version=CANONICAL_BATCH_SCHEMA_VERSION,
            **topology_fields,
            **constraint_fields,
        )
        self._publish_runtime_statistics()
        return result
