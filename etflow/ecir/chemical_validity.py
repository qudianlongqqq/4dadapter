"""Reference-free, thresholded chemical-validity diagnostics for MCVR."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .audit import field
from .geometry import (
    angle_triplets,
    bond_angles,
    bond_lengths,
    chirality_mismatch_fraction,
    clash_score,
    dihedral_angles,
    severe_clash,
    torsion_quads,
    unique_bonds,
)
from .rdkit_utils import chiral_center_quads
from .target_building import _record_to_rdkit_mapping


SCHEMA_VERSION = "mcvr-chemical-validity-v1"
DEFAULT_CONFIG = {
    "robust_z_threshold": 4.5,
    "lower_quantile": 0.005,
    "upper_quantile": 0.995,
    "minimum_sample_count": 20,
    "torsion_histogram_bins": 36,
    "severe_clash_distance_angstrom": 0.6,
    "clash_distance_angstrom": 1.0,
    "stereocenter_volume_epsilon": 1.0e-5,
    "score_weights": {
        "bond_outlier_rate": 1.0,
        "bond_outlier_magnitude": 0.25,
        "angle_outlier_rate": 1.0,
        "angle_outlier_magnitude": 0.25,
        "severe_clash_rate": 2.0,
        "clash_penetration": 1.0,
        "ring_bond_outlier_rate": 1.0,
        "ring_planarity_outlier_rate": 1.0,
        "stereocenter_degenerate_rate": 2.0,
        "torsion_prior_outlier_score": 0.0
    }
}


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atom_metadata(record: Any):
    mol, mapping = _record_to_rdkit_mapping(record)
    inverse = {rdkit: cache for cache, rdkit in mapping.items()}
    atom = {}
    for cache_index, rdkit_index in mapping.items():
        value = mol.GetAtomWithIdx(rdkit_index)
        atom[int(cache_index)] = (
            int(value.GetAtomicNum()),
            str(value.GetHybridization()),
            int(value.GetIsAromatic()),
            int(value.IsInRing()),
        )
    bond = {}
    for value in mol.GetBonds():
        left = inverse[value.GetBeginAtomIdx()]
        right = inverse[value.GetEndAtomIdx()]
        bond[tuple(sorted((left, right)))] = (
            str(value.GetBondType()), int(value.GetIsAromatic()), int(value.IsInRing())
        )
    rings = [tuple(inverse[index] for index in ring) for ring in mol.GetRingInfo().AtomRings()]
    return atom, bond, rings


def _bond_keys(left: int, right: int, atom, bond) -> list[tuple[str, str]]:
    pair = tuple(sorted((atom[left], atom[right])))
    zpair = tuple(sorted((atom[left][0], atom[right][0])))
    btype, aromatic, ring = bond[tuple(sorted((left, right)))]
    return [
        ("detailed", repr((pair, btype, aromatic, ring))),
        ("coarse", repr((zpair, btype, aromatic, ring))),
        ("basic", repr((btype, ring))),
        ("global", "all"),
    ]


def _angle_keys(left: int, center: int, right: int, atom) -> list[tuple[str, str]]:
    neighbors = tuple(sorted((atom[left], atom[right])))
    neighbor_z = tuple(sorted((atom[left][0], atom[right][0])))
    center_meta = atom[center]
    return [
        ("detailed", repr((center_meta, neighbors))),
        ("coarse", repr(((center_meta[0], center_meta[1], center_meta[3]), neighbor_z))),
        ("basic", repr((center_meta[0], center_meta[1], center_meta[3]))),
        ("global", "all"),
    ]


def _torsion_keys(a: int, b: int, atom, bond) -> list[tuple[str, str]]:
    endpoints = tuple(sorted((atom[a], atom[b])))
    zpair = tuple(sorted((atom[a][0], atom[b][0])))
    btype, aromatic, ring = bond[tuple(sorted((a, b)))]
    return [
        ("detailed", repr((endpoints, btype, aromatic, ring))),
        ("coarse", repr((zpair, btype, ring))),
        ("global", "all"),
    ]


def _planarity(coordinates: Tensor, indices: Sequence[int]) -> float:
    points = torch.as_tensor(coordinates, dtype=torch.float64)[list(indices)]
    if points.size(0) < 4:
        return 0.0
    centered = points - points.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    return float(singular[-1] / math.sqrt(points.size(0)))


def _robust_stat(values: Sequence[float], config: Mapping[str, Any]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = 1.4826 * mad + 1.0e-8
    z = float(config["robust_z_threshold"])
    q_low = float(np.quantile(array, float(config["lower_quantile"])))
    q_high = float(np.quantile(array, float(config["upper_quantile"])))
    return {
        "count": int(array.size), "median": median, "mad": mad, "robust_scale": scale,
        "quantile_lower": q_low, "quantile_upper": q_high,
        "lower": min(q_low, median - z * scale),
        "upper": max(q_high, median + z * scale),
    }


def build_validity_reference_statistics(
    records: Iterable[tuple[Any, Sequence[Tensor]]],
    *,
    train_split_sha256: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit robust environment statistics from train-reference conformers only."""

    settings = {**DEFAULT_CONFIG, **dict(config or {})}
    collected = {
        "bond": defaultdict(list), "angle": defaultdict(list),
        "ring_planarity": defaultdict(list), "torsion": defaultdict(list),
    }
    molecules = conformers = 0
    for record, references in records:
        atom, bond, rings = _atom_metadata(record)
        edge_index = torch.as_tensor(field(record, "edge_index"), dtype=torch.long)
        bonds = unique_bonds(edge_index)
        angles = angle_triplets(edge_index, len(atom))
        rotatable = torch.as_tensor(
            field(record, "rotatable_bond_index", torch.empty((2, 0))), dtype=torch.long
        )
        torsions = torsion_quads(edge_index, rotatable, len(atom))
        if isinstance(references, (list, tuple)):
            references = torch.stack([
                torch.as_tensor(value, dtype=torch.float32) for value in references
            ])
        else:
            references = torch.as_tensor(references, dtype=torch.float32)
        if references.ndim == 2:
            references = references.unsqueeze(0)
        molecules += 1
        for coordinates in references:
            conformers += 1
            lengths = bond_lengths(coordinates, bonds)
            for index, (left, right) in enumerate(bonds.t().tolist()):
                for level, key in _bond_keys(left, right, atom, bond):
                    collected["bond"][(level, key)].append(float(lengths[index]))
            values = bond_angles(coordinates, angles)
            for index, (left, center, right) in enumerate(angles.tolist()):
                for level, key in _angle_keys(left, center, right, atom):
                    collected["angle"][(level, key)].append(float(values[index]))
            for ring in rings:
                key = repr((len(ring), int(all(atom[index][2] for index in ring))))
                collected["ring_planarity"][("detailed", key)].append(_planarity(coordinates, ring))
                collected["ring_planarity"][("global", "all")].append(_planarity(coordinates, ring))
            values = dihedral_angles(coordinates, torsions)
            for index, (_, a, b, _) in enumerate(torsions.tolist()):
                for level, key in _torsion_keys(a, b, atom, bond):
                    collected["torsion"][(level, key)].append(float(values[index]))
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config": settings,
        "source": {
            "split": "train", "train_split_sha256": train_split_sha256,
            "molecules": molecules, "reference_conformers": conformers,
            "validation_used": False, "test_used": False,
        },
        "units": {"bond": "angstrom", "angle": "radian", "ring_planarity": "angstrom", "torsion": "radian"},
    }
    for mode in ("bond", "angle", "ring_planarity"):
        levels: dict[str, dict[str, Any]] = defaultdict(dict)
        for (level, key), values in collected[mode].items():
            levels[level][key] = _robust_stat(values, settings)
        result[mode] = dict(levels)
    bins = int(settings["torsion_histogram_bins"])
    torsion_levels: dict[str, dict[str, Any]] = defaultdict(dict)
    for (level, key), values in collected["torsion"].items():
        counts, _ = np.histogram(values, bins=bins, range=(-math.pi, math.pi))
        torsion_levels[level][key] = {"count": len(values), "histogram": counts.tolist()}
    result["torsion_prior"] = dict(torsion_levels)
    result["identity_sha256"] = _canonical_sha(result)
    return result


class ChemicalValidity:
    """Evaluate thresholded validity without references or labels."""

    def __init__(self, statistics: Mapping[str, Any] | str | Path) -> None:
        if isinstance(statistics, (str, Path)):
            statistics = json.loads(Path(statistics).read_text(encoding="utf-8"))
        self.statistics = dict(statistics)
        persisted = self.statistics.pop("identity_sha256", None)
        if persisted != _canonical_sha(self.statistics):
            raise ValueError("chemical validity statistics identity mismatch")
        self.statistics["identity_sha256"] = persisted
        self.config = dict(self.statistics["config"])
        self._environment_cache: dict[str, tuple[Any, Any, Any]] = {}
        self._prepared_cache: dict[str, dict[str, Any]] = {}

    def _environment(self, record: Any):
        key = str(field(record, "sample_id", field(record, "mol_id", "")))
        if key not in self._environment_cache:
            self._environment_cache[key] = _atom_metadata(record)
        return self._environment_cache[key]

    def _select(self, mode: str, keys: Sequence[tuple[str, str]]) -> tuple[Mapping[str, Any], str]:
        minimum = int(self.config["minimum_sample_count"])
        levels = self.statistics[mode]
        fallback = None
        for level, key in keys:
            value = levels.get(level, {}).get(key)
            if value is not None:
                fallback = (value, level)
                if int(value["count"]) >= minimum:
                    return value, level
        if fallback is None:
            raise KeyError(f"No {mode} fallback statistic is available")
        return fallback

    def _prepare(self, record: Any) -> dict[str, Any]:
        key = str(field(record, "sample_id", field(record, "mol_id", "")))
        if key in self._prepared_cache:
            return self._prepared_cache[key]
        atom, bond, rings = self._environment(record)
        edge_index = torch.as_tensor(field(record, "edge_index"), dtype=torch.long)
        bonds = unique_bonds(edge_index)
        angles = angle_triplets(edge_index, len(atom))
        rotatable = torch.as_tensor(
            field(record, "rotatable_bond_index", torch.empty((2, 0))), dtype=torch.long
        )
        torsions = torsion_quads(edge_index, rotatable, len(atom))
        bond_stats, ring_mask = [], []
        for left, right in bonds.t().tolist():
            stat, _ = self._select("bond", _bond_keys(left, right, atom, bond))
            bond_stats.append((stat["lower"], stat["upper"], stat["robust_scale"]))
            ring_mask.append(bool(bond[tuple(sorted((left, right)))][2]))
        angle_stats = []
        for left, center, right in angles.tolist():
            stat, _ = self._select("angle", _angle_keys(left, center, right, atom))
            angle_stats.append((stat["lower"], stat["upper"], stat["robust_scale"]))
        planarity_stats = []
        for ring in rings:
            stat, _ = self._select(
                "ring_planarity",
                (("detailed", repr((len(ring), int(all(atom[index][2] for index in ring))))), ("global", "all")),
            )
            planarity_stats.append(stat)
        torsion_histograms = []
        levels = self.statistics["torsion_prior"]
        for _, a, b, _ in torsions.tolist():
            chosen = None
            for level, environment_key in _torsion_keys(a, b, atom, bond):
                value = levels.get(level, {}).get(environment_key)
                if value is not None and int(value["count"]) >= int(self.config["minimum_sample_count"]):
                    chosen = value; break
            torsion_histograms.append(chosen or levels.get("global", {}).get("all"))
        prepared = {
            "edge_index": edge_index, "bonds": bonds, "angles": angles,
            "torsions": torsions, "rings": rings, "centers": chiral_center_quads(record),
            "bond_stats": torch.tensor(bond_stats, dtype=torch.float32).reshape(-1, 3),
            "ring_mask": torch.tensor(ring_mask, dtype=torch.bool),
            "angle_stats": torch.tensor(angle_stats, dtype=torch.float32).reshape(-1, 3),
            "planarity_stats": planarity_stats, "torsion_histograms": torsion_histograms,
        }
        self._prepared_cache[key] = prepared
        return prepared

    @staticmethod
    def _outlier(value: float, stat: Mapping[str, Any]) -> tuple[float, float]:
        distance = max(float(stat["lower"]) - value, value - float(stat["upper"]), 0.0)
        return (float(distance > 0.0), distance / max(float(stat["robust_scale"]), 1.0e-8))

    def evaluate(
        self, coordinates: Tensor, record: Any, *, baseline_coordinates: Tensor | None = None
    ) -> dict[str, float]:
        coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
        prepared = self._prepare(record)
        edge_index = prepared["edge_index"]
        bonds = prepared["bonds"]
        angles = prepared["angles"]
        torsions = prepared["torsions"]
        rings = prepared["rings"]
        lengths = bond_lengths(coordinates, bonds)
        bond_stats = prepared["bond_stats"]
        bond_distance = torch.maximum(bond_stats[:, 0] - lengths, lengths - bond_stats[:, 1]).clamp_min(0.0)
        bond_flags = bond_distance > 0
        bond_magnitudes = bond_distance / bond_stats[:, 2].clamp_min(1.0e-8)
        ring_flags = bond_flags[prepared["ring_mask"]]
        values = bond_angles(coordinates, angles)
        angle_stats = prepared["angle_stats"]
        angle_distance = torch.maximum(angle_stats[:, 0] - values, values - angle_stats[:, 1]).clamp_min(0.0)
        angle_flags = angle_distance > 0
        angle_magnitudes = angle_distance / angle_stats[:, 2].clamp_min(1.0e-8)
        planarity_flags = []
        for ring, stat in zip(rings, prepared["planarity_stats"]):
            flag, _ = self._outlier(_planarity(coordinates, ring), stat)
            planarity_flags.append(flag)
        bins = int(self.config["torsion_histogram_bins"])
        torsion_scores = []
        values = dihedral_angles(coordinates, torsions)
        for index, chosen in enumerate(prepared["torsion_histograms"]):
            if chosen:
                bin_index = min(bins - 1, int((float(values[index]) + math.pi) / (2 * math.pi) * bins))
                counts = chosen["histogram"]
                probability = (counts[bin_index] + 1.0) / (sum(counts) + bins)
                torsion_scores.append(-math.log(probability))
        centers = prepared["centers"]
        degenerate = []
        for center, first, second, third in centers:
            volume = torch.linalg.det(torch.stack([
                coordinates[first] - coordinates[center],
                coordinates[second] - coordinates[center],
                coordinates[third] - coordinates[center],
            ]))
            degenerate.append(float(abs(float(volume)) <= float(self.config["stereocenter_volume_epsilon"])))
        chirality_preserved = 1.0
        if baseline_coordinates is not None:
            chirality_preserved = 1.0 - float(
                chirality_mismatch_fraction(coordinates, torch.as_tensor(baseline_coordinates), record)
            )
        result = {
            "bond_outlier_rate": float(bond_flags.float().mean()) if bond_flags.numel() else 0.0,
            "bond_outlier_magnitude": float(bond_magnitudes.mean()) if bond_magnitudes.numel() else 0.0,
            "angle_outlier_rate": float(angle_flags.float().mean()) if angle_flags.numel() else 0.0,
            "angle_outlier_magnitude": float(angle_magnitudes.mean()) if angle_magnitudes.numel() else 0.0,
            "severe_clash_rate": float(severe_clash(coordinates, edge_index, float(self.config["severe_clash_distance_angstrom"]))),
            "clash_penetration": float(clash_score(coordinates, edge_index, float(self.config["clash_distance_angstrom"]))),
            "ring_bond_outlier_rate": float(ring_flags.float().mean()) if ring_flags.numel() else 0.0,
            "ring_planarity_outlier_rate": float(np.mean(planarity_flags)) if planarity_flags else 0.0,
            "chirality_preserved": chirality_preserved,
            "stereocenter_degenerate_rate": float(np.mean(degenerate)) if degenerate else 0.0,
            "torsion_prior_outlier_score": float(np.mean(torsion_scores)) if torsion_scores else 0.0,
        }
        result["total_thresholded_validity_score"] = sum(
            float(weight) * result[name]
            for name, weight in self.config["score_weights"].items()
        )
        return result
