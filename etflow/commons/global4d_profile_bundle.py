"""Safe, deterministic Global 4D profile bundle construction and verification."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import tarfile
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterable, Mapping

import torch
import yaml

from etflow.commons.global_coupled_4d_sampling import atomic_json_save, file_sha256
from etflow.formal_large import deterministic_molecule_order


BUNDLE_FORMAT_VERSION = "1.0"
BUNDLE_ROOT_NAME = "global4d_profile_bundle"
REQUIRED_STATIC_FILES = (
    "manifest/profile_manifest.json",
    "metadata/bundle_metadata.json",
    "metadata/file_hashes.json",
    "metadata/selection_report.json",
    "metadata/environment_source.json",
    "README.md",
)
SENSITIVE_KEY = re.compile(
    r"(^|[_-])(password|passwd|secret|token|api[_-]?key|access[_-]?key|private[_-]?key|ssh)([_-]|$)",
    re.IGNORECASE,
)
SENSITIVE_TEXT = (
    "BEGIN OPENSSH PRIVATE KEY",
    "BEGIN RSA PRIVATE KEY",
    "BEGIN EC PRIVATE KEY",
    "WANDB_API_KEY=",
    "AWS_SECRET_ACCESS_KEY=",
)


class BundleValidationError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_relative_path(value: str | Path) -> str:
    """Validate and normalize a bundle-relative portable path."""

    text = str(value).replace("\\", "/")
    if not text or "\x00" in text:
        raise BundleValidationError("Bundle path is empty or contains NUL")
    posix = PurePosixPath(text)
    windows = PureWindowsPath(str(value))
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise BundleValidationError(f"Absolute or drive path is forbidden: {value}")
    if any(part in ("", ".", "..") for part in posix.parts):
        raise BundleValidationError(f"Unsafe bundle path: {value}")
    return posix.as_posix()


def resolve_inside(root: Path, relative: str | Path) -> Path:
    normalized = safe_relative_path(relative)
    candidate = (root / Path(*PurePosixPath(normalized).parts)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise BundleValidationError(f"Path escapes bundle: {relative}") from exc
    return candidate


def _assert_regular_source(path: Path, label: str) -> None:
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise BundleValidationError(
                f"Refusing symlink component for {label}: {current}"
            )
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _safe_filename(value: str, suffix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip(" ._") or "record"
    stem = stem[:100]
    if re.search(r"[<>:\"/\\|?*]", stem) or stem.upper().split(".")[0] in {
        "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        stem = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    return stem + suffix


def _assert_no_sensitive_values(value: Any, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if SENSITIVE_KEY.search(key_text) and child not in (None, "", False):
                raise BundleValidationError(
                    f"Sensitive configuration key is not exportable: {location}.{key_text}"
                )
            _assert_no_sensitive_values(child, f"{location}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_sensitive_values(child, f"{location}[{index}]")
    elif isinstance(value, str) and any(marker in value for marker in SENSITIVE_TEXT):
        raise BundleValidationError(f"Sensitive text detected at {location}")


def select_manifest_records(
    manifest: Mapping[str, Any],
    *,
    max_molecules: int = 3,
    max_records: int = 30,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_molecules < 1 or max_records < 1:
        raise ValueError("max_molecules and max_records must be positive")
    rows = [dict(row) for row in manifest.get("records", [])]
    if not rows:
        raise ValueError("Manifest has no records")
    original_molecules = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        molecule = str(row["mol_id"])
        if molecule not in grouped:
            original_molecules.append(molecule)
        grouped[molecule].append(row)
    ranked = deterministic_molecule_order(original_molecules, seed)
    chosen = set(ranked[: min(max_molecules, max_records)])
    candidates = [row for row in rows if str(row["mol_id"]) in chosen]
    selected = candidates[:max_records]
    selected_molecules = list(dict.fromkeys(str(row["mol_id"]) for row in selected))
    selected_counts = Counter(str(row["mol_id"]) for row in selected)
    truncations = []
    for molecule in selected_molecules:
        original_count = len(grouped[molecule])
        kept = selected_counts[molecule]
        if kept < original_count:
            truncations.append(
                {
                    "mol_id": molecule,
                    "original_records": original_count,
                    "kept_records": kept,
                }
            )
    report = {
        "seed": int(seed),
        "selection_algorithm": (
            "existing deterministic_molecule_order SHA256(seed:molecule_id); "
            "selected records are emitted in original manifest order and capped by max_records"
        ),
        "original_molecule_count": len(original_molecules),
        "original_record_count": len(rows),
        "ranked_selected_molecule_ids": ranked[: min(max_molecules, max_records)],
        "selected_molecule_ids_in_manifest_order": selected_molecules,
        "selected_record_ids": [str(row["sample_id"]) for row in selected],
        "selected_molecule_count": len(selected_molecules),
        "selected_record_count": len(selected),
        "records_per_molecule": dict(selected_counts),
        "record_truncation_occurred": bool(truncations),
        "truncated_molecules": truncations,
    }
    return selected, report


def _git_value(arguments: list[str], default: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return default


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def source_environment() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "pyg_version": _package_version("torch-geometric"),
        "rdkit_version": _package_version("rdkit"),
    }


def _source_label(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return safe_relative_path(resolved.relative_to(root.resolve()).as_posix())
    except ValueError:
        return safe_relative_path(f"external/{path.name}")


def _copy_regular(source: Path, destination: Path) -> None:
    _assert_regular_source(source, "bundle input")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination, follow_symlinks=False)
    os.chmod(destination, 0o644)


def _cache_path_mapping(dataset: Any, wanted: set[str]) -> dict[str, tuple[Path, Any]]:
    mapping: dict[str, tuple[Path, Any]] = {}
    for index, path_value in enumerate(dataset.data_files):
        path = Path(path_value)
        _assert_regular_source(path, "cache record")
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(raw, Mapping):
            raise BundleValidationError(f"Cache record is not a mapping: {path}")
        _assert_no_sensitive_values(raw, f"cache.{path.name}")
        sample_id = str(raw.get("sample_id", raw.get("mol_id", "")))
        if sample_id in wanted:
            if sample_id in mapping:
                raise BundleValidationError(f"Duplicate cache sample_id: {sample_id}")
            mapping[sample_id] = (path, dataset.get(index))
            if len(mapping) == len(wanted):
                break
    missing = sorted(wanted.difference(mapping))
    if missing:
        raise BundleValidationError(f"Cache misses selected sample IDs: {missing[:10]}")
    return mapping


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.chmod(path, 0o644)


def _rewrite_config(source: Path, destination: Path) -> tuple[dict[str, Any], str]:
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise BundleValidationError("Config must be a mapping")
    _assert_no_sensitive_values(config, "config")
    bundled = dict(config)
    if isinstance(config.get("data"), Mapping):
        bundled["data"] = dict(config["data"])
        bundled["data"]["cache_dir"] = "cache"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(bundled, sort_keys=False), encoding="utf-8")
    os.chmod(destination, 0o644)
    return bundled, file_sha256(source)


def _bundle_readme(checkpoint_name: str, split: str) -> str:
    return f"""# Global Coupled 4D profile bundle

This is a minimal, real-data input bundle for bounded Global Coupled 4D
performance diagnosis. It is not a training dataset and must not be used to
replace the RTX 5090 server's timing with Windows timing.

## Linux export

```bash
python scripts/export_global4d_profile_bundle.py \\
  --checkpoint logs_global_coupled_4d/global4d_local025_seed42_5000step/checkpoints/{checkpoint_name} \\
  --config logs_global_coupled_4d/global4d_local025_seed42_5000step/config.resolved.yaml \\
  --cache_dir data/flexbond_inference_formal_small \\
  --manifest eval_manifest_formal_small.json --split {split} \\
  --max_molecules 3 --max_records 30 --seed 42 \\
  --output artifacts/global4d_profile_bundle.tar.gz
```

## Windows extraction and verification

```powershell
tar -xzf artifacts/global4d_profile_bundle.tar.gz -C artifacts
python scripts/verify_global4d_profile_bundle.py --bundle_dir artifacts/global4d_profile_bundle
```

## Windows bounded profile

```powershell
python scripts/profile_global4d_sampling.py `
  --checkpoint artifacts/global4d_profile_bundle/checkpoint/{checkpoint_name} `
  --config artifacts/global4d_profile_bundle/config/config.resolved.yaml `
  --cache_dir artifacts/global4d_profile_bundle/cache `
  --manifest artifacts/global4d_profile_bundle/manifest/profile_manifest.json `
  --split {split} --max_molecules 3 --max_records 30 `
  --refinement_steps 10 --device cuda --output_dir reports/profile_windows
```

Windows results are useful for call-chain and I/O diagnosis only. They cannot
substitute for RTX 5090 CUDA timing. Checkpoint, config, code commit, manifest,
and cache must retain the same semantic provenance.
"""


def collect_bundle_hashes(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "metadata/file_hashes.json":
            continue
        if path.is_symlink():
            raise BundleValidationError(f"Bundle contains symlink: {relative}")
        rows.append(
            {
                "path": safe_relative_path(relative),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "algorithm": "sha256",
        "files": rows,
        "self_excluded": "metadata/file_hashes.json",
    }


def _verify_model_and_checkpoint(config: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    from etflow.models.global_coupled_4d_flow import GlobalCoupled4DFlowLightningModule

    arguments = {
        **dict(config.get("model", {})),
        **dict(config.get("loss", {})),
        **dict(config.get("optimizer", {})),
        **dict(config.get("time_sampling", {})),
    }
    arguments.pop("scheduler", None)
    model = GlobalCoupled4DFlowLightningModule(**arguments)
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise BundleValidationError("Checkpoint has no state_dict mapping")
    model.load_state_dict(state, strict=True)
    return {
        "model_class": type(model).__name__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "state_dict_keys": len(state),
    }


def load_checkpoint_cpu(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise BundleValidationError("Checkpoint is not a mapping")
    _assert_no_sensitive_values(payload, "checkpoint")
    for value in payload.get("state_dict", {}).values():
        if torch.is_tensor(value) and value.device.type != "cpu":
            raise BundleValidationError("Checkpoint tensor did not load on CPU")
    return payload


def verify_bundle_directory(
    bundle_dir: str | Path,
    *,
    verify_model: bool = True,
    verify_dataset: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).resolve()
    errors = []
    checks: dict[str, Any] = {}
    try:
        if root.name != BUNDLE_ROOT_NAME:
            raise BundleValidationError(
                f"Bundle directory must be named {BUNDLE_ROOT_NAME}"
            )
        for relative in REQUIRED_STATIC_FILES:
            path = resolve_inside(root, relative)
            if not path.is_file() or path.is_symlink():
                raise BundleValidationError(f"Missing regular bundle file: {relative}")
        metadata = json.loads(
            (root / "metadata/bundle_metadata.json").read_text(encoding="utf-8")
        )
        hashes = json.loads(
            (root / "metadata/file_hashes.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (root / "manifest/profile_manifest.json").read_text(encoding="utf-8")
        )
        config_path = resolve_inside(root, metadata["paths"]["config"])
        checkpoint_path = resolve_inside(root, metadata["paths"]["checkpoint"])
        cache_root = resolve_inside(root, metadata["paths"]["cache_dir"])
        for key in ("checkpoint_original_path", "config_original_path", "manifest_original_path"):
            safe_relative_path(metadata[key])
        expected_paths = set()
        for row in hashes.get("files", []):
            relative = safe_relative_path(row["path"])
            path = resolve_inside(root, relative)
            if not path.is_file() or path.is_symlink():
                raise BundleValidationError(f"Hashed file missing or symlinked: {relative}")
            if path.stat().st_size != int(row["bytes"]):
                raise BundleValidationError(f"Size mismatch: {relative}")
            if file_sha256(path) != str(row["sha256"]):
                raise BundleValidationError(f"SHA256 mismatch: {relative}")
            expected_paths.add(relative)
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and path.relative_to(root).as_posix() != "metadata/file_hashes.json"
        }
        if actual_paths != expected_paths:
            raise BundleValidationError(
                f"Unhashed or missing files: extra={sorted(actual_paths - expected_paths)}, "
                f"missing={sorted(expected_paths - actual_paths)}"
            )
        if file_sha256(checkpoint_path) != metadata["checkpoint_sha256"]:
            raise BundleValidationError("Checkpoint metadata hash mismatch")
        if file_sha256(config_path) != metadata["config_sha256"]:
            raise BundleValidationError("Config metadata hash mismatch")
        manifest_path = root / "manifest/profile_manifest.json"
        if file_sha256(manifest_path) != metadata["reduced_manifest_sha256"]:
            raise BundleValidationError("Reduced manifest metadata hash mismatch")
        if str(manifest.get("manifest_version")) != "1.0":
            raise BundleValidationError("Unsupported profile manifest version")
        bundle_paths = manifest.get("bundle_paths", {})
        for value in bundle_paths.values():
            resolve_inside(root, value)
        rows = manifest.get("records")
        if not isinstance(rows, list) or not rows:
            raise BundleValidationError("Profile manifest has no records")
        ordered_ids = []
        molecule_ids = []
        for row in rows:
            cache_path = resolve_inside(root, row["cache_file"])
            if not cache_path.is_file() or cache_path.is_symlink():
                raise BundleValidationError(f"Missing cache file: {row['cache_file']}")
            raw = torch.load(cache_path, map_location="cpu", weights_only=False)
            if not isinstance(raw, Mapping):
                raise BundleValidationError(f"Cache is not a mapping: {row['cache_file']}")
            sample_id = str(row["sample_id"])
            if str(raw.get("sample_id", raw.get("mol_id"))) != sample_id:
                raise BundleValidationError(f"Cache sample mismatch: {sample_id}")
            if str(raw.get("x_init_hash")) != str(row["x_init_hash"]):
                raise BundleValidationError(f"Cache x_init_hash mismatch: {sample_id}")
            ordered_ids.append(sample_id)
            molecule_ids.append(str(row["mol_id"]))
        if len(ordered_ids) != len(set(ordered_ids)):
            raise BundleValidationError("Duplicate sample IDs in reduced manifest")
        if len(rows) != int(metadata["selected_record_count"]):
            raise BundleValidationError("Metadata selected_record_count mismatch")
        if ordered_ids != [str(value) for value in metadata["selected_record_ids"]]:
            raise BundleValidationError("Selected record order mismatch")
        ordered_molecules = list(dict.fromkeys(molecule_ids))
        if ordered_molecules != [str(value) for value in metadata["selected_molecule_ids"]]:
            raise BundleValidationError("Selected molecule order mismatch")
        cache_files = {safe_relative_path(row["cache_file"]) for row in rows}
        if len(cache_files) != int(metadata["cache_file_count"]):
            raise BundleValidationError("Cache file count mismatch")
        actual_cache_files = {
            path.relative_to(root).as_posix()
            for path in cache_root.rglob("*")
            if path.is_file()
        }
        if actual_cache_files != cache_files:
            raise BundleValidationError(
                "Bundle contains unreferenced cache files: "
                f"{sorted(actual_cache_files - cache_files)}"
            )
        if sum((root / path).stat().st_size for path in cache_files) != int(
            metadata["cache_total_bytes"]
        ):
            raise BundleValidationError("Cache total byte count mismatch")
        if len(ordered_molecules) != int(metadata["selected_molecule_count"]):
            raise BundleValidationError("Selected molecule count mismatch")
        optional = metadata.get("optional", {})
        allowed_paths = {
            metadata["paths"]["checkpoint"],
            metadata["paths"]["config"],
            metadata["paths"]["manifest"],
            metadata["paths"]["selection_report"],
            metadata["paths"]["environment_source"],
            "metadata/bundle_metadata.json",
            "README.md",
            *cache_files,
        }
        if optional.get("sampling_state_included"):
            allowed_paths.add("optional/sampling_state.json")
        if optional.get("partial_samples_included"):
            allowed_paths.add("optional/partial_samples.pt")
        if actual_paths != {safe_relative_path(path) for path in allowed_paths}:
            raise BundleValidationError(
                "Bundle contains files outside the minimal allowlist: "
                f"{sorted(actual_paths - allowed_paths)}"
            )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(config, Mapping):
            raise BundleValidationError("Config is not a mapping")
        checkpoint = load_checkpoint_cpu(checkpoint_path)
        if verify_dataset:
            from etflow.data.flexbond_eval_manifest import validate_dataset_against_manifest
            from etflow.data.flexbond_inference_dataset import FlexBondInferenceDataset

            validate_dataset_against_manifest(
                FlexBondInferenceDataset(cache_root, str(metadata["split"])), manifest
            )
        model_check = (
            _verify_model_and_checkpoint(config, checkpoint)
            if verify_model
            else {"skipped": True}
        )
        checks = {
            "hashes_verified": len(expected_paths),
            "manifest_records_verified": len(rows),
            "molecules_verified": len(ordered_molecules),
            "checkpoint_map_location": "cpu",
            "model": model_check,
            "cache_root": cache_root.relative_to(root).as_posix(),
        }
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "status": "VALID" if not errors else "INVALID",
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "bundle_dir": str(root),
        "verified_at": utc_now(),
        "checks": checks,
        "errors": errors,
    }


def _tar_directory(source: Path, destination: Path) -> None:
    with tarfile.open(destination, "w:gz", dereference=False) as archive:
        archive.add(source, arcname=BUNDLE_ROOT_NAME, recursive=True)


def create_profile_bundle(
    *,
    checkpoint: str | Path,
    config: str | Path,
    cache_dir: str | Path,
    manifest: str | Path,
    split: str,
    output: str | Path,
    max_molecules: int = 3,
    max_records: int = 30,
    seed: int = 42,
    include_sampling_state: bool = False,
    sampling_state: str | Path | None = None,
    include_partial_samples: bool = False,
    partial_samples: str | Path | None = None,
    force: bool = False,
    verification_callback: Callable[[Path], Mapping[str, Any]] | None = None,
    manifest_loader: Callable[[Path], Mapping[str, Any]] | None = None,
    dataset_factory: Callable[[Path, str], Any] | None = None,
    dataset_validator: Callable[[Iterable[Any], dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    config = Path(config)
    cache_dir = Path(cache_dir)
    manifest = Path(manifest)
    output = Path(output)
    for path, label in ((checkpoint, "checkpoint"), (config, "config"), (manifest, "manifest")):
        _assert_regular_source(path, label)
    cache_root = cache_dir / split if (cache_dir / split).is_dir() else cache_dir
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Missing cache directory: {cache_root}")
    absolute_cache = cache_root.expanduser().absolute()
    current = Path(absolute_cache.anchor)
    for part in absolute_cache.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise BundleValidationError(
                f"Refusing symlink component for cache directory: {current}"
            )
    if output.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if output.is_symlink():
        raise BundleValidationError(f"Refusing symlink output: {output}")
    if include_sampling_state and sampling_state is None:
        raise ValueError("--include_sampling_state requires --sampling_state")
    if include_partial_samples and partial_samples is None:
        raise ValueError("--include_partial_samples requires --partial_samples")
    optional_sources = []
    if include_sampling_state:
        optional_sources.append((Path(sampling_state), "optional/sampling_state.json"))
    if include_partial_samples:
        optional_sources.append((Path(partial_samples), "optional/partial_samples.pt"))
    for path, label in optional_sources:
        _assert_regular_source(path, label)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(tempfile.mkdtemp(prefix="global4d_bundle_", dir=output.parent))
    root = temporary_parent / BUNDLE_ROOT_NAME
    temporary_tar = output.with_name(output.name + f".tmp.{os.getpid()}")
    try:
        for directory in ("checkpoint", "config", "manifest", "cache", "metadata", "optional"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        if manifest_loader is None or dataset_validator is None:
            from etflow.data.flexbond_eval_manifest import (
                load_eval_manifest,
                validate_dataset_against_manifest,
            )
            manifest_loader = manifest_loader or load_eval_manifest
            dataset_validator = dataset_validator or validate_dataset_against_manifest
        if dataset_factory is None:
            from etflow.data.flexbond_inference_dataset import FlexBondInferenceDataset
            dataset_factory = FlexBondInferenceDataset

        source_manifest = dict(manifest_loader(manifest))
        _assert_no_sensitive_values(source_manifest, "manifest")
        selected, selection = select_manifest_records(
            source_manifest,
            max_molecules=max_molecules,
            max_records=max_records,
            seed=seed,
        )
        wanted = {str(row["sample_id"]) for row in selected}
        dataset = dataset_factory(cache_dir, split)
        cache_mapping = _cache_path_mapping(dataset, wanted)
        selected_data = [cache_mapping[str(row["sample_id"])][1] for row in selected]
        dataset_validator(
            selected_data, {**source_manifest, "records": selected}
        )
        checkpoint_name = _safe_filename(checkpoint.stem, checkpoint.suffix or ".ckpt")
        checkpoint_relative = f"checkpoint/{checkpoint_name}"
        config_relative = "config/config.resolved.yaml"
        _copy_regular(checkpoint, root / checkpoint_relative)
        _, source_config_hash = _rewrite_config(config, root / config_relative)
        cache_rows = []
        reduced_rows = []
        total_cache_bytes = 0
        for index, row in enumerate(selected):
            sample_id = str(row["sample_id"])
            source_path, data = cache_mapping[sample_id]
            name = f"{index:04d}_{_safe_filename(sample_id, '.pt')}"
            relative = f"cache/{safe_relative_path(split)}/{name}"
            destination = root / Path(*PurePosixPath(relative).parts)
            _copy_regular(source_path, destination)
            total_cache_bytes += destination.stat().st_size
            reduced_rows.append(
                {
                    "mol_id": str(row["mol_id"]),
                    "sample_id": sample_id,
                    "x_init_hash": str(row["x_init_hash"]),
                    "num_rotatable_bonds": int(row["num_rotatable_bonds"]),
                    "cache_file": relative,
                }
            )
            cache_rows.append(
                {
                    "sample_id": sample_id,
                    "mol_id": str(row["mol_id"]),
                    "source_cache_file": _source_label(source_path, Path.cwd()),
                    "bundle_cache_file": relative,
                    "atoms": int(data.atomic_numbers.numel()),
                    "rotatable_bonds": int(data.num_rotatable_bonds.item()),
                    "bytes": destination.stat().st_size,
                }
            )
        reduced_manifest = {
            "manifest_version": str(source_manifest["manifest_version"]),
            "created_at": utc_now(),
            "split": split,
            "selection_seed": int(seed),
            "bundle_paths": {
                "checkpoint": checkpoint_relative,
                "config": config_relative,
                "cache_dir": "cache",
            },
            "records": reduced_rows,
        }
        reduced_manifest_path = root / "manifest/profile_manifest.json"
        _write_json(reduced_manifest_path, reduced_manifest)
        for source, relative in optional_sources:
            if relative.endswith(".json"):
                value = json.loads(source.read_text(encoding="utf-8"))
                _assert_no_sensitive_values(value, relative)
            elif relative.endswith(".pt"):
                value = torch.load(source, map_location="cpu", weights_only=False)
                _assert_no_sensitive_values(value, relative)
            _copy_regular(source, root / relative)
        selection.update(
            {
                "physical_cache_file_count": len(cache_rows),
                "physical_copied_record_count": len(cache_rows),
                "manifest_enabled_record_count": len(reduced_rows),
                "per_record_cache_files": cache_rows,
            }
        )
        _write_json(root / "metadata/selection_report.json", selection)
        environment = source_environment()
        _write_json(
            root / "metadata/environment_source.json",
            {
                key: environment[key]
                for key in (
                    "python_version",
                    "torch_version",
                    "cuda_version",
                    "pyg_version",
                    "rdkit_version",
                )
            },
        )
        metadata = {
            "bundle_format_version": BUNDLE_FORMAT_VERSION,
            "source_git_commit": _git_value(["rev-parse", "HEAD"], "unknown"),
            "source_branch": _git_value(["branch", "--show-current"], "unknown"),
            "source_hostname": environment["hostname"],
            "source_platform": environment["platform"],
            "source_python_version": environment["python_version"],
            "source_torch_version": environment["torch_version"],
            "source_cuda_version": environment["cuda_version"],
            "source_pyg_version": environment["pyg_version"],
            "source_rdkit_version": environment["rdkit_version"],
            "checkpoint_original_path": _source_label(checkpoint, Path.cwd()),
            "config_original_path": _source_label(config, Path.cwd()),
            "manifest_original_path": _source_label(manifest, Path.cwd()),
            "checkpoint_filename": checkpoint_name,
            "checkpoint_sha256": file_sha256(root / checkpoint_relative),
            "config_sha256": file_sha256(root / config_relative),
            "source_config_sha256": source_config_hash,
            "original_manifest_sha256": file_sha256(manifest),
            "reduced_manifest_sha256": file_sha256(reduced_manifest_path),
            "cache_file_count": len(cache_rows),
            "cache_total_bytes": total_cache_bytes,
            "selected_molecule_count": selection["selected_molecule_count"],
            "selected_record_count": selection["selected_record_count"],
            "selected_molecule_ids": selection["selected_molecule_ids_in_manifest_order"],
            "selected_record_ids": selection["selected_record_ids"],
            "split": split,
            "seed": int(seed),
            "exported_at": utc_now(),
            "creation_command": (
                "python scripts/export_global4d_profile_bundle.py --checkpoint "
                f"{_source_label(checkpoint, Path.cwd())} --config {_source_label(config, Path.cwd())} "
                f"--cache_dir {_source_label(cache_dir.resolve() / split, Path.cwd())} "
                f"--manifest {_source_label(manifest, Path.cwd())} --split {split} "
                f"--max_molecules {max_molecules} --max_records {max_records} --seed {seed} "
                "--output artifacts/global4d_profile_bundle.tar.gz"
            ),
            "paths": {
                "checkpoint": checkpoint_relative,
                "config": config_relative,
                "manifest": "manifest/profile_manifest.json",
                "cache_dir": "cache",
                "selection_report": "metadata/selection_report.json",
                "environment_source": "metadata/environment_source.json",
            },
            "optional": {
                "sampling_state_included": include_sampling_state,
                "partial_samples_included": include_partial_samples,
            },
        }
        _write_json(root / "metadata/bundle_metadata.json", metadata)
        readme = root / "README.md"
        readme.write_text(_bundle_readme(checkpoint_name, split), encoding="utf-8")
        os.chmod(readme, 0o644)
        _write_json(root / "metadata/file_hashes.json", collect_bundle_hashes(root))
        verify = (
            verification_callback(root)
            if verification_callback is not None
            else verify_bundle_directory(root, verify_model=True)
        )
        if str(verify.get("status")) != "VALID":
            raise BundleValidationError(f"Pre-archive verification failed: {verify}")
        if temporary_tar.exists():
            temporary_tar.unlink()
        _tar_directory(root, temporary_tar)
        os.chmod(temporary_tar, 0o644)
        os.replace(temporary_tar, output)
        return {
            "status": "EXPORTED",
            "output": str(output.resolve()),
            "output_bytes": output.stat().st_size,
            "selected_molecule_count": selection["selected_molecule_count"],
            "selected_record_count": selection["selected_record_count"],
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "reduced_manifest_sha256": metadata["reduced_manifest_sha256"],
        }
    finally:
        if temporary_tar.exists():
            temporary_tar.unlink()
        shutil.rmtree(temporary_parent, ignore_errors=True)
