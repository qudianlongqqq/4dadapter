"""Resumable formal-large Minimal Validity Target asset construction."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
import yaml

from etflow.commons.record_identity import source_record_identity
from etflow.data.flexbond_optimizer_dataset import validate_cache_record
from etflow.ecir.audit import file_sha256
from etflow.ecir.minimal_validity_target import (
    MinimalValidityConfig,
    MinimalValidityTargetBuilder,
)
from etflow.ecir.formal_rdkit_adapter import adapt_formal_cache_record


CONFIG_SCHEMA = "ecir-mvr-formal-large-target-build-config-v1"
SOURCE_SCHEMA = "ecir-mvr-formal-large-real-sources-v1"
TARGET_SCHEMA = "ecir-mvr-formal-large-minimal-target-v1"
SUMMARY_SCHEMA = "ecir-mvr-formal-large-target-build-summary-v1"
VALIDATION_SCHEMA = "ecir-mvr-formal-large-target-validation-v1"
STAGE_D_VALIDITY_IDENTITY = (
    "66dd6ab6cf290057d8dea725e0042ba5a5fcc69ad0ec9e0db8971910df377cd3"
)
SPLITS = ("train", "val")
TELEMETRY_FIELDS = (
    "timestamp",
    "elapsed_seconds",
    "completed_records",
    "successful_records",
    "failed_records",
    "skipped_records",
    "records_per_minute",
    "average_seconds_per_record",
    "recent_100_average_seconds",
    "eta_seconds",
    "gpu_index",
    "gpu_uuid",
    "gpu_utilization_percent",
    "gpu_memory_used_mib",
    "gpu_memory_total_mib",
    "power_draw_w",
    "temperature_c",
    "cpu_percent",
    "rss_mib",
    "output_size_mib",
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = torch.as_tensor(value, dtype=torch.float32).detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        frame.to_parquet(temporary, index=False)
        pd.read_parquet(temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def forbid_test_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if "test" in {part.lower() for part in resolved.parts}:
        raise ValueError(f"test path is forbidden: {resolved}")
    return resolved


def require_parquet_engine() -> str:
    for name in ("pyarrow", "fastparquet"):
        if importlib.util.find_spec(name) is not None:
            return name
    raise RuntimeError(
        "formal target assets require pyarrow or fastparquet for Stage D-compatible manifests"
    )


def load_config(path: Path, *, output_root: Path | None = None) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unexpected formal target build config schema")
    if config.get("splits", {}).get("test", {}).get("enabled") is not False:
        raise ValueError("formal target config must disable test")
    if set(config["splits"]) != {"train", "val", "test"}:
        raise ValueError("formal target config may declare only train, val, and disabled test")
    expected = asdict(MinimalValidityConfig())
    if dict(config["target_builder"]) != expected:
        raise ValueError("target builder config differs from frozen Stage D defaults")
    config["config_path"] = str(path.resolve())
    config["config_file_sha256"] = file_sha256(path)
    config["output_root"] = str((output_root or Path(config["output_root"])).expanduser())
    return config


def verify_stage_d_identities(config: Mapping[str, Any]) -> dict[str, Any]:
    validity_path = forbid_test_path(config["validity_statistics"])
    validity = json.loads(validity_path.read_text(encoding="utf-8"))
    if validity.get("identity_sha256") != STAGE_D_VALIDITY_IDENTITY:
        raise ValueError("frozen Stage D validity statistics identity changed")
    metadata_path = forbid_test_path(config["stage_d_target_metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("decision") != "PASS":
        raise ValueError("Stage D target metadata is not a PASS")
    if metadata.get("test_paths_opened") != 0:
        raise ValueError("Stage D target metadata is not test-free")
    if dict(metadata.get("target_builder_config", {})) != dict(config["target_builder"]):
        raise ValueError("formal target builder config differs from Stage D metadata")
    builder_path = Path(__file__).with_name("minimal_validity_target.py")
    adapter_path = Path(__file__).with_name("formal_rdkit_adapter.py")
    return {
        "validity_statistics_path": str(validity_path),
        "validity_statistics_sha256": file_sha256(validity_path),
        "validity_statistics_identity_sha256": validity["identity_sha256"],
        "stage_d_target_metadata_path": str(metadata_path),
        "stage_d_target_metadata_sha256": file_sha256(metadata_path),
        "stage_d_target_identity_sha256": metadata["medium_target_identity_sha256"],
        "builder_code_path": str(builder_path.resolve()),
        "builder_code_sha256": file_sha256(builder_path),
        "formal_rdkit_adapter_path": str(adapter_path.resolve()),
        "formal_rdkit_adapter_sha256": file_sha256(adapter_path),
        "builder_config_sha256": canonical_sha256(config["target_builder"]),
    }


def _record(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise TypeError(f"formal source record must be a mapping: {path}")
    return dict(value)


def source_row(path: Path, split: str) -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError(f"unsupported source split: {split}")
    record = _record(path)
    validate_cache_record(record, require_persisted_pair=True)
    sample_id = str(record.get("sample_id", record.get("mol_id", "")))
    if not sample_id:
        raise ValueError(f"source has no sample_id: {path}")
    molecule_id = source_record_identity(record)
    coordinates = torch.as_tensor(record["x_init"], dtype=torch.float32)
    if not bool(torch.isfinite(coordinates).all()):
        raise ValueError(f"source coordinates are not finite: {path}")
    return {
        "schema_version": SOURCE_SCHEMA,
        "split": split,
        "sample_id": sample_id,
        "molecule_id": molecule_id,
        "generator_name": "ETFlow_formal_upstream",
        "source_severity": "normal",
        "source_path": str(path.resolve()),
        "coordinate_path": None,
        "coordinate_key": "x_init",
        "coordinate_sha256": tensor_sha256(coordinates),
        "source_file_sha256": file_sha256(path),
        "source_x_init_hash": str(record.get("x_init_hash", "")),
        "num_atoms": int(coordinates.size(0)),
        "num_rotatable_bonds": int(record.get("num_rotatable_bonds", 0)),
        "generator_checkpoint": str(record.get("generator_checkpoint", "")),
        "seed": int(record.get("sample_seed", 42)),
        "NFE": 10,
        "update_scale": 0.0,
        "reference_availability": bool(record.get("x_ref_candidates") is not None),
        "target_semantics": "offline_minimal_validity_target",
        "test_record": False,
    }


def _pair_distribution(frame: pd.DataFrame) -> dict[str, int]:
    values = frame.groupby("molecule_id").size().value_counts().sort_index()
    return {str(int(key)): int(value) for key, value in values.items()}


def build_source_manifests(
    config: Mapping[str, Any], *, resume: bool = True
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    input_root = forbid_test_path(config["input_cache"])
    output_root = Path(config["output_root"]).expanduser().resolve()
    source_dir = output_root / "real_sources"
    metadata_path = source_dir / "metadata.json"
    if resume and metadata_path.is_file() and all(
        (source_dir / f"{split}.parquet").is_file() for split in SPLITS
    ):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("input_cache") != str(input_root):
            raise ValueError("existing source manifests belong to a different input cache")
        return {
            split: pd.read_parquet(source_dir / f"{split}.parquet") for split in SPLITS
        }, metadata

    frames: dict[str, pd.DataFrame] = {}
    split_metadata = {}
    molecule_sets = {}
    for split in SPLITS:
        directory = input_root / split
        if not directory.is_dir():
            raise FileNotFoundError(f"formal cache split is missing: {directory}")
        files = sorted(directory.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"formal cache split has no .pt records: {directory}")
        frame = pd.DataFrame(source_row(path, split) for path in files)
        if frame.sample_id.duplicated().any():
            raise ValueError(f"duplicate {split} sample_id values")
        expected_molecules = int(config["splits"][split]["expected_molecules"])
        expected_pairs = int(config["splits"][split]["expected_records_per_molecule"])
        counts = frame.groupby("molecule_id").size()
        if frame.molecule_id.nunique() != expected_molecules:
            raise ValueError(
                f"{split} has {frame.molecule_id.nunique()} molecules; expected {expected_molecules}"
            )
        if not bool((counts == expected_pairs).all()):
            raise ValueError(f"{split} does not have exactly {expected_pairs} records per molecule")
        atomic_parquet(frame, source_dir / f"{split}.parquet")
        frames[split] = frame
        molecule_sets[split] = set(frame.molecule_id)
        split_metadata[split] = {
            "records": int(len(frame)),
            "molecules": int(frame.molecule_id.nunique()),
            "records_per_molecule": _pair_distribution(frame),
            "manifest_sha256": file_sha256(source_dir / f"{split}.parquet"),
            "aggregate_source_sha256": canonical_sha256(
                frame[["sample_id", "source_file_sha256", "coordinate_sha256"]]
                .sort_values("sample_id")
                .to_dict("records")
            ),
        }
    overlap = molecule_sets["train"] & molecule_sets["val"]
    if overlap:
        raise ValueError(f"train/val molecule leakage: {sorted(overlap)[:10]}")
    metadata = {
        "schema_version": SOURCE_SCHEMA,
        "created_at": utc_now(),
        "input_cache": str(input_root),
        "splits": split_metadata,
        "train_val_overlap": 0,
        "test_records_read": 0,
    }
    metadata["formal_source_identity_sha256"] = canonical_sha256(metadata)
    atomic_json(metadata, metadata_path)
    return frames, metadata


def target_key(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()


def target_paths(output_root: Path, split: str, sample_id: str) -> tuple[Path, Path]:
    key = target_key(sample_id)
    return (
        output_root / "minimal_targets" / split / f"{key}.pt",
        output_root / "manifests" / "records" / split / f"{key}.json",
    )


def _load_coordinates(row: Mapping[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
    record = _record(Path(str(row["source_path"])))
    coordinates = torch.as_tensor(record[str(row["coordinate_key"])], dtype=torch.float32)
    if tensor_sha256(coordinates) != str(row["coordinate_sha256"]):
        raise ValueError(f"source coordinate identity changed: {row['sample_id']}")
    if file_sha256(row["source_path"]) != str(row["source_file_sha256"]):
        raise ValueError(f"source file identity changed: {row['sample_id']}")
    return adapt_formal_cache_record(record), coordinates


def validate_target_payload(
    payload: Mapping[str, Any],
    row: Mapping[str, Any],
    identities: Mapping[str, Any],
) -> dict[str, Any]:
    if payload.get("schema_version") != TARGET_SCHEMA:
        raise ValueError("unexpected target payload schema")
    for name in ("sample_id", "molecule_id", "split"):
        if str(payload.get(name)) != str(row[name]):
            raise ValueError(f"target {name} does not match source")
    if payload.get("source_file_sha256") != row["source_file_sha256"]:
        raise ValueError("target source SHA does not match source")
    if payload.get("source_coordinate_sha256") != row["coordinate_sha256"]:
        raise ValueError("target source coordinate SHA does not match source")
    if payload.get("builder_code_sha256") != identities["builder_code_sha256"]:
        raise ValueError("target builder code identity changed")
    if payload.get("builder_config_sha256") != identities["builder_config_sha256"]:
        raise ValueError("target builder config identity changed")
    if payload.get("config_file_sha256") != identities.get(
        "config_file_sha256", payload.get("config_file_sha256")
    ):
        raise ValueError("target build config identity changed")
    adapter_sha = payload.get("formal_rdkit_adapter_sha256")
    if adapter_sha is not None and adapter_sha != identities.get(
        "formal_rdkit_adapter_sha256"
    ):
        raise ValueError("formal RDKit adapter identity changed")
    x_input = torch.as_tensor(payload["x_input"], dtype=torch.float32)
    x_target = torch.as_tensor(payload["x_target"], dtype=torch.float32)
    if tuple(x_input.shape) != tuple(x_target.shape) or tuple(x_target.shape) != (
        int(row["num_atoms"]),
        3,
    ):
        raise ValueError("target coordinate shape does not match source")
    if not bool(torch.isfinite(x_target).all()):
        raise ValueError("target contains NaN or Inf")
    if tensor_sha256(x_input) != row["coordinate_sha256"]:
        raise ValueError("persisted x_input identity changed")
    metadata = dict(payload["target_metadata"])
    if metadata.get("reference_fallback_used") is not False:
        raise ValueError("reference fallback is forbidden")
    if metadata.get("force_field_fallback_used") is not False:
        raise ValueError("force-field fallback is forbidden")
    if metadata.get("optimizer_config") != identities["target_builder_config"]:
        raise ValueError("target optimizer config differs from Stage D")
    if metadata.get("target_sha256") != tensor_sha256(x_target):
        raise ValueError("target tensor SHA mismatch")
    return metadata


def _manifest_row(
    source: Mapping[str, Any], payload: Mapping[str, Any], target_path: Path
) -> dict[str, Any]:
    metadata = dict(payload["target_metadata"])
    return {
        "schema_version": TARGET_SCHEMA,
        "split": str(source["split"]),
        "sample_id": str(source["sample_id"]),
        "molecule_id": str(source["molecule_id"]),
        "source": str(source["generator_name"]),
        "severity": str(source["source_severity"]),
        "source_path": str(source["source_path"]),
        "source_file_sha256": str(source["source_file_sha256"]),
        "source_coordinate_sha256": str(source["coordinate_sha256"]),
        "target_cache_path": str(target_path.resolve()),
        "target_file_sha256": file_sha256(target_path),
        "target_sha256": str(metadata["target_sha256"]),
        "target_status": str(metadata["target_status"]),
        "stop_reason": str(metadata["stop_reason"]),
        "validity_gain": float(metadata["validity_gain"]),
        "initial_to_target_rmsd": float(metadata["initial_to_target_rmsd"]),
        "max_atom_displacement": float(metadata["max_atom_displacement"]),
        "torsion_change": float(metadata["torsion_change"]),
        "max_rotatable_torsion_change": float(
            metadata["max_rotatable_torsion_change"]
        ),
        "selected_step": int(metadata["selected_step"]),
        "reference_fallback_used": bool(metadata["reference_fallback_used"]),
        "builder_code_sha256": str(payload["builder_code_sha256"]),
        "builder_config_sha256": str(payload["builder_config_sha256"]),
        "config_file_sha256": str(payload["config_file_sha256"]),
        "test_records_read": 0,
    }


def build_target(
    source: Mapping[str, Any],
    *,
    output_root: Path,
    builder: MinimalValidityTargetBuilder,
    identities: Mapping[str, Any],
    config_file_sha256: str,
) -> tuple[dict[str, Any], bool]:
    target_path, sidecar_path = target_paths(
        output_root, str(source["split"]), str(source["sample_id"])
    )
    if target_path.is_file() and sidecar_path.is_file():
        payload = torch.load(target_path, map_location="cpu", weights_only=False)
        validate_target_payload(payload, source, identities)
        persisted = json.loads(sidecar_path.read_text(encoding="utf-8"))
        current = _manifest_row(source, payload, target_path)
        if persisted != current:
            raise ValueError(f"persisted manifest row changed: {source['sample_id']}")
        return current, True
    if target_path.exists() or sidecar_path.exists():
        raise ValueError(f"partial target state exists: {source['sample_id']}")
    record, x_input = _load_coordinates(source)
    result = builder.build(x_input, record)
    payload = {
        "schema_version": TARGET_SCHEMA,
        "split": str(source["split"]),
        "sample_id": str(source["sample_id"]),
        "molecule_id": str(source["molecule_id"]),
        "x_input": x_input.cpu(),
        "x_target": torch.as_tensor(result["x_target"], dtype=torch.float32).cpu(),
        "source_atomic_numbers": torch.as_tensor(record["atomic_numbers"], dtype=torch.long),
        "source_file_sha256": str(source["source_file_sha256"]),
        "source_coordinate_sha256": str(source["coordinate_sha256"]),
        "target_metadata": result["target_metadata"],
        "builder_code_sha256": identities["builder_code_sha256"],
        "builder_config_sha256": identities["builder_config_sha256"],
        "config_file_sha256": config_file_sha256,
        "formal_rdkit_adapter_sha256": identities.get(
            "formal_rdkit_adapter_sha256"
        ),
        "test_records_read": 0,
    }
    validate_target_payload(payload, source, identities)
    atomic_torch(payload, target_path)
    row = _manifest_row(source, payload, target_path)
    atomic_json(row, sidecar_path)
    return row, False


def record_failure(
    source: Mapping[str, Any], output_root: Path, error: BaseException
) -> None:
    path = (
        output_root
        / "manifests"
        / "failures"
        / str(source["split"])
        / f"{target_key(str(source['sample_id']))}.json"
    )
    prior = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    attempts = list(prior.get("attempts", []))
    attempts.append(
        {
            "timestamp": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )
    atomic_json(
        {
            "schema_version": "ecir-mvr-formal-large-target-failure-v1",
            "split": str(source["split"]),
            "sample_id": str(source["sample_id"]),
            "molecule_id": str(source["molecule_id"]),
            "source_path": str(source["source_path"]),
            "attempts": attempts,
            "resolved": False,
            "test_records_read": 0,
        },
        path,
    )


def clear_failure(source: Mapping[str, Any], output_root: Path) -> None:
    path = (
        output_root
        / "manifests"
        / "failures"
        / str(source["split"])
        / f"{target_key(str(source['sample_id']))}.json"
    )
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        value["resolved"] = True
        value["resolved_at"] = utc_now()
        atomic_json(value, path)


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _gpu_metrics(index: str) -> dict[str, Any]:
    empty = {
        "gpu_index": index,
        "gpu_uuid": "",
        "gpu_utilization_percent": "",
        "gpu_memory_used_mib": "",
        "gpu_memory_total_mib": "",
        "power_draw_w": "",
        "temperature_c": "",
    }
    try:
        query = "index,uuid,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
                "-i",
                str(index),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
        fields = [part.strip() for part in output.splitlines()[0].split(",")]
        keys = list(empty)
        return dict(zip(keys, fields, strict=True))
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return empty


class RuntimeTelemetry:
    def __init__(
        self, output_root: Path, *, total_records: int, interval: float, gpu_index: str
    ) -> None:
        self.output_root = output_root
        self.path = output_root / "telemetry" / "runtime_telemetry.csv"
        self.total_records = int(total_records)
        self.interval = float(interval)
        self.gpu_index = str(gpu_index)
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.state = {
            "completed_records": 0,
            "successful_records": 0,
            "failed_records": 0,
            "skipped_records": 0,
            "durations": [],
        }
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()
        self._started = False
        self._stopped = False
        self.sample_errors: list[str] = []

    def update(self, *, success: bool, skipped: bool, seconds: float) -> None:
        with self.lock:
            self.state["completed_records"] += 1
            self.state["successful_records"] += int(success)
            self.state["failed_records"] += int(not success)
            self.state["skipped_records"] += int(skipped)
            self.state["durations"].append(float(seconds))

    def _sample(self) -> dict[str, Any]:
        with self.lock:
            state = {**self.state, "durations": list(self.state["durations"])}
        elapsed = time.monotonic() - self.started
        completed = int(state["completed_records"])
        durations = state["durations"]
        average = sum(durations) / len(durations) if durations else 0.0
        recent = durations[-100:]
        recent_average = sum(recent) / len(recent) if recent else 0.0
        remaining = max(0, self.total_records - completed)
        try:
            import psutil

            process = psutil.Process(os.getpid())
            cpu = process.cpu_percent(interval=None)
            rss = process.memory_info().rss / 2**20
        except (ImportError, OSError):
            cpu, rss = "", ""
        return {
            "timestamp": utc_now(),
            "elapsed_seconds": elapsed,
            "completed_records": completed,
            "successful_records": state["successful_records"],
            "failed_records": state["failed_records"],
            "skipped_records": state["skipped_records"],
            "records_per_minute": completed * 60.0 / elapsed if elapsed else 0.0,
            "average_seconds_per_record": average,
            "recent_100_average_seconds": recent_average,
            "eta_seconds": remaining * (recent_average or average),
            **_gpu_metrics(self.gpu_index),
            "cpu_percent": cpu,
            "rss_mib": rss,
            "output_size_mib": _directory_size(self.output_root) / 2**20,
        }

    def sample(self) -> None:
        row = self._sample()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.is_file()
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TELEMETRY_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _run(self) -> None:
        try:
            self.sample()
            while not self.stop_event.wait(self.interval):
                self.sample()
        except Exception as error:
            self.sample_errors.append(f"{type(error).__name__}: {error}")

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                raise RuntimeError("runtime telemetry has already been started")
            self._started = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            if self._stopped:
                return
            self._stopped = True
            self.stop_event.set()
            thread = self.thread
        if thread is not None:
            thread.join()
        try:
            self.sample()
        except Exception as error:
            self.sample_errors.append(f"{type(error).__name__}: {error}")
        if self.sample_errors:
            try:
                atomic_json(
                    {
                        "schema_version": "ecir-mvr-formal-target-telemetry-error-v1",
                        "errors": self.sample_errors,
                        "stopped_at": utc_now(),
                    },
                    self.output_root / "telemetry" / "telemetry_error.json",
                )
            except OSError:
                pass


def write_manifest_shards(
    output_root: Path, split: str, rows: list[dict[str, Any]], shard_size: int
) -> None:
    directory = output_root / "manifests" / "shards" / split
    directory.mkdir(parents=True, exist_ok=True)
    expected = set()
    for offset in range(0, len(rows), int(shard_size)):
        path = directory / f"shard-{offset // int(shard_size):06d}.jsonl"
        expected.add(path.name)
        content = "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows[offset : offset + int(shard_size)]
        )
        atomic_text(content, path)
    unexpected = {path.name for path in directory.glob("shard-*.jsonl")} - expected
    if unexpected:
        raise ValueError(f"unexpected stale manifest shards for {split}: {sorted(unexpected)}")


def collect_manifest_rows(output_root: Path, split: str) -> list[dict[str, Any]]:
    directory = output_root / "manifests" / "records" / split
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ] if directory.is_dir() else []


def finalize_manifests(
    output_root: Path, source_frames: Mapping[str, pd.DataFrame], shard_size: int
) -> dict[str, Any]:
    result = {}
    target_dir = output_root / "minimal_targets"
    pairing_dir = output_root / "manifests"
    for split in SPLITS:
        rows = collect_manifest_rows(output_root, split)
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values("sample_id").reset_index(drop=True)
        write_manifest_shards(output_root, split, rows, shard_size)
        atomic_parquet(frame, target_dir / f"{split}.parquet")
        atomic_parquet(frame, pairing_dir / f"pairing_{split}.parquet")
        planned = int(len(source_frames[split]))
        successful = int(len(frame))
        failed = failure_count(output_root, split=split)
        result[split] = {
            "planned_records": planned,
            "completed_records": successful,
            "successful_records": successful,
            "failed_records": failed,
            "success_rate": successful / planned if planned else 0.0,
            "failure_rate": failed / planned if planned else 0.0,
            "molecules": int(frame.molecule_id.nunique()) if not frame.empty else 0,
            "target_status_counts": {
                str(key): int(value)
                for key, value in frame.target_status.value_counts().items()
            }
            if not frame.empty
            else {},
            "target_manifest_sha256": file_sha256(target_dir / f"{split}.parquet"),
            "pairing_manifest_sha256": file_sha256(pairing_dir / f"pairing_{split}.parquet"),
            "aggregate_target_sha256": canonical_sha256(
                frame[["sample_id", "target_file_sha256", "target_sha256"]].to_dict("records")
            ) if not frame.empty else canonical_sha256([]),
        }
    return result


def write_asset_metadata_and_inventory(
    *,
    output_root: Path,
    source_frames: Mapping[str, pd.DataFrame],
    source_metadata: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
    identities: Mapping[str, Any],
    config_file_sha256: str,
    decision: str = "D1B_FORMAL_TARGETS_NOT_READY",
) -> dict[str, Any]:
    metadata = {
        "schema_version": "ecir-mvr-formal-large-minimal-targets-v1",
        "decision": decision,
        "algorithm": "Stage D MinimalValidityTargetBuilder, unchanged parameters and safety rules",
        "test_records_read": 0,
        "source_identity_sha256": source_metadata["formal_source_identity_sha256"],
        "validity_statistics_identity_sha256": identities[
            "validity_statistics_identity_sha256"
        ],
        "stage_d_target_identity_sha256": identities["stage_d_target_identity_sha256"],
        "builder_code_sha256": identities["builder_code_sha256"],
        "builder_config_sha256": identities["builder_config_sha256"],
        "formal_rdkit_adapter_sha256": identities.get(
            "formal_rdkit_adapter_sha256"
        ),
        "config_file_sha256": config_file_sha256,
        "target_builder_config": identities["target_builder_config"],
        "splits": dict(manifest_metadata),
    }
    metadata["formal_target_identity_sha256"] = canonical_sha256(metadata)
    atomic_json(metadata, output_root / "minimal_targets" / "metadata.json")
    atomic_json(
        {
            "schema_version": "ecir-mvr-formal-large-pairing-manifests-v1",
            "test_records_read": 0,
            "source_identity_sha256": source_metadata["formal_source_identity_sha256"],
            "target_identity_sha256": metadata["formal_target_identity_sha256"],
            "splits": dict(manifest_metadata),
        },
        output_root / "manifests" / "metadata.json",
    )

    inventory: dict[str, str] = {
        identities["builder_code_path"]: identities["builder_code_sha256"],
        identities["validity_statistics_path"]: identities[
            "validity_statistics_sha256"
        ],
    }
    adapter_path = identities.get("formal_rdkit_adapter_path")
    adapter_sha = identities.get("formal_rdkit_adapter_sha256")
    if adapter_path and adapter_sha:
        inventory[str(adapter_path)] = str(adapter_sha)
    for split in SPLITS:
        for row in source_frames[split].to_dict("records"):
            inventory[str(row["source_path"])] = str(row["source_file_sha256"])
        for row in collect_manifest_rows(output_root, split):
            inventory[str(row["target_cache_path"])] = str(row["target_file_sha256"])
        for path in (
            output_root / "real_sources" / f"{split}.parquet",
            output_root / "minimal_targets" / f"{split}.parquet",
            output_root / "manifests" / f"pairing_{split}.parquet",
        ):
            if path.is_file():
                inventory[str(path.resolve())] = file_sha256(path)
        for path in sorted((output_root / "manifests" / "shards" / split).glob("*.jsonl")):
            inventory[str(path.resolve())] = file_sha256(path)
    content = "".join(
        f"{digest}  {path}\n" for path, digest in sorted(inventory.items())
    )
    atomic_text(content, output_root / "SHA256SUMS.txt")
    return metadata


def _inventory(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        if name in result:
            raise ValueError(f"duplicate SHA inventory path: {name}")
        result[name] = digest
    return result


def environment_identity() -> dict[str, Any]:
    try:
        import rdkit
        import torch_geometric

        rdkit_version = rdkit.__version__
        pyg_version = torch_geometric.__version__
    except ImportError:
        rdkit_version = pyg_version = "unavailable"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unavailable"
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "torch_geometric": pyg_version,
        "rdkit": rdkit_version,
        "git_commit": commit,
    }


def failure_count(output_root: Path, *, split: str | None = None) -> int:
    root = output_root / "manifests" / "failures"
    if split is not None:
        root = root / split
    if not root.is_dir():
        return 0
    return sum(
        not json.loads(path.read_text(encoding="utf-8")).get("resolved", False)
        for path in root.rglob("*.json")
    )


def unresolved_failure_sample_ids(output_root: Path) -> set[str]:
    root = output_root / "manifests" / "failures"
    if not root.is_dir():
        return set()
    result = set()
    for path in root.rglob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not value.get("resolved", False):
            result.add(str(value["sample_id"]))
    return result


def _strict_pair_check(
    source: Mapping[str, Any],
    target_row: Mapping[str, Any],
    identities: Mapping[str, Any],
) -> None:
    target_path = Path(str(target_row["target_cache_path"]))
    if not target_path.is_file():
        raise FileNotFoundError(f"target payload is missing: {target_path}")
    if file_sha256(target_path) != str(target_row["target_file_sha256"]):
        raise ValueError(f"target file SHA mismatch: {source['sample_id']}")
    payload = torch.load(target_path, map_location="cpu", weights_only=False)
    validate_target_payload(payload, source, identities)
    source_path = Path(str(source["source_path"]))
    if file_sha256(source_path) != str(source["source_file_sha256"]):
        raise ValueError(f"source file SHA mismatch: {source['sample_id']}")
    record = _record(source_path)
    validate_cache_record(record, require_persisted_pair=True)
    if payload.get("formal_rdkit_adapter_sha256") is not None:
        adapt_formal_cache_record(record)
    source_atoms = torch.as_tensor(record["atomic_numbers"], dtype=torch.long).view(-1)
    target_atoms = torch.as_tensor(payload["source_atomic_numbers"], dtype=torch.long).view(-1)
    if not torch.equal(source_atoms, target_atoms):
        raise ValueError(f"target atom order differs from source: {source['sample_id']}")
    if source_record_identity(record) != str(source["molecule_id"]):
        raise ValueError(f"target mol_id differs from source: {source['sample_id']}")
    if str(record.get("sample_id", record.get("mol_id"))) != str(source["sample_id"]):
        raise ValueError(f"target sample_id differs from source: {source['sample_id']}")


def _shard_rows(output_root: Path, split: str) -> list[dict[str, Any]]:
    rows = []
    directory = output_root / "manifests" / "shards" / split
    for path in sorted(directory.glob("shard-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def strict_mvr_dataset_load(
    output_root: Path, identities: Mapping[str, Any], sample_count: int
) -> dict[str, int]:
    """Construct actual Stage D dataset items without opening the test split."""

    from etflow.ecir.chemical_validity import ChemicalValidity
    from etflow.ecir.mvr_dataset import MCVRMixedDataset

    validity = ChemicalValidity(identities["validity_statistics_path"])
    loaded = {}
    for offset, split in enumerate(SPLITS):
        sources = output_root / "real_sources" / f"{split}.parquet"
        targets = output_root / "minimal_targets" / f"{split}.parquet"
        source_frame = pd.read_parquet(sources)
        count = min(int(sample_count), len(source_frame))
        dataset = MCVRMixedDataset(
            sources,
            targets,
            validity,
            length=count,
            seed=42 + offset * 100_000,
        )
        for index in range(count):
            item = dataset[index]
            for name in (
                "x_input",
                "x_target",
                "active_mode_mask",
                "affected_atom_mask",
                "deterministic_error_features",
            ):
                value = torch.as_tensor(getattr(item, name))
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"{split} MCVR item {index} has non-finite {name}")
        loaded[split] = count
    return loaded


def validate_formal_assets(
    *,
    output_root: Path,
    source_frames: Mapping[str, pd.DataFrame],
    identities: Mapping[str, Any],
    require_complete: bool,
    strict_sample_count: int,
) -> dict[str, Any]:
    """Validate every expected pair and independently strict-load a stable sample."""

    output_root = Path(output_root).expanduser().resolve()
    criteria: dict[str, bool] = {
        "test_records_read_zero": True,
        "builder_config_matches_stage_d": identities.get("target_builder_config")
        == asdict(MinimalValidityConfig()),
        "builder_code_identity_present": bool(identities.get("builder_code_sha256")),
        "train_val_disjoint": True,
        "all_sources_have_exactly_one_target": True,
        "no_unexpected_targets": True,
        "all_target_payloads_strict_valid": True,
        "pairing_manifest_matches_target_manifest": True,
        "manifest_shards_match": True,
        "metadata_identities_match": True,
        "sha256_inventory_valid": True,
        "no_unresolved_failures": failure_count(output_root) == 0,
        "strict_sample_load_pass": True,
        "mcvr_dataset_strict_load_pass": True,
    }
    errors: list[str] = []
    metadata_path = output_root / "minimal_targets" / "metadata.json"
    inventory_path = output_root / "SHA256SUMS.txt"
    inventory: dict[str, str] = {}
    metadata = None
    if require_complete:
        if not metadata_path.is_file():
            criteria["metadata_identities_match"] = False
            errors.append("minimal target metadata is missing")
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_metadata = {
                "test_records_read": 0,
                "builder_code_sha256": identities["builder_code_sha256"],
                "builder_config_sha256": identities["builder_config_sha256"],
                "validity_statistics_identity_sha256": identities[
                    "validity_statistics_identity_sha256"
                ],
                "formal_rdkit_adapter_sha256": identities.get(
                    "formal_rdkit_adapter_sha256"
                ),
            }
            if any(metadata.get(key) != value for key, value in expected_metadata.items()):
                criteria["metadata_identities_match"] = False
                errors.append("minimal target metadata identity mismatch")
        if not inventory_path.is_file():
            criteria["sha256_inventory_valid"] = False
            errors.append("SHA256SUMS.txt is missing")
        else:
            try:
                inventory = _inventory(inventory_path)
            except Exception as error:
                criteria["sha256_inventory_valid"] = False
                errors.append(f"invalid SHA inventory: {error}")
    source_sets = {
        split: set(map(str, frame.get("molecule_id", [])))
        for split, frame in source_frames.items()
    }
    if source_sets.get("train", set()) & source_sets.get("val", set()):
        criteria["train_val_disjoint"] = False
        errors.append("train/val molecule overlap")
    split_results = {}
    strict_candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for split in SPLITS:
        sources = source_frames[split].copy()
        if sources.empty:
            expected_ids: set[str] = set()
        else:
            if sources.sample_id.duplicated().any():
                criteria["all_sources_have_exactly_one_target"] = False
                errors.append(f"duplicate source sample_id in {split}")
            expected_ids = set(map(str, sources.sample_id))
        target_manifest = output_root / "minimal_targets" / f"{split}.parquet"
        pairing_manifest = output_root / "manifests" / f"pairing_{split}.parquet"
        if not target_manifest.is_file() or not pairing_manifest.is_file():
            targets = pd.DataFrame()
            if expected_ids:
                criteria["all_sources_have_exactly_one_target"] = False
                errors.append(f"missing {split} target or pairing manifest")
        else:
            targets = pd.read_parquet(target_manifest)
            pairing = pd.read_parquet(pairing_manifest)
            if targets.to_dict("records") != pairing.to_dict("records"):
                criteria["pairing_manifest_matches_target_manifest"] = False
                errors.append(f"{split} pairing manifest differs from target manifest")
        if not targets.empty and targets.sample_id.duplicated().any():
            criteria["all_sources_have_exactly_one_target"] = False
            errors.append(f"duplicate target sample_id in {split}")
        actual_ids = set(map(str, targets.sample_id)) if not targets.empty else set()
        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids
        if missing:
            criteria["all_sources_have_exactly_one_target"] = False
            errors.append(f"{split} missing {len(missing)} targets")
        if require_complete and extra:
            criteria["no_unexpected_targets"] = False
            errors.append(f"{split} has {len(extra)} unexpected targets")
        sidecars = collect_manifest_rows(output_root, split)
        shards = _shard_rows(output_root, split)
        sidecar_ids = {str(row["sample_id"]) for row in sidecars}
        shard_ids = {str(row["sample_id"]) for row in shards}
        relevant_actual = actual_ids if require_complete else actual_ids & expected_ids
        if not relevant_actual.issubset(sidecar_ids) or not relevant_actual.issubset(shard_ids):
            criteria["manifest_shards_match"] = False
            errors.append(f"{split} sidecar/shard coverage differs")
        if expected_ids:
            target_lookup = targets.set_index("sample_id", drop=False)
            for source in sources.to_dict("records"):
                sample_id = str(source["sample_id"])
                if sample_id not in actual_ids:
                    continue
                target_row = target_lookup.loc[sample_id]
                if isinstance(target_row, pd.DataFrame):
                    criteria["all_sources_have_exactly_one_target"] = False
                    continue
                target_dict = target_row.to_dict()
                try:
                    _strict_pair_check(source, target_dict, identities)
                except Exception as error:
                    criteria["all_target_payloads_strict_valid"] = False
                    errors.append(f"{split}:{sample_id}: {type(error).__name__}: {error}")
                if require_complete:
                    expected_inventory = {
                        str(source["source_path"]): str(source["source_file_sha256"]),
                        str(target_dict["target_cache_path"]): str(
                            target_dict["target_file_sha256"]
                        ),
                    }
                    if any(inventory.get(path) != digest for path, digest in expected_inventory.items()):
                        criteria["sha256_inventory_valid"] = False
                        errors.append(f"{split}:{sample_id}: SHA inventory mismatch")
                strict_candidates.append((source, target_dict))
        if require_complete:
            source_manifest = output_root / "real_sources" / f"{split}.parquet"
            for path in (source_manifest, target_manifest, pairing_manifest):
                if path.is_file() and inventory.get(str(path.resolve())) != file_sha256(path):
                    criteria["sha256_inventory_valid"] = False
                    errors.append(f"{split} manifest SHA inventory mismatch: {path.name}")
                elif not path.is_file():
                    criteria["sha256_inventory_valid"] = False
                    errors.append(f"{split} manifest is missing: {path.name}")
            for path in (output_root / "manifests" / "shards" / split).glob(
                "shard-*.jsonl"
            ):
                if inventory.get(str(path.resolve())) != file_sha256(path):
                    criteria["sha256_inventory_valid"] = False
                    errors.append(f"{split} shard SHA inventory mismatch: {path.name}")
            if metadata is not None:
                persisted_split = metadata.get("splits", {}).get(split, {})
                expected_split = {
                    "target_manifest_sha256": file_sha256(target_manifest)
                    if target_manifest.is_file()
                    else None,
                    "pairing_manifest_sha256": file_sha256(pairing_manifest)
                    if pairing_manifest.is_file()
                    else None,
                }
                if any(
                    persisted_split.get(key) != value
                    for key, value in expected_split.items()
                ):
                    criteria["metadata_identities_match"] = False
                    errors.append(f"{split} metadata manifest identity mismatch")
        split_results[split] = {
            "planned_records": int(len(sources)),
            "target_records": int(len(actual_ids & expected_ids)),
            "planned_molecules": int(sources.molecule_id.nunique()) if not sources.empty else 0,
            "missing_targets": int(len(missing)),
            "unexpected_targets": int(len(extra)) if require_complete else 0,
            "target_manifest_sha256": file_sha256(target_manifest)
            if target_manifest.is_file()
            else None,
            "pairing_manifest_sha256": file_sha256(pairing_manifest)
            if pairing_manifest.is_file()
            else None,
        }
    strict_candidates.sort(
        key=lambda pair: canonical_sha256(
            {"sample_id": pair[0]["sample_id"], "seed": 42}
        )
    )
    for source, target_row in strict_candidates[: int(strict_sample_count)]:
        try:
            _strict_pair_check(source, target_row, identities)
        except Exception as error:
            criteria["strict_sample_load_pass"] = False
            errors.append(
                f"strict sample {source['sample_id']}: {type(error).__name__}: {error}"
            )
    if len(strict_candidates) < min(
        int(strict_sample_count), sum(len(frame) for frame in source_frames.values())
    ):
        criteria["strict_sample_load_pass"] = False
        errors.append("too few completed records for strict sample validation")
    mcvr_dataset_loaded = {"train": 0, "val": 0}
    if require_complete:
        try:
            mcvr_dataset_loaded = strict_mvr_dataset_load(
                output_root, identities, int(strict_sample_count)
            )
        except Exception as error:
            criteria["mcvr_dataset_strict_load_pass"] = False
            errors.append(f"MCVR dataset strict load failed: {type(error).__name__}: {error}")
    passed = all(criteria.values())
    decision = (
        "D1B_FORMAL_TARGETS_READY"
        if require_complete and passed
        else (
            "D1B_FORMAL_TARGET_PILOT_PASS"
            if not require_complete and passed
            else "D1B_FORMAL_TARGETS_NOT_READY"
        )
    )
    result = {
        "schema_version": VALIDATION_SCHEMA,
        "decision": decision,
        "require_complete": bool(require_complete),
        "test_records_read": 0,
        "splits": split_results,
        "criteria": criteria,
        "errors": errors[:1000],
        "unresolved_failure_records": failure_count(output_root),
        "mcvr_dataset_loaded_records": mcvr_dataset_loaded,
        "builder_code_sha256": identities["builder_code_sha256"],
        "builder_config_sha256": identities["builder_config_sha256"],
        "validity_statistics_identity_sha256": identities[
            "validity_statistics_identity_sha256"
        ],
    }
    result["validation_identity_sha256"] = canonical_sha256(result)
    return result


def build_summary(
    *,
    output_root: Path,
    source_metadata: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
    identities: Mapping[str, Any],
    status: str,
    started_at: str,
    elapsed_seconds: float,
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    telemetry_path = output_root / "telemetry" / "runtime_telemetry.csv"
    telemetry = pd.read_csv(telemetry_path) if telemetry_path.is_file() else pd.DataFrame()
    def numeric(name: str) -> pd.Series:
        return pd.to_numeric(
            telemetry.get(name, pd.Series(dtype=float)), errors="coerce"
        ).dropna()

    gpu_memory = numeric("gpu_memory_used_mib")
    gpu_util = numeric("gpu_utilization_percent")
    cpu = numeric("cpu_percent")
    rss = numeric("rss_mib")
    target_seconds = numeric("average_seconds_per_record")
    skipped = numeric("skipped_records")
    completed = sum(int(value["completed_records"]) for value in manifest_metadata.values())
    failures = failure_count(output_root)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": status,
        "decision": (
            "D1B_FORMAL_TARGETS_READY"
            if validation and validation.get("decision") == "D1B_FORMAL_TARGETS_READY"
            else "D1B_FORMAL_TARGETS_NOT_READY"
        ),
        "started_at": started_at,
        "completed_at": utc_now(),
        "total_elapsed_seconds": float(elapsed_seconds),
        "splits": manifest_metadata,
        "planned_records": sum(int(value["records"]) for value in source_metadata["splits"].values()),
        "completed_records": completed,
        "successful_records": completed,
        "failed_records": failures,
        "skipped_records": int(skipped.iloc[-1]) if len(skipped) else 0,
        "test_records_read": 0,
        "seconds_per_1000_records": float(elapsed_seconds) * 1000.0 / max(completed, 1),
        "average_seconds_per_target": float(target_seconds.iloc[-1])
        if len(target_seconds)
        else None,
        "average_gpu_memory_mib": float(gpu_memory.mean()) if len(gpu_memory) else None,
        "peak_gpu_memory_mib": float(gpu_memory.max()) if len(gpu_memory) else None,
        "average_gpu_utilization_percent": float(gpu_util.mean()) if len(gpu_util) else None,
        "average_cpu_percent": float(cpu.mean()) if len(cpu) else None,
        "peak_cpu_rss_mib": float(rss.max()) if len(rss) else None,
        "disk_usage_mib": _directory_size(output_root) / 2**20,
        "source_identity_sha256": source_metadata["formal_source_identity_sha256"],
        "builder_code_sha256": identities["builder_code_sha256"],
        "builder_config_sha256": identities["builder_config_sha256"],
        "validity_statistics_identity_sha256": identities["validity_statistics_identity_sha256"],
        "estimated_same_scale_seconds": float(elapsed_seconds) if completed else None,
        "validation": validation,
        "environment": environment_identity(),
    }
    summary["summary_identity_sha256"] = canonical_sha256(summary)
    return summary


def summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# D1-B Formal-Large Target Build Summary",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        f"- Status: `{summary['status']}`",
        f"- Planned records: {summary['planned_records']}",
        f"- Completed records: {summary['completed_records']}",
        f"- Failed records: {summary['failed_records']}",
        f"- Success rate: {summary['successful_records'] / max(summary['planned_records'], 1):.6f}",
        f"- Failure rate: {summary['failed_records'] / max(summary['planned_records'], 1):.6f}",
        f"- Test records read: {summary['test_records_read']}",
        f"- Total elapsed seconds: {summary['total_elapsed_seconds']:.3f}",
        f"- Seconds per 1,000 records: {summary['seconds_per_1000_records']:.3f}",
        f"- Average seconds per target: {summary['average_seconds_per_target']}",
        f"- Average GPU memory MiB: {summary['average_gpu_memory_mib']}",
        f"- Peak GPU memory MiB: {summary['peak_gpu_memory_mib']}",
        f"- Average GPU utilization percent: {summary['average_gpu_utilization_percent']}",
        f"- Disk usage MiB: {summary['disk_usage_mib']:.3f}",
        f"- Source identity SHA256: `{summary['source_identity_sha256']}`",
        f"- Builder code SHA256: `{summary['builder_code_sha256']}`",
        f"- Builder config SHA256: `{summary['builder_config_sha256']}`",
        "",
        "## Splits",
        "",
    ]
    for split, values in summary["splits"].items():
        lines.extend(
            [
                f"### {split}",
                "",
                f"- Planned records: {values['planned_records']}",
                f"- Completed records: {values['completed_records']}",
                f"- Successful records: {values['successful_records']}",
                f"- Failed records: {values['failed_records']}",
                f"- Success rate: {values['success_rate']:.6f}",
                f"- Failure rate: {values['failure_rate']:.6f}",
                f"- Molecules: {values['molecules']}",
                f"- Target manifest SHA256: `{values['target_manifest_sha256']}`",
                f"- Pairing manifest SHA256: `{values['pairing_manifest_sha256']}`",
                f"- Aggregate target SHA256: `{values['aggregate_target_sha256']}`",
                "",
            ]
        )
    return "\n".join(lines)
