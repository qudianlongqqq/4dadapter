from __future__ import annotations

import csv
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from etflow.ecir import formal_target_assets as assets
from etflow.ecir.minimal_validity_target import MinimalValidityConfig


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ecir_mvr_formal_large_minimal_targets.yaml"
PARQUET_AVAILABLE = any(
    importlib.util.find_spec(name) is not None for name in ("pyarrow", "fastparquet")
)


class FakeBuilder:
    def __init__(self):
        self.calls = 0

    def build(self, coordinates, record):
        self.calls += 1
        target = torch.as_tensor(coordinates, dtype=torch.float32).clone()
        return {
            "x_target": target,
            "target_metadata": {
                "target_status": "identity_clean",
                "stop_reason": "already_valid",
                "validity_gain": 0.0,
                "initial_to_target_rmsd": 0.0,
                "max_atom_displacement": 0.0,
                "torsion_change": 0.0,
                "max_rotatable_torsion_change": 0.0,
                "selected_step": 0,
                "target_sha256": assets.tensor_sha256(target),
                "reference_fallback_used": False,
                "force_field_fallback_used": False,
                "optimizer_config": asdict(MinimalValidityConfig()),
            },
        }


def _identities():
    builder_path = ROOT / "etflow/ecir/minimal_validity_target.py"
    validity_path = ROOT / "data/ecir_mvr/validity_reference_stats.json"
    return {
        "builder_code_path": str(builder_path.resolve()),
        "builder_code_sha256": assets.file_sha256(builder_path),
        "builder_config_sha256": assets.canonical_sha256(
            asdict(MinimalValidityConfig())
        ),
        "target_builder_config": asdict(MinimalValidityConfig()),
        "validity_statistics_path": str(validity_path.resolve()),
        "validity_statistics_sha256": assets.file_sha256(validity_path),
        "validity_statistics_identity_sha256": assets.STAGE_D_VALIDITY_IDENTITY,
        "stage_d_target_identity_sha256": "c" * 64,
        "config_file_sha256": "b" * 64,
    }


def _source(tmp_path: Path, split: str, suffix: str):
    sample_id = f"{split}::sample-{suffix}"
    molecule_id = f"molecule-{split}-{suffix}"
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    path = tmp_path / f"{split}-{suffix}.pt"
    torch.save(
        {
            "sample_id": sample_id,
            "mol_id": sample_id,
            "source_record_id": molecule_id,
            "atomic_numbers": torch.tensor([6, 6]),
            "x_init": coordinates,
        },
        path,
    )
    return {
        "schema_version": assets.SOURCE_SCHEMA,
        "split": split,
        "sample_id": sample_id,
        "molecule_id": molecule_id,
        "generator_name": "ETFlow_formal_upstream",
        "source_severity": "normal",
        "source_path": str(path.resolve()),
        "coordinate_path": None,
        "coordinate_key": "x_init",
        "coordinate_sha256": assets.tensor_sha256(coordinates),
        "source_file_sha256": assets.file_sha256(path),
        "num_atoms": 2,
        "test_record": False,
    }


def _write_inventory(output: Path, frames, identities):
    for split, frame in frames.items():
        assets.atomic_parquet(frame, output / "real_sources" / f"{split}.parquet")
    manifest_metadata = assets.finalize_manifests(output, frames, shard_size=1)
    source_metadata = {
        "formal_source_identity_sha256": assets.canonical_sha256(
            {
                split: frame[["sample_id", "source_file_sha256"]].to_dict("records")
                for split, frame in frames.items()
            }
        )
    }
    assets.write_asset_metadata_and_inventory(
        output_root=output,
        source_frames=frames,
        source_metadata=source_metadata,
        manifest_metadata=manifest_metadata,
        identities=identities,
        config_file_sha256="b" * 64,
    )


def test_config_freezes_stage_d_builder_and_formal_counts(tmp_path):
    config = assets.load_config(CONFIG, output_root=tmp_path)
    assert config["target_builder"] == asdict(MinimalValidityConfig())
    assert config["splits"]["train"] == {
        "expected_molecules": 50_000,
        "expected_records_per_molecule": 3,
    }
    assert config["splits"]["val"] == {
        "expected_molecules": 5_000,
        "expected_records_per_molecule": 2,
    }
    assert config["splits"]["test"] == {"enabled": False}
    assert config["pilot_records"] == 100


def test_config_rejects_any_builder_parameter_change(tmp_path):
    config = yaml.safe_load(CONFIG.read_text())
    config["target_builder"]["learning_rate"] = 0.002
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="differs from frozen Stage D"):
        assets.load_config(path, output_root=tmp_path)


def test_stage_d_metadata_and_validity_identities_are_reused(tmp_path):
    config = assets.load_config(CONFIG, output_root=tmp_path)
    identities = assets.verify_stage_d_identities(config)
    assert identities["validity_statistics_identity_sha256"] == assets.STAGE_D_VALIDITY_IDENTITY
    assert len(identities["builder_code_sha256"]) == 64
    assert len(identities["stage_d_target_identity_sha256"]) == 64


def test_test_paths_are_rejected_without_enumeration(tmp_path):
    with pytest.raises(ValueError, match="test path is forbidden"):
        assets.forbid_test_path(tmp_path / "test")
    assert assets.forbid_test_path(tmp_path / "train").name == "train"


def test_target_build_is_atomic_resumable_and_never_recomputed(tmp_path):
    source = _source(tmp_path, "train", "a")
    builder = FakeBuilder()
    identities = _identities()
    row, skipped = assets.build_target(
        source,
        output_root=tmp_path / "output",
        builder=builder,
        identities=identities,
        config_file_sha256="b" * 64,
    )
    assert not skipped and builder.calls == 1
    target_path = Path(row["target_cache_path"])
    original_mtime = target_path.stat().st_mtime_ns
    second, skipped = assets.build_target(
        source,
        output_root=tmp_path / "output",
        builder=builder,
        identities=identities,
        config_file_sha256="b" * 64,
    )
    assert skipped and builder.calls == 1
    assert second == row
    assert target_path.stat().st_mtime_ns == original_mtime
    assert not list(target_path.parent.glob("*.tmp.*"))


def test_partial_target_state_is_rejected(tmp_path):
    source = _source(tmp_path, "train", "partial")
    output = tmp_path / "output"
    target, _ = assets.target_paths(output, "train", source["sample_id"])
    target.parent.mkdir(parents=True)
    torch.save({"partial": True}, target)
    with pytest.raises(ValueError, match="partial target state"):
        assets.build_target(
            source,
            output_root=output,
            builder=FakeBuilder(),
            identities=_identities(),
            config_file_sha256="b" * 64,
        )


def test_failure_reason_is_persisted_and_can_be_resolved(tmp_path):
    source = _source(tmp_path, "train", "failure")
    output = tmp_path / "output"
    assets.record_failure(source, output, RuntimeError("first"))
    assets.record_failure(source, output, ValueError("second"))
    assert assets.failure_count(output) == 1
    path = next((output / "manifests/failures/train").glob("*.json"))
    value = json.loads(path.read_text())
    assert [row["error"] for row in value["attempts"]] == ["first", "second"]
    assets.clear_failure(source, output)
    assert assets.failure_count(output) == 0


@pytest.mark.skipif(not PARQUET_AVAILABLE, reason="Parquet engine is not installed")
def test_full_validator_requires_pairing_atom_order_sha_and_no_failures(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(assets, "validate_cache_record", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        assets,
        "strict_mvr_dataset_load",
        lambda output_root, identities, sample_count: {"train": 1, "val": 1},
    )
    output = tmp_path / "output"
    train = _source(tmp_path, "train", "one")
    val = _source(tmp_path, "val", "one")
    builder = FakeBuilder()
    identities = _identities()
    for source in (train, val):
        assets.build_target(
            source,
            output_root=output,
            builder=builder,
            identities=identities,
            config_file_sha256="b" * 64,
        )
    frames = {"train": pd.DataFrame([train]), "val": pd.DataFrame([val])}
    _write_inventory(output, frames, identities)
    result = assets.validate_formal_assets(
        output_root=output,
        source_frames=frames,
        identities=identities,
        require_complete=True,
        strict_sample_count=2,
    )
    assert result["decision"] == "D1B_FORMAL_TARGETS_READY"
    assert all(result["criteria"].values())
    assert result["test_records_read"] == 0

    target_path = Path(
        pd.read_parquet(output / "minimal_targets/train.parquet")
        .iloc[0]
        .target_cache_path
    )
    payload = torch.load(target_path, map_location="cpu", weights_only=False)
    payload["source_atomic_numbers"] = torch.tensor([6, 8])
    torch.save(payload, target_path)
    broken = assets.validate_formal_assets(
        output_root=output,
        source_frames=frames,
        identities=identities,
        require_complete=True,
        strict_sample_count=2,
    )
    assert broken["decision"] == "D1B_FORMAL_TARGETS_NOT_READY"
    assert not broken["criteria"]["all_target_payloads_strict_valid"]


@pytest.mark.skipif(not PARQUET_AVAILABLE, reason="Parquet engine is not installed")
def test_pilot_gate_accepts_exact_completed_subset(monkeypatch, tmp_path):
    monkeypatch.setattr(assets, "validate_cache_record", lambda *args, **kwargs: {})
    output = tmp_path / "output"
    sources = [_source(tmp_path, "train", str(index)) for index in range(3)]
    for source in sources:
        assets.build_target(
            source,
            output_root=output,
            builder=FakeBuilder(),
            identities=_identities(),
            config_file_sha256="b" * 64,
        )
    frames = {"train": pd.DataFrame(sources), "val": pd.DataFrame(columns=sources[0])}
    _write_inventory(output, frames, _identities())
    result = assets.validate_formal_assets(
        output_root=output,
        source_frames=frames,
        identities=_identities(),
        require_complete=False,
        strict_sample_count=3,
    )
    assert result["decision"] == "D1B_FORMAL_TARGET_PILOT_PASS"


@pytest.mark.skipif(not PARQUET_AVAILABLE, reason="Parquet engine is not installed")
def test_train_val_overlap_blocks_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(assets, "validate_cache_record", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        assets,
        "strict_mvr_dataset_load",
        lambda output_root, identities, sample_count: {"train": 1, "val": 1},
    )
    output = tmp_path / "output"
    train = _source(tmp_path, "train", "overlap")
    val = _source(tmp_path, "val", "overlap")
    val["molecule_id"] = train["molecule_id"]
    record = torch.load(val["source_path"], weights_only=False)
    record["source_record_id"] = train["molecule_id"]
    torch.save(record, val["source_path"])
    val["source_file_sha256"] = assets.file_sha256(val["source_path"])
    for source in (train, val):
        assets.build_target(
            source,
            output_root=output,
            builder=FakeBuilder(),
            identities=_identities(),
            config_file_sha256="b" * 64,
        )
    frames = {"train": pd.DataFrame([train]), "val": pd.DataFrame([val])}
    _write_inventory(output, frames, _identities())
    result = assets.validate_formal_assets(
        output_root=output,
        source_frames=frames,
        identities=_identities(),
        require_complete=True,
        strict_sample_count=2,
    )
    assert not result["criteria"]["train_val_disjoint"]
    assert result["decision"] == "D1B_FORMAL_TARGETS_NOT_READY"


def test_runtime_telemetry_has_frozen_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(assets, "_gpu_metrics", lambda index: {
        "gpu_index": index,
        "gpu_uuid": "GPU-test",
        "gpu_utilization_percent": 1,
        "gpu_memory_used_mib": 2,
        "gpu_memory_total_mib": 3,
        "power_draw_w": 4,
        "temperature_c": 5,
    })
    monitor = assets.RuntimeTelemetry(
        tmp_path, total_records=10, interval=30, gpu_index="1"
    )
    monitor.update(success=True, skipped=False, seconds=0.5)
    monitor.sample()
    with (tmp_path / "telemetry/runtime_telemetry.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == assets.TELEMETRY_FIELDS
        row = next(reader)
    assert row["gpu_index"] == "1" and row["gpu_uuid"] == "GPU-test"
    assert row["completed_records"] == "1"


def test_runtime_telemetry_thread_stops_once_and_writes_final_sample(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(assets, "_gpu_metrics", lambda index: {
        "gpu_index": index,
        "gpu_uuid": "GPU-test",
        "gpu_utilization_percent": 0,
        "gpu_memory_used_mib": 0,
        "gpu_memory_total_mib": 1,
        "power_draw_w": 0,
        "temperature_c": 0,
    })
    monitor = assets.RuntimeTelemetry(
        tmp_path, total_records=1, interval=0.01, gpu_index="0"
    )
    monitor.start()
    monitor.update(success=True, skipped=False, seconds=0.25)
    monitor.stop()
    monitor.stop()
    assert monitor.thread is not None and not monitor.thread.is_alive()
    rows = list(
        csv.DictReader(
            (tmp_path / "telemetry/runtime_telemetry.csv").open(newline="")
        )
    )
    assert rows[-1]["completed_records"] == "1"


def test_runner_only_builds_targets_and_never_starts_training():
    runner = (ROOT / "scripts/run_ecir_mvr_formal_large_target_build.sh").read_text()
    assert "build_ecir_mvr_formal_large_targets.py" in runner
    assert "train_ecir" not in runner
    assert "test" not in runner.lower()
    builder = (ROOT / "scripts/build_ecir_mvr_formal_large_targets.py").read_text()
    assert "D1B_FORMAL_TARGET_PILOT_PASS" in builder
    assert "MinimalValidityTargetBuilder" in builder
    assert "build_real_error_target" not in builder
