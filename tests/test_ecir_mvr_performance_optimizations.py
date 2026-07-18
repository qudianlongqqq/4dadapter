from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch_geometric.data import Batch, Data

from etflow.ecir import formal_rdkit_adapter
from etflow.ecir.mvr_dataset import (
    FormalAdapterLRU,
    RuntimeCacheStatistics,
    canonical_static_topology_fields,
    formal_adapter_cache_key,
    runtime_statistics_identity,
)
from etflow.ecir.mvr_loss import MCVRLoss
from etflow.ecir.mvr_model import MCVRModel
from etflow.ecir import formal_runtime_readiness as runtime_readiness
from scripts import finalize_ecir_mvr_formal64_config as finalizer
from scripts import train_ecir_mvr_medium_rescue_v2 as formal_training
from scripts import train_ecir_mvr_run_a as train_run_a


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/ecir_mvr_formal_large_d1b_base.yaml"


def _row(**changes):
    values = {
        "sample_id": "train::cache-record",
        "coordinate_sha256": "a" * 64,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _cache_record(**changes):
    values = {
        "sample_id": "train::cache-record",
        "source_record_id": "molecule-a",
        "smiles": "CC",
        "atomic_numbers": torch.tensor([6, 6]),
        "topology_signature": "b" * 64,
    }
    values.update(changes)
    return values


def test_formal_adapter_lru_is_identity_bound_and_evictable(monkeypatch):
    calls = []

    def adapt(record):
        calls.append(record["topology_signature"])
        return {
            **record,
            "_formal_rdkit_adapter_schema": "test",
            "_formal_runtime_value": len(calls),
        }

    monkeypatch.setattr(formal_rdkit_adapter, "adapt_formal_cache_record", adapt)
    cache = FormalAdapterLRU(1)
    first = _cache_record()
    assert cache.adapt(_row(), first)["_formal_runtime_value"] == 1
    assert cache.adapt(_row(), first)["_formal_runtime_value"] == 1
    assert cache.hits == 1 and cache.misses == 1 and len(calls) == 1
    entry = next(iter(cache._values.values()))
    assert entry["schema_version"] == "ecir-mvr-formal-adapter-worker-lru-v1"
    assert entry["feature_version"] == "formal-rdkit-static-v1"
    assert len(entry["identity_sha256"]) == 64
    assert all(name.startswith("_formal_") for name in entry["runtime_fields"])
    assert not any(
        forbidden in json.dumps(entry).lower()
        for forbidden in ("hidden_dim", "model_weight", "embedding")
    )

    changed = _cache_record(topology_signature="c" * 64)
    assert formal_adapter_cache_key(_row(), first) != formal_adapter_cache_key(
        _row(), changed
    )
    for different_row, different_record in (
        (_row(sample_id="train::other"), first),
        (_row(coordinate_sha256="d" * 64), first),
        (_row(), _cache_record(ordered_smiles="[CH3][CH3]")),
        (_row(), _cache_record(atomic_numbers=torch.tensor([6, 7]))),
    ):
        assert formal_adapter_cache_key(
            different_row, different_record
        ) != formal_adapter_cache_key(_row(), first)
    assert cache.adapt(_row(), changed)["_formal_runtime_value"] == 2
    assert cache.adapt(_row(), first)["_formal_runtime_value"] == 3
    assert len(calls) == 3


def test_formal_training_entry_passes_runtime_optimization_parameters(monkeypatch):
    captured = {}

    def dataset_factory(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(train_run_a, "MCVRMixedDataset", dataset_factory)
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config["data"]["runtime_optimizations"] = {
        "formal_adapter_lru_size": 512,
        "precompute_training_topology": True,
    }
    assert formal_training._dataset is train_run_a._dataset
    formal_training._dataset(config, "train", validity=object())
    assert captured["formal_adapter_lru_size"] == 512
    assert captured["precompute_training_topology"] is True

    config["data"].pop("runtime_optimizations")
    formal_training._dataset(config, "train", validity=object())
    assert captured["formal_adapter_lru_size"] == 0
    assert captured["precompute_training_topology"] is False


def test_static_topology_and_statistics_are_versioned_and_model_independent():
    graph = _graph(5, 0.0)
    record = {
        "atomic_numbers": torch.tensor([6, 6, 7, 6, 8]),
        "rotatable_bond_index": graph.rotatable_bond_index,
        "bond_is_in_ring": graph.bond_is_in_ring,
    }
    fields = canonical_static_topology_fields(
        record, graph.edge_index, graph.num_nodes
    )
    assert fields["canonical_static_topology_schema_version"] == (
        "ecir-mvr-static-topology-cache-v1"
    )
    assert fields["canonical_static_topology_feature_version"] == (
        "molecular-static-topology-v1"
    )
    assert len(fields["canonical_static_topology_identity_sha256"]) == 64
    assert not any(
        forbidden in " ".join(fields).lower()
        for forbidden in ("hidden", "weight", "embedding", "layer")
    )

    identity = runtime_statistics_identity(512, True)
    statistics = RuntimeCacheStatistics(2, identity)
    statistics.publish(
        worker_id=1,
        pid=42,
        identity_sha256=identity,
        cache_hits=3,
        cache_misses=2,
        rdkit_adapter_build_count=2,
        topology_build_count=5,
    )
    assert statistics.snapshot()[0]["cache_hits"] == 3
    with pytest.raises(RuntimeError, match="identity changed"):
        statistics.publish(
            worker_id=0,
            pid=41,
            identity_sha256="0" * 64,
            cache_hits=0,
            cache_misses=0,
            rdkit_adapter_build_count=0,
            topology_build_count=0,
        )


def _graph(nodes: int, offset: float) -> Data:
    undirected = [(index, index + 1) for index in range(nodes - 1)]
    directed = undirected + [(right, left) for left, right in undirected]
    edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
    generator = torch.Generator().manual_seed(100 + nodes)
    x_input = torch.randn(nodes, 3, generator=generator) + offset
    x_target = x_input + 0.01 * torch.randn(nodes, 3, generator=generator)
    data = Data(
        num_nodes=nodes,
        node_attr=torch.randn(nodes, 10, generator=generator),
        edge_index=edge_index,
        edge_attr=torch.ones(edge_index.size(1), 1),
        bond_is_in_ring=torch.zeros(edge_index.size(1), dtype=torch.bool),
        rotatable_bond_index=torch.tensor([[1], [2]], dtype=torch.long),
        x_init=x_input,
        x_input=x_input,
        x_target=x_target,
        active_mode_mask=torch.tensor([[1, 1, 0, 0, 1, 0]], dtype=torch.float32),
        affected_atom_mask=torch.ones(nodes),
        deterministic_error_features=torch.tensor(
            [[0.2, 0.1, 0, 0, 0, 0, 0.2, 0.1, 0.1, 0.0]],
            dtype=torch.float32,
        ),
        metadata_availability=torch.ones(1, 4),
        upstream_metadata=torch.zeros(1, 4),
        difficulty_target=torch.tensor([0.25]),
        num_rotatable_bonds=torch.tensor([1]),
    )
    return data


def _with_precomputed(data: Data) -> Data:
    result = data.clone()
    fields = canonical_static_topology_fields(
        {
            "atomic_numbers": torch.full((result.num_nodes,), 6),
            "rotatable_bond_index": result.rotatable_bond_index,
            "bond_is_in_ring": result.bond_is_in_ring,
        },
        result.edge_index,
        result.num_nodes,
    )
    for name, value in fields.items():
        setattr(result, name, value)
    return result


def _maximum_tensor_difference(left, right) -> float:
    values = []
    for key in left:
        if torch.is_tensor(left[key]) and left[key].shape == right[key].shape:
            values.append(
                float((left[key] - right[key]).detach().abs().max())
                if left[key].numel()
                else 0.0
            )
    return max(values, default=0.0)


def test_precomputed_topology_is_numerically_training_equivalent():
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    baseline_batch = Batch.from_data_list([_graph(5, 0.0), _graph(6, 2.0)])
    optimized_batch = Batch.from_data_list(
        [_with_precomputed(_graph(5, 0.0)), _with_precomputed(_graph(6, 2.0))]
    )
    baseline = MCVRModel(**config["model"])
    optimized = MCVRModel(**config["model"])
    optimized.load_state_dict(baseline.state_dict(), strict=True)
    baseline_loss = MCVRLoss(config["loss"])
    optimized_loss = MCVRLoss(config["loss"])
    baseline_optimizer = torch.optim.AdamW(
        baseline.parameters(), lr=2.0e-4, weight_decay=1.0e-6
    )
    optimized_optimizer = torch.optim.AdamW(
        optimized.parameters(), lr=2.0e-4, weight_decay=1.0e-6
    )

    time_value = torch.full((2,), 0.5)
    torch.manual_seed(42)
    baseline_output = baseline(
        baseline_batch, baseline_batch.x_input, time_value
    )
    torch.manual_seed(42)
    optimized_output = optimized(
        optimized_batch, optimized_batch.x_input, time_value
    )
    assert _maximum_tensor_difference(baseline_output, optimized_output) == 0.0

    baseline_optimizer.zero_grad(set_to_none=True)
    optimized_optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(123)
    baseline_losses = baseline_loss(baseline, baseline_batch)
    torch.manual_seed(123)
    optimized_losses = optimized_loss(optimized, optimized_batch)
    assert set(baseline_losses) == set(optimized_losses)
    assert max(
        float(
            (baseline_losses[name] - optimized_losses[name]).detach().abs()
        )
        for name in baseline_losses
    ) == 0.0
    baseline_losses["loss"].backward()
    optimized_losses["loss"].backward()
    gradient_difference = max(
        float((left.grad - right.grad).detach().abs().max())
        for left, right in zip(baseline.parameters(), optimized.parameters())
        if left.grad is not None
    )
    assert gradient_difference <= 1.0e-8
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in baseline.parameters()
        if parameter.grad is not None
    )
    baseline_optimizer.step()
    optimized_optimizer.step()
    parameter_difference = max(
        float((left - right).detach().abs().max())
        for left, right in zip(baseline.parameters(), optimized.parameters())
    )
    assert parameter_difference <= 1.0e-8

    torch.manual_seed(777)
    first = optimized(optimized_batch, optimized_batch.x_input, time_value)
    torch.manual_seed(777)
    second = optimized(optimized_batch, optimized_batch.x_input, time_value)
    assert _maximum_tensor_difference(first, second) == 0.0


def _formal_evidence(tmp_path: Path):
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    source_metadata = tmp_path / "source_metadata.json"
    target_metadata = tmp_path / "target_metadata.json"
    validation = tmp_path / "validation.json"
    identities = {
        "validity_statistics_identity_sha256": "1" * 64,
        "formal_source_identity_sha256": "2" * 64,
        "formal_target_identity_sha256": "3" * 64,
        "builder_code_sha256": "4" * 64,
        "builder_config_sha256": "5" * 64,
        "formal_rdkit_adapter_sha256": "6" * 64,
    }
    source_metadata.write_text(
        json.dumps(
            {
                "formal_source_identity_sha256": identities[
                    "formal_source_identity_sha256"
                ]
            }
        )
    )
    target_metadata.write_text(json.dumps(identities))
    validation.write_text(
        json.dumps(
            {
                "decision": "D1B_FORMAL_TARGETS_READY",
                "test_records_read": 0,
                "criteria": {"complete": True},
                "splits": {
                    "train": {"target_records": 150000},
                    "val": {"target_records": 10000},
                },
            }
        )
    )
    base["data"]["source_metadata"] = str(source_metadata)
    base["data"]["target_metadata"] = str(target_metadata)
    base["data"]["target_validation"] = str(validation)
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False))
    return base_path, identities


def _runtime_evidence(base_path: Path, tmp_path: Path) -> tuple[Path, dict]:
    config = yaml.safe_load(base_path.read_text())
    code = runtime_readiness.runtime_code_identity()
    report = {
        "schema_version": "ecir-mvr-formal-runtime-validation-v1",
        "decision": runtime_readiness.RUNTIME_READY,
        "train_checked": 150000,
        "val_checked": 10000,
        "test_records_read": 0,
        "passed_count": 160000,
        "failed_count": 0,
        "failure_classifications": {},
        "failures": [],
        "base_config_sha256": runtime_readiness.file_sha256(base_path),
        "runtime_adapter_sha256": runtime_readiness.file_sha256(
            ROOT / "etflow/ecir/formal_rdkit_adapter.py"
        ),
        "runtime_code_identity_sha256": code["identity_sha256"],
        "runtime_code_files": code["files"],
        "formal_asset_identities": runtime_readiness.formal_asset_identities(
            config
        ),
        "git_commit": runtime_readiness.git_commit(),
        "formal_target_modified": False,
        "checkpoint_created": False,
    }
    report["runtime_validation_identity_sha256"] = (
        runtime_readiness.canonical_sha256(report)
    )
    path = tmp_path / "D1B_FORMAL_RUNTIME_VALIDATION.json"
    path.write_text(json.dumps(report))
    return path, report


def test_formal64_finalizer_rejects_capacity_and_pins_real_evidence(tmp_path):
    base_path, identities = _formal_evidence(tmp_path)
    runtime_path, runtime_report = _runtime_evidence(base_path, tmp_path)
    report = {
        "status": "D1B_FORMAL_PREFLIGHT_PASS",
        "mode": "formal_preflight",
        "capacity_only": False,
        "target_effective_batch": 64,
        "test_records_read": 0,
        "formal_training_started": False,
        "formal_checkpoint_created": False,
        "config_sha256": finalizer._sha256(base_path),
        "commit_sha": finalizer._git_commit(),
        "frozen_identities": identities,
        "runtime_validation_report_sha256": runtime_readiness.file_sha256(
            runtime_path
        ),
        "runtime_validation_identity_sha256": runtime_report[
            "runtime_validation_identity_sha256"
        ],
        "recommended": {
            "micro_batch_size": 64,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": 64,
        },
    }
    report_path = tmp_path / "formal64_preflight/D1B_FORMAL_PREFLIGHT.json"
    report_path.parent.mkdir()
    report_path.write_text(json.dumps(report))
    output = tmp_path / "D1B_FORMAL_RECOMMENDED_CONFIG.yaml"
    resolved = finalizer.finalize_formal64_config(
        base_path, report_path, output, runtime_path
    )
    assert {key: resolved["training"][key] for key in finalizer.FORMAL64} == (
        finalizer.FORMAL64
    )
    assert resolved["preflight"]["report_sha256"] == finalizer._sha256(
        report_path
    )
    assert resolved["preflight"]["capacity_report_used"] is False
    assert resolved["preflight"]["test_records_read"] == 0
    assert resolved["frozen_identities"] == identities
    assert resolved["runtime_validation"]["report_sha256"] == (
        runtime_readiness.file_sha256(runtime_path)
    )
    assert resolved["data"]["runtime_optimizations"] == {
        "formal_adapter_lru_size": 0,
        "precompute_training_topology": False,
    }

    capacity = copy.deepcopy(report)
    capacity.update(
        {
            "status": "D1B_FORMAL_CAPACITY_PASS",
            "mode": "capacity_only",
            "capacity_only": True,
            "target_effective_batch": 256,
        }
    )
    with pytest.raises(RuntimeError, match="non-capacity"):
        finalizer.validate_formal64_preflight(
            capacity,
            identities,
            base_config_sha256=finalizer._sha256(base_path),
            expected_commit=finalizer._git_commit(),
        )

    old_commit = copy.deepcopy(report)
    old_commit["commit_sha"] = "0" * 40
    with pytest.raises(RuntimeError, match="non-capacity"):
        finalizer.validate_formal64_preflight(
            old_commit,
            identities,
            base_config_sha256=finalizer._sha256(base_path),
            expected_commit=finalizer._git_commit(),
        )
