import json
import importlib
import sys
from types import ModuleType, SimpleNamespace
from importlib.machinery import ModuleSpec

import pytest
import torch

from etflow.commons.global4d_performance import (
    StageAccumulator,
    compact_json,
    recover_record_chunks,
    run_save_policy_benchmark,
    synthetic_sample_record,
)
from scripts import profile_global4d_sampling as profiler
from scripts import analyze_global4d_record_distribution as distribution
from scripts import benchmark_global4d_sampling_io as io_benchmark


def test_full_rewrite_is_quadratic_while_chunks_are_linear_and_recoverable(tmp_path):
    records = [synthetic_sample_record(index, atoms=5) for index in range(11)]
    full = run_save_policy_benchmark(
        records, tmp_path / "full", save_every=1, mode="full_rewrite"
    )
    chunks = run_save_policy_benchmark(
        records, tmp_path / "chunk", save_every=3, mode="chunk"
    )
    assert full["save_count"] == 11
    assert chunks["save_count"] == 4
    assert full["total_serialized_bytes"] > 3 * full["final_partial_bytes"]
    recovered = recover_record_chunks(tmp_path / "chunk")
    assert [row["sample_id"] for row in recovered] == [
        row["sample_id"] for row in records
    ]
    chunk_state = json.loads(
        (tmp_path / "chunk" / "sampling_state.json").read_text()
    )
    assert "completed_ordered_sample_ids" not in chunk_state
    assert chunk_state["last_chunk"].endswith(".pt")


def test_save_frequency_and_disable_partial_policy():
    decisions = [
        profiler.should_save_partial(False, 3, index, 8) for index in range(1, 9)
    ]
    assert decisions == [False, False, True, False, False, True, False, True]
    assert not any(
        profiler.should_save_partial(True, 1, index, 3) for index in range(1, 4)
    )
    with pytest.raises(ValueError):
        profiler.should_save_partial(False, 0, 1, 1)


def test_profiled_refine_does_not_change_model_coordinates():
    class Model:
        def refine(self, data, **kwargs):
            return data.x_init * 2 + 1, {"stable": True}

    data = SimpleNamespace(x_init=torch.arange(6, dtype=torch.float32).reshape(2, 3))
    expected, _ = Model().refine(data)
    actual, diagnostics, timing = profiler.profiled_refine(
        Model(),
        data,
        device="cpu",
        cuda_sync_timing=False,
        refinement_steps=10,
        update_scale=0.2,
        max_displacement=0.1,
        max_coordinate_norm=1000.0,
    )
    torch.testing.assert_close(actual, expected)
    assert diagnostics["stable"] is True
    assert set(timing) == {
        "cpu_wall_seconds",
        "cuda_seconds",
        "synchronize_seconds",
    }


def test_compact_report_rejects_large_record_fields_and_stage_fields_are_complete(tmp_path):
    stages = StageAccumulator()
    stages.add("rollout", calls=2, cpu_wall_seconds=1.0, cuda_seconds=0.75)
    row = stages.compact(records=2, steps=20, total_seconds=2.0)[0]
    assert {
        "stage",
        "calls",
        "cpu_wall_seconds",
        "cuda_seconds",
        "self_seconds",
        "seconds_per_record",
        "seconds_per_refinement_step",
        "wall_time_fraction",
    }.issubset(row)
    compact_json({"summary": {"count": 2}}, tmp_path / "compact.json")
    with pytest.raises(ValueError, match="large fields"):
        compact_json({"records": [{"large": True}]}, tmp_path / "bad.json")


def test_profiler_cli_runs_two_fake_records_without_writes(tmp_path, monkeypatch):
    manifest = {
        "manifest_version": "1.0",
        "records": [
            {
                "mol_id": "mol-1",
                "sample_id": f"sample-{index}",
                "x_init_hash": f"hash-{index}",
                "num_rotatable_bonds": 1,
            }
            for index in (1, 2)
        ],
    }

    class Data:
        def __init__(self, index):
            self.sample_id = f"sample-{index}"
            self.atomic_numbers = torch.tensor([6, 6, 8])
            self.num_rotatable_bonds = torch.tensor(1)
            self.x_init = torch.full((3, 3), float(index))

        def to(self, device):
            return self

    by_id = {f"sample-{index}": Data(index) for index in (1, 2)}

    class FakeModel:
        @classmethod
        def load_from_checkpoint(cls, *args, **kwargs):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def refine(self, data, **kwargs):
            return data.x_init + 0.25, {
                "stable": True,
                "mean_timing": {
                    "egnn_forward_time": 0.001,
                    "jacobian_construction_time": 0.0002,
                    "svd_time": 0.0003,
                    "cartesian_projection_time": 0.0001,
                },
                "preparation_timing": {"total_preparation_time": 0.0001},
                "solver_backend_counts": {"svd_fallback": 10},
                "linear_algebra": [
                    {
                        "rollout_step": step,
                        "graph": 0,
                        "num_atoms": 3,
                        "num_joints": 1,
                        "jacobian_rows": 9,
                        "jacobian_columns": 4,
                        "effective_rank": 3,
                        "condition_number": 2.0,
                        "solver_backend": "svd_fallback",
                        "solver_fallback_count": 1,
                        "attempted_backends": ["rank_check", "svd"],
                        "timing": {"svd_time": 0.0003},
                    }
                    for step in range(10)
                ],
            }

    if "torch_cluster" not in sys.modules:
        cluster = ModuleType("torch_cluster")
        cluster.__spec__ = ModuleSpec("torch_cluster", loader=None)
        cluster.radius_graph = lambda x, r, **kwargs: torch.empty(
            (2, 0), dtype=torch.long, device=x.device
        )
        monkeypatch.setitem(sys.modules, "torch_cluster", cluster)
    manifest_module = importlib.import_module("etflow.data.flexbond_eval_manifest")
    inference_module = importlib.import_module("etflow.data.flexbond_inference_dataset")
    monkeypatch.setattr(manifest_module, "load_eval_manifest", lambda path: manifest)
    monkeypatch.setattr(
        manifest_module,
        "validate_dataset_against_manifest",
        lambda dataset, selected: by_id,
    )
    monkeypatch.setattr(inference_module, "FlexBondInferenceDataset", lambda *args: [])
    fake_model_module = ModuleType("etflow.models.global_coupled_4d_flow")
    fake_model_module.GlobalCoupled4DFlowLightningModule = FakeModel
    monkeypatch.setitem(
        sys.modules, "etflow.models.global_coupled_4d_flow", fake_model_module
    )

    config = tmp_path / "config.yaml"
    config.write_text("model: {}\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "profile"
    argv = [
        "profile_global4d_sampling.py",
        "--checkpoint",
        str(tmp_path / "unused.ckpt"),
        "--config",
        str(config),
        "--cache_dir",
        str(tmp_path / "cache"),
        "--manifest",
        str(manifest_path),
        "--device",
        "cpu",
        "--max_records",
        "2",
        "--warmup_records",
        "0",
        "--disable_partial_save",
        "--skip_batch_benchmark",
        "--output_dir",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    profiler.main()
    payload = json.loads((output / "global4d_sampling_profile.json").read_text())
    assert payload["counts"]["profiled_records"] == 2
    assert payload["save_policy"]["enabled"] is False
    assert not (output / "profile_partial_samples.pt").exists()
    assert (output / "global4d_sampling_profile.csv").is_file()
    assert all("records" not in key for key in payload)


def test_io_benchmark_cli_is_explicitly_bounded(tmp_path, monkeypatch):
    output = tmp_path / "io"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_global4d_sampling_io.py",
            "--records",
            "4",
            "--atoms",
            "3",
            "--save_every_records",
            "1",
            "2",
            "--output_dir",
            str(output),
        ],
    )
    io_benchmark.main()
    payload = json.loads(
        (output / "global4d_sampling_io_benchmark.json").read_text()
    )
    assert payload["record_count"] == 4
    assert max(row["save_count"] for row in payload["policies"]) <= 4


def test_distribution_report_counts_manifest_records_without_raw_payload(tmp_path, monkeypatch):
    manifest = {
        "records": [
            {
                "mol_id": "mol-1" if index < 2 else "mol-2",
                "sample_id": f"sample-{index}",
                "x_init_hash": f"hash-{index}",
                "num_rotatable_bonds": index + 1,
            }
            for index in range(3)
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cache = tmp_path / "cache" / "test"
    cache.mkdir(parents=True)
    for index, row in enumerate(manifest["records"]):
        torch.save(
            {
                **row,
                "source_mol_id": row["mol_id"],
                "atomic_numbers": torch.tensor([6, 6, 8]),
            },
            cache / f"sample-{index}.pt",
        )
    output = tmp_path / "distribution"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_global4d_record_distribution.py",
            "--manifest",
            str(manifest_path),
            "--cache_dir",
            str(tmp_path / "cache"),
            "--output_dir",
            str(output),
        ],
    )
    distribution.main()
    payload = json.loads((output / "global4d_record_distribution.json").read_text())
    assert payload["counts"]["independent_molecules"] == 2
    assert payload["counts"]["generated_records"] == 3
    assert "records" not in payload
    assert (output / "global4d_record_distribution.csv").is_file()
