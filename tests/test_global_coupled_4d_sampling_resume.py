import json
import sys
from types import SimpleNamespace

import pytest
import torch

from etflow.commons.global_coupled_4d_sampling import (
    atomic_json_save,
    atomic_torch_save,
    checkpoint_inference_identity,
)
from etflow.data.flexbond_eval_manifest import build_manifest_aware_sample_payload
from scripts.eval_flexbond_optimizer import _load_method_records
from scripts.sample_global_coupled_4d_flow import _validate_resume_records
from scripts import sample_global_coupled_4d_flow as sampler


def _manifest():
    return {
        "manifest_version": "1.0",
        "records": [
            {
                "mol_id": "mol-1",
                "sample_id": "sample-1",
                "x_init_hash": "hash-1",
                "num_rotatable_bonds": 1,
            },
            {
                "mol_id": "mol-2",
                "sample_id": "sample-2",
                "x_init_hash": "hash-2",
                "num_rotatable_bonds": 1,
            },
        ],
    }


def _inference():
    return {
        f"sample-{index}": SimpleNamespace(
            source_mol_id=f"mol-{index}",
            sample_id=f"sample-{index}",
            x_init_hash=f"hash-{index}",
            num_rotatable_bonds=torch.tensor([1]),
        )
        for index in (1, 2)
    }


def _record(index):
    return {
        "mol_id": f"sample-{index}",
        "source_mol_id": f"mol-{index}",
        "sample_id": f"sample-{index}",
        "x_init_hash": f"hash-{index}",
        "method_name": "global_coupled_4d_adapter",
        "status": "success",
        "x_refined": torch.zeros(2, 3),
    }


def test_partial_payload_is_ordered_resumable_and_rejected_by_evaluator(tmp_path):
    manifest = _manifest()
    selected = {**manifest, "records": manifest["records"][:1]}
    partial = build_manifest_aware_sample_payload(
        records=[_record(1)],
        manifest=manifest,
        manifest_path=tmp_path / "manifest.json",
        selected_manifest=selected,
        split="test",
        inference_cache_path=tmp_path / "cache",
        inference_by_id=_inference(),
        extra={"partial": True, "run_identity": {"alpha": 0.2}},
    )
    path = tmp_path / "partial_samples.pt"
    atomic_torch_save(partial, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _validate_resume_records(loaded["records"], manifest["records"])
    with pytest.raises(ValueError, match="Partial sample payloads"):
        _load_method_records(
            path,
            "global_coupled_4d_adapter",
            manifest,
            manifest_path=tmp_path / "manifest.json",
            split="test",
            inference_cache_path=tmp_path / "cache",
            inference_by_id=_inference(),
        )

    loaded["records"][0]["sample_id"] = "sample-2"
    with pytest.raises(ValueError, match="ordered manifest prefix"):
        _validate_resume_records(loaded["records"], manifest["records"])


def test_atomic_state_and_checkpoint_content_identity(tmp_path):
    state_path = tmp_path / "sampling_state.json"
    atomic_json_save({"completed_count": 1}, state_path)
    assert json.loads(state_path.read_text(encoding="utf-8"))["completed_count"] == 1

    state_dict = {"layer.weight": torch.arange(6).reshape(2, 3).float()}
    first = tmp_path / "step5000.ckpt"
    second = tmp_path / "last.ckpt"
    torch.save({
        "state_dict": state_dict,
        "global_step": 5000,
        "hyper_parameters": {"hidden_dim": 32},
        "optimizer_states": [{"different": 1}],
    }, first)
    torch.save({
        "state_dict": state_dict,
        "global_step": 5000,
        "hyper_parameters": {"hidden_dim": 32},
        "optimizer_states": [{"different": 2}],
    }, second)
    first_identity = checkpoint_inference_identity(first)
    second_identity = checkpoint_inference_identity(second)
    assert first_identity["file_sha256"] != second_identity["file_sha256"]
    assert first_identity["inference_sha256"] == second_identity["inference_sha256"]


@pytest.mark.parametrize(
    ("partial_format", "save_every"),
    [("legacy", 1), ("chunked", 1), ("chunked", 2)],
)
def test_sampler_resumes_after_one_molecule_without_duplicate_or_omission(
    tmp_path, monkeypatch, partial_format, save_every
):
    atomic_numbers = torch.tensor([6, 8])
    x_init = torch.zeros(2, 3)
    from etflow.data.flexbond_cache_schema import x_init_sha256

    hashes = {
        f"sample-{index}": x_init_sha256(x_init + index, atomic_numbers)
        for index in (1, 2)
    }
    manifest = _manifest()
    for row in manifest["records"]:
        row["x_init_hash"] = hashes[row["sample_id"]]

    class Data:
        def __init__(self, index):
            self.mol_id = f"sample-{index}"
            self.sample_id = f"sample-{index}"
            self.source_mol_id = f"mol-{index}"
            self.smiles = "CO"
            self.atomic_numbers = atomic_numbers
            self.x_init = x_init + index
            self.x_init_hash = hashes[self.sample_id]
            self.num_rotatable_bonds = torch.tensor([1])

        def to(self, device):
            return self

    inference = {f"sample-{index}": Data(index) for index in (1, 2)}
    fail_second = {"enabled": True}

    class FakeModel:
        motion_mode = "global_coupled_4d_joint_deformation"

        def __init__(self):
            self.calls = 0
            self.topology_cache = SimpleNamespace(
                stats=SimpleNamespace(hits=0, misses=1)
            )

        def to(self, device):
            return self

        def eval(self):
            return self

        def refine(self, data, *args, **kwargs):
            self.calls += 1
            if fail_second["enabled"] and data.sample_id == "sample-2":
                raise RuntimeError("simulated interruption")
            diagnostics = {
                "stable": True,
                "failure_reason": "",
                "trajectory": [],
                "update_scale": 0.2,
                "joint_mode": "full_4d",
                "solver_fallback_rate": 1.0,
                "solver_backend_counts": {"svd_fallback": 10},
                "devices": {"backbone": "cpu", "jacobian": "cpu", "gram": "cpu", "solver": "cpu"},
                "step_times": [0.01] * 10,
                "mean_step_time": 0.01,
                "mean_timing": {"svd_time": 0.001},
                "preparation_timing": {"cache_hit": False},
                "topology_cache_hit_rate": 0.0,
            }
            return data.x_init + 0.01, diagnostics

    monkeypatch.setattr(sampler, "FlexBondInferenceDataset", lambda *args: list(inference.values()))
    monkeypatch.setattr(sampler, "load_eval_manifest", lambda path: manifest)
    monkeypatch.setattr(sampler, "validate_dataset_against_manifest", lambda *args: inference)
    monkeypatch.setattr(sampler, "file_sha256", lambda path: "file-hash")
    monkeypatch.setattr(
        sampler,
        "checkpoint_inference_identity",
        lambda path: {
            "path": str(tmp_path / "model.ckpt"),
            "file_sha256": "checkpoint-file-hash",
            "inference_sha256": "checkpoint-inference-hash",
            "global_step": 5000,
        },
    )
    monkeypatch.setattr(
        sampler.GlobalCoupled4DFlowLightningModule,
        "load_from_checkpoint",
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(sampler, "collect_run_provenance", lambda **kwargs: {})
    output = tmp_path / "group" / "samples.pt"
    argv = [
        "sample_global_coupled_4d_flow.py",
        "--checkpoint", str(tmp_path / "model.ckpt"),
        "--config", str(tmp_path / "config.yaml"),
        "--cache_dir", str(tmp_path / "cache"),
        "--manifest", str(tmp_path / "manifest.json"),
        "--output", str(output),
        "--max_molecules", "2",
        "--update_scale", "0.2",
        "--device", "cpu",
        "--partial_format", partial_format,
        "--save_every_records", str(save_every),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        sampler.main()
    partial_path = output.parent / "partial_samples.pt"
    if partial_format == "legacy":
        partial = torch.load(partial_path, map_location="cpu", weights_only=False)
        assert [row["sample_id"] for row in partial["records"]] == ["sample-1"]
    else:
        assert not partial_path.exists()
        first_chunk = output.parent / "partial_chunks/chunk_000000.pt"
        if save_every == 1:
            chunk = torch.load(
                first_chunk,
                map_location="cpu",
                weights_only=False,
            )
            assert [row["sample_id"] for row in chunk["records"]] == ["sample-1"]
        else:
            assert not first_chunk.exists()

    fail_second["enabled"] = False
    monkeypatch.setattr(sys, "argv", argv)
    sampler.main()
    final = torch.load(output, map_location="cpu", weights_only=False)
    assert [row["sample_id"] for row in final["records"]] == [
        "sample-1", "sample-2"
    ]
    assert not partial_path.exists()
    loaded_records, missing, failed = _load_method_records(
        output,
        "global_coupled_4d_adapter",
        manifest,
        manifest_path=tmp_path / "different-evaluator-manifest-path.json",
        split="test",
        inference_cache_path=tmp_path / "different-evaluator-cache-path",
        inference_by_id=inference,
    )
    assert list(loaded_records) == ["sample-1", "sample-2"]
    assert missing == [] and failed == []
    state = json.loads((output.parent / "sampling_state.json").read_text())
    assert state["status"].lower() == "completed" and state["completed_count"] == 2
    if partial_format == "chunked":
        assert state["completed_chunk_count"] == (2 if save_every == 1 else 1)
        assert "completed_ordered_sample_ids" not in state

    monkeypatch.setattr(
        sampler.GlobalCoupled4DFlowLightningModule,
        "load_from_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("completed output should skip model loading")
        ),
    )
    changed_argv = list(argv)
    changed_argv[changed_argv.index("--update_scale") + 1] = "0.3"
    monkeypatch.setattr(sys, "argv", changed_argv)
    with pytest.raises(ValueError, match="different sampling command"):
        sampler.main()
    monkeypatch.setattr(sys, "argv", argv)
    sampler.main()
    resumed_state = json.loads(
        (output.parent / "sampling_state.json").read_text(encoding="utf-8")
    )
    assert resumed_state["status"] == "COMPLETED"
    assert resumed_state["resumed_completed_output"] is True
