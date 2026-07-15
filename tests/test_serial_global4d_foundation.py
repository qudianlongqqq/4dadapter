from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from etflow.serial_global4d.cache import (
    LABEL_FIELDS,
    SerialGlobal4DResidualDataset,
    assert_teacher_identity,
    build_stage2_training_record,
    label_free_cartesian_view,
    resolve_cartesian_teacher_selection,
    rollout_frozen_cartesian,
    validate_stage2_inference_record,
    validate_stage2_training_record,
)
from etflow.serial_global4d.model import SerialGlobal4DResidualRefiner
from etflow.serial_global4d.safety import safe_serial_update, trust_region_clip
from scripts.build_serial_global4d_residual_cache import (
    _audit_partial_cache,
    _resume_identity,
)
from scripts.evaluate_serial_global4d_confirm30 import _apply as apply_serial_step
from etflow.serial_global4d.oracle import (
    benefit_aware_gate_target,
    solve_serial_residual_oracle,
)
from etflow.serial_global4d.targets import materialize_stage2_targets


def source_record():
    return {
        "mol_id": "mol-1",
        "sample_id": "sample-1",
        "source_mol_id": "mol-1",
        "atomic_numbers": torch.tensor([6, 6, 8, 1]),
        "node_attr": torch.randn(4, 10),
        "edge_index": torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        ),
        "edge_attr": torch.ones(6, 1),
        "rotatable_bond_mask": torch.tensor([False, False, True, True, False, False]),
        "rotatable_bond_index": torch.tensor([[1], [2]], dtype=torch.long),
        "atom_bond_influence_index": torch.tensor([[2, 3], [0, 0]], dtype=torch.long),
        "num_rotatable_bonds": torch.tensor([1]),
        "x_init": torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.2, 0.0], [3.0, 0.4, 0.2]]
        ),
        "x_init_hash": "source-hash",
        "x_ref": torch.full((4, 3), 50.0),
        "x_ref_aligned": torch.tensor(
            [[0.0, 0.1, 0.0], [1.0, 0.1, 0.0], [2.0, 0.3, 0.1], [3.0, 0.5, 0.3]]
        ),
        "x_ref_candidates": torch.full((8, 3), 99.0),
        "target_velocity": torch.full((4, 3), 77.0),
    }


class LabelRejectingTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()), requires_grad=False)
        self.eval()

    def refine(self, data, **kwargs):
        assert not any(
            key in LABEL_FIELDS or key.startswith("x_ref") for key in data.keys()
        )
        return data.x_init + 0.05, {"stable": True, **kwargs}


def identity(name="teacher-a"):
    return {"checkpoint": name, "identity_sha256": f"sha-{name}"}


def canonical_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_stage1_rollout_receives_a_structurally_label_free_view():
    source = source_record()
    view = label_free_cartesian_view(source)
    assert "x_ref_aligned" not in view
    assert "target_velocity" not in view
    refined, diagnostics = rollout_frozen_cartesian(
        LabelRejectingTeacher(),
        source,
        refinement_steps=10,
        update_scale=0.5,
        max_displacement=0.1,
        max_coordinate_norm=1000.0,
        device="cpu",
    )
    torch.testing.assert_close(refined, source["x_init"] + 0.05)
    assert diagnostics["stable"]


def test_stage2_training_record_uses_frozen_cartesian_output_and_fixed_residual():
    source = source_record()
    x_cart = source["x_init"] + 0.05
    record = build_stage2_training_record(
        source,
        x_cart,
        teacher_sampling_identity=identity(),
        original_manifest_identity="manifest-sha",
        split="train",
    )
    checked = validate_stage2_training_record(record)
    torch.testing.assert_close(checked["x_cart"], x_cart)
    torch.testing.assert_close(checked["u_stage2"], source["x_ref_aligned"] - x_cart)


def test_stage2_cache_refuses_test_training_and_teacher_identity_reuse():
    source = source_record()
    with pytest.raises(ValueError, match="train or val"):
        build_stage2_training_record(
            source,
            source["x_init"] + 0.05,
            teacher_sampling_identity=identity(),
            original_manifest_identity="manifest-sha",
            split="test",
        )
    record = build_stage2_training_record(
        source,
        source["x_init"] + 0.05,
        teacher_sampling_identity=identity(),
        original_manifest_identity="manifest-sha",
        split="val",
    )
    with pytest.raises(ValueError, match="different Cartesian teacher"):
        assert_teacher_identity(record, identity("teacher-b"))


def test_stage2_inference_schema_rejects_every_reference_label():
    source = source_record()
    record = build_stage2_training_record(
        source,
        source["x_init"] + 0.05,
        teacher_sampling_identity=identity(),
        original_manifest_identity="manifest-sha",
        split="val",
    )
    with pytest.raises(ValueError, match="contains labels"):
        validate_stage2_inference_record(record)


def test_damped_oracle_matches_closed_form_solution():
    jacobian = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]]
    )
    residual = torch.tensor([[2.0, 4.0, 6.0]])
    ridge = 0.5
    axes = torch.tensor([[1.0, 0.0, 0.0]])
    result = solve_serial_residual_oracle(jacobian, residual, axes, ridge=ridge)
    expected = torch.linalg.solve(
        jacobian.T @ jacobian + ridge * torch.eye(4),
        jacobian.T @ residual.reshape(-1),
    )
    torch.testing.assert_close(result.q_res_star.reshape(-1), expected)
    torch.testing.assert_close(result.r_j_star.reshape(-1), jacobian @ expected)
    assert torch.isfinite(result.projection_energy_ratio)


def test_rank_deficient_and_no_joint_oracles_are_finite():
    residual = torch.randn(3, 3)
    duplicate = torch.randn(9, 1).repeat(1, 4)
    result = solve_serial_residual_oracle(
        duplicate, residual, torch.tensor([[1.0, 0.0, 0.0]]), ridge=1.0e-5
    )
    assert torch.isfinite(result.q_res_star).all()
    empty = solve_serial_residual_oracle(
        torch.empty(9, 0), residual, torch.empty(0, 3), ridge=1.0e-5
    )
    assert empty.q_res_star.shape == (0, 4)
    torch.testing.assert_close(empty.r_j_star, torch.zeros_like(residual))


def test_benefit_gate_analytic_solution_and_adverse_direction():
    residual = torch.tensor([[2.0, 0.0, 0.0]])
    prediction = torch.tensor([[4.0, 0.0, 0.0]])
    gate, gain, beneficial = benefit_aware_gate_target(residual, prediction, beta=1.0)
    torch.testing.assert_close(gate, torch.tensor(0.5))
    torch.testing.assert_close(gain, torch.tensor(4.0))
    assert bool(beneficial)
    adverse, adverse_gain, adverse_beneficial = benefit_aware_gate_target(
        residual, -prediction, beta=1.0
    )
    torch.testing.assert_close(adverse, torch.tensor(0.0))
    torch.testing.assert_close(adverse_gain, torch.tensor(0.0))
    assert not bool(adverse_beneficial)


def test_benefit_gate_is_always_bounded_and_zero_norm_is_finite():
    residual = torch.randn(7, 5, 3)
    prediction = torch.randn(7, 5, 3)
    gate, gain, beneficial = benefit_aware_gate_target(residual, prediction, beta=0.7)
    assert gate.shape == gain.shape == beneficial.shape == (7,)
    assert torch.isfinite(gate).all()
    assert bool(((gate >= 0) & (gate <= 1)).all())
    zero, zero_gain, _ = benefit_aware_gate_target(torch.zeros(2, 3), torch.zeros(2, 3))
    assert torch.isfinite(zero) and torch.isfinite(zero_gain)


def test_teacher_selection_uses_explicit_paths_and_sha_not_stale_linux_paths(tmp_path):
    import hashlib
    import json

    checkpoint = tmp_path / "step100000.ckpt"
    config = tmp_path / "config.resolved.yaml"
    checkpoint.write_bytes(b"checkpoint")
    config.write_text("model: cartesian\n", encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    selection = tmp_path / "formal_large_best_configs.json"
    selection.write_text(
        json.dumps(
            {
                "configs": {
                    "cartesian": {
                        "checkpoint_path": "/stale/linux/step100000.ckpt",
                        "config_path": "/stale/linux/config.resolved.yaml",
                        "checkpoint_file_sha256": digest(checkpoint),
                        "config_file_sha256": digest(config),
                        "validation_manifest_sha256": "manifest-sha",
                        "selection_split": "validation",
                        "test_used_for_selection": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    resolved_checkpoint, resolved_config, metadata = (
        resolve_cartesian_teacher_selection(
            best_configs=selection, checkpoint=checkpoint, config=config
        )
    )
    assert resolved_checkpoint == checkpoint.resolve()
    assert resolved_config == config.resolve()
    assert metadata["validation_manifest_sha256"] == "manifest-sha"


def test_teacher_selection_rejects_wrong_explicit_checkpoint(tmp_path):
    import hashlib
    import json

    checkpoint = tmp_path / "wrong.ckpt"
    config = tmp_path / "config.yaml"
    checkpoint.write_bytes(b"wrong")
    config.write_text("model: cartesian\n", encoding="utf-8")
    selection = tmp_path / "best.json"
    selection.write_text(
        json.dumps(
            {
                "configs": {
                    "cartesian": {
                        "checkpoint_file_sha256": hashlib.sha256(
                            b"selected"
                        ).hexdigest(),
                        "config_file_sha256": hashlib.sha256(
                            config.read_bytes()
                        ).hexdigest(),
                        "selection_split": "validation",
                        "test_used_for_selection": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checkpoint.*selection SHA256"):
        resolve_cartesian_teacher_selection(
            best_configs=selection, checkpoint=checkpoint, config=config
        )


def test_project_canonical_manifest_identity_is_not_raw_file_hash(tmp_path):
    import hashlib
    import json

    from etflow.formal_large import canonical_sha256

    payload = {"manifest_version": "1.0", "records": [{"sample_id": "x"}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    raw = hashlib.sha256(path.read_bytes()).hexdigest()
    assert canonical_sha256(json.loads(path.read_text(encoding="utf-8"))) != raw


def test_materialized_stage2_targets_reconstruct_jq_and_are_finite():
    source = source_record()
    record = build_stage2_training_record(
        source,
        source["x_init"] + 0.05,
        teacher_sampling_identity=identity(),
        original_manifest_identity="manifest-sha",
        split="train",
    )
    targets = materialize_stage2_targets(record, target_time=0.125, ridge=1.0e-5)
    record.update(targets)
    checked = validate_stage2_training_record(record, require_targets=True)
    assert torch.isfinite(checked["q_res_star"]).all()
    assert torch.isfinite(checked["r_J_star"]).all()
    assert targets["target_reconstruction_error"] < 1.0e-6


def test_train_dataset_refuses_partial_cache_until_completed_marker(tmp_path):
    cache_root = tmp_path / "cache"
    split_dir = cache_root / "train"
    split_dir.mkdir(parents=True)
    record = build_stage2_training_record(
        source_record(),
        source_record()["x_init"] + 0.05,
        teacher_sampling_identity=identity(),
        original_manifest_identity="manifest-sha",
        split="train",
    )
    record.update(materialize_stage2_targets(record, target_time=0.0))
    torch.save(record, split_dir / "00000000.pt")
    with pytest.raises(ValueError, match="COMPLETED.json"):
        SerialGlobal4DResidualDataset(cache_root, "train")

    completed = {
        "status": "COMPLETED",
        "split": "train",
        "record_count": 1,
        "teacher_sampling_identity_sha256": identity()["identity_sha256"],
        "cohort_manifest_sha256": "manifest-sha",
        "cache_manifest_sha256": "cache-manifest-sha",
        "generation_code_commits": ["commit-a"],
    }
    completed["cache_identity_sha256"] = canonical_sha(completed)
    (cache_root / "COMPLETED.json").write_text(json.dumps(completed), encoding="utf-8")
    dataset = SerialGlobal4DResidualDataset(cache_root, "train")
    assert len(dataset) == 1
    assert dataset.completion["status"] == "COMPLETED"


def test_partial_cache_audit_proves_exact_contiguous_prefix(tmp_path):
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    teacher = {
        "checkpoint": "teacher-a",
        "code_commit": "commit-a",
        "identity_sha256": "teacher-sha",
    }
    source = source_record()
    record = build_stage2_training_record(
        source,
        source["x_init"] + 0.05,
        teacher_sampling_identity=teacher,
        original_manifest_identity="manifest-sha",
        split="train",
        pilot_manifest_identity="manifest-sha",
    )
    record.update(materialize_stage2_targets(record, target_time=0.0))
    torch.save(record, split_dir / "00000000.pt")
    rows = [
        {
            "sample_id": "sample-1",
            "mol_id": "mol-1",
            "x_init_hash": "source-hash",
            "num_atoms": 4,
            "num_edges": 6,
            "num_joints": 1,
            "flexibility_cohort": "low",
        }
    ]
    audited = _audit_partial_cache(
        split_dir,
        rows,
        [17],
        limit=1,
        split="train",
        identity=teacher,
        manifest_sha="manifest-sha",
        target_times=[0.0, 0.125, 0.25],
    )
    assert audited[0]["source_dataset_index"] == 17
    (split_dir / "orphan.tmp.1").write_bytes(b"partial")
    with pytest.raises(ValueError, match="unexpected files"):
        _audit_partial_cache(
            split_dir,
            rows,
            [17],
            limit=1,
            split="train",
            identity=teacher,
            manifest_sha="manifest-sha",
            target_times=[0.0, 0.125, 0.25],
        )


def test_resume_identity_allows_only_code_commit_change(tmp_path):
    path = tmp_path / "identity.json"
    previous = {
        "checkpoint": "teacher-a",
        "code_commit": "commit-a",
        "identity_sha256": "identity-a",
    }
    path.write_text(json.dumps(previous), encoding="utf-8")
    candidate = {
        "checkpoint": "teacher-a",
        "code_commit": "commit-b",
        "identity_sha256": "identity-b",
    }
    selected, commits = _resume_identity(candidate, path)
    assert selected == previous
    assert commits == ["commit-a", "commit-b"]
    candidate["checkpoint"] = "teacher-b"
    with pytest.raises(ValueError, match="another teacher/command"):
        _resume_identity(candidate, path)


def _serial_training_batch(no_joints=False):
    source = source_record()
    if no_joints:
        source["rotatable_bond_index"] = torch.empty((2, 0), dtype=torch.long)
        source["rotatable_bond_mask"] = torch.zeros(6, dtype=torch.bool)
        source["atom_bond_influence_index"] = torch.empty((2, 0), dtype=torch.long)
        source["num_rotatable_bonds"] = torch.tensor([0])
    record = build_stage2_training_record(
        source,
        source["x_init"] + 0.05,
        teacher_sampling_identity=identity(),
        original_manifest_identity="manifest-sha",
        split="train",
    )
    record.update(materialize_stage2_targets(record, target_time=0.125))
    record["batch"] = torch.zeros(4, dtype=torch.long)
    return SimpleNamespace(**record)


def _small_serial_model():
    return SerialGlobal4DResidualRefiner(
        hidden_dim=16,
        edge_hidden_dim=16,
        time_embedding_dim=8,
        num_layers=2,
        gate_hidden_dim=8,
    )


def test_serial_model_has_no_cartesian_head_and_delta_is_only_gated_jq():
    model = _small_serial_model()
    assert not any("cartesian" in name for name, _ in model.named_parameters())
    batch = _serial_training_batch()
    pos, time = model.stage2_positions(batch)
    output = model(batch, pos, time)
    torch.testing.assert_close(
        output["delta"],
        model.internal_beta
        * output["gate"][output["atom_batch"], None]
        * output["v_internal"],
    )
    rejected = model(batch, pos, time, gate_override=0.0)
    torch.testing.assert_close(rejected["delta"], torch.zeros_like(pos))


def test_serial_phase_a_real_loss_backward_and_no_joint_are_finite():
    model = _small_serial_model()
    result = model.phase_a_loss(_serial_training_batch())
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    empty = model.phase_a_loss(_serial_training_batch(no_joints=True))
    assert empty["q_pred"].shape == (0, 4)
    assert torch.isfinite(empty["loss"])
    empty["loss"].backward()


def test_serial_phase_b_freezes_everything_except_gate_and_target_is_bounded():
    model = _small_serial_model()
    model.freeze_for_phase_b()
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert trainable and all(name.startswith("gate_head.") for name in trainable)
    result = model.phase_b_loss(_serial_training_batch())
    assert bool(((result["gate_target"] >= 0) & (result["gate_target"] <= 1)).all())
    result["loss"].backward()
    assert all(parameter.grad is not None for parameter in model.gate_head.parameters())


def test_serial_trust_region_clips_atom_and_graph_rms_displacement():
    delta = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    clipped, diagnostics = trust_region_clip(
        delta,
        torch.zeros(2, dtype=torch.long),
        max_atom_displacement=0.5,
        max_graph_rms_displacement=0.25,
        max_internal_velocity_norm=None,
    )
    assert diagnostics["atom_clipped"]
    assert diagnostics["graph_rms_clipped"]
    assert torch.linalg.vector_norm(clipped, dim=-1).max() <= 0.25 + 1.0e-7


def test_geometry_violation_backtracks_until_safe():
    current = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    delta = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    result = safe_serial_update(
        current,
        delta,
        edge_index,
        torch.zeros(3, dtype=torch.long),
        max_atom_displacement=3.0,
        max_graph_rms_displacement=3.0,
        min_nonbond_distance=0.1,
        max_backtracks=4,
    )
    assert result.accepted
    assert result.backtracking_count == 2
    assert torch.isfinite(result.coordinates).all()


def test_backtracking_failure_rejects_and_keeps_coordinates_unchanged():
    current = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    delta = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    result = safe_serial_update(
        current,
        delta,
        edge_index,
        torch.zeros(3, dtype=torch.long),
        max_atom_displacement=3.0,
        max_graph_rms_displacement=3.0,
        min_nonbond_distance=0.1,
        max_backtracks=0,
    )
    assert not result.accepted
    assert result.reject_reason == "bond_stretch"
    torch.testing.assert_close(result.coordinates, current)
    torch.testing.assert_close(result.accepted_delta, torch.zeros_like(delta))


def test_two_step_serial_rollout_recomputes_model_and_jacobian_each_step():
    model = _small_serial_model()
    batch = _serial_training_batch()
    calls = []
    original = model.forward

    def counted(*args, **kwargs):
        calls.append(torch.as_tensor(args[1]).clone())
        return original(*args, **kwargs)

    model.forward = counted
    current = batch.x_cart
    for _ in range(2):
        current, _ = apply_serial_step(
            model,
            batch,
            current,
            gate_override=None,
            full_safety=True,
        )
    assert len(calls) == 2
    assert torch.isfinite(current).all()
