import pytest
import torch
from pathlib import Path
import yaml

from etflow.commons.global_coupled_4d_jacobian import apply_global_coupled_4d_jacobian
from etflow.models.global_coupled_4d_flow import GlobalCoupled4DFlowLightningModule


def batch(with_reference=False):
    value = {
        "x_init": torch.tensor([[0., 0, 0], [1., 0, 0], [1., 1, 0.], [2., 1, .5]]),
        "node_attr": torch.randn(4, 10),
        "edge_index": torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        "edge_attr": torch.zeros(6, 1),
        "rotatable_bond_index": torch.tensor([[1, 2], [2, 3]]),
        "batch": torch.zeros(4, dtype=torch.long),
    }
    if with_reference:
        value["x_ref_aligned"] = value["x_init"] + torch.tensor(
            [[0., .1, 0], [0., 0, .1], [.1, 0, 0], [0., -.1, .1]]
        )
    return value


def model():
    torch.manual_seed(5)
    return GlobalCoupled4DFlowLightningModule(
        hidden_dim=24, edge_hidden_dim=24, time_embedding_dim=16, num_layers=2
    )


def test_forward_uses_complete_mapping_and_orthogonal_residual():
    network = model()
    output = network(batch())
    assert output["q"].shape == (2, 4) and output["v_final"].shape == (4, 3)
    detail = output["_graph_details"][0]
    mapped, _ = apply_global_coupled_4d_jacobian(
        batch()["x_init"], output["q"], detail["topology"]
    )
    torch.testing.assert_close(mapped, output["v_internal"])
    normal = detail["jacobian"].T @ output["v_residual"].reshape(-1)
    assert torch.linalg.norm(normal) < 2e-4


def test_stretch_invariant_omega_and_velocity_rotate_equivariantly():
    network = model().eval()
    data = batch()
    first = network(data)
    rotation = torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1.]])
    moved = dict(data)
    moved["x_init"] = data["x_init"] @ rotation.T + torch.tensor([2., -1, .5])
    second = network(moved)
    torch.testing.assert_close(second["q"][:, 0], first["q"][:, 0], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(second["q"][:, 1:], first["q"][:, 1:] @ rotation.T,
                               atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(second["v_final"], first["v_final"] @ rotation.T,
                               atol=3e-4, rtol=3e-4)


def test_empty_joint_degrades_to_cartesian_residual():
    network = model()
    data = batch(); data["rotatable_bond_index"] = torch.empty((2, 0), dtype=torch.long)
    output = network(data)
    assert output["q"].numel() == 0 and not output["v_internal"].any()
    torch.testing.assert_close(output["v_final"], output["v_residual"])


def test_ablation_modes_and_training_sampling_mapping_are_finite():
    network = model()
    data = batch(with_reference=True)
    for mode in ("full_4d", "torsion_only", "angular_only", "stretch_only", "internal_zero"):
        output = network(data, joint_mode=mode)
        assert torch.isfinite(output["v_final"]).all()
        if mode == "internal_zero":
            assert not output["v_internal"].any()
    loss = network._shared_step(data, "train")
    assert torch.isfinite(loss)
    loss.backward()
    assert network.backbone.joint_head[-1].weight.grad is not None
    assert torch.isfinite(network.backbone.joint_head[-1].weight.grad).all()
    assert network.backbone.cartesian_layer_weights.grad is not None
    assert torch.isfinite(network.backbone.cartesian_layer_weights.grad).all()


def test_topology_is_cached_across_rollout_steps():
    network = model()
    data = batch()
    refined, diagnostics = network.refine(data, refinement_steps=3, update_scale=.1)
    assert diagnostics["stable"] and torch.isfinite(refined).all()
    assert network.topology_cache.stats.misses == 1
    assert network.topology_cache.stats.hits == 0
    assert diagnostics["preparation_timing"]["cache_hit"] is False
    assert diagnostics["topology_cache_hit_rate"] == pytest.approx(2 / 3)


def test_optimized_rollout_is_numerically_equivalent_to_reference_path():
    network = model().eval()
    data = batch()
    reference, reference_diagnostics = network.refine(
        data,
        refinement_steps=3,
        update_scale=.2,
        save_trajectory_metrics=True,
        use_rollout_cache=False,
        optimized=False,
    )
    network.topology_cache.clear()
    optimized, optimized_diagnostics = network.refine(
        data,
        refinement_steps=3,
        update_scale=.2,
        save_trajectory_metrics=True,
        use_rollout_cache=True,
        optimized=True,
    )
    torch.testing.assert_close(optimized, reference, atol=2e-6, rtol=2e-6)
    assert optimized_diagnostics["solver_backend_counts"] == {"svd_fallback": 3}
    assert max(
        row["orthogonality_error"] for row in optimized_diagnostics["trajectory"]
    ) < 2e-4
    assert max(
        row["reconstruction_error"] for row in optimized_diagnostics["trajectory"]
    ) < 1e-6


def test_one_command_pipeline_has_fixed_smoke_and_formal_budget():
    root = Path(__file__).resolve().parents[1]
    unified = (root / "scripts/run_global_coupled_4d_smoke_and_matched.sh").read_text(encoding="utf-8")
    formal = (root / "scripts/run_global_coupled_4d_formal_matched.sh").read_text(encoding="utf-8")
    config = yaml.safe_load((root / "configs/global_coupled_4d_local025_matched.yaml").read_text(encoding="utf-8"))
    assert "set -Eeuo pipefail" in unified
    assert "run_global_coupled_4d_smoke.sh" in unified
    assert "run_global_coupled_4d_formal_matched.sh" in unified
    assert "train_flexbond" not in unified and "train_jacobian_4d" not in unified
    assert "--max_steps 5000" in formal
    assert "1000,2000,3000,4000,5000" in formal
    assert "checkpoints=(step1000 step2000 step3000 step4000 step5000 last)" in formal
    assert config["trainer"]["max_steps"] == 5000
    assert config["data"]["batch_size"] == 4
    assert config["trainer"]["accumulate_grad_batches"] == 2
    assert config["optimizer"]["lr"] == 0.0002


def test_ablation_runner_has_exact_twelve_new_model_groups():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/run_global_coupled_4d_ablation_all.sh").read_text(encoding="utf-8")
    for mode in ("full_4d", "torsion_only", "bending_torsion", "angular_only", "stretch_only", "internal_zero"):
        assert mode in script
    assert "for alpha_code in 02 05" in script
    assert "train_flexbond" not in script and "train_jacobian_4d" not in script
