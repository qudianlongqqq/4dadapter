import torch

from etflow.models.gated_kinematic_flow import GatedKinematicFlowLightningModule
from etflow.models.flexbond_optimizer import FlexBondOptimizerLightningModule
from etflow.models.motion_factory import build_motion_model


def batch():
    return {"x_init":torch.tensor([[0.,0,0],[1.,0,0],[1.,1,0.],[2.,1,.5]]),
        "node_attr":torch.randn(4,10),"edge_index":torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]]),
        "edge_attr":torch.zeros(6,1),"rotatable_bond_index":torch.tensor([[1,2],[2,3]]),
        "batch":torch.zeros(4,dtype=torch.long)}


def model():
    torch.manual_seed(4);return GatedKinematicFlowLightningModule(hidden_dim=24,edge_hidden_dim=24,time_embedding_dim=16,num_layers=2)


def test_gate_overrides_and_bounded_rate():
    network=model();data=batch();learned=network(data);zero=network(data,gate_override="all_zero");one=network(data,gate_override="all_one")
    assert not zero["v_kin"].any();torch.testing.assert_close(one["effective_torsion_rate"],one["bounded_torsion_rate"])
    assert float(learned["bounded_torsion_rate"].detach().abs().max())<=network.hparams.torsion_rate_scale


def test_gate_and_rate_head_receive_finite_gradients():
    network=model();output=network(batch());loss=output["effective_torsion_rate"].square().mean()+output["gate"].mean();loss.backward()
    final=network.backbone.motion_head[-1]
    assert final.weight.grad is not None and torch.isfinite(final.weight.grad).all()
    assert bool((final.weight.grad[0].abs().sum()>0)&(final.weight.grad[1].abs().sum()>0))


def test_scalar_outputs_invariant_and_velocity_equivariant():
    network=model().eval();data=batch();rotation=torch.tensor([[0.,-1,0],[1,0,0],[0,0,1.]])
    first=network(data);moved=dict(data);moved["x_init"]=data["x_init"]@rotation.T+torch.tensor([2.,-1,.5])
    second=network(moved)
    torch.testing.assert_close(first["gate"],second["gate"],atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(first["bounded_torsion_rate"],second["bounded_torsion_rate"],atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(second["v_final"],first["v_final"]@rotation.T,atol=2e-4,rtol=2e-4)


def test_no_joint_falls_back_to_cartesian_residual():
    network=model();data=batch();data["rotatable_bond_index"]=torch.empty((2,0),dtype=torch.long)
    output=network(data);assert output["gate"].numel()==0 and not output["v_kin"].any()
    torch.testing.assert_close(output["v_final"],output["v_residual"])


def test_rollout_rebuilds_geometry_and_limits_displacement():
    network=model();data=batch();before=network._last_topology_build_count
    refined,diagnostics=network.refine(data,refinement_steps=5,update_scale=.2,max_displacement=.01,save_trajectory_metrics=True)
    assert network._last_topology_build_count-before==5
    assert len(diagnostics["trajectory"])==5 and torch.isfinite(refined).all()
    assert max(row["update_norm"] for row in diagnostics["trajectory"])<=.010001


def test_zero_update_scale_and_twenty_steps_remain_finite():
    network=model();data=batch();refined,diagnostics=network.refine(data,refinement_steps=20,update_scale=0.0,max_displacement=.01)
    torch.testing.assert_close(refined,data["x_init"]);assert diagnostics["stable"]


def test_large_rate_scale_remains_finite_through_tanh():
    network=model();output=network(batch(),torsion_rate_scale_override=1e4)
    assert torch.isfinite(output["bounded_torsion_rate"]).all() and torch.isfinite(output["v_final"]).all()


def test_motion_factory_preserves_legacy_and_cartesian_modes():
    legacy=build_motion_model({"motion_mode":"legacy_flexbond4d","hidden_dim":16,"edge_hidden_dim":16,"num_layers":1})
    cartesian=build_motion_model({"motion_mode":"cartesian","hidden_dim":16,"edge_hidden_dim":16,"num_layers":1})
    assert isinstance(legacy,FlexBondOptimizerLightningModule) and legacy.optimizer_mode=="flexbond4d_hybrid_optimizer"
    assert cartesian.optimizer_mode=="cartesian_optimizer"


def test_new_checkpoint_class_rejects_legacy_motion_mode():
    try:GatedKinematicFlowLightningModule(motion_mode="legacy_flexbond4d")
    except ValueError:pass
    else:raise AssertionError("new checkpoint class accepted a legacy mode")
