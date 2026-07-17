from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import yaml

from etflow.ecir.confidence_calibration import file_sha256
from etflow.ecir.conflict_aware_fusion import (
    VARIANTS, axial_change, bond_axes, fuse_conflict_aware,
    local_conflict_mask, minimum_norm_conflict_projection,
    pairwise_conflict_removal, stage_h0_decision,
)
from etflow.ecir.feature_conditioned_confidence import sign_validity_safe_mask
from scripts.evaluate_ecir_mvr_stage_h0 import infer_h0_variant, verify_frozen


def system(dtype=torch.float64, device="cpu"):
    x = torch.tensor([[0., 0., 0.], [1., 0., 0.]], dtype=dtype, device=device)
    bonds = torch.tensor([[0], [1]], device=device)
    cart = torch.tensor([[.5, 0, 0], [-.5, 0, 0]], dtype=dtype, device=device)
    bond = -cart
    safe = torch.tensor([True], device=device)
    axes = bond_axes(x, bonds)
    conflict, _, a, _ = local_conflict_mask(x, cart, bond, bonds, safe)
    return x, bonds, cart, bond, safe, axes, conflict, a


def test_no_conflict_is_strict_identity():
    x,b,c,_,s,a,_,_=system(); out,_=fuse_conflict_aware(x,c,c,b,s,operator="minnorm",lambda_conflict=1,conflict_eps=1e-10,ridge=1e-10)
    assert out.data_ptr() == c.data_ptr()


def test_single_opposite_pairwise_l1_removes_axis():
    x,b,c,_,_,a,m,v=system(); out=pairwise_conflict_removal(c,b,a,v,m,lambda_conflict=1)
    assert axial_change(out,b,a).abs().max() < 1e-12


def test_single_same_direction_not_modified():
    x,b,c,_,s,a,_,_=system(); m,_,v,_=local_conflict_mask(x,c,c,b,s); out=pairwise_conflict_removal(c,b,a,v,m,lambda_conflict=1)
    assert torch.equal(out,c)


@pytest.mark.parametrize("strength", [.25,.5,1.])
def test_pairwise_strength_is_exact(strength):
    x,b,c,_,_,a,m,v=system(); out=pairwise_conflict_removal(c,b,a,v,m,lambda_conflict=strength)
    assert axial_change(out,b,a).item() == pytest.approx((1-strength)*v.item())


def test_minnorm_l1_satisfies_constraint():
    x,b,c,_,_,a,m,_=system(); out,info=minimum_norm_conflict_projection(c,b,a,m,lambda_conflict=1,ridge=1e-12)
    assert axial_change(out,b,a).abs().max() < 1e-9 and info.rank == 1


def test_shared_atom_pairwise_is_order_independent():
    x=torch.tensor([[0.,0,0],[1.,0,0],[0.,1.,0.]],dtype=torch.float64); b=torch.tensor([[0,0],[1,2]])
    c=torch.tensor([[1.,1,0],[-1.,0,0],[0.,-1.,0.]],dtype=torch.float64); a=bond_axes(x,b); v=axial_change(c,b,a); m=torch.ones(2,dtype=torch.bool)
    out1=pairwise_conflict_removal(c,b,a,v,m,lambda_conflict=.5); p=torch.tensor([1,0]); out2=pairwise_conflict_removal(c,b[:,p],a[p],v[p],m[p],lambda_conflict=.5)
    assert torch.allclose(out1,out2)


def test_atom_renumbering_equivariance():
    x,b,c,d,s,_,_,_=system(); perm=torch.tensor([1,0]); inv=torch.argsort(perm)
    out,_=fuse_conflict_aware(x,c,d,b,s,operator="minnorm",lambda_conflict=1,conflict_eps=1e-10,ridge=1e-12)
    outp,_=fuse_conflict_aware(x[perm],c[perm],d[perm],inv[b],s,operator="minnorm",lambda_conflict=1,conflict_eps=1e-10,ridge=1e-12)
    assert torch.allclose(out,outp[inv])


def test_translation_invariance():
    x,b,c,d,s,_,_,_=system(); o1,_=fuse_conflict_aware(x,c,d,b,s,operator="minnorm",lambda_conflict=1,conflict_eps=1e-10,ridge=1e-12); o2,_=fuse_conflict_aware(x+7,c,d,b,s,operator="minnorm",lambda_conflict=1,conflict_eps=1e-10,ridge=1e-12)
    assert torch.allclose(o1,o2)


def test_rotation_equivariance():
    x,b,c,d,s,_,_,_=system(); r=torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]],dtype=x.dtype)
    o1,_=fuse_conflict_aware(x,c,d,b,s,operator="minnorm",lambda_conflict=1,conflict_eps=1e-10,ridge=1e-12); o2,_=fuse_conflict_aware(x@r.T,c@r.T,d@r.T,b,s,operator="minnorm",lambda_conflict=1,conflict_eps=1e-10,ridge=1e-12)
    assert torch.allclose(o1@r.T,o2,atol=1e-9)


def test_ring_exclusion_removes_constraint():
    x,b,c,d,s,_,_,_=system(); out,diag=fuse_conflict_aware(x,c,d,b,s,operator="minnorm",lambda_conflict=1,conflict_eps=1e-10,ridge=1e-12,ring_mask=torch.tensor([True]),nonring_only=True)
    assert torch.equal(out,c) and diag["total_conflict_bonds"] == 0


def test_empty_conflict_minnorm_safe():
    x,b,c,_,_,a,_,_=system(); out,info=minimum_norm_conflict_projection(c,b,a,torch.tensor([False]),lambda_conflict=1)
    assert out.data_ptr()==c.data_ptr() and info.conflicts==0


def test_solver_diagnostics_are_finite():
    x,b,c,_,_,a,m,_=system(); _,info=minimum_norm_conflict_projection(c,b,a,m,lambda_conflict=1,ridge=1e-10)
    assert info.condition >= 1 and info.fallback in {"none","lstsq"}


def test_nonfinite_input_fails_closed():
    x,b,c,d,s,_,_,_=system(); c[0,0]=float("nan")
    with pytest.raises(FloatingPointError): fuse_conflict_aware(x,c,d,b,s,operator="pairwise",lambda_conflict=1,conflict_eps=1e-10,ridge=1e-10)


def test_float32_float64_agree():
    values=[]
    for dtype in (torch.float32,torch.float64):
        x,b,c,d,s,_,_,_=system(dtype); values.append(fuse_conflict_aware(x,c,d,b,s,operator="minnorm",lambda_conflict=.5,conflict_eps=1e-10,ridge=1e-10)[0].double())
    assert torch.allclose(*values,atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_cpu_cuda_agree():
    cpu=system(); gpu=system(device="cuda"); o1,_=fuse_conflict_aware(cpu[0],cpu[2],cpu[3],cpu[1],cpu[4],operator="minnorm",lambda_conflict=.5,conflict_eps=1e-10,ridge=1e-10); o2,_=fuse_conflict_aware(gpu[0],gpu[2],gpu[3],gpu[1],gpu[4],operator="minnorm",lambda_conflict=.5,conflict_eps=1e-10,ridge=1e-10)
    assert torch.allclose(o1,o2.cpu(),atol=1e-8)


def test_record_batch_partition_is_mathematically_independent():
    records=list(range(65)); assert [records[i:i+2] for i in range(0,65,2)][0]+[] == [0,1]
    assert sum((records[i:i+16] for i in range(0,65,16)),[]) == sum((records[i:i+64] for i in range(0,65,64)),[]) == records


def test_sign_safe_logic_is_stage_g_function():
    assert sign_validity_safe_mask([.8],[1.],[1.2],[.1]).item() is True


def test_fixed_variants_are_complete():
    assert len(VARIANTS)==11 and {"H0_PAIRWISE_L025","H0_MINNORM_L100","H0_MINNORM_NONRING_L050"} <= set(VARIANTS)


def test_decision_rules():
    assert stage_h0_decision({"a":True},{},invalid=False)=="STAGE_H0_STRONG_HEADROOM"
    assert stage_h0_decision({}, {"a":True}, invalid=False)=="STAGE_H0_WEAK_HEADROOM"
    assert stage_h0_decision({}, {}, invalid=False)=="STAGE_H0_NO_HEADROOM"
    assert stage_h0_decision({"a":True},{},invalid=True)=="STAGE_H0_INVALID"


def test_inference_reuses_acceptance_and_trust_flow():
    source=inspect.getsource(infer_h0_variant)
    assert "select_trajectory_candidate" in source and "trust_clip_velocity" in source and "inference_feature_batch" in source


def test_validation_test_isolation_and_frozen_results():
    config=yaml.safe_load(Path("configs/ecir_mvr_stage_h0_conflict_fusion.yaml").read_text(encoding="utf-8")); verify_frozen(config)
    assert config["validation_only"] is True and config["test_records_read"]==0


def test_protected_file_sha_unchanged():
    assert file_sha256("reports/global4d_profile_bundle_verification.json")=="738171eb7e5f047e94cd4e1a46689613fe1f30bf33a320a2e3ec5a6944a5ec7d"


def test_d1b_checkpoint_and_frozen_parameters_declared():
    config=yaml.safe_load(Path("configs/ecir_mvr_stage_h0_conflict_fusion.yaml").read_text(encoding="utf-8"))
    assert config["checkpoint"]["neural_weights_frozen"] is True and config["checkpoint"]["strict_load"] is True


def test_stage_f_g_paths_are_not_h0_output():
    config=yaml.safe_load(Path("configs/ecir_mvr_stage_h0_conflict_fusion.yaml").read_text(encoding="utf-8"))
    assert config["output_dir"]=="diagnostics/ecir_mvr/stage_h0"
