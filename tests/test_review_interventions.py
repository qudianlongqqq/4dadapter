from types import SimpleNamespace
import pytest
import torch
from rdkit import Chem
from etflow.ecir.review_interventions import (primitive_direction, fixed_rms_proposal,
                                             stereo_diagnostics, rms,bounded_stereo_proposal)
from etflow.ecir.learned_geometry import geometry_values
from etflow.ecir.review_interventions import hybrid_mmff_direction, normalize_direction


def sample():
    x = torch.tensor([[1.,0.,0.],[0.,0.,0.],[0.,1.1,0.],[0.,0.,1.2]], dtype=torch.float64)
    g = SimpleNamespace(bonds=torch.tensor([[0,1,1],[1,2,3]]),
                        angles=torch.tensor([[0,1,2],[0,1,3],[2,1,3]]))
    b,a=geometry_values(x,g)
    p=dict(bond_mu=b*.98, angle_mu=a+.02,
           bond_sigma=b*.02, angle_sigma=torch.ones_like(a)*.04)
    return x,g,p,(torch.ones_like(b),torch.ones_like(a)),torch.tensor([.5,.5])


def test_exact_budget_and_zero():
    x,g,p,r,w=sample()
    d=primitive_direction(x,g,p,r,w)
    y=fixed_rms_proposal(x,d,.01)
    assert abs(float(rms(y-x))-.01)<1e-12
    assert torch.equal(fixed_rms_proposal(x,torch.zeros_like(x),.01),x)
    with pytest.raises(ValueError): fixed_rms_proposal(x,d,float('nan'))


def test_hybrid_mixes_direction_without_inheriting_input_force_scale():
    source, graph, prediction, reliability, weights = sample()
    pina = primitive_direction(source, graph, prediction, reliability, weights)
    mmff = normalize_direction(source.square(), source)
    expected = normalize_direction(.25 * pina + .75 * mmff, source)
    mixed = hybrid_mmff_direction(source, 40 * pina, .001 * mmff, .25)
    assert torch.allclose(mixed, expected, atol=1e-12)
    assert torch.allclose(hybrid_mmff_direction(source, pina, mmff, 0), mmff)
    assert torch.equal(hybrid_mmff_direction(source, pina, -pina, .5), torch.zeros_like(pina))


def test_frozen_coefficient_negative_gradient_and_sigma_intervention():
    x,g,p,r,w=sample(); x.requires_grad_(True)
    b,a=geometry_values(x,g)
    energy=.5*((b-p['bond_mu'])/p['bond_sigma']).square().mean()*.5
    energy+=.5*((a-p['angle_mu'])/p['angle_sigma']).square().mean()*.5
    grad=-torch.autograd.grad(energy,x)[0]
    d=primitive_direction(x,g,p,r,w)
    assert torch.allclose(d,grad/rms(grad),atol=1e-12)
    original={k:v.clone() for k,v in p.items()}
    alt=(p['bond_sigma']*2,p['angle_sigma'])
    changed=primitive_direction(x,g,p,r,w,coefficient_sigma=alt)
    assert not torch.allclose(d,changed)
    assert all(torch.equal(p[k],original[k]) for k in p)


def test_stereo_detects_mirror_but_not_translation():
    mol=Chem.AddHs(Chem.MolFromSmiles('F[C@](Cl)(Br)I'))
    x=torch.tensor([[1.,0.,0.],[0.,0.,0.],[0.,1.,0.],[0.,0.,1.],[-1.,-1.,-1.]])
    assert stereo_diagnostics(mol,x,x+10)['tetra_flips']==0
    y=x.clone(); y[:,0]*=-1
    assert stereo_diagnostics(mol,x,y)['tetra_flips']==1
    y=torch.zeros_like(x)
    assert stereo_diagnostics(mol,x,y)['tetra_candidate_degenerate']==1


def test_ez_detects_coordinate_flip():
    mol=Chem.MolFromSmiles('F/C=C/F')
    x=torch.tensor([[-1.,1.,0.],[0.,0.,0.],[1.,0.,0.],[2.,-1.,0.]])
    assert stereo_diagnostics(mol,x,x)['ez_flips']==0
    y=x.clone(); y[3,1]=1
    assert stereo_diagnostics(mol,x,y)['ez_flips']==1


def test_bounded_wrapper_caps_and_preserves_source_stereo():
    mol=Chem.MolFromSmiles('F[C@](Cl)(Br)I')
    x=torch.tensor([[1.,0.,0.],[0.,0.,0.],[0.,1.,0.],[0.,0.,1.],[-1.,-1.,-1.]])
    y=x.clone();y[:,0]*=-1
    z,info=bounded_stereo_proposal(mol,x,y,max_atom_displacement=10.,max_rms=10.)
    assert info['backtracks']>0
    assert stereo_diagnostics(mol,x,z)['tetra_flips']==0
    assert stereo_diagnostics(mol,x,z)['tetra_candidate_degenerate']==0
    z,info=bounded_stereo_proposal(mol,x,x+100,max_atom_displacement=.1,max_rms=.03)
    assert float(torch.linalg.vector_norm(z-x,dim=-1).max())<=.100001
    assert float(rms(z-x))<=.030001
    z,info=bounded_stereo_proposal(mol,x,x*float('nan'),max_atom_displacement=.1)
    assert torch.equal(z,x)
    with pytest.raises(ValueError):bounded_stereo_proposal(mol,x,x,max_atom_displacement=float('inf'))
