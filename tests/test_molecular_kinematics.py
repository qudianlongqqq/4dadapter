import torch

from etflow.commons.kinematic_projection import decompose_target,soft_gate_target
from etflow.commons.molecular_kinematics import build_molecular_kinematic_topology
from etflow.commons.torsion_kinematic_jacobian import (
    apply_jacobian,apply_jacobian_transpose,build_dense_jacobian,
)


def directed(pairs):
    values=[]
    for a,b in pairs: values.extend(((a,b),(b,a)))
    return torch.tensor(values,dtype=torch.long).t()


def chain_topology(reverse=False):
    edge=directed([(0,1),(1,2),(2,3)])
    bonds=torch.tensor([[2,3],[1,2]],dtype=torch.long) if reverse else torch.tensor([[1,2],[2,3]],dtype=torch.long)
    return build_molecular_kinematic_topology(4,edge,bonds)


def test_single_joint_matches_cross_product():
    pos=torch.tensor([[0.,0,0],[1,0,0],[2,0,0.],[2,1,0.]])
    topology=build_molecular_kinematic_topology(4,directed([(0,1),(1,2),(2,3)]),torch.tensor([[1],[2]]))
    velocity,valid=apply_jacobian(pos,torch.tensor([2.]),topology)
    expected=torch.cross(torch.tensor([1.,0,0]),pos[3]-pos[1],dim=0)*2
    torch.testing.assert_close(velocity[3],expected);assert valid.all()


def test_nested_joint_contributions_add_without_overlap_average():
    pos=torch.tensor([[0.,0,0],[1,0,0],[1,1,0.],[2,1,1.]])
    topology=chain_topology(); rates=torch.tensor([.3,-.4])
    velocity,_=apply_jacobian(pos,rates,topology);dense,_=build_dense_jacobian(pos,topology)
    torch.testing.assert_close(velocity.reshape(-1),dense@rates)
    terminal=dense.reshape(4,3,2)[3]@rates
    torch.testing.assert_close(velocity[3],terminal)


def test_branch_joints_only_affect_their_descendants():
    edge=directed([(0,1),(1,2),(1,3)])
    topology=build_molecular_kinematic_topology(4,edge,torch.tensor([[1,1],[2,3]]))
    affected=[set(topology.affected_atom_index[topology.affected_joint_index==j].tolist()) for j in range(2)]
    assert {frozenset(v) for v in affected}=={frozenset({2}),frozenset({3})}


def test_dense_and_matrix_free_and_adjoint_identity():
    torch.manual_seed(2);pos=torch.randn(4,3);topology=chain_topology();rates=torch.randn(2);vector=torch.randn(4,3)
    dense,_=build_dense_jacobian(pos,topology);velocity,_=apply_jacobian(pos,rates,topology);transpose,_=apply_jacobian_transpose(pos,vector,topology)
    torch.testing.assert_close(velocity.reshape(-1),dense@rates)
    torch.testing.assert_close((velocity*vector).sum(),(rates*transpose).sum())


def test_reversed_input_bond_direction_keeps_physical_orientation():
    pos=torch.randn(4,3);normal=chain_topology();reversed_topology=chain_topology(True);rate=torch.tensor([.2,-.1])
    assert torch.equal(normal.parent_atom,reversed_topology.parent_atom)
    left,_=apply_jacobian(pos,rate,normal);right,_=apply_jacobian(pos,rate,reversed_topology)
    torch.testing.assert_close(left,right)


def test_non_tree_and_disconnected_fail_closed():
    ring=directed([(0,1),(1,2),(2,0)])
    topology=build_molecular_kinematic_topology(3,ring,torch.tensor([[0],[1]]))
    assert topology.num_joints==0 and topology.status=="non_tree_fragment_graph"
    disconnected=build_molecular_kinematic_topology(3,directed([(0,1)]),torch.empty((2,0),dtype=torch.long))
    assert disconnected.status=="disconnected" and not disconnected.valid


def test_projection_reconstructs_and_is_orthogonal():
    pos=torch.randn(4,3);jacobian,_=build_dense_jacobian(pos,chain_topology());target=torch.randn(4,3)
    result=decompose_target(jacobian,target)
    torch.testing.assert_close(result.u_kin_star+result.u_res_star,target)
    assert abs(float((result.u_kin_star*result.u_res_star).sum()))<1e-5


def test_rank_deficient_and_no_joint_projection_are_safe():
    target=torch.randn(3,3);zero=torch.zeros(9,2);result=decompose_target(zero,target)
    assert result.rank==0;torch.testing.assert_close(result.u_res_star,target)
    empty=decompose_target(torch.empty(9,0),target)
    assert empty.rank==0 and empty.rate_star_damped.numel()==0


def test_exact_projection_and_damped_target_are_distinct():
    jacobian=torch.tensor([[1.,0],[0,1.],[0,0.]])*.01;target=torch.tensor([[1.,2,3.]])
    result=decompose_target(jacobian,target,rate_target_ridge=1.)
    assert not torch.allclose(result.rate_star_exact,result.rate_star_damped)
    torch.testing.assert_close(result.u_kin_star,torch.tensor([[1.,2,0.]]))


def test_soft_gate_target_is_not_hard_binary():
    target=soft_gate_target(torch.tensor([0.,.05,.1]),threshold=.05,temperature=.02)
    assert bool(((target>0)&(target<1)).all())


def test_translation_and_rotation_equivariance_of_jacobian():
    pos=torch.randn(4,3);topology=chain_topology();rate=torch.tensor([.2,-.3]);rotation=torch.tensor([[0.,-1,0],[1,0,0],[0,0,1.]])
    velocity,_=apply_jacobian(pos,rate,topology);translated,_=apply_jacobian(pos+torch.tensor([3.,-2,1.]),rate,topology)
    rotated,_=apply_jacobian(pos@rotation.T,rate,topology)
    torch.testing.assert_close(translated,velocity);torch.testing.assert_close(rotated,velocity@rotation.T)


def test_short_bond_is_safely_skipped():
    pos=torch.tensor([[0.,0,0],[0,0,0],[1.,0,0]])
    topology=build_molecular_kinematic_topology(3,directed([(0,1),(1,2)]),torch.tensor([[0],[1]]))
    velocity,valid=apply_jacobian(pos,torch.ones(1),topology)
    assert not valid.any() and torch.isfinite(velocity).all() and not velocity.any()


def test_global_least_squares_is_no_worse_than_independent_column_fits():
    torch.manual_seed(9);jacobian=torch.randn(18,3);target=torch.randn(18)
    global_rate=torch.linalg.pinv(jacobian)@target
    independent=torch.stack([(jacobian[:,i]@target)/(jacobian[:,i]@jacobian[:,i]) for i in range(3)])
    assert torch.linalg.norm(jacobian@global_rate-target)<=torch.linalg.norm(jacobian@independent-target)+1e-6


def test_symmetric_torsion_basis_does_not_explain_rigid_translation():
    pos=torch.tensor([[0.,0,0],[1.,0,0],[2.,1,0],[2.,-1,0.]])
    topology=build_molecular_kinematic_topology(4,directed([(0,1),(1,2),(1,3)]),torch.tensor([[0],[1]]))
    jacobian,_=build_dense_jacobian(pos,topology);translation=torch.ones(4,3)
    result=decompose_target(jacobian,translation)
    assert float(result.u_kin_star.abs().max())<1e-6


def test_atom_permutation_permutes_kinematic_velocity():
    pos=torch.tensor([[0.,0,0],[1.,0,0],[2.,0,0],[2.,1,0.],[2.,0,1.]])
    edges=[(0,1),(1,2),(2,3),(2,4)];rot=torch.tensor([[1],[2]])
    topology=build_molecular_kinematic_topology(5,directed(edges),rot);velocity,_=apply_jacobian(pos,torch.tensor([.4]),topology)
    permutation=torch.tensor([3,0,4,1,2]);inverse=torch.empty_like(permutation);inverse[permutation]=torch.arange(5)
    permuted_pos=pos[permutation];permuted_edges=[(int(inverse[a]),int(inverse[b])) for a,b in edges]
    permuted_rot=torch.tensor([[int(inverse[1])],[int(inverse[2])]])
    permuted_topology=build_molecular_kinematic_topology(5,directed(permuted_edges),permuted_rot)
    permuted_velocity,_=apply_jacobian(permuted_pos,torch.tensor([.4]),permuted_topology)
    torch.testing.assert_close(permuted_velocity,velocity[permutation])
