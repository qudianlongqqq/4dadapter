import torch

from etflow.commons.global_coupled_4d_topology import (
    GlobalCoupled4DTopologyCache,
    build_global_coupled_4d_topology,
)


def directed(pairs):
    values = []
    for left, right in pairs:
        values.extend(((left, right), (right, left)))
    return torch.tensor(values, dtype=torch.long).t()


def test_chain_contains_complete_descendant_subtrees():
    topology = build_global_coupled_4d_topology(
        5,
        directed([(0, 1), (1, 2), (2, 3), (3, 4)]),
        torch.tensor([[1, 2], [2, 3]]),
    )
    assert topology.status == "ok" and topology.num_joints == 2
    affected = [
        set(topology.affected_atom_index[topology.affected_joint_index == joint].tolist())
        for joint in range(2)
    ]
    assert any({3, 4}.issubset(atoms) for atoms in affected)
    assert 4 in affected[0].intersection(affected[1])


def test_branch_children_do_not_cross_affect():
    topology = build_global_coupled_4d_topology(
        4,
        directed([(0, 1), (1, 2), (1, 3)]),
        torch.tensor([[1, 1], [2, 3]]),
    )
    affected = {
        frozenset(topology.affected_atom_index[topology.affected_joint_index == joint].tolist())
        for joint in range(topology.num_joints)
    }
    assert affected == {frozenset({2}), frozenset({3})}


def test_empty_joint_is_valid_residual_only_topology():
    topology = build_global_coupled_4d_topology(
        3, directed([(0, 1), (1, 2)]), torch.empty((2, 0), dtype=torch.long)
    )
    assert topology.valid and topology.status == "no_rotatable_bonds"
    assert topology.num_joints == 0


def test_ring_and_disconnected_graphs_fail_closed():
    ring = build_global_coupled_4d_topology(
        3, directed([(0, 1), (1, 2), (2, 0)]), torch.tensor([[0], [1]])
    )
    assert not ring.valid and ring.num_joints == 0
    disconnected = build_global_coupled_4d_topology(
        3, directed([(0, 1)]), torch.empty((2, 0), dtype=torch.long)
    )
    assert not disconnected.valid and disconnected.num_joints == 0


def test_coordinate_independent_topology_cache_hits():
    cache = GlobalCoupled4DTopologyCache()
    edge = directed([(0, 1), (1, 2)])
    rotatable = torch.tensor([[0], [1]])
    first = cache.get(3, edge, rotatable)
    second = cache.get(3, edge, rotatable)
    assert first.status == second.status
    assert cache.stats.misses == 1 and cache.stats.hits == 1


def test_prepared_topology_caches_masks_incidence_and_fixed_indices():
    cache = GlobalCoupled4DTopologyCache()
    edge = directed([(0, 1), (1, 2), (2, 3), (3, 4)])
    rotatable = torch.tensor([[1, 2], [2, 3]])
    first = cache.get_prepared(5, edge, rotatable)
    second = cache.get_prepared(5, edge, rotatable)
    assert first is second
    assert first.downstream_mask.shape == (2, 5)
    assert torch.equal(first.joint_to_atom_incidence, first.downstream_mask)
    assert first.ancestor_mask.shape == (2, 2)
    assert first.jacobian_flat_index.numel() == first.topology.affected_atom_index.numel()
