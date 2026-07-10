"""Deterministic rigid-fragment trees for molecular torsion kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import torch
from torch import Tensor


@dataclass
class MolecularKinematicTopology:
    """Coordinate-independent topology; all indices are local to one molecule."""

    num_atoms: int
    fragments: tuple[tuple[int, ...], ...]
    root_fragment: int
    canonical_bond_id: Tensor
    parent_fragment: Tensor
    child_fragment: Tensor
    parent_atom: Tensor
    child_atom: Tensor
    affected_atom_index: Tensor
    affected_joint_index: Tensor
    affected_ptr: Tensor
    orientation_sign: Tensor
    status: str

    @property
    def num_joints(self) -> int:
        return int(self.parent_atom.numel())

    @property
    def valid(self) -> bool:
        return self.status in {"ok", "no_rotatable_bonds"}

    def to(self, device) -> "MolecularKinematicTopology":
        values = {
            name: getattr(self, name).to(device)
            for name in (
                "canonical_bond_id", "parent_fragment", "child_fragment",
                "parent_atom", "child_atom", "affected_atom_index",
                "affected_joint_index", "affected_ptr", "orientation_sign",
            )
        }
        return MolecularKinematicTopology(
            self.num_atoms, self.fragments, self.root_fragment, status=self.status, **values
        )


def _empty(num_atoms: int, fragments, root: int, status: str, device) -> MolecularKinematicTopology:
    empty = torch.empty(0, dtype=torch.long, device=device)
    return MolecularKinematicTopology(
        num_atoms=num_atoms, fragments=tuple(fragments), root_fragment=root,
        canonical_bond_id=torch.empty((0, 2), dtype=torch.long, device=device),
        parent_fragment=empty, child_fragment=empty, parent_atom=empty,
        child_atom=empty, affected_atom_index=empty, affected_joint_index=empty,
        affected_ptr=torch.zeros(1, dtype=torch.long, device=device),
        orientation_sign=torch.empty(0, dtype=torch.float32, device=device),
        status=status,
    )


def _components(num_atoms: int, edges: set[tuple[int, int]]) -> list[tuple[int, ...]]:
    adjacency = [set() for _ in range(num_atoms)]
    for left, right in edges:
        adjacency[left].add(right); adjacency[right].add(left)
    unseen = set(range(num_atoms)); components = []
    while unseen:
        start = min(unseen); queue = [start]; unseen.remove(start); atoms = []
        while queue:
            atom = queue.pop(); atoms.append(atom)
            for neighbor in sorted(adjacency[atom], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor); queue.append(neighbor)
        components.append(tuple(sorted(atoms)))
    return components


def build_molecular_kinematic_topology(
    num_atoms: int,
    edge_index: Tensor,
    rotatable_bond_index: Tensor,
) -> MolecularKinematicTopology:
    """Cut rotatable bonds, root the fragment tree, and orient every joint.

    Invalid/disconnected/non-tree inputs return an explicit status and no joint,
    ensuring callers fall back to their Cartesian residual instead of guessing.
    """

    device = edge_index.device
    if num_atoms < 2:
        return _empty(num_atoms, [tuple(range(num_atoms))], 0, "too_few_atoms", device)
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        return _empty(num_atoms, [], -1, "invalid_edge_index", device)
    graph_edges = set()
    for left, right in edge_index.detach().cpu().t().tolist():
        left, right = int(left), int(right)
        if left == right or min(left, right) < 0 or max(left, right) >= num_atoms:
            continue
        graph_edges.add(tuple(sorted((left, right))))
    original_components = _components(num_atoms, graph_edges)
    if len(original_components) != 1:
        return _empty(num_atoms, original_components, -1, "disconnected", device)
    raw_bonds = []
    if rotatable_bond_index.ndim == 2 and rotatable_bond_index.size(0) == 2:
        raw_bonds = [(int(a), int(b)) for a, b in rotatable_bond_index.detach().cpu().t().tolist()]
    canonical = sorted(set(tuple(sorted(pair)) for pair in raw_bonds if pair[0] != pair[1]))
    if not canonical:
        return _empty(num_atoms, [tuple(range(num_atoms))], 0, "no_rotatable_bonds", device)
    if any(pair not in graph_edges for pair in canonical):
        return _empty(num_atoms, original_components, -1, "rotatable_bond_missing", device)
    fragments = _components(num_atoms, graph_edges.difference(canonical))
    atom_to_fragment = {}
    for fragment_id, atoms in enumerate(fragments):
        atom_to_fragment.update({atom: fragment_id for atom in atoms})
    fragment_edges = []
    for bond in canonical:
        left_fragment, right_fragment = atom_to_fragment[bond[0]], atom_to_fragment[bond[1]]
        if left_fragment == right_fragment:
            return _empty(num_atoms, fragments, -1, "non_tree_fragment_graph", device)
        fragment_edges.append((left_fragment, right_fragment, bond))
    if len(fragment_edges) != len(fragments) - 1:
        return _empty(num_atoms, fragments, -1, "non_tree_fragment_graph", device)
    root = min(range(len(fragments)), key=lambda index: (-len(fragments[index]), min(fragments[index]), fragments[index]))
    adjacency = [[] for _ in fragments]
    for left, right, bond in fragment_edges:
        adjacency[left].append((right, bond)); adjacency[right].append((left, bond))
    parent = {root: -1}; parent_joint = {}; order = []; queue = deque([root])
    while queue:
        fragment = queue.popleft()
        for neighbor, bond in sorted(adjacency[fragment], key=lambda item: (item[0], item[1])):
            if neighbor in parent:
                continue
            parent[neighbor] = fragment; parent_joint[neighbor] = bond
            order.append(neighbor); queue.append(neighbor)
    if len(parent) != len(fragments):
        return _empty(num_atoms, fragments, root, "disconnected_fragment_graph", device)
    children = [[] for _ in fragments]
    for child, parent_id in parent.items():
        if parent_id >= 0:
            children[parent_id].append(child)
    descendants = {}
    def collect(fragment):
        result = set(fragments[fragment])
        for child in children[fragment]:
            result.update(collect(child))
        descendants[fragment] = tuple(sorted(result)); return result
    collect(root)
    canonical_ids=[]; parent_fragments=[]; child_fragments=[]; parent_atoms=[]; child_atoms=[]
    affected_atoms=[]; affected_joints=[]; ptr=[0]; signs=[]
    raw_lookup = {tuple(sorted(pair)): pair for pair in raw_bonds}
    for joint_id, child_fragment in enumerate(order):
        parent_fragment = parent[child_fragment]; bond = parent_joint[child_fragment]
        if atom_to_fragment[bond[0]] == parent_fragment:
            parent_atom, child_atom = bond
        else:
            child_atom, parent_atom = bond
        canonical_ids.append(bond); parent_fragments.append(parent_fragment)
        child_fragments.append(child_fragment); parent_atoms.append(parent_atom); child_atoms.append(child_atom)
        atoms = descendants[child_fragment]; affected_atoms.extend(atoms)
        affected_joints.extend([joint_id] * len(atoms)); ptr.append(len(affected_atoms))
        original = raw_lookup.get(bond, bond)
        signs.append(1.0 if original == (parent_atom, child_atom) else -1.0)
    return MolecularKinematicTopology(
        num_atoms=num_atoms, fragments=tuple(fragments), root_fragment=root,
        canonical_bond_id=torch.tensor(canonical_ids, dtype=torch.long, device=device),
        parent_fragment=torch.tensor(parent_fragments, dtype=torch.long, device=device),
        child_fragment=torch.tensor(child_fragments, dtype=torch.long, device=device),
        parent_atom=torch.tensor(parent_atoms, dtype=torch.long, device=device),
        child_atom=torch.tensor(child_atoms, dtype=torch.long, device=device),
        affected_atom_index=torch.tensor(affected_atoms, dtype=torch.long, device=device),
        affected_joint_index=torch.tensor(affected_joints, dtype=torch.long, device=device),
        affected_ptr=torch.tensor(ptr, dtype=torch.long, device=device),
        orientation_sign=torch.tensor(signs, dtype=torch.float32, device=device),
        status="ok",
    )
