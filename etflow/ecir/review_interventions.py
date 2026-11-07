"""Explicit, inference-only scientific interventions for frozen BA refiners.

No checkpoint or production inference behavior is changed. In particular,
``coefficient_sigma`` changes only the analytic coefficient; callers compute
reliability from the original prediction before calling this function.
"""
from __future__ import annotations

import torch
from .learned_geometry import geometry_values, remove_rigid_component
from .musigma_reliability import _bond_descent, _angle_descent


def rms(vector):
    return vector.square().sum(-1).mean().sqrt()


def normalize_direction(vector, coordinates):
    if vector.shape != coordinates.shape or not bool(torch.isfinite(vector).all()):
        raise ValueError("direction must be finite and have the coordinate shape")
    projected = remove_rigid_component(vector, coordinates)
    scale = rms(projected)
    if float(scale) <= 1e-14:
        return torch.zeros_like(projected)
    return projected / scale


def primitive_direction(source, graph, prediction, reliability, family_weights,
                        *, coefficient_sigma=None):
    """Frozen-coefficient direction, with independently supplied gates/scales."""
    bq, aq = geometry_values(source, graph)
    bs, ass = coefficient_sigma if coefficient_sigma is not None else (
        prediction['bond_sigma'], prediction['angle_sigma'])
    bs, ass = bs.to(source), ass.to(source)
    if (not bool(torch.isfinite(bs).all() and torch.isfinite(ass).all())
            or bool((bs <= 0).any() or (ass <= 0).any())):
        raise ValueError("coefficient scales must be finite and positive")
    br, ar = reliability
    bc = (family_weights[0] / max(1, bq.numel())) * br.to(source) * (
        bq - prediction['bond_mu'].to(source)) / bs.square()
    ac = (family_weights[1] / max(1, aq.numel())) * ar.to(source) * (
        aq - prediction['angle_mu'].to(source)) / ass.square()
    raw = _bond_descent(source, graph, bc) + _angle_descent(source, graph, ac)
    return normalize_direction(raw, source)


def fixed_rms_proposal(source, direction, budget):
    """Exact per-conformer raw Cartesian RMS budget; zero stays zero."""
    if not torch.isfinite(torch.tensor(budget)) or budget < 0:
        raise ValueError("budget must be finite and nonnegative")
    unit = normalize_direction(direction, source)
    return source + float(budget) * unit


def stereo_diagnostics(molecule, source, candidate, epsilon=1e-5):
    """Compare coordinate-derived stereo using a fixed graph/atom order.

    Tagged tetrahedral centers and specified E/Z double bonds are audited.
    This is a diagnostic, not a completeness claim for all stereochemistry.
    """
    from rdkit import Chem
    source = torch.as_tensor(source, dtype=torch.float64)
    candidate = torch.as_tensor(candidate, dtype=torch.float64)
    if source.shape != candidate.shape or source.shape != (molecule.GetNumAtoms(), 3):
        raise ValueError("atom count/order binding required")
    if not bool(torch.isfinite(source).all() and torch.isfinite(candidate).all()):
        raise ValueError("nonfinite coordinates")
    values = dict(tetra_centers=0, tetra_comparable=0, tetra_flips=0,
                  tetra_source_degenerate=0, tetra_candidate_degenerate=0,
                  ez_bonds=0, ez_comparable=0, ez_flips=0,
                  ez_source_degenerate=0, ez_candidate_degenerate=0)
    for atom in molecule.GetAtoms():
        if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        ns = sorted(n.GetIdx() for n in atom.GetNeighbors())
        if len(ns) not in (3, 4):
            continue
        values['tetra_centers'] += 1
        origin = ns[3] if len(ns) == 4 else atom.GetIdx()
        def volume(x):
            return float(torch.linalg.det(torch.stack([x[n]-x[origin] for n in ns[:3]])))
        a, b = volume(source), volume(candidate)
        values['tetra_source_degenerate'] += abs(a) <= epsilon
        values['tetra_candidate_degenerate'] += abs(b) <= epsilon
        if abs(a) > epsilon and abs(b) > epsilon:
            values['tetra_comparable'] += 1
            values['tetra_flips'] += a*b < 0
    for bond in molecule.GetBonds():
        if bond.GetStereo() not in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
                                   Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS):
            continue
        ns = list(bond.GetStereoAtoms())
        if len(ns) != 2:
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        def score(x):
            axis = x[j]-x[i]
            axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-14)
            u, v = x[ns[0]]-x[i], x[ns[1]]-x[j]
            u, v = u-(u*axis).sum()*axis, v-(v*axis).sum()*axis
            norm = float(torch.linalg.vector_norm(u)*torch.linalg.vector_norm(v))
            return float((u*v).sum())/max(norm, 1e-14), norm
        a, an = score(source); b, bn = score(candidate)
        values['ez_bonds'] += 1
        bad_a = an <= epsilon or abs(a) <= epsilon
        bad_b = bn <= epsilon or abs(b) <= epsilon
        values['ez_source_degenerate'] += bad_a
        values['ez_candidate_degenerate'] += bad_b
        if not bad_a and not bad_b:
            values['ez_comparable'] += 1
            values['ez_flips'] += a*b < 0
    return values


def bounded_stereo_proposal(molecule, source, proposal, *, max_atom_displacement,
                            max_rms=0.03, max_backtracks=12):
    """Optional inference wrapper: uniformly shrink, then check output stereo.

    This is a new, separately evaluated variant. It uses no Reference geometry,
    V3D outcome, learned target, or energy oracle. It does not guarantee energy
    improvement or correctness of the Source's chemical stereochemistry.
    """
    if max_atom_displacement <= 0 or max_rms <= 0 or max_backtracks < 0:
        raise ValueError('positive limits and nonnegative backtrack count required')
    if not bool(torch.isfinite(torch.tensor([max_atom_displacement,max_rms])).all()):
        raise ValueError('finite limits required')
    if source.shape != proposal.shape or not bool(torch.isfinite(source).all()):
        raise ValueError('finite source and equal coordinate shapes required')
    if not bool(torch.isfinite(proposal).all()):
        return source.clone(),dict(scale=0.,reason='nonfinite_proposal',backtracks=0)
    delta=proposal-source
    peak=float(torch.linalg.vector_norm(delta,dim=-1).max())
    scale=min(1.,max_atom_displacement/max(peak,1e-14),max_rms/max(float(rms(delta)),1e-14))
    for count in range(max_backtracks+1):
        candidate=source+scale*delta
        s=stereo_diagnostics(molecule,source,candidate)
        if s['tetra_source_degenerate'] or s['ez_source_degenerate']:
            return source.clone(),dict(scale=0.,reason='source_stereo_degenerate',backtracks=count)
        if not (s['tetra_flips'] or s['ez_flips'] or s['tetra_candidate_degenerate'] or s['ez_candidate_degenerate']):
            return candidate,dict(scale=scale,reason='unchanged' if scale==1. else 'shrunk',backtracks=count)
        scale*=.5
    return source.clone(),dict(scale=0.,reason='stereo_backtracking_exhausted',backtracks=max_backtracks)
