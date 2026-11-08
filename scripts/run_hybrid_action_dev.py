"""Compare three guarded actions on an outcome-independent subset of frozen DEV."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_review_gaps import ORIGINAL, ROOT, SEEDS, cache_for, evaluate, sha, write_json
from scripts.review_asset_paths import ASSET_ROOT, asset_path

BUDGETS = (0.00432154, 0.01, 0.02)
PINA_WEIGHT = 0.25
ATOM_CAP = 0.10
PRIMARY_BUDGET = 0.01
SERIALIZATION_MARGIN = 0.00018
METHODS = ('pina', 'mmff', 'hybrid')
DEV_SIZE = 256


def require_hash(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Frozen asset hash mismatch: {path}')
    return path


def prepare(args):
    import numpy as np
    import pandas as pd
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from etflow.ecir.formal_rdkit_adapter import adapt_formal_cache_record
    from etflow.ecir.j1r1_full_joint_unrestricted import (
        UnrestrictedFullJointModel, unrestricted_full_joint_action,
    )
    from etflow.ecir.review_interventions import (
        bounded_stereo_proposal, fixed_rms_proposal, hybrid_mmff_direction,
        normalize_direction, rms, stereo_diagnostics,
    )

    torch.set_num_threads(2)
    config_path = ROOT / 'configs/sixs_j1r1_full_joint_unrestricted_movement.json'
    config = json.loads(config_path.read_text(encoding='utf8'))['data']
    dev_path = require_hash(ORIGINAL / config['dev_manifest'], config['dev_manifest_sha256'])
    source_path = require_hash(asset_path(config['source_payload']), config['source_payload_sha256'])
    val_path = require_hash(asset_path(config['val_manifest']), config['val_manifest_sha256'])
    graph_path = (ASSET_ROOT / '4dadapter-lsgo-formal/reports/ecir_mvr/lsgo_formal/cache/VAL_PREPARED.pt')
    require_hash(graph_path, 'e9cff35fb3bc8467ca3d19d20318c349472c27bc6cf3e2c7ef07a5db70214d93')
    registry_path = ORIGINAL / 'reports/ecir_mvr/sixs_primary_final_evaluation/00_FROZEN_FINAL_EVALUATION_PROTOCOL.json'
    registry = json.loads(registry_path.read_text(encoding='utf8'))
    checkpoint_entries = {row['seed']: row for row in registry['model_methods']
                          if row['formulation'] == 'Unrestricted'}
    dev = json.loads(dev_path.read_text(encoding='utf8'))['rows']
    selected = sorted(dev, key=lambda row: hashlib.sha256(
        ('pina-hybrid-dev-v1|' + row['molecule_id']).encode('utf8')).hexdigest())[:DEV_SIZE]
    record_ids = [record for row in selected for record in row['sample_ids']]
    record_id_set = set(record_ids)
    if (len(selected) != DEV_SIZE or len(record_id_set) != DEV_SIZE * 2
            or any(len(row['sample_ids']) != 2 for row in selected)):
        raise ValueError('DEV denominator mismatch')
    protocol = {
        'cohort': 'frozen ETFlow DEV, first 256 by salted molecule identity hash, both Source records',
        'selected': selected, 'record_count': len(record_ids), 'seeds': list(SEEDS),
        'methods': list(METHODS), 'budgets_angstrom': list(BUDGETS),
        'primary_budget_angstrom': PRIMARY_BUDGET, 'pina_weight': PINA_WEIGHT,
        'max_atom_displacement_angstrom': ATOM_CAP, 'precision_decimal_places': 4,
        'serialization_margin_angstrom': SERIALIZATION_MARGIN,
        'primary_gate': 'hybrid minus MMFF molecule bootstrap lower 95% bound > 0 at 0.01 angstrom',
        'secondary_gate': 'also review energy, stereo, realized displacement, and action latency before any training',
        'energy': 'same MMFF94s potential at fixed geometry, diagnostic favorable to MMFF',
        'runtime': 'CPU, cached graph, per-record inference and guard. MMFF setup and gradient included. Excludes IO and evaluator.',
        'dev_sha256': sha(dev_path), 'val_manifest_sha256': config['val_manifest_sha256'],
        'graph_asset_sha256': sha(graph_path), 'source_binding_sha256': config['source_payload_sha256'],
        'checkpoint_hashes': {str(seed): checkpoint_entries[seed]['checkpoint_sha256'] for seed in SEEDS},
    }
    protocol_path = args.out / 'PROTOCOL.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf8')) != protocol:
        raise ValueError('Output directory belongs to a different protocol')
    write_json(protocol_path, protocol)

    graphs = {row['molecule_id']: row['graph'] for row in
              torch.load(graph_path, map_location='cpu', weights_only=False)['items']}
    bound_sources = {row['sample_id']: row for row in
                     torch.load(source_path, map_location='cpu', weights_only=False)['val']
                     if row['sample_id'] in record_id_set}
    val_records = pd.read_parquet(val_path).set_index('sample_id')
    if not val_records.index.is_unique or set(bound_sources) != record_id_set:
        raise ValueError('VAL source manifest is incomplete or ambiguous')
    models = {}
    for seed, entry in checkpoint_entries.items():
        path = require_hash(ORIGINAL / entry['checkpoint'], entry['checkpoint_sha256'])
        payload = torch.load(path, map_location='cpu', weights_only=False)
        model = UnrestrictedFullJointModel(128, 3)
        model.load_state_dict(payload['model_state'], strict=True)
        models[seed] = model.eval()

    def mmff_setup(molecule, source):
        work_mol = Chem.Mol(molecule)
        work_mol.RemoveAllConformers()
        conformer = Chem.Conformer(molecule.GetNumAtoms())
        for index, point in enumerate(source.tolist()):
            conformer.SetAtomPosition(index, point)
        work_mol.AddConformer(conformer, assignId=True)
        if not AllChem.MMFFHasAllMoleculeParams(work_mol):
            return work_mol, None, torch.zeros_like(source)
        props = AllChem.MMFFGetMoleculeProperties(work_mol, mmffVariant='MMFF94s')
        force_field = AllChem.MMFFGetMoleculeForceField(work_mol, props, confId=0)
        gradient = torch.tensor(force_field.CalcGrad(), dtype=source.dtype).reshape_as(source)
        return work_mol, force_field, -gradient

    jobs = cache_for(args.out) / 'jobs'
    jobs.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(selected):
        job_path = jobs / f'{index:04d}.pkl'
        if job_path.exists():
            continue
        molecule_id = row['molecule_id']
        graph = graphs[molecule_id]
        job = dict(molecule_id=molecule_id, records=[])
        for record_id in row['sample_ids']:
            record_row = val_records.loc[record_id]
            record_path = asset_path(config['val_cache']) / Path(str(record_row.source_path).replace('\\', '/')).name
            require_hash(record_path, record_row.source_file_sha256)
            record = torch.load(record_path, map_location='cpu', weights_only=False)
            if record['sample_id'] != record_id or bound_sources[record_id]['molecule_id'] != molecule_id:
                raise ValueError(f'Source identity mismatch for {record_id}')
            molecule = adapt_formal_cache_record(record)['_formal_rdkit_mol']
            source = torch.as_tensor(bound_sources[record_id]['source'], dtype=torch.float64)
            raw = torch.as_tensor(record['x_init'], dtype=torch.float64)
            if raw.shape != source.shape or not torch.allclose(raw - raw.mean(0), source - source.mean(0), atol=3e-6, rtol=0):
                raise ValueError(f'Source coordinate mismatch for {record_id}')
            # The two DEV conformers of a molecule share one indexed topology.
            mol_binary = molecule.ToBinary()
            if 'mol_binary' in job and mol_binary != job['mol_binary']:
                raise ValueError(f'Indexed molecule changed between Source records: {record_id}')
            job['mol_binary'] = mol_binary
            serialized_source = torch.from_numpy(np.round(source.numpy(), 4))
            start = perf_counter()
            # Keep the RDKit molecule alive while its force field holds atom positions.
            energy_mol, force_field, mmff_direction = mmff_setup(molecule, source)
            mmff_seconds = perf_counter() - start
            source_energy = (force_field.CalcEnergy(tuple(serialized_source.flatten().tolist()))
                             if force_field is not None else None)
            cases = [dict(case='source', coordinates=serialized_source.numpy())]

            def guarded_case(method, seed, budget, direction, direction_seconds):
                start = perf_counter()
                proposal = fixed_rms_proposal(source, direction, budget)
                candidate, guard = bounded_stereo_proposal(
                    molecule, source, proposal, max_atom_displacement=ATOM_CAP - SERIALIZATION_MARGIN,
                    max_rms=budget - SERIALIZATION_MARGIN,
                )
                rounded = torch.from_numpy(np.round(candidate.numpy(), 4))
                diagnostics = stereo_diagnostics(molecule, serialized_source, rounded)
                if (diagnostics['tetra_flips'] or diagnostics['ez_flips'] or
                        diagnostics['tetra_candidate_degenerate'] > diagnostics['tetra_source_degenerate'] or
                        diagnostics['ez_candidate_degenerate'] > diagnostics['ez_source_degenerate']):
                    rounded = serialized_source
                    guard = dict(scale=0., reason='serialization_stereo_fallback')
                    diagnostics = stereo_diagnostics(molecule, serialized_source, rounded)
                delta = rounded - serialized_source
                realized = float(rms(delta))
                peak = float(torch.linalg.vector_norm(delta, dim=-1).max())
                if realized > budget + 1e-12 or peak > ATOM_CAP + 1e-12:
                    raise ValueError(f'Serialized displacement exceeded cap for {record_id}')
                elapsed = direction_seconds + perf_counter() - start
                energy = (force_field.CalcEnergy(tuple(rounded.flatten().tolist())) - source_energy
                          if force_field is not None else None)
                cases.append(dict(case=f'{method}_{seed}@{budget:.8f}', coordinates=rounded.numpy(),
                                  method=method, seed=seed, budget=budget, rms=realized, peak=peak,
                                  mmff_delta_kcal_mol=energy, action_ms=1000 * elapsed,
                                  guard_scale=guard['scale'], guard_reason=guard['reason'], **diagnostics))

            for budget in BUDGETS:
                guarded_case('mmff', 0, budget, mmff_direction, mmff_seconds)
            for seed in SEEDS:
                start = perf_counter()
                with torch.inference_mode():
                    model = models[seed]
                    prediction = model.belief(graph)
                    action = unrestricted_full_joint_action(model, source, [graph], prediction)
                    pina_direction = action.direction
                pina_seconds = perf_counter() - start
                start = perf_counter()
                mixed = hybrid_mmff_direction(source, pina_direction, mmff_direction, PINA_WEIGHT)
                hybrid_seconds = pina_seconds + mmff_seconds + perf_counter() - start
                for budget in BUDGETS:
                    guarded_case('pina', seed, budget, pina_direction, pina_seconds)
                    guarded_case('hybrid', seed, budget, mixed, hybrid_seconds)
            job['records'].append(dict(record_id=record_id, cases=cases,
                                       mmff_supported=force_field is not None))
        temporary = job_path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            pickle.dump(job, stream, pickle.HIGHEST_PROTOCOL)
        temporary.replace(job_path)
        if (index + 1) % 32 == 0:
            print(f'PREPARED {index + 1}/{DEV_SIZE} molecules', flush=True)


def summarize(args):
    import numpy as np
    import pandas as pd

    frame = pd.read_parquet(args.out / 'PER_RECORD_INTERVENTIONS.parquet')
    protocol = json.loads((args.out / 'PROTOCOL.json').read_text(encoding='utf8'))
    expected_ids = {record for item in protocol['selected'] for record in item['sample_ids']}
    if (set(frame.record_id) != expected_ids or frame.duplicated(['record_id', 'case']).any()
            or not frame.evaluator_success.all()):
        raise ValueError('Evaluation has missing, duplicated, or failed records')
    expected_cases = 1 + len(BUDGETS) * (1 + 2 * len(SEEDS))
    if not frame.groupby('record_id').size().eq(expected_cases).all():
        raise ValueError('A record lacks required method outcomes')
    source = frame[frame['case'] == 'source'].set_index('record_id')
    actions = frame[frame['case'] != 'source'].copy()
    actions['source_pass'] = actions.record_id.map(source.validity3d)
    actions['damaged'] = actions.source_pass & ~actions.validity3d
    actions['repaired'] = ~actions.source_pass & actions.validity3d
    summary = actions.groupby(['method', 'budget']).agg(
        v3d=('validity3d', 'mean'), bond_pass=('bond_valid', 'mean'), angle_pass=('angle_valid', 'mean'),
        repaired=('repaired', 'mean'), damaged=('damaged', 'mean'), mean_rms=('rms', 'mean'),
        atom_shift_p99=('peak', lambda values: values.quantile(.99)), max_atom_shift=('peak', 'max'),
        median_mmff_delta_kcal_mol=('mmff_delta_kcal_mol', 'median'),
        median_cpu_action_ms=('action_ms', 'median'),
        tetra_flips=('tetra_flips', 'sum'), ez_flips=('ez_flips', 'sum'),
    )
    summary.to_csv(args.out / 'METHOD_SUMMARY.csv')
    seed_summary = actions.groupby(['method', 'budget', 'seed'])['validity3d'].mean()
    seed_summary.to_csv(args.out / 'SEED_V3D.csv')
    rng = np.random.default_rng(20261108)
    effects = []
    for budget in BUDGETS:
        per_molecule = (actions[actions.budget == budget]
                        .groupby(['molecule_id', 'method']).validity3d.mean().unstack())
        indices = rng.integers(len(per_molecule), size=(10000, len(per_molecule)))
        for comparator in ('mmff', 'pina'):
            difference = (per_molecule.hybrid - per_molecule[comparator]).to_numpy() * 100
            low, high = np.quantile(difference[indices].mean(1), [.025, .975])
            effects.append(dict(budget=budget, comparator=comparator, delta_v3d_pp=difference.mean(),
                                ci_low_pp=low, ci_high_pp=high, molecules=len(per_molecule)))
    pd.DataFrame(effects).to_csv(args.out / 'PAIRED_V3D.csv', index=False)
    primary = next(row for row in effects if row['budget'] == PRIMARY_BUDGET and row['comparator'] == 'mmff')
    write_json(args.out / 'RESULT.json', dict(
        source_v3d=float(source.validity3d.mean()), molecules=int(frame.molecule_id.nunique()),
        source_records=len(source), evaluated_rows=len(frame),
        mmff_supported_records=int(source.mmff_supported.sum()),
        primary=primary, primary_v3d_gate_pass=bool(primary['ci_low_pp'] > 0),
    ))
    print(summary.to_string())
    print(pd.DataFrame(effects).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'evaluate', 'summarize'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'evaluate': evaluate, 'summarize': summarize}[args.stage](args)
