# MCVR External Refinement Environment Audit

Status: `MCVR_EXTERNAL_REFINEMENT_ENVIRONMENT_READY`

## Repository isolation

- Branch: `eval/mcvr-v8-external-refinement-baselines`
- Parent clean/pushed HEAD: `ab2eb8e8fb04e9f5e9ced9e6ab4fdd7f9b387bf5`
- The final Matched D1 12.5K report was committed and pushed before this branch was created.
- No neural model, checkpoint, loss, optimizer, scheduler, sampler, or training process was changed or started.

## RDKit and MMFF94s

- Python: `E:\miniconda\envs\etflow-5080-v2\python.exe`
- RDKit: `2026.03.4`
- `MMFFGetMoleculeProperties`: available
- `MMFFGetMoleculeForceField`: available
- `MMFFOptimizeMolecule`: available
- Explicit variant: `mmffVariant="MMFF94s"`
- Frozen explicit-hydrogen policy: preserve the Source cache atom set and cache order; never embed.
- Cache-to-RDKit atom mapping verified: 10,000/10,000 records.
- MMFF parameter coverage: 9,998/10,000 (99.98%). Records 7726 and 7727 are retained and fall back bitwise to Source.
- Formal charge distribution: charge 0 = 9,556; +1 = 420; +2 = 6; -1 = 14; -2 = 4.
- Radical/open-shell audit: 0/10,000 records have nonzero radical electrons.
- Elements present (atomic number): H, B, C, N, O, F, P, S, Cl, Br, I.
- RDKit was already suitable; it was not reinstalled, upgraded, or downgraded.

## GFN2-xTB

- Official stable release: Grimme Lab xTB 6.7.1.
- Official release page: <https://github.com/grimme-lab/xtb/releases/tag/v6.7.1>
- Installed archive source: <https://github.com/grimme-lab/xtb/releases/download/v6.7.1/xtb-6.7.1-linux-x86_64.tar.xz>
- Archive SHA256: `62a8d18778286e815292ee53d76ce447daf460a4dea3782c0f25cbac7019b5df`
- Executable: `E:\tools\xtb\6.7.1-linux-x86_64\bin\xtb`
- Executable SHA256: `debf27a9e0fa4bfb5ca75aafe4b90d8211f08ec2f4a482f375a4987212eaa12a`
- Architecture: statically linked ELF 64-bit x86-64, GNU/Linux 3.2.0 or newer.
- Runtime: existing WSL2 distribution `Ubuntu-22.04`; invoked by absolute executable path, not global PATH.
- Version output: `xtb version 6.7.1 (edcfbbe)`.
- The official Windows asset is described upstream as `6.7.1pre`, so it was intentionally not used.
- Minimal GFN2 single-point/geometry-optimization smoke: normal termination; `xtbopt.xyz` produced; convergence text present; atom order and finite coordinates verified.
- Fixed command semantics: GFN2, gas phase, no solvent, `--opt normal`, 250 maximum cycles, per-record charge and UHF.
- Worker benchmark on the same eight records: worker 1 = 3.671 wall s/record (8/8); worker 2 = 1.890 (8/8); worker 4 was unstable (6/8). Frozen choice: 2 workers, 1 OMP/MKL/OpenBLAS thread per worker.

## Environment changes

- Added only the official stable xTB distribution under `E:\tools\xtb\6.7.1-linux-x86_64`.
- Did not change PATH, system files, conda packages, PyTorch, or RDKit.
- xTB binaries and archives are outside Git and are not committed.

## Isolation

- `formal_test_records_read = 0`
- `formal_test_assets_opened = false`
- `frozen_holdout_records_read = 0`
- `minimal_validity_target_test_used = false`
- `parameter_selection_from_formal_test = false`
