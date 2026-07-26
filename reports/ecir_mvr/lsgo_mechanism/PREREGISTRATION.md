# LSGO-B Mechanism & Sufficiency Audit — preregistration

Status: **FROZEN BEFORE MECHANISM_CONFIRM xTB ACCESS**  
Base: `a2a4b5d65cd358caaba6185121bb7e62aac8d2ae`  
Branch: `research/lsgo-ba-mechanism-audit`

## Scope and invariants

This is a frozen-inference mechanism audit. It does not train a primary model, tune a checkpoint, optimize coordinates with xTB, or access formal test/frozen holdout. xTB energy and force are diagnostic outcomes only. A Reference ensemble is a sampled set of plausible conformers; no single Reference is treated as the unique target and no Source is assumed to be worse than every Reference.

The frozen BA checkpoints are seeds 173/181/193; seed 181 is the preregistered representative for thresholds, abnormal-only and force analysis. The update is one direct normalized-gradient step with graph RMS budget `0.003 Å`, per-atom cap `0.03 Å`, followed by the unchanged topology/chirality/ring/hard-steric safety acceptance. B-only, A-only and BA differ only in which primitive family participates in the objective. BA-all must be exactly equivalent to the historical solver.

## Cohort and matching

`MECHANISM_CONFIRM` contains 48 TRAIN-only, multiply referenced molecules and three Source records per molecule (144 Source records). It is SHA-ranked before outcomes, split equally into canonical free-rotor bins 0–2, 3–4 and ≥5. It has zero identity overlap with the 1,400 historically exposed molecules, including BAT external. All available 857 References are evaluated. Nearest Reference means minimum aligned RMSD under the existing frozen matching rule.

## Thresholds frozen without xTB

Representative seed 181 was evaluated on all 46,581 TRAIN Reference conformers (2,000 molecules):

- high/low BA threshold: median mean group-wise squared z = `0.5751809014217901`;
- abnormal-only Bond threshold: TRAIN Reference primitive `|z|` p95 = `0.9558624036568723`;
- abnormal-only Angle threshold: TRAIN Reference primitive `|z|` p95 = `2.393193580500847`;
- Source/Reference energy axis: `E_Source - median(E_Reference ensemble)`; approximately tied is `|Δ| ≤ 0.1 kcal/mol`.

## Frozen analyses

- Source vs Reference minimum, median and aligned-RMSD-nearest energy, with molecule-cluster bootstrap (5,000 replicates, seed 47021).
- Four preregistered BA-abnormality/Source-energy quadrants.
- B/A/BA energy ablation on the same coordinates, checkpoints, normalization and trust region.
- Spearman primary and Pearson diagnostic associations of initial B/A/BA abnormality with energy gain.
- Chemical strata: bond type, hybridization, aromatic, amide-like, ring, heavy atoms, rotatable bonds and flexibility.
- xTB CLI `--grad` feasibility: energy identity ≤1e-7 Eh, finite values, repeatability ≤1e-7 Eh/bohr, atom order identity, and central finite-difference sign/unit audit at `1e-4 Å` before force conclusions.
- Force subset: Source index 0 from six molecules in each flex bin (18 records), before and after seed-181 BA. Independent B/A/T projections plus BA and BAT joint SVD union projections; relative/absolute rank tolerances `1e-8`/`1e-10`.
- Frozen seed-223 K3 torsion surprise association with ensemble energy excess, remaining BA energy excess, and physical torsion-force projection.
- BA-all vs TRAIN-Reference-p95 abnormal-only; no-op is diagnostic only.

## Decision rules

- Component dominance: a single component retains ≥80% of BA median gain while the other retains ≤50%.
- BA synergy: BA exceeds the best single component by ≥10%, with molecule-bootstrap lower CI above zero.
- Torsion is physically actionable if incremental BAT-union projection has median ≥0.05 overall or ≥0.10 in high-flex. If that occurs while torsion-surprise Spearman is weak (`|ρ|<0.20`), output `TORSION_REDESIGN_WARRANTED`; if both are weak, output `TORSION_LOW_ACTIONABILITY`.
- Abnormality gate candidate: ≥30% movement reduction, ≥80% BA median benefit retained, and harmful p99 worsens by no more than 0.02 kcal/mol. Otherwise `KEEP_BA_ALL`.
- BA local strain support requires median BA-union force fraction ≥0.25 plus tail-safe energy improvement. This is explicitly limited to current Source and the 0.003 Å micro-refinement regime.

Allowed final labels are only: `BA_LOCAL_STRAIN_CONFIRMED`, `B_DOMINANT`, `A_DOMINANT`, `BA_SYNERGISTIC`, `TORSION_REDESIGN_WARRANTED`, `TORSION_LOW_ACTIONABILITY`, `ABNORMALITY_GATE_CANDIDATE`, `KEEP_BA_ALL`.

Formal test reads = **0**. Frozen holdout reads = **0**.
