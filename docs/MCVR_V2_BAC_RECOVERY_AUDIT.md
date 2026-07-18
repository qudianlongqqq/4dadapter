# MCVR V2-BAC Phase-1 Recovery Audit

## Frozen scope

Phase 1 audits the completed Cartesian V2-BAC implementation without reopening
the frozen validation holdout or formal test. The protected implementation is
commit `9528e8f0ea558bb5bcb742708aa3f4b45172206a` on local protection branch
`wip/mcvr-v2-bac-completed`; this audit runs on `audit/fix-mcvr-v2-bac`.

The preregistered recovery cohort has identity
`3241a30c47a532dfbee4448bcd6556f0e5a67b752d17f85505e1f99f8a48ec51`.
It is drawn deterministically from the non-holdout validation-tune pool with
seed 43017. Development contains 512 molecules/1024 records and the nested
diagnostic cohort contains 128 molecules/256 records. Molecule overlap with the
frozen holdout is zero. All recovery artifacts must keep:

- `test_records_read=0`
- `test_assets_opened=false`
- `frozen_holdout_records_opened=0`
- `validation_only=true`

## Preregistered compute budget

At most five Phase-1 GPU optimizer invocations are permitted: one two-batch
diagnostic, one smoke of at most 200 steps, and one 1k pilot each for A0, D0,
and D1. No retry is implicit. There is no 10k/formal-large run, model expansion,
holdout reuse, or formal-test access. Hidden dimension remains 64 and the shared
backbone remains four layers.

## Audited chain and static findings

The chain is source coordinates -> canonical constraints -> offline target ->
canonical PyG batch -> D1-B base field plus Angle/Clash branches -> fusion ->
trust clipping -> global safety gate -> four teacher steps -> per-trajectory
safety evaluation -> accept or exact-source rollback.

Units and shapes are internally consistent at the public boundaries. Bond
coordinates and thresholds are Angstrom; angles and angle statistics are
radians; coordinates and Cartesian proposals are `[N, 3]`; unique bonds are
`[2, B]`; canonical angles are stored `[3, A]` and converted to `[A, 3]` in the
model/loss. Constraint fields are static, versioned, model-independent data.
No hidden-width, layer, weight, or intermediate representation enters a cache.

The following findings require quantitative confirmation before a repair is
selected:

1. `ChemicalValidity` clash evaluation excludes directly bonded 1-2 pairs but
   includes 1-3 pairs. The sparse target/model detector excludes both 1-2 and
   1-3. The optimized and accepted clash definitions therefore differ.
2. `MCVRBACLoss` performs a second forward at `t=0`. It supervises a proposal
   made only from fused Angle+Clash branches, while inference integrates the
   complete D1-B base plus fused branch field at nonzero teacher times.
3. The new branch loss applies a Bond residual objective to an Angle+Clash-only
   proposal. Bond is also supervised independently by the unchanged base loss.
4. Each branch is attenuated by `bac_constraint_scale=0.05`, learned strength,
   confidence, branch gate, per-atom constraint-count averaging, fusion gate,
   trust clipping, global safety gate, and inference `step_size=0.25`.
5. The registered constraint-type embedding is not consumed in forward.
6. An existing finite backtracking helper is tested but `infer_bac` evaluates
   only the four integrated trajectory states and never calls it.
7. Safety treats every positive Bond/Angle/Clash/Ring delta as regression with
   default epsilon zero, while meaningful improvement uses a separate fixed
   `1e-8` aggregate threshold. Weighted objective regression and lack of
   meaningful improvement are not represented as separate reasons.
8. Per-atom scatter divides by every incident constraint, including inactive
   constraints whose learned weight is zero. Constraint-rich atoms can thus be
   attenuated by inactive neighbors.

## Frozen repair candidates

Before quantitative selection, the permissible minimal candidates are:

1. Align clash target/model and evaluator topology exclusions.
2. Introduce explicit absolute/relative non-regression tolerance only if the
   precision audit proves false rejection from numerical noise.
3. Activate finite predefined backtracking while retaining hard ring,
   chirality, identity, finite-value, and trust protections.
4. Remove a proven duplicate attenuation or train/eval proposal mismatch.

No loss-weight sweep or capacity change is permitted. The diagnostic outputs
will classify evidence as DATA_SUPPORT, TARGET_SCALE, TARGET_CONFLICT,
LEARNING_SIGNAL, PROPOSAL_ATTENUATION, SAFETY_BOTTLENECK, CAPACITY_LIMIT, or
METRIC_POWER before selecting the smallest repair.

## Diagnostic contract

`scripts/audit_ecir_mvr_v2_bac_recovery.py` reads all 256 diagnostic records
for CPU data/support statistics and at most two batches for model, gradient,
proposal, precision, and rollback diagnostics. It uses `autograd.grad`, creates
no optimizer, and does not mutate optimizer or checkpoint state. Local
Bond/Angle/Clash component vectors are explicitly diagnostic negative-residual
directions; they are not serialized targets and do not alter training semantics.

This document is the run-before-repair audit. Quantitative evidence and the
selected repair will be appended only after the preregistered diagnostic
finishes.

## Quantitative diagnostic result

The single preregistered diagnostic completed successfully. All 256 records
were used for read-only data/target statistics and exactly two batches (128
records) were used for model, gradient, proposal, and rollback diagnostics.
No optimizer was created or mutated.

Data support is asymmetric:

- Bond: mean 5.33 active constraints/graph; 254/256 graphs have Bond activity.
- Angle: mean 1.18 active constraints/graph and mean active ratio 1.47%; 128/256
  graphs have Angle activity, so Angle has limited but usable development power.
- Clash: zero active sparse clash constraints in all 256 real diagnostic
  records. Clash scientific efficacy cannot be estimated on this cohort.
- Active combinations are 126 Bond-only, 127 Bond+Angle, one Angle-only, two
  none, and zero combinations containing Clash.

The clash exclusion mismatch is real in code but did not create an active
metric/model disagreement at the 1.0 Angstrom threshold in this cohort. The
only sub-threshold pairs found were 69 directly bonded pairs, which both
definitions exclude from their final score. No penetrating 1-3 pair occurred.
The mismatch remains a semantic debt, not the demonstrated cause of this
development result.

Local diagnostic target directions show moderate rather than catastrophic
conflict. Bond-Angle cosine has mean -0.0114 and p95 0.0561. Cancellation ratio
has median 0.00676, mean 0.0860, and p95 0.296. This does not support a primary
TARGET_CONFLICT diagnosis. Clash component norm is exactly zero on every real
diagnostic record.

The loss signal reaches the model. Across the two batches, global gradient
norms are 0.060/0.037 for Bond, 0.082/0.218 for Angle, and 0.018/0.027 for
Clash. Angle is not gradient-starved. Bond-Angle gradient cosine is mildly
negative (-0.071/-0.046), and Bond-Clash is -0.204/-0.263; these conflicts are
measurable but not evidence of capacity failure. The Clash loss/gradient in
these training-path batches comes from deterministic synthetic corruptions,
not active real validation clashes.

The proposal chain shows attenuation:

- base raw proposal norm mean: 0.02996
- fused full proposal norm mean: 0.03026
- raw-to-fused scale mean: 1.00462
- fused-to-trust-clipped scale mean: 0.9621
- graph clipping fraction: 9.375%
- clipped-to-global-gated scale median: 0.01157, mean: 0.13635
- target cosine changes only from 0.23194 raw to 0.23493 fused

Thus the learned Angle/Clash additions barely alter the existing D1-B field,
and the global safety gate attenuates the full field much more than fusion or
clipping on the median graph. The current branch scatter also counts inactive
constraints in its per-atom denominator even though their numerator weight is
zero; this is a direct, evidence-consistent dilution when only 1.47% of angles
are active.

Current four-step acceptance is 12/128 (9.375%) on the nested diagnostic
subset. Of 116 rejected records:

- 70 exceed both atom and molecule trust limits at the final considered state;
- 65 have no meaningful BAC gain;
- 51 rejected records nevertheless have positive BAC gain;
- 33 have only the two trust-limit reasons;
- 26 worsen Angle, 18 Bond, 12 Clash, 17 Ring, and six change chirality, often
  co-occurring with trust failure.

Absolute tolerance probes from zero through `1e-6` leave acceptance exactly
12/128. This rejects numerical epsilon relaxation as the Phase-1 repair. The
float64-input probe also produces identical decisions because
`ChemicalValidity.evaluate` explicitly converts coordinates to float32.

## Root-cause classification

- **DATA_SUPPORT_FAILURE (Clash): supported.** Zero real active Clash records.
- **METRIC_POWER_FAILURE (Clash): supported.** No Clash-active subset exists.
- **TARGET_SCALE_FAILURE: partial.** Angle target directions exist; Clash does
  not. The dominant measured loss is downstream proposal attenuation rather
  than absent Angle target magnitude.
- **TARGET_CONFLICT_FAILURE: not primary.** Median cancellation is below 1%.
- **LEARNING_SIGNAL_FAILURE: rejected.** Angle and Clash gradients are nonzero.
- **PROPOSAL_ATTENUATION_FAILURE: supported.** Fused field differs from base by
  about 0.46% on average; inactive constraints dilute scatter; the global gate
  has median scale 0.0116.
- **SAFETY_BOTTLENECK: supported.** 51 positive-gain proposals are rejected and
  33 are trust-only failures, while the existing finite backtracking helper is
  not used by inference.
- **CAPACITY_LIMIT: rejected.** Target/fusion/safety failures precede any
  defensible capacity diagnosis; width 64/layers 4 remain frozen.

## Selected minimum repair D1

Three linked changes are selected, each behind a new flag with the legacy
default unchanged:

1. **Active-only constraint scatter normalization.** For D1, the per-atom
   denominator counts active constraints only. The numerator and equivariant
   directions are unchanged. This removes proven zero-weight dilution without
   changing the canonical dataset schema or serialized targets.
2. **Inference-aligned BAC proposal loss.** Legacy loss uses an Angle+Clash-only
   proposal from a second `t=0` forward. D1 supervises the actual full
   `v_final` first-step field at teacher time 1.0 and step size 0.25. This makes
   training gradients reach the same base+branch+clip+gate path evaluated at
   inference.
3. **Finite proposal backtracking.** D1 tries only preregistered scales 1.0,
   0.5, and 0.25 for each trajectory proposal and retains all hard trust,
   Bond/Angle/Clash/Ring, chirality, identity, and finite checks. It cannot
   accept a proposal with no BAC gain and cannot remove safety checks.

No epsilon, metric threshold, loss weight, model width/layer count, target
asset, or dataset schema changes. D1-B and legacy V2 checkpoint strict-load
surfaces are unchanged because the new controls add no parameters. Clash is
explicitly out of scope for a positive Phase-1 efficacy claim due to zero data
support.
