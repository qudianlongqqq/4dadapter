# ECIR progressive experiment plan

The thresholds below are project configuration, not universal chemistry
standards.

1. Stage 1: five molecules, 200 CPU steps and 200 CUDA steps.
2. Stage 2: 500 train / 100 val, 5,000 steps.
3. Stage 3: 5,000 / 500, 20,000 steps, only after Stage 2 GO.
4. Stage 4: frozen formal-large split, 100,000 steps, only after independent
   Stage 3 GO and user confirmation.

The default gate requires at least two internal metrics with directionally
consistent paired molecule-bootstrap CIs, COV within 0.02, RMSD/MAT within 0.02,
clean identity stability and an unseen checkpoint/NFE/seed pass.

## Executed result

Both Stage 1 CPU and RTX 5080 CUDA smokes completed. The final heterogeneous
Stage 2 checkpoint is
`logs_ecir/stage2_heterogeneous_500_100_5k/step005000.ckpt`.

Stage 2 improved bond, angle, torsion and ring metrics, but aligned RMSD changed
by +0.01917 on average with paired 95% CI `[+0.01170,+0.02738]`; the CI exceeds
the 0.02 margin. No genuine unseen-checkpoint/NFE/seed cohort is locally
available. Decision: `NO_GO`. Neither 20k nor 100k training may start.

The progressive Windows and Linux scripts enforce this fail-closed decision by
reading the evaluation JSON before printing later commands.
