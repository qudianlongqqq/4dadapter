# Formal training summary

Final formal variant: **BA+C**.

No new formal training is required: BA uses the already frozen 2,500-step LSGO-B checkpoints for seeds 173/181/193 (473,674 parameters each), and C is an analytic steric barrier with zero trainable parameters. The 83,337-parameter torsion pilot heads are retained as failed internal artifacts and are not formal checkpoints.

Formal added steps: 0. Formal added parameters: 0. The original frozen BA training identity and checkpoint SHA values are recorded in `CHECKPOINT_FREEZE_MANIFEST.json`.
