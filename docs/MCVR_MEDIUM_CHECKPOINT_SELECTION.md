# MCVR Medium Checkpoint Selection

## Selection result

No scheduled medium checkpoint was eligible for selection.

The frozen schedule required complete validation at steps 5000, 10000, 15000, and 20000. Training stopped at step 2000 under `velocity_norm_sustained_growth`, so none of those checkpoints existed and `best_noninferior_validity.ckpt` was not created.

The saved `last.ckpt` at step 2000 was evaluated only to complete the required early-stop Gate 2 audit:

- SHA256: `4a82898156ffddebfab9ad409d1cd68c4890f2bf7116ff20da63831daa9030e3`
- Scheduled validation checkpoint: false
- Accuracy non-inferiority in post-stop evaluation: pass
- Gate 2: fail
- Selection status: diagnostic only; not a medium candidate

No checkpoint was selected by loss. No missing 5000/10000/15000/20000 checkpoint was synthesized, copied, or relabeled.
