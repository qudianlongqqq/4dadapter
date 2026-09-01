# Checkpoint and recovery audit

## Endpoint and selection

- Scientific endpoint is exactly step 17,500 in config, final checkpoint, final status, and runners.
- No early stopping, best-DEV, best-V3D, or best-PB checkpoint branch was found.
- DEV is evaluated after the final checkpoint.
- Step22,500 is a separate capacity diagnostic and is not the selected endpoint.

## Recovery contents

Both seed307 recovery checkpoints are at step17,500 and contain:

- model state;
- optimizer state;
- scheduler state;
- global step;
- config SHA256;
- Python RNG state;
- NumPy RNG state;
- Torch CPU RNG state;
- Torch CUDA RNG state;
- dedicated sampling generator state.

The sampler is stateless apart from that generator; no separate epoch/cursor state exists. Evidence: checkpoint key inspection and runner lines `496-509,570-584` (Restricted) and `345-355,385-392` (Unrestricted).

## Continuity

Training logs are monotone and complete at their logging cadence. Final checkpoint and recovery checkpoint both report 17,500. Restricted evaluation recovery preflight is PASS, including checkpoint, coordinate, comparator, and record identity checks.

```text
FINAL_STEP_17500 = VERIFIED
DEV_CHECKPOINT_SELECTION = NO_EVIDENCE
RECOVERY_STATE_COMPLETE = YES
RECOVERY_DISCONTINUITY = NO_EVIDENCE
EXACT_PROCESS_INTERRUPTION_HISTORY = UNKNOWN
```

The absence of logged discontinuity is not proof that no process restart occurred; it means no scientific state discontinuity is visible in the stored sequence and recovery metadata.


