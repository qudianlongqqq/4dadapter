# Failure modes

## Known failure modes and limits

| mode | evidence | status |
|---|---|---|
| Rare unrestricted energetic catastrophe | one seed307 Unrestricted record exceeds +100 kcal/mol; max about +570 | VERIFIED |
| Rare unrestricted movement tail | tau max 0.03855 Å; serialized per-atom displacement max about 0.182 Å | VERIFIED |
| Restricted bound/cap weakly active | tau reaches 0.010 and about 0.58% records activate atom cap | VERIFIED |
| Global conformation largely unchanged | Reference RMSD changes are tiny despite local validity/energy gains | VERIFIED |
| Predictive mu degradation under full-joint training | mechanism audit reports Bond/Angle mu degradation versus frozen J1-R1 | VERIFIED, downstream Proposal still improves |
| Bond/Angle per-record reversals | paired component transitions include candidate wins and losses | VERIFIED |
| Clash/aromatic/PB components mostly unchanged | seed307 component rates tie for many components | VERIFIED; some endpoints have low event power |
| Size/flexibility/poor-Source dependence | quintile analyses exist and generally retain gains | PARTIALLY CHARACTERIZED, seed307 DEV only |

## Uncharacterized modes

- seed-to-seed movement and xTB-tail instability;
- cross-upstream failure under different Source defect distributions;
- very large/highly flexible molecules beyond current support;
- explicit torsional correction failures and conformational-basin changes;
- wrong topology, bond order, atom mapping, or protonation at input;
- symmetry-equivalent atom effects in Reference RMSD;
- rare stereochemical/ring failure rates with adequate power;
- calibration failure of sigma or Reliability under distribution shift;
- runtime/memory scaling with molecule size.

## Required limitations language

The paper must proactively state that the method is a fixed-topology local refiner, not a conformer search or topology repair method; that current Reference RMSD gains are negligible; that Unrestricted has a rare positive-energy tail at seed307; and that cross-upstream/seed/protected robustness remains unknown until completed.

```text
KNOWN_FAILURE_MODES = RARE_UNRESTRICTED_ENERGY_AND_MOVEMENT_TAIL; NEGLIGIBLE_GLOBAL_REFERENCE_RMSD_CHANGE; PER_RECORD_LOCAL_REVERSALS; FULL_JOINT_MU_DEGRADATION
UNCHARACTERIZED_FAILURE_MODES = SEED_TAIL_STABILITY; CROSS_UPSTREAM; EXTREME_SIZE_FLEXIBILITY; TORSIONAL_GLOBAL_REFINEMENT; WRONG_TOPOLOGY; SIGMA_CALIBRATION_SHIFT
```

