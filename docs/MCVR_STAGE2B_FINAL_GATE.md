# MCVR Stage 2b Final Gate 1

## Final decision

- `run_a_decision`: `RUN_A_PASS`
- `run_b_decision`: `RUN_B_HARMS`
- `selected_medium_configuration`: `RUN_A`
- `current_decision`: `GO_20K_WITH_RUN_A`
- `20k_permitted`: `true` — permission is for an explicit future human launch only
- `100k_permitted`: `false`
- Run C started: `false`
- Test records read: `0`

Run A's frozen result, selected checkpoint, identities, clean identity controls, and all 22 upstream conditions remain valid. Therefore the pre-registered Gate 1 rule retains Run A even though Run B is harmful relative to it.

## Selected medium configuration

The generated candidate is `configs/ecir_mvr_medium_20k_run_a_selected.yaml`. It preserves the Run A rigid-only architecture, losses, data mixture, seed, teacher steps, inference settings, and non-inferiority margins; only the medium-run identity/output locations and 20,000-step schedule are declared. Torsion repair remains disabled, with torsion scale and gate fixed at zero.

The recorded next command is:

```text
python scripts/train_ecir_mvr_medium_20k.py --config configs/ecir_mvr_medium_20k_run_a_selected.yaml
```

This command was **not executed**. The command is a handoff contract for the separately authorized medium-run launcher; the current Stage 2b scripts deliberately remain capped at 5k and are not repurposed here.

## Stop boundary

Stage 2b ends at this Gate 1 record. No actual 20k run, 100k run, Run C, or test evaluation is authorized or started by this experiment completion.
