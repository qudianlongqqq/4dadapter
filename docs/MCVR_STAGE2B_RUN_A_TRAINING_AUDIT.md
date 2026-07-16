# MCVR Stage 2b Run A Training Audit

## Scope and authorization

Only `ecir_mvr_stage2b_run_a_rigid_only_seed42_5k` was executed. Run B, Run C, 20k, and 100k were never started. The protected untracked file `reports/global4d_profile_bundle_verification.json` was not modified, staged, or committed.

## Preflight gates

- Stage C decision: PASS
- Train/validation molecules: 500 / 100
- Train/validation records: 750 / 130
- Train/validation molecule overlap: none
- Training mixture plan: 45% real, 30% synthetic, 25% clean identity
- Real-source contribution: 22.5% ETFlow and 22.5% Cartesian of the 1000-record epoch plan
- Training Cartesian scale: 0.50 only; validation Cartesian scale: unseen 0.35 only
- Out-of-domain extreme fraction: 0
- Test records and test paths read: 0
- Data audit identity: `b698ab70d2873e4d72140009402ac850f5ba76a1f43619cd3c9183462a692e21`

The targeted ECIR/MCVR suite passed 47 tests. The full repository suite passed 308 tests with 23 known warnings. The old ECIR checkpoint loaded strictly, and the frozen Stage B rescued decision remained reproducible and test-free.

## Run configuration

- Seed 42; 5000 optimizer steps; batch size 8; AdamW, learning rate 2e-4, weight decay 1e-6
- Four teacher/inference steps over t=[0,1]
- Cartesian velocity, rigid/local repair, rigid gate, global safety gate, trust clipping, deterministic validity features, learned error embedding, uncertainty head, identity protection, and minimal-validity targets enabled
- Torsion repair disabled; torsion gate fixed to zero; torsion velocity scales zero; torsion code and torsion anchor retained

No Global4D q output, Jacobian q-target inversion, Strict Global4D fusion, restrained-MMFF training target, nearest-reference fallback, or Cartesian ten-step extreme source was used.

## Execution record

The first process exited before optimizer step 1 because PyTorch 2.11 exposes its version as a string-like object that PyYAML would not serialize. Converting environment version fields to native strings fixed the fail-closed metadata write; targeted tests passed before restart.

At step 2000, the initial identity diagnostic reported 75% unchanged and stopped. Investigation showed every clean candidate was rejected and exactly equal to its input. The apparent failures came only from self-Kabsch float residuals around 1e-6 Å. The diagnostic was corrected to use direct coordinate equality; independent inference verified 20/20 identity at both steps 1000 and 2000. Training then resumed from the saved step-2000 model and optimizer state. This correction did not alter data, targets, model weights, configuration, or non-inferiority margins.

The run completed at step 5000 without a genuine stop condition. Active segment times were 274.458 and 401.715 seconds, totaling 676.173 seconds. Run metadata records the resume checkpoint and preserves the original start time.

## Monitoring outcome

- No NaN/Inf or gradient explosion
- Velocity norm did not show sustained abnormal growth
- Rigid gate remained selective and did not collapse to all-zero or all-one
- Global safety gate had brief low values but recovered and did not remain collapsed
- Clean controls remained exact identity under acceptance
- Severe clash and chirality did not worsen
- Torsion gate and torsion contribution remained exactly zero
- All five full validations passed accuracy non-inferiority
- No data or frozen identity changed

The complete metric history is in `logs_ecir_mvr/stage2b/run_a_rigid_only_seed42_5k/metrics.csv`; checkpoint selection and loss summaries are in `diagnostics/ecir_mvr/stage2b/run_a/`.
