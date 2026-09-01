# Runtime, resources and downstream usefulness

## Current runtime evidence

Training logs record GPU use, training elapsed time and approximately 275 MB peak allocated training memory for new multiseed runs. The evaluation resource audit shows that post-SDF V3D/PB/table work is CPU-only and gives evaluator probe timings. These do **not** establish final-model inference latency or fair end-to-end throughput.

Missing final evidence:

- trainable parameter count;
- neural forward plus coordinate-generation time per record/molecule;
- batch throughput, batch size, device and synchronized GPU timing;
- peak allocated/reserved inference memory;
- separately measured graph/RDKit preprocessing and SDF serialization where material;
- matched MMFF94s CPU runtime and success rate;
- xTB single-point wall time; and, only if run, xTB optimization time/steps/convergence/failures.

Minimum benchmark: one frozen non-outcome cohort, warm-up followed by repeated synchronized CUDA measurements, fixed batches, explicit preprocessing boundary, exact hardware/software/thread counts, and identical record denominators. MMFF/xTB run with frozen CPU/thread settings. Report median plus spread and total throughput; never infer efficiency from training time.

## Downstream value

Computational efficiency of refinement and utility to a downstream property/docking model are separate claims. The current scoped paper is about conformer refinement and has no frozen downstream pipeline. A new property, docking or representation study would introduce another task, dataset and selection loop without being necessary to establish the core method.

```text
RUNTIME_EVIDENCE = MISSING
DOWNSTREAM_TASK_REQUIRED = NO
DOWNSTREAM_EFFICIENCY_REQUIRED = NO
RUNTIME_ONLY_SUFFICIENT = YES_FOR_CURRENT_SCOPE
DO_NOT_CLAIM = DOWNSTREAM_PERFORMANCE_OR_EFFICIENCY
```
