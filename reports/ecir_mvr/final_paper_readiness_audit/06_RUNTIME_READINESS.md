# Runtime and compute readiness

```text
RUNTIME_EVIDENCE = MISSING_FOR_CURRENT_FINAL_FAIR_COMPARISON
RUNTIME_BENCHMARK_REQUIRED = YES
```

## Evidence that exists

- Current final model parameter-group audits report 766,288 trainable parameters.
- Frozen environment: NVIDIA GeForce RTX 5080, PyTorch 2.11.0+cu128, CUDA runtime 12.8, RDKit 2026.03.4; evaluator environment uses RDKit 2025.09.6 and PoseBusters 0.6.5.
- Training summaries record about 275 MB peak allocated GPU memory for the new Unrestricted runs. This is training evidence, not inference peak VRAM.
- The primary coordinate pipeline reports 605.998 s total, but that interval combines topology/reference work and six neural methods; it is not a per-method latency benchmark.
- Historical external-baseline runtime exists (`external_refinement_baselines/RUNTIME_ANALYSIS.md`), but it is not the valid current-final MMFF comparison and must not be used as the headline final-cohort speed table.
- xTB single-point job runtimes exist in caches/status artifacts. No current-final GFN2-xTB optimization runtime exists.

Missing authoritative final evidence: warm inference latency, throughput, ms/conformer, inference peak VRAM, valid current-final MMFF runtime, and GFN2-xTB optimization runtime if that baseline is claimed.

## Minimum fair benchmark protocol (do not run in this audit)

1. Freeze the exact 5,000 final records or a predeclared deterministic subset for expensive xTB optimization.
2. Report hardware, OS, software versions, process/thread counts, and whether preprocessing/serialization is included.
3. SIXS: RTX 5080, fixed production batch size, at least one warm-up pass, synchronized wall clock, parameter count, throughput, median and P95 per-record latency, and peak inference VRAM.
4. MMFF94s: CPU only, fixed threads and existing 500-iteration protocol, success/failure denominator, median and P95 successful-job latency plus all-record wall time.
5. GFN2-xTB optimization: CPU, fixed workers/threads/cycles/convergence, success/failure denominator, median and P95 job latency plus all-record wall time.
6. Never describe GPU SIXS and CPU optimizers as identical hardware execution. Compare wall-clock operating points with hardware differences explicit.

This benchmark needs no model training.
