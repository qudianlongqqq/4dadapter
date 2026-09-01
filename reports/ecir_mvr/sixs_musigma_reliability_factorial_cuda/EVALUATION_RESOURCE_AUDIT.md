# SIXS factorial evaluation resource audit

> Audit date: 2026-08-29 (Asia/Shanghai)  
> Scope: frozen seed307 internal-development factorial only. No Formal, large-holdout or xTB access; no metric, cohort, checkpoint, model or Sigma-v2 change.

## Decision

The post-SDF scientific evaluation is CPU-only in the actual implementation. Calling the stable environment a “CPU fallback” is misleading: it is a separate evaluation worker for RDKit, PoseBusters, NumPy, pandas and PyArrow work that never creates a CUDA tensor. Training and the currently optimized model-forward/coordinate stage benefit materially from CUDA, but their mathematics do not intrinsically require a GPU.

```text
MODEL_INFERENCE_REQUIRES_GPU = NO
COORDINATE_GENERATION_REQUIRES_GPU = NO
V3D_REQUIRES_GPU = NO
POSEBUSTERS_REQUIRES_GPU = NO
TABLE_MATERIALIZATION_REQUIRES_GPU = NO
BOOTSTRAP_REQUIRES_GPU = NO
EVALUATION_AFTER_SDF_IS_CPU_ONLY = YES
CPU_EVALUATION_IS_COMPUTE_DOWNGRADE = NO
```

“Requires GPU = NO” means portable to CPU, not equally fast. The production training/model-forward stage remains assigned to CUDA for throughput.

## Exact lifecycle and resource ownership

| Stage | Actual implementation/libraries | Torch / CUDA behavior | Primary resource |
|---|---|---|---|
| A. Training | PyTorch, project graph/geometry code | Torch tensors and CUDA kernels in the CUDA run; model/optimizer occupy GPU memory | GPU plus CPU batch assembly and RAM |
| B. Checkpoint load | `torch.load`, model state load | CPU deserialization followed by CUDA parameter copies in optimized worker | CPU I/O + GPU memory |
| C. Model inference | frozen graph batch, mu/sigma/reliability heads | CUDA tensors/kernels in optimized worker; CPU execution is mathematically available | GPU |
| D. Coordinate generation | differentiable geometry, VJP, rigid projection, cap/safety | CUDA kernels in optimized worker; `create_graph=False`; per-record RDKit/cache work also uses CPU | GPU + CPU |
| E. SDF writing | RDKit `SDWriter`, atomic rename | Coordinates copied to CPU; no CUDA computation in writer | CPU + disk |
| F. SDF parsing | RDKit `SDMolSupplier` | No CUDA tensor/kernel | CPU + RAM |
| G. Bond/Angle extraction | already frozen evaluation payload, NumPy/pandas summaries | No CUDA tensor/kernel after split | CPU + RAM |
| H. Validity3D | external worker, RDKit/NumPy | External environment has no Torch installed | CPU |
| I. PoseBusters | PoseBusters 0.6.5, RDKit/pandas; worker count 1 | External environment has no Torch installed | CPU |
| J. Table construction | NumPy/pandas | No CUDA tensor/kernel | CPU + RAM |
| K. Bootstrap preparation/final bootstrap | pandas/NumPy cluster arrays and resampling | No CUDA tensor/kernel | CPU + RAM |
| L. JSON/CSV/Parquet writes | stdlib JSON, pandas, PyArrow | No CUDA tensor/kernel | CPU + disk |
| M. DONE freeze | hash validation and atomic JSON | No CUDA tensor/kernel | CPU + disk |

## Minimal dynamic observations

The frozen J0-R1 CUDA-produced SDF was used without model forward at N=32, N=128 and N=5000. The probe executed imports, first/all-molecule parsing, property extraction, NumPy creation, pandas construction, merge, Parquet/CSV/JSON writes and cleanup.

| Environment / N | Result | Representative timings and memory |
|---|---|---|
| CUDA Python, N=5000 | PASS | import 1.501 s; full parse 0.945 s; DataFrame 0.057 s; peak observed RSS about 733 MB; CUDA allocated bytes remained 0 |
| CPU orchestration Python, N=5000 | PASS | import 1.190 s; full parse 0.975 s; DataFrame 0.053 s; peak observed RSS about 421 MB |
| CUDA Python with frozen model loaded, N=5000 | PASS | same post-SDF operations passed; model allocation about 2.86 MB; no post-SDF CUDA allocation or kernel was introduced |

CPU and CUDA probes produced the same IDs, ordering, schema and values for all tested core frames. Full probe artifacts are under `resource_audit_runtime/`.

The exact external evaluator was also exercised twice at N=32 and N=128. Per-record V3D and every V3D component were identical. Per-record PB and every PB scientific component were identical; only the non-scientific temporary `file` path differed. At N=128, PB took 5.66 s and V3D 4.26 s. The external process reported `torch_imported=false`.

## Environment inventory

| Component | CUDA training environment | CPU orchestration environment | Stable external evaluator |
|---|---|---|---|
| Python | 3.11.15, MSC 1944 | 3.11.8, MSC 1937 | 3.11.15, MSC 1944 |
| Torch | 2.11.0+cu128 | 2.13.0+cpu | not installed |
| CUDA runtime | Torch CUDA 12.8, available | none | none |
| RDKit | 2026.03.4 | 2026.03.3 | 2025.09.6 |
| NumPy | 1.26.4 | 2.3.3 | 2.0.2 |
| pandas | 3.0.3 | 2.3.3 | 2.2.3 |
| SciPy | 1.17.1 | 1.16.2 | 1.14.1 |
| PyArrow | 25.0.0 | 25.0.0 | 20.0.0 |
| PoseBusters | not installed | not installed | 0.6.5 |
| BLAS | conda BLAS/OpenBLAS 3.9.0 reporting | scipy-openblas 0.3.30 | scipy-openblas 0.3.27 |

The scientific separation is already explicit: the main worker freezes SDF and record/predictive inputs, then launches `evaluate_sixs_musigma_external.py` in the external-validity environment. That worker contains no model forward and no Torch dependency.

## J0-R1 frozen status

J0-R1 training completed 17,500/17,500 steps and its final checkpoint hash is `60db7bef39e4a91d98906750fd02563491f1e641aa7d04bb9812b0f8f781a6a7`. Stable evaluation completed 5,000 records and froze DONE. The recovered result is Proposal V3D 0.4802 and Proposal PB 0.9320.

The audit did not retrain J0-R0/J0-R1 and did not regenerate coordinates. A recovery that had already begun before this audit did regenerate J0-R1 coordinates once. The original CUDA SDF (`9d7eb31b...`) and recovery SDF (`58ea4665...`) have equal IDs/order/atom counts; 4,954 records are coordinate-identical and 46 differ by at most 0.0001 Å, consistent with an SDF decimal serialization boundary. This distinction is retained rather than silently treating regeneration as byte-equivalent.

