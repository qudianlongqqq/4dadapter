# ETFlow + 4D Jacobian evaluation plan

## 1. Formal `scripts/eval.py` output

`scripts/eval.py` builds `EuclideanDataset(partition=..., split="test")`, loads a
model from `config["model"]` and `config["model_args"]`, loads the checkpoint
`state_dict`, and calls `model.sample(...)` with the batched rotatable-bond and
atom-bond-influence tensors.

For each molecule it reads every reference conformer from the processed `.pt`
file and generates twice that many conformers. The saved file is:

```text
<output_dir>/samples/<task_name>/<timestamp>/generated_files.pkl
```

It is a pickled list of `torch_geometric.data.Data` objects. Each item contains:

```text
smiles:  ordered SMILES
pos_ref: [num_reference, num_atoms, 3]
pos_gen: [2 * num_reference, num_atoms, 3]
rdmol:   the molecule object from the dataset item
```

`times.pkl` contains sampling times per generated conformer.

## 2. Input required by `scripts/eval_cov_mat.py`

`eval_cov_mat.py --path` calls `load_pkl(path)` and passes the resulting list to
`CovMatEvaluator`. The evaluator requires `smiles`, `pos_ref`, and `pos_gen`.
It reconstructs ordered RDKit molecules from `smiles`; the stored `rdmol` is not
read by the current evaluator.

The default evaluator ratio is 2. Therefore a molecule is skipped unless:

```text
num_generated >= 2 * num_reference
```

The script sends COV-R/COV-P and MAT-R/MAT-P to an offline or online W&B run.
It logs evaluation progress, but the current code does not print the returned
metric tables or write a standalone summary CSV itself.

## 3. Can subset output feed `eval_cov_mat.py` directly?

Yes, use the subset directory's `generated_files.pkl`:

```bash
WANDB_MODE=offline python scripts/eval_cov_mat.py \
  --path logs_eval_subset/seed42_scale001_q0001_n20/generated_files.pkl \
  --num_workers 1
```

`eval_jacobian_4d_subset.py` deliberately preserves the formal list-of-Data
format and the `2 * num_reference` generation ratio. It saves incrementally,
so the file may contain fewer molecules after an interrupted run, but every
stored molecule is individually compatible.

`subset_output.pt` is a diagnostic artifact and is **not** a direct
`eval_cov_mat.py` input. It contains tensors, timings, model type, head-call
counts, failures, and paths. No conversion script is required while
`generated_files.pkl` is present.

## 4. If only `subset_output.pt` remains

The current diagnostic file does not contain the RDKit object, but the current
CovMat evaluator reconstructs molecules from SMILES. A small conversion could
create one `Data(smiles=..., pos_ref=..., pos_gen=...)` per successful record
and pickle the list. This is only a recovery path; normal subset runs already
write the compatible file.

## 5. Formal base versus 4D commands

Generate base conformers:

```bash
WANDB_MODE=offline python scripts/eval.py \
  --config <base-config.resolved.yaml> \
  --checkpoint <base-checkpoint.ckpt> \
  --output_dir logs_eval_full/base
```

Generate 4D conformers:

```bash
WANDB_MODE=offline python scripts/eval.py \
  --config <jacobian-config.resolved.yaml> \
  --checkpoint <jacobian-checkpoint.ckpt> \
  --output_dir logs_eval_full/jacobian_4d
```

Then run, once for each discovered `generated_files.pkl`:

```bash
WANDB_MODE=offline python scripts/eval_cov_mat.py \
  --path <generated_files.pkl> \
  --num_workers 1
```

The base and 4D runs must use the same test split, sampler arguments, generated
conformer ratio, and evaluator options.

## 6. Why formal debug mode saves no result

In `scripts/eval.py`, `--debug` breaks only after the first molecule has
finished all of its requested conformers. The save block is guarded by:

```python
if not debug:
    save_pkl(...)
```

Consequently debug mode can spend substantial time generating the first
molecule and intentionally writes neither `generated_files.pkl` nor
`times.pkl`. The dedicated smoke and subset scripts always save diagnostics.

## 7. Recommended minimum evaluation sequence

1. Run `run_sampling_smoke_pair.sh` and require both models to succeed.
2. Run paired 20-molecule subset sampling.
3. Inspect `subset_output.pt` with `summarize_jacobian_4d_subset.py`.
4. Run `eval_cov_mat.py` on both subset `generated_files.pkl` files with the
   same options.
5. Only after the subset path succeeds, run full `scripts/eval.py` for base and
   4D and evaluate both formal output files.

`scripts/run_jacobian_4d_eval_pair.sh` automates steps 2–4 by default and keeps
an explicit `--full` mode for the 1000-molecule evaluation.
