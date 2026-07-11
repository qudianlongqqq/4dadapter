# Reference FlexBond-4D training budget

Confidence: **low**

Ambiguous: **False**

## Selected evidence

- `reference_run`: `E:\3dconformergenerationcode\4dadapter\scripts\run_jacobian_4d_formal_multiseed_100k.sh`
- `config_path`: `E:\3dconformergenerationcode\4dadapter\configs\drugs-so3-jacobian-4d-bs4.yaml`
- `checkpoint_path`: ``
- `max_steps`: `100000`
- `checkpoint_global_step`: `0`
- `batch_size`: `4`
- `accumulate_grad_batches`: `2`
- `effective_batch_size`: `8`
- `learning_rate`: `0.0008`
- `scheduler`: `CosineAnnealingWarmupRestarts`
- `optimizer`: `AdamW`
- `t_min`: `0.0001`
- `t_max`: `0.9999`
- `seed`: `42`
- `precision`: `unknown`
- `gpu_count`: `1`
- `train_split`: `train`
- `val_split`: `val`
- `train_num_molecules`: `0`
- `val_num_molecules`: `0`
- `validation_frequency`: `500`
- `checkpoint_interval`: `0`
- `start_time`: `unknown`
- `end_time`: `unknown`
- `git_commit`: `unknown`
- `confidence`: `low`
- `evidence`: `['launch script declaration only; not proof of a completed run']`

## Candidates

| score | kind | max steps | checkpoint step | path |
|---:|---|---:|---:|---|
| 13 | launch_declaration | 100000 | 0 | `E:\3dconformergenerationcode\4dadapter\scripts\run_jacobian_4d_formal_multiseed_100k.sh` |
| 5 | launch_declaration | 100000 | 0 | `E:\3dconformergenerationcode\4dadapter\scripts\run_jacobian_4d_longtrain_seed42.sh` |

> Formal training is blocked: a launch script is not proof of a completed reference run.
