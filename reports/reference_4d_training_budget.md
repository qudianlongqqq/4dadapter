# Reference FlexBond-4D small-run budget

Reference status: **OLD_RESULT_MISSING**

> The old model is read-only and is never retrained by the Global Coupled 4D pipeline.

## Fixed matched budget

- `max_steps`: `5000`
- `batch_size`: `4`
- `accumulate_grad_batches`: `2`
- `effective_batch_size`: `8`
- `learning_rate`: `0.0002`

## Optional fields

- `t_min`: `0.0` (Global4D fallback; old field missing)
- `t_max`: `0.25` (Global4D fallback; old field missing)
- `hidden_dim`: `128` (Global4D fallback; old field missing)
- `edge_hidden_dim`: `128` (Global4D fallback; old field missing)
- `num_layers`: `6` (Global4D fallback; old field missing)
- `optimizer`: `AdamW` (Global4D fallback; old field missing)
- `scheduler`: `none` (Global4D fallback; old field missing)
- `precision`: `32-true` (Global4D fallback; old field missing)
- `train_data`: `data/flexbond_cache_formal_small` (Global4D fallback; old field missing)
- `val_data`: `data/flexbond_cache_formal_small` (Global4D fallback; old field missing)
- `seed`: `42` (Global4D fallback; old field missing)
- `validation_frequency`: `250` (Global4D fallback; old field missing)
