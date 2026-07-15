# Serial Global4D Stage 0 Preflight

Status: **PASS**.

- Branch is `feat/serial-global4d-residual-v1`.
- Existing Oracle SHA256 matches `B135D8E5...F1877BA`.
- Cartesian step100000 loads strictly with no missing or unexpected keys.
- Teacher mode is `cartesian_optimizer`, eval mode is active, all parameters
  are frozen, and `v_final == v_cart` on a real two-graph CUDA forward.
- The teacher input view exposes no reference or target fields.
- Resolved config and validation best-config identities match the checkpoint.
- Confirm30 raw and canonical manifest identities match the frozen selection.
- All 60 sample IDs exist in formal validation cache and all 60 persisted,
  manifest, and recomputed `x_init_hash` values match.
- `27 passed`; no test split was read.
- Protected report SHA256 is unchanged.
