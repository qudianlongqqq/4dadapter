# End-to-end data-flow audit

```text
prepared graph/reference payload + ETFlow source-binding payload
  -> frozen TRAIN/VAL manifests (hash checked)
  -> TRAIN: molecule draw -> one of 3 source records -> one reference conformer
  -> GraphGeometry + source/reference coordinates
  -> collated batch on CUDA
  -> shared 3-layer 128D graph backbone
  -> Bond mu / Angle-cosine mu / direct sigma
  -> R1 Reliability + Adaptive BA
  -> analytic Bond/Angle Cartesian first-order contributions
  -> rigid-mode projection
  -> per-graph RMS direction normalization
  -> magnitude tau
  -> Restricted capped proposal OR Unrestricted uncapped proposal
  -> frozen DEV SDF + PER_RECORD/PER_PRIMITIVE payloads
  -> Validity3D and PoseBusters external CPU evaluator
  -> fixed-order all-atom Kabsch Reference RMSD
  -> GFN2-xTB frozen-coordinate single points
  -> CSV/Parquet/JSON/Markdown summaries
```

## Data identities

| stage | input/output | verified size/shape/units | identity evidence |
|---|---|---|---|
| prepared payload | TRAIN/VAL molecule items | 50,000 / 5,000 molecules | hash `b19bd8...` |
| source binding | ETFlow source rows | TRAIN 150,000; VAL 10,000 | hash `c6b2ea...` and lineage audit |
| frozen DEV manifest | deterministic subset of VAL | 2,500 molecules, 5,000 records, 2 records/molecule | hash `15db86...` |
| graph | atoms/edges/bonds/angles | variable N; edge index [2,E], bonds [2,B], angles [A,3] | `GraphGeometry`, `prepare_graph` |
| coordinates | source/reference/proposal | [sum atoms,3], Å, float64 in prepared/evidence paths | runners and correspondence audit |
| Bond target | length | Å | `geometry_values` |
| Angle target | cosine | dimensionless | `stable_angle_cosine` |
| model outputs | mu/sigma/features | one value per Bond/Angle primitive | model forward |
| Reliability | primitive gate | [B] and [A], (0,1) | `PrimitiveReliabilityHead` |
| Adaptive BA | family weights | [molecules,2], sum 1 | `AdaptiveBAHead` |
| direction | Cartesian | [sum atoms,3], per-graph RMS 1 or zero fallback | action code |
| tau | movement scale | one Å scalar/molecule-record | magnitude heads |
| evaluator | SDF -> per-record tables | 5,000 ordered record IDs | both `COORDINATES_READY.json` share record hash `2690c66e...` |

TRAIN sampling is explicit in `run_sixs_musigma_reliability_factorial.py:340-351`: uniformly draw a molecule index, then independently draw one source and one reference conformer. DEV evaluation uses the frozen manifest order and hash checks (`run_sixs_j1r1_full_joint_adaptive_ba_movement.py:839-949`).

The same record-ID hash appears in Restricted and Unrestricted coordinate-freeze artifacts. External evaluator rows are rejoined by record ID/order. No record-identity mismatch is present in completed seed307 artifacts.

DATA_FLOW_STATUS = PASS_FOR_COMPLETED_SEED307  
MULTISEED_DATA_FLOW_STATUS = INCOMPLETE


