# ECIR Stage 2 reproduction

Stage A reproduced the frozen heterogeneous 5k ECIR evaluation on `val` only. No checkpoint, atlas, target, or legacy diagnostic was modified.

## Identities

- ECIR checkpoint: `step005000.ckpt`, SHA256 `232e47865d01a71543cf2cd16ede577764fd3d94ac843d78dcdcf8c9789fa98d`
- resolved config SHA256: `2060d765031fc2bdb4f73cf7008b40906e90aef0d24912354c529536ee1ed79d`
- validation atlas SHA256: `8501185f916cf6f048bd56fc4343e5c2b2f38b9ca96523f2f6b6351628654820`
- atlas identity: `aa0db9d67d57cc2077557fac76270bbf1322f295d76852b2d3d310d309f2e985`
- records/molecules: 100/50; test was not used.

## Reproduction result

| Metric | frozen result delta | reproduced delta | absolute drift |
|---|---:|---:|---:|
| bond violation | -0.040251025 | -0.040251024 | 9.41e-10 |
| angle violation | -0.025805545 | -0.025805545 | 3.73e-11 |
| torsion circular error | -0.017354166 | -0.017354163 | 2.93e-09 |
| ring invalidity | -0.034864549 | -0.034864548 | 5.12e-10 |
| aligned RMSD | +0.019169605 | +0.019169589 | 1.55e-08 |
| MAT-P | +0.024506558 | +0.024506550 | 8.34e-09 |
| MAT-R | +0.021283658 | +0.021283659 | 7.08e-10 |

The maximum drift is below the declared `5e-7` absolute tolerance. Reproduction status is **PASS**; scientific status remains **NO_GO** because RMSD, MAT-P and MAT-R fail the 0.02 project noninferiority gate and unseen evidence is absent.

Machine-readable evidence is in `diagnostics/ecir_mvr/stage_a_reproduction/result.json`.
