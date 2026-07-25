# Old external exclusion audit

Status: **PASS**

The prospective `BAT_EXTERNAL_CONFIRM` cohort was selected before any coordinate generation or external evaluation. It contains 200 molecule identities and 600 Source records.

The exclusion union contains 1,200 previously exposed molecule identities:

| Historical cohort | Molecules excluded |
|---|---:|
| LSGO external confirm | 200 |
| NSSM external confirm | 500 |
| Unified external utility | 500 |
| Union | 1,200 |

The selected BAT external cohort has zero identity overlap with that union. It is also molecule-disjoint from `BAT_TRAIN`, `BAT_DEV_A`, and `BAT_DEV_B` by construction.

Frozen identities:

| Cohort | Molecules | Source records | References | Molecule identity SHA256 |
|---|---:|---:|---:|---|
| BAT_TRAIN | 2,000 | 6,000 | 46,581 | `fc008c0a128d4fd1290b620b16b0d80bb585da747a82c54dbfa409e7a9ac298f` |
| BAT_DEV_A | 250 | 750 | 6,015 | `aa47d9d40c9780d8c724d4103f79d1470f97230edb36649f383ae4d49702cac2` |
| BAT_DEV_B | 250 | 750 | 5,988 | `1190455a601e04dea2e634e0f6ca8f4860a40fc769a751e2e00f453f79951208` |
| BAT_EXTERNAL_CONFIRM | 200 | 600 | 4,982 | `06b09f2d730339f3efdf33da2a0e48e3a0d21f7de4cada3510414c174a7fb79b` |

Identity manifest SHA256: `b2cb6ff4d108f71841a5d90bd0f6a1958c2ad99638b84e42594b53ee30f2baea`

External compact SHA256: `23cf14b0ec354aee97b8f0d8486dd03be1877d7139b2a9ca4e1983e0a54cadec`

No PoseBusters or xTB result was inspected during selection. `formal_test_records_read = 0` and `frozen_holdout_records_read = 0`.
