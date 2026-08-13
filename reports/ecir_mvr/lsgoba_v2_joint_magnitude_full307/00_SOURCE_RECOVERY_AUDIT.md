# Source recovery audit

Status: **PASS**. Every formal-large TRAIN and legal development-VALIDATION source cache was verified against its frozen manifest file SHA, raw coordinate SHA, molecule/sample identity, explicit atom identity/order, graph edge identity, atom count, and the exact v1 Reference-ensemble hashes.

- TRAIN Source: 150,000 records / 50,000 molecules (3 uniformly sampleable Sources per molecule)
- legal development VALIDATION Source: 10,000 records / 5,000 molecules
- TRAIN/VALIDATION molecule overlap: 0
- synthetic/corrupted Source: none
- formal test / frozen holdout records read: 0 / 0
- frozen Source payload SHA256: `c6b2eab6e56eab90e68e0f84ab465ef2308a3d13efdd4da5f0ac88823de2657d`
