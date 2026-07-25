# Torsion canonicalization audit

Decision: **TORSION_LABEL_GO**

- Molecules: `2500`; raw/canonical rotors: `19312` / `16998`.
- Source/Reference angles audited: `50994` / `436280`.
- Forward/reverse failures: `0`; maximum circular error: `8.88e-16`.
- Duplicate central bonds / invalid indices: `0` / `0`.
- Ring/non-single/amide-like restricted inclusions: `0` / `0` / `0`. Raw project rotors contained `2002` amide-like definitions, all explicitly excluded before label construction.
- Symmetric terminal environments audited: `9494`. Canonical ranks resolve a deterministic representative; circular labels remain tied to the frozen atom order.
- Degenerate definitions are returned as finite zero by the dihedral primitive; none creates a non-finite or out-of-range label.

The labels are periodic angles directly computed from internal TRAIN/DEV Reference coordinates. No MVT, Cartesian Source→Reference delta, PB, xTB, formal test, or frozen holdout was read.
