# Controlled DEV protocol

- Seed: 307.
- TRAIN: the exact molecule-disjoint v1 TRAIN source/reference pairs.
- Objective: Delta-Q J1 beta-NLL plus unchanged post-update geometry loss, 1:1.
- Optimization: matched v1 AdamW, batch 64, learning rates, clipping, cosine
  scheduler, and 17,500-step endpoint.
- ETFlow-like DEV: the existing frozen 2,500-molecule/5,000-record DEV manifest.
- AvgFlow low-headroom DEV: must be newly frozen and molecule-disjoint from
  TRAIN and from every identity in the already-inspected 4,408-molecule
  AvgFlow final cohort. If no legal unused cohort can be proven, this endpoint
  fails closed rather than reusing the discovery set.
- No early stopping or DEV checkpoint selection. No Formal/large-holdout read.
- xTB is omitted from this first mechanistic pilot.
