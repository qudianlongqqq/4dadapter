# LSGO-BA v2 architecture

The frozen v1 graph encoder and Bond/cosine-Angle conditional-mean heads are retained and warm-started from seed307. Frozen DRCSR sigmas remain buffers supplied by each graph. The existing 128-dimensional node embedding is mean-pooled per molecule and concatenated with 17 TRAIN-normalized Source-state scalars. The only added network is `Linear(145,64) -> GELU -> Linear(64,32) -> GELU -> Linear(32,1)`.

- magnitude-head parameters: 11,457
- total v2 trainable parameters: 485,131
- parameter increase: 2.418752%
- tau: `0.010 * sigmoid(a)`; final weights zero; initial bias `-0.8472978603872037`; initial tau 0.003 Å
- structured direction / rigid removal / atom cap / deployment safety: unchanged
- direction recomputed from current theta each optimizer step and detached before post loss
- Cartesian Reference loss / learned sigma / second GNN / second-order direction backprop: absent
