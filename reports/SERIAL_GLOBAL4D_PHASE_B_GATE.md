# Serial Global4D Phase B Gate

Phase B **PASS** after 750 gate-only steps. The Phase A backbone and q head
were frozen. On fixed Confirm30, the learned gate had mean/std `0.501/0.0685`
and range `0.376–0.738`, so it collapsed to neither zero nor one.

Ungated negative-gain fraction fell from 40% to 35% after gating; positive
fraction rose from 60% to 65%, and mean gated gain was `+0.6938`. Mean gate
increased with flexibility: low `0.433`, medium `0.468`, high `0.545`.

External checkpoint:
`E:/3dconformergenerationcode/serial_global4d_work/logs/phase_b_gate_750/step000750.ckpt`
(SHA256 `0c75a5b7421ef9f4b75870582c6af9b41342c75fe29a2ca96949d13e3685a210`).
