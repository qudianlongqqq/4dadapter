# Current architecture reconstructed from executable code

## Method identity

A shared 3-layer 128-dimensional invariant message-passing geometry model predicts Bond length means and Angle cosine means plus direct heteroscedastic sigma; source-conditioned Reliability and graph-conditioned Adaptive Bond/Angle allocation weight analytic first-order primitive descent contributions; a graph/state magnitude head scales a rigid-mode-projected, graph-RMS-normalized Cartesian direction.

This is an implementation description, not a claim that the method is exactly Gauss–Newton.

| module | file | class/function | input | output | role | trainable | shared |
|---|---|---|---|---|---|---|---|
| graph preparation | `etflow/ecir/learned_geometry.py:166-224` | `prepare_graph` | molecule record | `GraphGeometry` | atom/bond/angle categorical tensors and frozen statistics | no | yes |
| shared backbone | `learned_geometry.py:254-272,275-350` | `MessageLayer`, `LearnedGeometryObjective` | graph categories/topology | node embedding [atoms,128] | three message layers | yes | yes |
| Bond mu | `learned_geometry.py:296-326` | `bond_head` | pair node features + edge features | positive Bond mean in Å | `softplus(raw)` | yes | yes |
| Angle mu | `learned_geometry.py:300-338` | `angle_head` | triplet node/edge features | cosine mean | `tanh(raw)` | yes | yes |
| direct sigma | `musigma_reliability.py:66-145` | `DirectMuSigmaModel` | primitive features + frozen sigma_stat | family sigma | `sigma_stat*exp(6*tanh(raw/6))` | yes | yes |
| Reliability (R1) | `musigma_reliability.py:148-227` | `PrimitiveReliabilityHead` | local feature, q, mu, log sigma, signed/absolute/standardized defect, family | scalar sigmoid gate per primitive | source-conditioned action reliability | yes | yes |
| Adaptive BA | `j1r1_full_joint.py:29-41` | `AdaptiveBAHead` | detached mean graph embedding [128] | [w_B,w_A] softmax | family allocation | yes | yes |
| primitive descent | `musigma_reliability.py:301-340` | `_bond_descent`, `_angle_descent` | source coordinates + coefficients | Cartesian vector | analytic first-order primitive derivatives | no coordinate Hessian | yes |
| full action | `j1r1_full_joint.py:98-209` | `full_joint_action` | source, graph, prediction | direction/proposal/mechanisms | combine Reliability, BA, sigma precision | differentiable to controller modules | restricted |
| bounded magnitude | `lsgoba_v2_joint_magnitude.py:35-47` | `AdaptiveTrustMagnitudeHead` | detached graph embedding + 17-state | tau in (0,0.010) Å | sigmoid magnitude | yes | restricted |
| atom cap | `lsgoba_v2_joint_magnitude.py:142-148` | `scaled_proposal` | tau-scaled per-atom delta | capped proposal | 0.03 Å per-atom cap | no | restricted |
| unbounded magnitude | `j1r1_full_joint_unrestricted.py:28-51` | `UnboundedSoftplusMagnitudeHead` | same graph/state input | positive unbounded tau | softplus, exact initial tau 0.003 Å | yes | unrestricted |
| unrestricted action | `j1r1_full_joint_unrestricted.py:72-168` | `unrestricted_action` | same source/graph/prediction | source + tau*direction | no bound/cap/rollback | yes | unrestricted |

## Exact correction path

For family primitive p, the code forms a coefficient proportional to

[
rac{w_f}{n_f} r_p rac{q_p-mu_p}{sigma_p^2}.
]

Bond and Angle analytic derivatives are multiplied by the negative coefficient and accumulated. The vector is projected by `remove_rigid_component` (`j1r1_full_joint.py:154-156`), normalized by graph RMS (`:157-164`), and set to zero when nonfinite or numerically degenerate.

There is no explicit `torch.autograd` VJP in the action implementation. The only `autograd.grad` calls in the current runners are gradient-route preflight checks. Paper wording should say “analytic first-order primitive Jacobian-transpose contributions” or equivalent, not claim an explicit autograd VJP implementation.

## Detach/stop-gradient/fallback inventory

- q, mu, sigma entering action coefficients are detached (`j1r1_full_joint.py:113-130`).
- primitive derivative tensors are detached (`musigma_reliability.py:301-340`).
- graph embedding into Adaptive BA and magnitude is detached (`j1r1_full_joint.py:89-95,187-189`).
- magnitude state is detached.
- post loss divides by detached sigma (`musigma_reliability.py:449-466`).
- nonfinite/zero direction becomes exactly zero.
- Restricted applies a 0.03 Å atom cap. Unrestricted does not.
- Restricted computes a `safety_accept` fallback flag for diagnostics during evaluation, but ignores the returned fallback coordinates and evaluates/writes `action.proposal`; it therefore has no scientific rollback. Evidence: runner `:914-937` and `safety_accept` in `learned_geometry.py:599-616`.
- No hidden mask beyond family/empty-angle handling and graph slicing was found.


