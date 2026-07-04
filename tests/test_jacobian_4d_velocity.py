import torch

from etflow.commons.jacobian_4d_selection import select_jacobian_4d_bonds
from etflow.commons.jacobian_4d_velocity import (
    apply_jacobian_4d_correction,
    build_atom_jacobian,
    build_bond_frames,
    build_local_frame,
    combine_jacobian_4d_velocity,
    solve_q_targets,
)


def test_local_frame_is_orthonormal_for_axis_and_fallback_cases():
    axis = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 2.0, 3.0]]
    )
    frame = build_local_frame(axis)
    identity = torch.eye(3).expand(axis.size(0), -1, -1)
    torch.testing.assert_close(frame.transpose(1, 2) @ frame, identity)


def test_stretch_sign_and_angular_mapping():
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    )
    anchor = torch.tensor([0])
    moving = torch.tensor([1])
    affected_atom = torch.tensor([2])
    affected_bond = torch.tensor([0])

    positive, _, _ = apply_jacobian_4d_correction(
        pos,
        torch.tensor([[2.0, 0.0, 0.0, 0.0]]),
        anchor,
        moving,
        affected_atom,
        affected_bond,
    )
    negative, _, _ = apply_jacobian_4d_correction(
        pos,
        torch.tensor([[-2.0, 0.0, 0.0, 0.0]]),
        anchor,
        moving,
        affected_atom,
        affected_bond,
    )
    # The affected-side reference gives e1=+y and e2=+z. For omega=e2 and
    # r=(1, 1, 0), omega x r=(-1, 1, 0).
    angular, _, _ = apply_jacobian_4d_correction(
        pos,
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        anchor,
        moving,
        affected_atom,
        affected_bond,
    )

    torch.testing.assert_close(positive[2], torch.tensor([2.0, 0.0, 0.0]))
    torch.testing.assert_close(negative[2], torch.tensor([-2.0, 0.0, 0.0]))
    torch.testing.assert_close(angular[2], torch.tensor([-1.0, 1.0, 0.0]))


def test_explicit_jacobian_matches_direct_velocity_formula():
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, -0.5, 1.0],
            [-2.0, 3.0, 1.0],
        ],
        dtype=torch.float64,
    )
    anchor = torch.tensor([0])
    moving = torch.tensor([1])
    affected_atom = torch.tensor([1, 2, 3])
    affected_bond = torch.zeros(3, dtype=torch.long)
    q = torch.tensor([[0.2, -0.3, 0.4, 0.5]], dtype=torch.float64)
    frame, valid = build_bond_frames(
        pos, anchor, moving, affected_atom, affected_bond
    )
    assert valid.tolist() == [True]

    lever = pos[affected_atom] - pos[anchor[0]]
    e0 = frame[0, :, 0].expand(affected_atom.numel(), -1)
    atom_frame = frame.expand(affected_atom.numel(), -1, -1)
    jacobian = build_atom_jacobian(e0, atom_frame, lever)
    via_jacobian = torch.matmul(jacobian, q[0])
    omega_global = frame[0] @ q[0, 1:]
    direct = q[0, 0] * e0 + torch.cross(
        omega_global.expand_as(lever), lever, dim=-1
    )
    correction, _, _ = apply_jacobian_4d_correction(
        pos, q, anchor, moving, affected_atom, affected_bond
    )

    torch.testing.assert_close(via_jacobian, direct)
    torch.testing.assert_close(correction[affected_atom], direct)
    torch.testing.assert_close(correction[4], torch.zeros(3, dtype=pos.dtype))


def test_final_velocity_loss_backpropagates_when_q_loss_weight_is_zero():
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    )
    anchor = torch.tensor([0])
    moving = torch.tensor([1])
    affected_atom = torch.tensor([1, 2])
    affected_bond = torch.zeros(2, dtype=torch.long)
    q_pred = torch.zeros((1, 4), requires_grad=True)
    correction, _, valid = apply_jacobian_4d_correction(
        pos, q_pred, anchor, moving, affected_atom, affected_bond
    )
    v_final, _ = combine_jacobian_4d_velocity(
        torch.zeros_like(pos), correction, correction_scale=0.03
    )
    target = torch.zeros_like(pos)
    target[affected_atom, 0] = 1.0
    flow_loss = (v_final - target).square().mean()
    flow_loss.backward()

    assert valid.tolist() == [True]
    assert q_pred.grad is not None
    assert q_pred.grad.abs().sum() > 0


def test_affected_side_frame_and_correction_rotate_covariantly():
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 0.0, 1.0],
        ]
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotated_pos = pos @ rotation.transpose(0, 1)
    anchor = torch.tensor([0])
    moving = torch.tensor([1])
    affected_atom = torch.tensor([1, 2, 3])
    affected_bond = torch.zeros(3, dtype=torch.long)
    q = torch.tensor([[0.2, -0.3, 0.4, 0.5]])

    frame, valid = build_bond_frames(
        pos, anchor, moving, affected_atom, affected_bond
    )
    rotated_frame, rotated_valid = build_bond_frames(
        rotated_pos, anchor, moving, affected_atom, affected_bond
    )
    correction, _, _ = apply_jacobian_4d_correction(
        pos, q, anchor, moving, affected_atom, affected_bond
    )
    rotated_correction, _, _ = apply_jacobian_4d_correction(
        rotated_pos, q, anchor, moving, affected_atom, affected_bond
    )

    assert valid.tolist() == rotated_valid.tolist() == [True]
    torch.testing.assert_close(
        frame.transpose(1, 2) @ frame, torch.eye(3).unsqueeze(0)
    )
    torch.testing.assert_close(rotated_frame, rotation @ frame)
    torch.testing.assert_close(
        rotated_correction, correction @ rotation.transpose(0, 1)
    )


def test_synthetic_least_squares_recovers_q_true():
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 2.0],
            [2.0, -1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    anchor = torch.tensor([0])
    moving = torch.tensor([1])
    affected_atom = torch.tensor([1, 2, 3, 4])
    affected_bond = torch.zeros(4, dtype=torch.long)
    frame, frame_valid = build_bond_frames(
        pos, anchor, moving, affected_atom, affected_bond
    )
    assert frame_valid.tolist() == [True]
    frame = frame[0]
    e0 = frame[:, 0].expand(4, -1)
    jacobian = build_atom_jacobian(
        e0,
        frame.expand(4, -1, -1),
        pos[affected_atom] - pos[0],
    ).reshape(-1, 4)
    q_true = torch.tensor([0.1, 0.2, -0.3, 0.4], dtype=torch.float64)
    residual = torch.zeros_like(pos)
    residual[affected_atom] = (jacobian @ q_true).reshape(-1, 3)

    q_target, valid, condition = solve_q_targets(
        pos,
        residual,
        anchor,
        moving,
        affected_atom,
        affected_bond,
        ridge_eps=1.0e-10,
        max_condition=1.0e12,
    )

    assert valid.tolist() == [True]
    assert torch.isfinite(condition).all()
    torch.testing.assert_close(q_target[0], q_true, rtol=1.0e-7, atol=1.0e-7)


def test_q_target_filters_large_nonfinite_and_ill_conditioned_solutions():
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 2.0],
        ],
        dtype=torch.float64,
    )
    anchor = torch.tensor([0])
    moving = torch.tensor([1])
    affected_atom = torch.tensor([1, 2, 3])
    affected_bond = torch.zeros(3, dtype=torch.long)
    residual = torch.ones_like(pos)

    _, _, condition = solve_q_targets(
        pos,
        residual,
        anchor,
        moving,
        affected_atom,
        affected_bond,
        max_condition=1.0e12,
    )
    large_target, large_valid, _ = solve_q_targets(
        pos,
        residual * 1000.0,
        anchor,
        moving,
        affected_atom,
        affected_bond,
        max_q_norm=1.0e-3,
        max_condition=1.0e12,
    )
    _, condition_valid, _ = solve_q_targets(
        pos,
        residual,
        anchor,
        moving,
        affected_atom,
        affected_bond,
        max_condition=float(condition.item()) * 0.5,
    )
    nonfinite_residual = residual.clone()
    nonfinite_residual[2, 0] = float("nan")
    nonfinite_residual[3, 1] = float("inf")
    nonfinite_target, nonfinite_valid, _ = solve_q_targets(
        pos,
        nonfinite_residual,
        anchor,
        moving,
        affected_atom,
        affected_bond,
    )

    assert large_valid.tolist() == [False]
    assert condition_valid.tolist() == [False]
    assert nonfinite_valid.tolist() == [False]
    torch.testing.assert_close(large_target, torch.zeros_like(large_target))
    torch.testing.assert_close(
        nonfinite_target, torch.zeros_like(nonfinite_target)
    )


def test_disabled_and_empty_selection_fall_back_to_atom_velocity():
    v_atom = torch.randn(5, 3)
    pos = torch.randn(5, 3)
    empty = torch.empty(0, dtype=torch.long)
    v_corr, counts, valid = apply_jacobian_4d_correction(
        pos,
        torch.empty(0, 4),
        empty,
        empty,
        empty,
        empty,
    )
    q_target, q_valid, _ = solve_q_targets(
        pos,
        torch.zeros_like(pos),
        empty,
        empty,
        empty,
        empty,
    )
    q_loss = torch.empty(0, 4, requires_grad=True).sum() * 0.0
    v_final, scaled = combine_jacobian_4d_velocity(
        v_atom, v_corr, 0.3, enabled=False
    )

    torch.testing.assert_close(v_corr, torch.zeros_like(v_corr))
    torch.testing.assert_close(counts, torch.zeros_like(counts))
    assert valid.numel() == 0
    assert q_target.shape == (0, 4)
    assert q_valid.numel() == 0
    torch.testing.assert_close(q_loss, torch.tensor(0.0))
    assert v_final is v_atom
    torch.testing.assert_close(scaled, torch.zeros_like(scaled))


def test_nonempty_but_invalid_bond_is_a_finite_noop():
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    anchor = torch.tensor([0])
    moving = torch.tensor([1])
    affected_atom = torch.tensor([1, 2])
    affected_bond = torch.zeros(2, dtype=torch.long)
    q_pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    correction, counts, geometry_valid = apply_jacobian_4d_correction(
        pos, q_pred, anchor, moving, affected_atom, affected_bond
    )
    q_target, q_valid, _ = solve_q_targets(
        pos,
        torch.ones_like(pos),
        anchor,
        moving,
        affected_atom,
        affected_bond,
    )

    assert geometry_valid.tolist() == [False]
    assert q_valid.tolist() == [False]
    torch.testing.assert_close(correction, torch.zeros_like(correction))
    torch.testing.assert_close(counts, torch.zeros_like(counts))
    torch.testing.assert_close(q_target, torch.zeros_like(q_target))
    assert torch.isfinite(correction).all()


def test_overlapping_bond_corrections_are_averaged():
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.5, 0.0],
        ]
    )
    correction, counts, valid = apply_jacobian_4d_correction(
        pos,
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]),
        torch.tensor([0, 2]),
        torch.tensor([1, 3]),
        torch.tensor([4, 4]),
        torch.tensor([0, 1]),
    )

    assert valid.tolist() == [True, True]
    torch.testing.assert_close(counts[4], torch.tensor([2.0]))
    torch.testing.assert_close(correction[4], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.isfinite(correction).all()


def test_selection_filters_small_sides_and_caps_per_molecule():
    rotatable = torch.tensor([[0, 1, 5], [1, 2, 6]], dtype=torch.long)
    influence = torch.tensor(
        [[1, 2, 3, 2, 6, 7], [0, 0, 0, 1, 2, 2]], dtype=torch.long
    )
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
    selected = select_jacobian_4d_bonds(
        rotatable,
        influence,
        batch,
        min_affected_atoms=2,
        max_bonds_per_mol=1,
    )

    # Bond 1 is too small. Bond 0 and bond 2 are retained, one in each graph.
    assert selected["original_bond_index"].tolist() == [0, 2]
    assert selected["affected_count"].tolist() == [3, 2]
    assert selected["affected_bond_index"].tolist() == [0, 0, 0, 1, 1]
