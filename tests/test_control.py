"""The controller, checked against a chain whose answers are known in closed form.

The point of the `Kinematics` protocol is that this file needs no GPU, no
Newton and no robot. A three-link planar arm has an analytic Jacobian and an
analytic reachable set, so the solver can be checked against the truth rather
than against itself -- which is the failure mode this project keeps finding
(reader and writer sharing one misconception and agreeing perfectly).

The regression that matters most here is `test_solver_actually_moves_off_its_seed`.
An earlier version of this solver differentiated the residual instead of the
forward map, which flips the sign of every step: each one increased the cost,
the line search rejected all of them, and the solver returned its seed while
reporting a residual. It looked like a hard problem. It was a sign error.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from labgen.control import (IK_EPS, YAM_JAWS, YAM_PAD, ArmSpec, BimanualRig,
                            FingerPad, GraspTooWide, IKResult, JawModel,
                            Trajectory, UnsourcedPlacement, is_placeholder,
                            jacobian, lerp_path, pad_to_longest, solve_pose)

LINKS = (0.30, 0.25, 0.20)
REACH = sum(LINKS)


class PlanarChain:
    """Three revolute joints in the XY plane. Everything about it is analytic."""

    def __init__(self, links=LINKS):
        self.links = tuple(links)
        self.n_joints = len(self.links)

    def _angles(self, q):
        return np.cumsum(np.asarray(q, float)[:self.n_joints])

    def pose(self, q):
        a = self._angles(q)
        x = float(np.sum(np.asarray(self.links) * np.cos(a)))
        y = float(np.sum(np.asarray(self.links) * np.sin(a)))
        return np.array([x, y, 0.0]), np.array([math.cos(a[-1]),
                                                math.sin(a[-1]), 0.0])

    def analytic_jacobian(self, q):
        a = self._angles(q)
        J = np.zeros((3, self.n_joints))
        for j in range(self.n_joints):
            J[0, j] = -sum(self.links[i] * math.sin(a[i])
                           for i in range(j, self.n_joints))
            J[1, j] = sum(self.links[i] * math.cos(a[i])
                          for i in range(j, self.n_joints))
        return J


@pytest.fixture
def chain():
    return PlanarChain()


@pytest.fixture
def limits(chain):
    n = chain.n_joints
    return -np.pi * np.ones(n), np.pi * np.ones(n)


# --- the Jacobian is the real thing --------------------------------------

def test_central_difference_jacobian_matches_the_analytic_one(chain):
    q = np.array([0.3, -0.7, 0.45])
    num = jacobian(chain, q)
    assert num == pytest.approx(chain.analytic_jacobian(q), abs=1e-7)


def test_jacobian_step_is_not_machine_epsilon():
    """float32 joint_q makes a 1e-8 perturbation pure rounding noise.

    This is a PIN, not a behavioural check, and the distinction is worth being
    explicit about: the planar double above is float64, so it cannot reproduce
    the failure -- setting IK_EPS to 1e-8 leaves every other test in this file
    green. The bug only appears against Newton, whose joint_q is float32, where
    a solver descending on noise still returns a number and the number is wrong
    by hundreds of millimetres. So this constant is guarded by name because no
    test here can guard it by behaviour.
    """
    assert IK_EPS == 1e-4


def test_jacobian_gains_axis_rows_when_orientation_is_solved(chain):
    q = np.array([0.2, 0.1, -0.3])
    assert jacobian(chain, q).shape == (3, 3)
    assert jacobian(chain, q, use_axis=True).shape == (6, 3)


# --- the solver ----------------------------------------------------------

def reachable_target(chain, q):
    return chain.pose(q)[0]


def test_solves_a_reachable_target(chain, limits):
    target = reachable_target(chain, np.array([0.4, -0.6, 0.35]))
    res = solve_pose(chain, target, *limits)
    assert res.ok and bool(res)
    assert res.position_error_m < 0.004
    assert np.linalg.norm(chain.pose(res.q)[0] - target) < 0.004


def test_solves_position_and_approach_axis_together(chain, limits):
    q_true = np.array([0.5, -0.4, 0.2])
    target, axis = chain.pose(q_true)
    res = solve_pose(chain, target, *limits, approach=axis)
    assert res.ok
    assert res.position_error_m < 0.004
    assert res.axis_error_deg < 12.0


def test_solver_actually_moves_off_its_seed(chain, limits):
    """The sign-error regression.

    Differentiating the residual rather than the forward map makes every step
    increase the cost, so the solver returns its seed and reports the seed's
    error. The tell was bit-identical residuals across four different
    weightings. Here: the answer must differ from the seed AND be better.
    """
    seed = np.array([0.0, 0.0, 0.0])
    target = reachable_target(chain, np.array([0.6, -0.5, 0.3]))
    seed_err = float(np.linalg.norm(chain.pose(seed)[0] - target))

    res = solve_pose(chain, target, *limits, seed=seed, restarts=1)

    assert not np.allclose(res.q, seed), "solver returned its own seed"
    assert res.position_error_m < seed_err


def test_an_unreachable_target_fails_rather_than_reporting_a_pose(chain, limits):
    """Outside the workspace the honest answer is "no", with the real distance."""
    target = np.array([REACH + 0.25, 0.0, 0.0])
    res = solve_pose(chain, target, *limits)
    assert not res.ok and not bool(res)
    # The closest the arm can get is full extension, so the residual is exactly
    # the overshoot -- a number the caller can act on.
    assert res.position_error_m == pytest.approx(0.25, abs=5e-3)


def test_out_of_plane_target_is_unreachable_and_says_so(chain, limits):
    res = solve_pose(chain, np.array([0.2, 0.2, 0.4]), *limits)
    assert not res.ok
    assert res.position_error_m >= 0.4 - 1e-6


def test_the_result_never_leaves_the_joint_limits(chain):
    """Tight limits must bind even when the unconstrained solution is better."""
    lower = np.array([-0.10, -0.10, -0.10])
    upper = np.array([0.10, 0.10, 0.10])
    target = np.array([0.0, REACH, 0.0])          # straight up: needs ~90 deg
    res = solve_pose(chain, target, lower, upper)
    assert not res.ok
    assert np.all(res.q >= lower - 1e-12) and np.all(res.q <= upper + 1e-12)


def test_the_same_seed_gives_the_same_answer(chain, limits):
    target = np.array([0.35, 0.28, 0.0])
    a = solve_pose(chain, target, *limits, rng=np.random.default_rng(7))
    b = solve_pose(chain, target, *limits, rng=np.random.default_rng(7))
    assert np.array_equal(a.q, b.q)
    assert a.position_error_m == b.position_error_m


def test_restarts_are_reported_so_a_marginal_solve_is_visible(chain, limits):
    res = solve_pose(chain, reachable_target(chain, np.array([0.1, 0.2, 0.1])),
                     *limits)
    assert res.restarts_used >= 1


def test_a_zero_approach_vector_is_rejected(chain, limits):
    with pytest.raises(ValueError, match="zero vector"):
        solve_pose(chain, np.array([0.2, 0.2, 0.0]), *limits,
                   approach=np.zeros(3))


def test_result_renders_both_tolerances(chain, limits):
    res = solve_pose(chain, reachable_target(chain, np.array([0.2, 0.1, 0.1])),
                     *limits)
    assert "mm" in str(res) and "deg" in str(res)


# --- interpolation -------------------------------------------------------

def test_path_hits_both_endpoints_exactly():
    a, b = np.zeros(3), np.array([1.0, -1.0, 0.5])
    path = lerp_path([a, b], [1.0], fps=10)
    assert len(path) == 10
    assert path[-1] == pytest.approx(b)


def test_path_eases_in_and_out():
    """Smoothstep, not linear: a velocity step is a commanded-torque spike, and
    on an arm at its real effort limit a spike is a saturation event."""
    path = lerp_path([np.zeros(1), np.ones(1)], [1.0], fps=60)
    steps = np.diff([float(p[0]) for p in [np.zeros(1), *path]])
    assert steps[0] < steps[len(steps) // 2]
    assert steps[-1] < steps[len(steps) // 2]


def test_path_rejects_a_duration_count_that_cannot_be_right():
    with pytest.raises(ValueError, match="durations"):
        lerp_path([np.zeros(2), np.ones(2), np.zeros(2)], [1.0], fps=30)


def test_a_zero_length_segment_still_produces_a_frame():
    assert len(lerp_path([np.zeros(1), np.ones(1)], [0.0], fps=60)) == 1


# --- the jaws ------------------------------------------------------------

def test_jaw_endpoints_round_trip():
    assert YAM_JAWS.q_for_gap(YAM_JAWS.gap_open_m) == pytest.approx(YAM_JAWS.q_open)
    assert YAM_JAWS.q_for_gap(YAM_JAWS.gap_closed_m) == pytest.approx(YAM_JAWS.q_closed)
    assert YAM_JAWS.gap_for_q(YAM_JAWS.q_open) == pytest.approx(YAM_JAWS.gap_open_m)


def test_jaw_model_is_monotone_the_right_way_round():
    """The previous model was exactly backwards, so "open" shut the hand.

    q runs NEGATIVE as the jaws open, which is the counter-intuitive direction
    and precisely why it was got wrong by reading the finger body origins.
    """
    assert YAM_JAWS.q_open < YAM_JAWS.q_closed
    assert YAM_JAWS.gap_for_q(-0.02) > YAM_JAWS.gap_for_q(-0.01)


def test_too_wide_raises_rather_than_clamping():
    """Clamping gives a hand that closes on nothing and a task that reports
    success while the object never moves."""
    with pytest.raises(GraspTooWide, match="cannot be grasped"):
        YAM_JAWS.q_for_gap(0.120)


def test_the_jaw_calibration_is_sourced():
    assert YAM_JAWS.source and not YAM_JAWS.source.upper().startswith("TODO")


def test_a_mid_stroke_gap_is_linear():
    mid = 0.5 * (YAM_JAWS.gap_closed_m + YAM_JAWS.gap_open_m)
    assert YAM_JAWS.q_for_gap(mid) == pytest.approx(
        0.5 * (YAM_JAWS.q_closed + YAM_JAWS.q_open))


# --- arm placement, which is where a bimanual rig goes wrong quietly ------

def spec(**kw):
    base = dict(name="left", n_arm=6, n_finger=2,
                joint_lower=-np.ones(8), joint_upper=np.ones(8), jaws=YAM_JAWS)
    base.update(kw)
    return ArmSpec(**base)


def test_an_arm_at_the_origin_needs_no_measurement():
    """The single-arm rig defines the frame, so the origin is not a claim."""
    assert spec().at_origin


def test_a_second_arm_without_a_measured_offset_raises():
    """Inventing a mounting offset produces a scene that looks right and puts
    every bimanual reach in the wrong place."""
    with pytest.raises(UnsourcedPlacement, match="measured off the real rig"):
        spec(name="right", base_position_m=(0.0, 0.6, 0.0))


def test_a_todo_is_not_a_source():
    with pytest.raises(UnsourcedPlacement):
        spec(name="right", base_position_m=(0.0, 0.6, 0.0),
             placement_source="TODO measure the mount plate")


def test_a_measured_offset_is_accepted():
    arm = spec(name="right", base_position_m=(0.0, 0.6, 0.0),
               placement_source="calipers on the mount plate, 2026-09-21")
    assert not arm.at_origin
    assert arm.base_position_m == (0.0, 0.6, 0.0)


def test_a_rotated_base_also_counts_as_a_placement():
    h = math.sqrt(0.5)
    with pytest.raises(UnsourcedPlacement):
        spec(name="right", base_orientation_wxyz=(h, 0.0, 0.0, h))


def test_mismatched_limit_arrays_are_rejected():
    with pytest.raises(ValueError, match="lower limits against"):
        spec(joint_lower=-np.ones(8), joint_upper=np.ones(6))


def test_too_few_limits_for_the_joint_count_is_rejected():
    with pytest.raises(ValueError, match="joint limits for an arm"):
        spec(n_arm=6, joint_lower=-np.ones(4), joint_upper=np.ones(4))


def test_inverted_limits_are_rejected_by_joint_number():
    lo, hi = -np.ones(8), np.ones(8)
    lo[2], hi[2] = 0.5, -0.5
    with pytest.raises(ValueError, match="joint 3"):
        spec(joint_lower=lo, joint_upper=hi)


def test_with_fingers_sets_both_fingers_together():
    q = spec().with_fingers(np.arange(6, dtype=float), -0.047)
    assert q.size == 8
    assert q[6] == q[7] == -0.047


def test_clamp_respects_every_limit():
    arm = spec()
    q = arm.clamp(np.full(8, 5.0))
    assert np.all(q <= arm.joint_upper + 1e-12)


# --- trajectories, including two arms on one clock ------------------------

def test_trajectory_lengths_and_duration():
    t = Trajectory(arm="left", fps=60)
    t.start(np.zeros(3))
    t.move_to(np.ones(3), 1.0, "reach")
    t.hold(0.5)
    assert len(t) == 1 + 60 + 30
    assert t.seconds == pytest.approx(91 / 60)
    assert t.q == pytest.approx(np.ones(3))


def test_an_empty_trajectory_refuses_to_report_a_configuration():
    with pytest.raises(ValueError, match="trajectory is empty"):
        _ = Trajectory(arm="left").q


def test_a_trajectory_cannot_be_started_twice():
    t = Trajectory(arm="left")
    t.start(np.zeros(2))
    with pytest.raises(ValueError, match="already started"):
        t.start(np.ones(2))


def test_waypoints_carry_their_arm_and_label():
    t = Trajectory(arm="right", fps=10)
    t.start(np.zeros(2))
    t.move_to(np.ones(2), 0.2, "approach")
    assert t.steps[-1].arm == "right"
    assert t.steps[-1].label == "approach"


def test_two_arms_are_padded_onto_one_clock():
    """A bimanual plan is built per arm and run on one clock. Without padding
    the shorter arm's command array just ends, and whatever reads it next gets
    an index error or a stale row."""
    left, right = Trajectory("left", fps=10), Trajectory("right", fps=10)
    left.start(np.zeros(2))
    left.move_to(np.ones(2), 1.0)
    right.start(np.zeros(2))
    right.move_to(np.ones(2), 0.3)

    pad_to_longest([left, right])

    assert len(left) == len(right)
    assert right.steps[-1].label == "wait"
    assert right.steps[-1].cfg == pytest.approx(np.ones(2))


def test_padding_an_empty_trajectory_is_an_error():
    t = Trajectory("left")
    t.start(np.zeros(1))
    with pytest.raises(ValueError, match="cannot pad an empty"):
        pad_to_longest([t, Trajectory("right")])


# --- the constraint that keeps this testable -----------------------------

def test_the_controller_imports_without_a_physics_engine():
    """Run in a CLEAN interpreter: another test module in this session imports
    pxr, so inspecting this process's sys.modules tests the wrong thing.

    A controller that can only be exercised on the one machine with a GPU is a
    controller nobody checks.
    """
    import subprocess
    import sys

    code = (
        "import sys; import labgen.control;"
        "banned=('newton','warp','torch','isaaclab','omni','pxr');"
        "print(','.join(m for m in banned if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=".")
    assert out.returncode == 0, out.stderr
    leaked = [m for m in out.stdout.strip().split(",") if m]
    assert not leaked, f"importing the controller dragged in {leaked}"


# --- the rig, where a bimanual setup goes quietly wrong --------------------

def rig(second_source="calipers on the mount plate, 2026-09-21"):
    left = spec(name="arm_left")
    right = spec(name="arm_right", base_position_m=(0.44, 0.0, 0.0),
                 placement_source=second_source)
    return BimanualRig([left, right])


def test_a_measured_rig_is_verified():
    r = rig()
    assert r.verified and r.unverified_arms == []
    assert r.why_unverified() == ""


def test_a_placeholder_offset_builds_but_does_not_verify():
    """A PLACEHOLDER is the difference between "cannot proceed" and "can
    proceed but must not be trusted".

    A TODO blocks, because it means there is no number. A placeholder passes,
    because the rest of the stack has to be buildable before the tape measure
    comes out -- and it poisons `verified` so nothing that writes training data
    will accept it.
    """
    r = rig("PLACEHOLDER 440 mm along +x, not a measurement")
    assert not r.verified
    assert r.unverified_arms == ["arm_right"]
    assert "NOT measured" in r.why_unverified()
    assert "arm_right" in r.why_unverified()


def test_a_todo_offset_does_not_even_build():
    with pytest.raises(UnsourcedPlacement):
        rig("TODO ask the lab")


def test_recording_is_refused_on_an_unmeasured_rig():
    """The whole point of tracking this. A demo recorded on a guessed base
    offset trains a policy on a robot that does not exist, and the error is
    invisible in the recorded data."""
    r = rig("PLACEHOLDER 440 mm along +x")
    with pytest.raises(UnsourcedPlacement, match="refusing to record"):
        r.require_verified("record a demonstration")
    rig().require_verified("record a demonstration")      # measured: fine


def test_exactly_one_arm_may_define_the_frame():
    with pytest.raises(ValueError, match="at the origin"):
        BimanualRig([spec(name="a"), spec(name="b")])


def test_two_arms_cannot_share_a_name():
    with pytest.raises(ValueError, match="share a name"):
        BimanualRig([spec(name="arm"), spec(name="arm",
                                            base_position_m=(0.44, 0.0, 0.0),
                                            placement_source="measured")])


def test_a_rig_needs_at_least_one_arm():
    with pytest.raises(ValueError, match="at least one arm"):
        BimanualRig([])


def test_arms_are_addressable_by_name():
    r = rig()
    assert r["arm_right"].base_position_m == (0.44, 0.0, 0.0)
    assert r.names == ["arm_left", "arm_right"]
    assert len(r) == 2
    with pytest.raises(KeyError):
        _ = r["arm_middle"]


def test_a_single_arm_rig_is_verified_without_any_measurement():
    """One arm defines the frame, so the origin is not a claim about anything."""
    assert BimanualRig([spec(name="arm_left")]).verified


def test_placeholder_detection_is_not_fooled_by_case_or_padding():
    assert is_placeholder("  placeholder whatever ")
    assert not is_placeholder("measured with calipers")
    assert not is_placeholder("")


# --- the finger pad -------------------------------------------------------

def test_the_shipped_pad_is_flagged_unverified():
    """It came off the URDF mesh, not off real hardware. Until calipers say
    otherwise it must not pass as a measurement."""
    assert not YAM_PAD.verified
    assert "NOT caliper-verified" in YAM_PAD.source


def test_the_pad_is_far_smaller_than_the_hull_it_replaces():
    """The hull spans +/-36 mm, i.e. 72 mm across, and engulfs a 70 mm beaker.
    The real jaw is a quarter of that."""
    assert YAM_PAD.width_m < 0.030
    assert YAM_PAD.length_m < 0.030


def test_pad_half_extents_are_ordered_width_thickness_length():
    hx, hy, hz = YAM_PAD.half_extents_m
    assert hx == pytest.approx(YAM_PAD.width_m / 2)
    assert hy == pytest.approx(YAM_PAD.thickness_m / 2)
    assert hz == pytest.approx(YAM_PAD.length_m / 2)


def test_the_two_pads_are_mirrored():
    left, right = YAM_PAD.centre_for(1), YAM_PAD.centre_for(-1)
    assert left[0] == pytest.approx(-right[0])
    assert left[1] == pytest.approx(-right[1])
    assert left[2] == pytest.approx(right[2]), "both sit at the same depth"


def test_a_pad_without_a_source_is_rejected():
    with pytest.raises(ValueError, match="needs a source"):
        FingerPad(width_m=0.02, length_m=0.02, thickness_m=0.005,
                  centre_m=(0, 0, 0), source="")


def test_a_pad_cannot_claim_verified_on_a_todo():
    with pytest.raises(ValueError, match="needs a real source"):
        FingerPad(width_m=0.02, length_m=0.02, thickness_m=0.005,
                  centre_m=(0, 0, 0), source="TODO calipers", verified=True)


def test_a_zero_sized_pad_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        FingerPad(width_m=0.0, length_m=0.02, thickness_m=0.005,
                  centre_m=(0, 0, 0), source="measured")
