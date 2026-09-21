"""Retargeting, checked without a phone, a network or a GPU.

The device interface exists so the phone and the Quest are the same thing to
everything downstream. The consequence worth having is this file: every rule
the operator will actually feel -- press to engage, release to freeze, re-press
without the arm jumping, a dropout not flinging the arm across the bench -- is
arithmetic, and arithmetic can be pinned.

These are the failures that would otherwise be found by watching a real arm
move wrongly while holding a phone, which is the worst debugging loop available.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from labgen.devices import (MAX_JUMP_DEG, MAX_JUMP_M, HandTarget, PoseEvent,
                            RelativeRetargeter, ReplayPoseSource, Workspace)

HAND0 = np.array([0.20, 0.30, 0.15])


def ev(x=0.0, y=0.0, z=0.0, engaged=True, grip=0.0, scale=1.0,
       quat=(1.0, 0.0, 0.0, 0.0), **kw) -> PoseEvent:
    return PoseEvent(position_m=(x, y, z), orientation_wxyz=quat,
                     engaged=engaged, grip=grip, scale=scale, **kw)


def quat_z(deg: float):
    h = math.radians(deg) / 2.0
    return (math.cos(h), 0.0, 0.0, math.sin(h))


# --- engage / release, the part the operator feels ------------------------

def test_not_engaged_gives_no_target_at_all():
    """A hand that drifts while the operator is not pressing is unusable."""
    r = RelativeRetargeter()
    assert r.update(ev(0.1, 0.1, 0.1, engaged=False), HAND0) is None
    assert not r.engaged


def test_first_engaged_sample_anchors_and_does_not_move_the_hand():
    """Pressing must not itself command motion, wherever the phone happens
    to be when the operator presses."""
    r = RelativeRetargeter()
    t = r.update(ev(0.9, -0.4, 0.2), HAND0)
    assert t.position_m == pytest.approx(HAND0)
    assert t.engaged and r.engaged


def test_the_hand_moves_by_the_device_delta():
    r = RelativeRetargeter()
    r.update(ev(0.0, 0.0, 0.0), HAND0)
    t = r.update(ev(0.03, -0.02, 0.01), HAND0)
    assert t.position_m == pytest.approx(HAND0 + np.array([0.03, -0.02, 0.01]))


def test_releasing_freezes_and_repressing_does_not_jump():
    """Clutching. Without it a phone-sized motion range drives a bench-sized
    workspace only once, and the operator cannot recentre."""
    r = RelativeRetargeter()
    r.update(ev(0.0, 0.0, 0.0), HAND0)
    moved = r.update(ev(0.04, 0.0, 0.0), HAND0).position_m
    assert moved == pytest.approx(HAND0 + np.array([0.04, 0.0, 0.0]))

    assert r.update(ev(0.04, 0.0, 0.0, engaged=False), moved) is None

    # Phone recentred far away; re-press must anchor there, not lunge.
    t = r.update(ev(-0.5, 0.3, 0.0), moved)
    assert t.position_m == pytest.approx(moved)


def test_the_target_is_computed_from_the_anchor_not_the_live_hand():
    """The caller passes the measured hand pose every tick. If that fed the
    target the loop would chase its own tracking error and run away."""
    r = RelativeRetargeter()
    r.update(ev(0.0, 0.0, 0.0), HAND0)
    drifting = HAND0 + np.array([0.05, 0.05, 0.05])
    t = r.update(ev(0.01, 0.0, 0.0), drifting)
    assert t.position_m == pytest.approx(HAND0 + np.array([0.01, 0.0, 0.0]))


# --- scaling -------------------------------------------------------------

def test_position_scale_multiplies_the_delta():
    r = RelativeRetargeter(position_scale=2.0)
    r.update(ev(0.0, 0.0, 0.0), HAND0)
    t = r.update(ev(0.01, 0.0, 0.0), HAND0)
    assert t.position_m == pytest.approx(HAND0 + np.array([0.02, 0.0, 0.0]))


def test_the_operators_own_scale_from_the_device_also_applies():
    r = RelativeRetargeter()
    r.update(ev(0.0, 0.0, 0.0, scale=0.5), HAND0)
    t = r.update(ev(0.02, 0.0, 0.0, scale=0.5), HAND0)
    assert t.position_m == pytest.approx(HAND0 + np.array([0.01, 0.0, 0.0]))


def test_a_negative_scale_cannot_invert_the_mapping():
    """Inverted control is the single most disorienting failure available."""
    r = RelativeRetargeter()
    r.update(ev(0.0, 0.0, 0.0, scale=-1.0), HAND0)
    t = r.update(ev(0.02, 0.0, 0.0, scale=-1.0), HAND0)
    assert t.position_m == pytest.approx(HAND0)


# --- orientation ---------------------------------------------------------

def test_device_rotation_turns_the_hand_by_the_same_delta():
    r = RelativeRetargeter()
    r.update(ev(quat=quat_z(0.0)), HAND0, np.eye(3))
    t = r.update(ev(quat=quat_z(30.0)), HAND0, np.eye(3))
    yaw = math.degrees(math.atan2(t.rotation[1, 0], t.rotation[0, 0]))
    assert yaw == pytest.approx(30.0, abs=1e-6)


def test_orientation_can_be_switched_off_for_position_only_driving():
    r = RelativeRetargeter(use_orientation=False)
    r.update(ev(quat=quat_z(0.0)), HAND0, np.eye(3))
    t = r.update(ev(0.01, 0.0, 0.0, quat=quat_z(30.0)), HAND0, np.eye(3))
    assert t.rotation == pytest.approx(np.eye(3))
    assert t.position_m == pytest.approx(HAND0 + np.array([0.01, 0.0, 0.0]))


def test_the_approach_axis_is_the_device_forward_axis():
    t = HandTarget(np.zeros(3), np.eye(3), 0.0, True)
    assert t.approach == pytest.approx(np.array([0.0, 0.0, 1.0]))


# --- dropouts ------------------------------------------------------------

def test_a_tracking_dropout_re_anchors_instead_of_flinging_the_arm():
    """A phone losing tracking emits one enormous delta. Forwarding it
    commands the arm across the bench in a single tick."""
    r = RelativeRetargeter()
    r.update(ev(0.0, 0.0, 0.0), HAND0)
    assert r.update(ev(0.01, 0.0, 0.0), HAND0) is not None

    assert r.update(ev(3.0, 0.0, 0.0), HAND0) is None      # the dropout
    assert r.jumps_rejected == 1

    # and it re-anchors there, so the next small motion is relative to it
    t = r.update(ev(3.01, 0.0, 0.0), HAND0)
    assert t.position_m == pytest.approx(HAND0 + np.array([0.01, 0.0, 0.0]))


def test_a_large_rotation_in_one_sample_is_also_a_dropout():
    r = RelativeRetargeter()
    r.update(ev(quat=quat_z(0.0)), HAND0, np.eye(3))
    assert r.update(ev(quat=quat_z(90.0)), HAND0, np.eye(3)) is None
    assert r.jumps_rejected == 1


def test_motion_just_under_the_threshold_is_not_rejected():
    r = RelativeRetargeter()
    r.update(ev(0.0, 0.0, 0.0), HAND0)
    assert r.update(ev(MAX_JUMP_M * 0.99, 0.0, 0.0), HAND0) is not None
    assert r.jumps_rejected == 0


def test_the_thresholds_match_the_upstream_package():
    """Same numbers the teleop package uses for its own jump protection.

    Pinned because a different value here and there means the two disagree
    about what a dropout is, and the symptom is intermittent.
    """
    assert MAX_JUMP_M == 0.05
    assert MAX_JUMP_DEG == 35.0


# --- workspace -----------------------------------------------------------

def test_the_target_is_clamped_into_the_workspace_and_says_so():
    """Clamped rather than refused: the operator is watching a screen, and an
    arm that stops responding at the edge reads as a bug."""
    ws = Workspace(lower_m=(-0.5, 0.0, 0.0), upper_m=(0.5, 0.6, 0.4))
    r = RelativeRetargeter(workspace=ws)
    r.update(ev(0.0, 0.0, 0.0), HAND0)
    # Climb in realistic increments. A single 1 m sample is a DROPOUT, not a
    # command -- the first version of this test asserted on the clamp and got
    # None back, which is the jump guard doing its job.
    t = None
    for i in range(1, 9):
        t = r.update(ev(0.0, 0.0, 0.04 * i), HAND0)
    assert t.position_m[2] == pytest.approx(0.4)
    assert r.clamped


def test_a_target_inside_the_workspace_is_not_flagged():
    ws = Workspace(lower_m=(-0.5, 0.0, 0.0), upper_m=(0.5, 0.6, 0.4))
    r = RelativeRetargeter(workspace=ws)
    r.update(ev(0.0, 0.0, 0.0), HAND0)
    r.update(ev(0.0, 0.0, 0.01), HAND0)
    assert not r.clamped


# --- the wire format -----------------------------------------------------

def good_payload(**kw):
    p = {"position": {"x": 0.1, "y": 0.2, "z": 0.3},
         "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
         "move": True, "gripper": 0.5, "scale": 1.0, "seq": 7}
    p.update(kw)
    return p


def test_a_well_formed_payload_round_trips():
    e = PoseEvent.from_json(good_payload())
    assert e.position_m == (0.1, 0.2, 0.3)
    assert e.engaged and e.grip == 0.5 and e.seq == 7


def test_a_missing_field_raises_rather_than_defaulting_to_zero():
    """A dropped coordinate defaulting to 0 is a hand that lunges to the
    origin, and the operator finds out by watching it happen."""
    bad = good_payload()
    del bad["position"]["y"]
    with pytest.raises(ValueError, match="malformed pose payload"):
        PoseEvent.from_json(bad)


def test_a_non_unit_quaternion_is_rejected():
    with pytest.raises(ValueError, match="unit quaternion"):
        PoseEvent.from_json(good_payload(
            orientation={"w": 0.5, "x": 0.0, "y": 0.0, "z": 0.0}))


def test_a_slightly_off_unit_quaternion_is_normalised():
    e = PoseEvent.from_json(good_payload(
        orientation={"w": 1.0004, "x": 0.0, "y": 0.0, "z": 0.0}))
    assert sum(c * c for c in e.orientation_wxyz) == pytest.approx(1.0)


def test_the_engage_button_defaults_to_not_engaged():
    """If the field is absent the safe reading is 'not pressed'."""
    p = good_payload()
    del p["move"]
    assert not PoseEvent.from_json(p).engaged


def test_the_hand_defaults_to_right_but_can_be_named():
    assert PoseEvent.from_json(good_payload()).hand == "right"
    assert PoseEvent.from_json(good_payload(hand="left")).hand == "left"


# --- a whole recorded session --------------------------------------------

def test_a_recorded_session_replays_to_the_same_path():
    """A source is a source: the same retargeting runs on a recording, which
    is what makes any of this checkable without hardware."""
    recorded = [ev(0.0, 0.0, 0.0), ev(0.01, 0.0, 0.0), ev(0.02, 0.0, 0.0),
                ev(0.02, 0.0, 0.0, engaged=False), ev(0.5, 0.5, 0.5)]
    r = RelativeRetargeter()
    hand = HAND0.copy()
    path = []
    for e in ReplayPoseSource(recorded).events():
        t = r.update(e, hand)
        if t is not None:
            hand = t.position_m
            path.append(hand.copy())

    assert len(path) == 4
    assert path[2] == pytest.approx(HAND0 + np.array([0.02, 0.0, 0.0]))
    # the release/re-press pair must leave the hand exactly where it was
    assert path[3] == pytest.approx(path[2])


def test_two_hands_keep_separate_anchors():
    """Bimanual: one accumulated pose in the bridge cannot express this, which
    is why the anchoring is here and not there."""
    left, right = RelativeRetargeter(), RelativeRetargeter()
    lh, rh = np.array([0.1, 0.3, 0.2]), np.array([-0.1, 0.3, 0.2])
    left.update(ev(0.0, 0.0, 0.0, hand="left"), lh)
    right.update(ev(5.0, 5.0, 5.0, hand="right"), rh)

    lt = left.update(ev(0.01, 0.0, 0.0, hand="left"), lh)
    rt = right.update(ev(5.0, 5.02, 5.0, hand="right"), rh)

    assert lt.position_m == pytest.approx(lh + np.array([0.01, 0.0, 0.0]))
    assert rt.position_m == pytest.approx(rh + np.array([0.0, 0.02, 0.0]))


def test_devices_import_without_a_physics_engine_or_a_web_stack():
    """This module runs inside the Isaac Lab environment, which CLAUDE.md rule
    5 says not to install into. The FastAPI/uvicorn/WebXR half is a separate
    process in its own venv; this half must need nothing.
    """
    import subprocess
    import sys

    code = (
        "import sys; import labgen.devices;"
        "banned=('newton','warp','torch','isaaclab','fastapi','uvicorn','teleop');"
        "print(','.join(m for m in banned if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=".")
    assert out.returncode == 0, out.stderr
    leaked = [m for m in out.stdout.strip().split(",") if m]
    assert not leaked, f"importing the device layer dragged in {leaked}"
