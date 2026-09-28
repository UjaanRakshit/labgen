"""The sim-to-real harness, checked against logs whose answers are known.

A comparison tool that cannot tell two different robots apart, or that reports
agreement between logs of different experiments, is worse than none: it would
certify a sim as accurate. So the core checks here are that it DISTINGUISHES --
identical logs agree, a known offset shows up as that offset, a slower servo
shows up as a slower rise -- and that it refuses comparisons that would be
meaningless.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from labgen.sim2real import (CONTROL_HZ, RESET_POSE_RAD, Log, Protocol, Segment,
                             compare, current_pose_protocol, default_protocol,
                             step_metrics)


def first_order_log(protocol: Protocol, tau_s: float, source: str,
                    offset_rad: float = 0.0) -> Log:
    """A robot whose every joint is a first-order lag of time constant tau."""
    q_cmd, seg = protocol.commands()
    dt = 1.0 / protocol.hz
    q = np.empty_like(q_cmd)
    q[0] = q_cmd[0]
    a = dt / (tau_s + dt)
    for k in range(1, len(q_cmd)):
        q[k] = q[k - 1] + a * (q_cmd[k] - q[k - 1])
    q = q + offset_rad
    t = np.arange(len(q_cmd)) * dt
    return Log(t=t, q_cmd=q_cmd, q=q, qd=np.gradient(q, dt, axis=0),
               eff=np.full_like(q, np.nan), segment=seg, source=source,
               protocol_json=protocol.to_json())


# --- the protocol ---------------------------------------------------------

def test_the_default_protocol_starts_at_the_rigs_reset_pose():
    q_cmd, _ = default_protocol().commands()
    assert q_cmd[0] == pytest.approx(np.array(RESET_POSE_RAD))


def test_it_runs_at_the_real_teleop_rate():
    assert CONTROL_HZ == 100.0
    p = default_protocol()
    q_cmd, _ = p.commands()
    assert len(q_cmd) == pytest.approx(p.seconds * CONTROL_HZ, abs=len(p.segments))


def test_every_joint_is_excited_once():
    joints = [s.joint for s in default_protocol().segments if s.kind == "step"]
    assert sorted(joints) == [0, 1, 2, 3, 4, 5]


def test_excitation_is_small_enough_for_hardware():
    """First-instant PD torque must sit well inside each motor's peak."""
    from labgen.hardware import YAM_V1
    for s in default_protocol().segments:
        if s.kind in ("step", "sine"):
            tau0 = abs(s.amplitude) * YAM_V1.kp[s.joint]
            assert tau0 < 0.5 * YAM_V1.torque_max_nm[s.joint], s


def test_a_large_amplitude_is_refused():
    """The protocol runs on a real arm; a typo must not become a 1 rad jump."""
    with pytest.raises(ValueError, match="too large"):
        Segment("step", 1.0, joint=1, amplitude=0.5)


def test_steps_are_eased_not_jumped():
    p = Protocol(segments=[Segment("step", 1.0, joint=2, amplitude=0.1, ease_s=0.05)])
    q_cmd, _ = p.commands()
    d = np.diff(q_cmd[:, 2])
    assert d.max() < 0.1, "the whole step must not happen in one tick"
    assert q_cmd[-1, 2] - q_cmd[0, 2] == pytest.approx(0.1)


def test_current_pose_steps_return_smoothly_to_the_measured_start():
    base = np.array([-0.05, 0.0, 0.0, -0.08, 0.0, 0.0])
    p = current_pose_protocol(base)
    commands, _ = p.commands()
    assert commands[0] == pytest.approx(base)
    assert commands[-1] == pytest.approx(base)
    assert np.max(commands - base, axis=0) == pytest.approx(np.full(6, 0.10))
    assert np.min(commands - base) >= -1e-12
    assert np.max(np.abs(np.diff(commands, axis=0))) < 0.02
    assert all(s.kind != "sine" for s in p.segments)


def test_current_pose_protocol_refuses_a_large_or_negative_step():
    for amplitude in (-0.1, 0.0, 0.11):
        with pytest.raises(ValueError, match="amplitude"):
            current_pose_protocol(np.zeros(6), amplitude)


def test_the_protocol_round_trips_through_json():
    p = default_protocol()
    assert Protocol.from_json(p.to_json()).to_json() == p.to_json()


def test_an_unknown_segment_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown segment kind"):
        Segment("jerk", 1.0, joint=0, amplitude=0.1)


# --- step metrics ---------------------------------------------------------

def test_step_metrics_on_a_known_first_order_response():
    """A first-order lag rises 10-90% in tau*ln(9) and never overshoots."""
    tau = 0.1
    t = np.arange(0, 2.0, 0.001)
    y = 1.0 - np.exp(-t / tau)
    m = step_metrics(y, t, 0.0, 1.0)
    assert m.rise_s == pytest.approx(tau * math.log(9), abs=0.002)
    assert m.overshoot_pct == 0.0
    assert m.steady_err_rad < 1e-6


def test_step_metrics_see_overshoot():
    t = np.arange(0, 3.0, 0.001)
    y = 1.0 - np.exp(-2 * t) * np.cos(8 * t)
    assert step_metrics(y, t, 0.0, 1.0).overshoot_pct > 10


def test_a_zero_step_is_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        step_metrics(np.zeros(5), np.arange(5.0), 0.3, 0.3)


# --- the comparison, which must distinguish --------------------------------

def test_identical_logs_agree_exactly():
    p = default_protocol()
    a = first_order_log(p, 0.08, "real")
    b = first_order_log(p, 0.08, "sim")
    c = compare(a, b)
    assert c.ok
    assert c.rms_diff_deg.max() < 1e-9


def test_a_constant_offset_shows_up_as_that_offset():
    p = default_protocol()
    c = compare(first_order_log(p, 0.08, "real"),
                first_order_log(p, 0.08, "sim", offset_rad=math.radians(1.0)))
    assert c.rms_diff_deg == pytest.approx(np.ones(6), abs=1e-6)
    assert not c.ok, "1 deg everywhere must fail a 0.5 deg tolerance"


def test_a_slower_servo_shows_up_as_a_slower_rise():
    """The whole point: a sim whose joints respond slower than the robot's must
    be caught, and caught in the step figures, not just the RMS."""
    p = default_protocol()
    c = compare(first_order_log(p, 0.05, "real"), first_order_log(p, 0.15, "sim"))
    for sa, sb in zip(c.steps_a, c.steps_b):
        assert sb.rise_s > 2.5 * sa.rise_s


def test_logs_of_different_protocols_are_refused():
    """Otherwise the difference measures the experiments, not the robots."""
    a = first_order_log(default_protocol(0.10), 0.08, "real")
    b = first_order_log(default_protocol(0.05), 0.08, "sim")
    with pytest.raises(ValueError, match="different protocols"):
        compare(a, b)


def test_misaligned_logs_are_refused():
    p = default_protocol()
    a = first_order_log(p, 0.08, "real")
    b = first_order_log(p, 0.08, "sim")
    cut = len(b.t) // 2
    b = Log(t=b.t[:cut], q_cmd=b.q_cmd[:cut], q=b.q[:cut], qd=b.qd[:cut],
            eff=b.eff[:cut], segment=b.segment[:cut], source=b.source,
            protocol_json=b.protocol_json)
    with pytest.raises(ValueError, match="not aligned"):
        compare(a, b)


def test_the_report_names_both_sources_and_a_verdict():
    p = default_protocol()
    text = str(compare(first_order_log(p, 0.08, "real"), first_order_log(p, 0.08, "sim")))
    assert "real" in text and "sim" in text and "AGREE" in text


def test_logs_round_trip_through_disk(tmp_path):
    p = default_protocol()
    a = first_order_log(p, 0.08, "real")
    a.save(tmp_path / "a.npz")
    b = Log.load(tmp_path / "a.npz")
    assert b.source == "real"
    assert np.array_equal(a.q, b.q)
    assert b.protocol_json == a.protocol_json


def test_the_harness_needs_no_physics_engine():
    import subprocess
    import sys
    code = ("import sys; import labgen.sim2real;"
            "banned=('newton','warp','torch','isaaclab','mujoco','i2rt');"
            "print(','.join(m for m in banned if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".")
    assert out.returncode == 0, out.stderr
    assert not [m for m in out.stdout.strip().split(",") if m]
