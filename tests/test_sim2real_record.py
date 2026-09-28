"""Real recorder must reject ambiguous adapters and unmeasured gripper limits."""

import json
import io
import sys
import types
from enum import Enum

import numpy as np
import pytest

from scripts.sim2real_record import (REAL_MAX_START_DELTA_RAD, expected_adapter_serial,
                                     require_joint_margin, saved_gripper_limits, start_pose_error,
                                     wait_for_guided_reset)


def test_persistent_link_identifies_the_adapter(tmp_path):
    (tmp_path / "right.link").write_text(
        "[Match]\nDriver=gs_usb\n"
        "Property=ID_SERIAL=canable.io_canable2_gs_usb_RIGHT123\n"
        "[Link]\nName=can_right\n"
    )
    assert expected_adapter_serial({}, "can_right", tmp_path) == "RIGHT123"
    with pytest.raises(ValueError, match="exactly one serial-pinned"):
        expected_adapter_serial({}, "can_left", tmp_path)


def test_config_mapping_cannot_be_marked_unverified(tmp_path):
    with pytest.raises(ValueError, match="mapping_verified"):
        expected_adapter_serial({"adapter_serial": "RIGHT123", "mapping_verified": False},
                                "can_right", tmp_path)


def test_saved_gripper_limits_are_bound_to_channel_and_model(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("robot: {}\n")
    limits = tmp_path / "limits.json"
    limits.write_text(json.dumps({"channel": "can_right", "gripper_type": "linear_4310",
                                  "gripper_limits_rad": [6.42, 1.11]}))
    cfg = {"gripper_limits_path": "limits.json"}
    assert saved_gripper_limits(cfg, config, "can_right", "linear_4310") == pytest.approx(
        np.array([6.42, 1.11]))
    with pytest.raises(ValueError, match="do not match"):
        saved_gripper_limits(cfg, config, "can_left", "linear_4310")


def test_missing_or_bad_limits_refuse_before_opening_the_arm(tmp_path):
    config = tmp_path / "config.yaml"
    with pytest.raises(ValueError, match="missing"):
        saved_gripper_limits({}, config, "can_right", "linear_4310")
    (tmp_path / "limits.json").write_text(json.dumps({
        "channel": "can_right", "gripper_type": "linear_4310",
        "gripper_limits_rad": [1.0, 1.0]}))
    with pytest.raises(ValueError, match="distinct"):
        saved_gripper_limits({"gripper_limits_path": "limits.json"}, config,
                             "can_right", "linear_4310")


def test_real_start_pose_guard_is_bounded():
    from labgen.sim2real import RESET_POSE_RAD
    near = np.array(RESET_POSE_RAD)
    near[1] += REAL_MAX_START_DELTA_RAD
    assert start_pose_error(near) == pytest.approx(REAL_MAX_START_DELTA_RAD)
    near[1] += 0.01
    assert start_pose_error(near) > REAL_MAX_START_DELTA_RAD
    with pytest.raises(ValueError, match="six finite"):
        start_pose_error([float("nan")] * 6)


def test_guided_reset_requires_ready_and_a_close_pose(monkeypatch):
    from labgen.sim2real import RESET_POSE_RAD

    class Arm:
        def __init__(self, q):
            self.q = q

        def get_observations(self):
            return {"joint_pos": self.q}

    monkeypatch.setattr("scripts.sim2real_record.select.select",
                        lambda *_: ([True], [], []))
    goal = np.asarray(RESET_POSE_RAD)
    far = Arm(goal + np.array([0.2, 0, 0, 0, 0, 0]))
    monkeypatch.setattr("scripts.sim2real_record.sys.stdin", io.StringIO("READY\nABORT\n"))
    assert not wait_for_guided_reset(far, goal)
    close = Arm(goal.copy())
    monkeypatch.setattr("scripts.sim2real_record.sys.stdin", io.StringIO("READY\n"))
    assert wait_for_guided_reset(close, goal)


def test_joint_margin_rejects_a_command_near_a_limit():
    bounds = np.tile([-0.15, 2.0], (6, 1))
    q = np.zeros((2, 6))
    q[1, :] = 0.10
    require_joint_margin(q, bounds)
    q[1, 2] = 1.97
    with pytest.raises(ValueError, match="j3"):
        require_joint_margin(q, bounds)


def test_current_pose_real_path_never_moves_to_the_distant_reset(monkeypatch, tmp_path):
    from scripts import sim2real_record as recorder
    from labgen.sim2real import Log

    class ArmType(Enum):
        YAM = 1

    class GripperType(Enum):
        LINEAR_4310 = 1

    class Robot:
        def __init__(self):
            self.q = np.zeros(7)
            self.closed = False

        def get_joint_pos(self):
            return self.q.copy()

        def get_observations(self):
            return {"joint_pos": self.q.copy(), "joint_vel": np.zeros(7)}

        def get_robot_info(self):
            return {"joint_limits": np.tile([-1.0, 2.0], (7, 1))}

        def command_joint_pos(self, q):
            self.q = np.asarray(q, dtype=float).copy()

        def close(self):
            self.closed = True

    robot = Robot()
    i2rt = types.ModuleType("i2rt")
    robots = types.ModuleType("i2rt.robots")
    factory = types.ModuleType("i2rt.robots.get_robot")
    utils = types.ModuleType("i2rt.robots.utils")
    factory.get_yam_robot = lambda **kwargs: robot
    utils.ArmType, utils.GripperType = ArmType, GripperType
    yaml = types.ModuleType("yaml")
    yaml.safe_load = lambda _: {"robot": {"channel": "can_right",
                                           "gripper_type": "linear_4310"}}
    for name, module in (("i2rt", i2rt), ("i2rt.robots", robots),
                         ("i2rt.robots.get_robot", factory),
                         ("i2rt.robots.utils", utils), ("yaml", yaml)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(recorder, "saved_gripper_limits", lambda *_: np.array([6.0, 1.0]))
    monkeypatch.setattr(recorder, "expected_adapter_serial", lambda *_: "RIGHT")
    monkeypatch.setattr(recorder, "claim_bus", lambda *_: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(recorder.time, "sleep", lambda *_: None)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("robot: {}\n")
    out = tmp_path / "real.npz"
    monkeypatch.setattr(sys, "argv", ["sim2real_record.py", "--backend", "real",
                                    "--teleop-config", str(cfg),
                                    "--i-have-cleared-the-workspace", "--start-at-current",
                                    "--out", str(out)])
    assert recorder.main() == 0
    assert robot.closed
    log = Log.load(out)
    assert log.protocol.base == (0.0,) * 6
    assert log.q_cmd.max() == pytest.approx(0.10)
