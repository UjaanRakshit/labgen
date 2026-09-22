"""The real robot's actuator parameters, pinned against the source they came from.

These are copied, not derived, so the tests cannot check them against physics.
What they CAN do is make a change to any of them loud: every number here decides
how the simulated arm responds to a command, and a quiet edit would make the
sim diverge from the hardware in a way no other test would notice.

The regression that matters most is `test_the_shoulder_is_not_a_10_nm_joint`.
This project spent a round claiming 10 N.m was the YAM's rating on every joint
and that gravity used 81% of it at the shoulder. Both halves were wrong. Joints
1-3 are DM4340s with a 28 N.m peak, and the 81% came from a gravity figure
(8.06 N.m) computed with each body's weight at its frame origin instead of its
centre of mass. Corrected, gravity at the worst pose of the task plan is
10.60 N.m -- over the simplified 10 N.m MJCF limit, 38% of the real motor.
"""

from __future__ import annotations

import numpy as np
import pytest

from labgen.hardware import (DM4310, DM4340, I2RT_REAL_VERSION, YAM_V1,
                             ArmHardware)


def test_the_real_rig_runs_i2rt_1_1_2():
    """Pinned so an upgrade of the deployment venv is a visible change."""
    assert I2RT_REAL_VERSION == "1.1.2"
    assert I2RT_REAL_VERSION in YAM_V1.source


def test_motor_layout_matches_the_real_config():
    assert [m.name for m in YAM_V1.motors] == [
        "DM4340", "DM4340", "DM4340", "DM4310", "DM4310", "DM4310"]


def test_the_shoulder_is_not_a_10_nm_joint():
    """The correction. Gravity at the worst pose in the task plan is 10.60 N.m
    at joint 2 (centre-of-mass lever arms): over a uniform 10 N.m limit, but
    38% of the DM4340's peak."""
    assert list(YAM_V1.torque_max_nm) == [28.0, 28.0, 28.0, 10.0, 10.0, 10.0]
    assert YAM_V1.headroom([0, 10.60, 0, 0, 0, 0])[1] == pytest.approx(0.379, abs=1e-3)
    assert 10.60 > 10.0, "infeasible under the simplified MJCF limit"


def test_motor_constants_match_the_driver():
    assert (DM4340.torque_max_nm, DM4340.velocity_max_rad_s, DM4340.gear_ratio) == (28.0, 10.0, 40.0)
    assert (DM4310.torque_max_nm, DM4310.velocity_max_rad_s, DM4310.gear_ratio) == (10.0, 30.0, 10.0)


def test_controller_gains_are_the_real_ones():
    """kp 80 on the shoulder and 10 on the wrist. The sim previously ran 3000
    everywhere -- 37x and 300x too stiff -- which tracks beautifully and says
    nothing about the real arm."""
    assert YAM_V1.kp == (80.0, 80.0, 80.0, 10.0, 10.0, 10.0)
    assert YAM_V1.kd == (5.0, 5.0, 5.0, 1.5, 1.5, 1.5)


def test_joint_friction_is_the_real_ones():
    """Recorded by i2rt, not guessed here. An earlier report said joint friction
    'needs system identification on hardware'; i2rt had already done it."""
    assert YAM_V1.coulomb_friction_nm == (0.3, 0.3, 0.3, 0.06, 0.06, 0.06)


def test_the_real_arm_needs_more_gravity_torque_than_its_model():
    """The hardware factor is above 1 on joints 2-4, so the real arm is heavier
    in effect than the URDF. The sim applies 1.0 because its plant IS the model;
    this number is the evidence of the gap, kept rather than silently used."""
    f = YAM_V1.gravity_comp_factor
    assert f == (1.0, 1.1, 1.1, 1.2, 1.0, 1.0)
    assert max(f) > 1.0


def test_armature_is_recorded_as_unknown_not_guessed():
    """Armature is rotor inertia x gear^2, and i2rt has the gears but not the
    rotor inertia. A number here would be invented."""
    assert not YAM_V1.armature_known
    assert YAM_V1.armature_kg_m2 is None


def test_every_parameter_array_has_one_entry_per_joint():
    for name in ("kp", "kd", "gravity_comp_factor", "grav_comp_kd",
                 "coulomb_friction_nm", "directions"):
        assert len(getattr(YAM_V1, name)) == YAM_V1.n_joints == 6


def test_a_mismatched_array_is_rejected():
    with pytest.raises(ValueError, match="entries for 6 motors"):
        ArmHardware(name="bad", motors=YAM_V1.motors, kp=(1.0,) * 5,
                    kd=YAM_V1.kd, gravity_comp_factor=YAM_V1.gravity_comp_factor,
                    grav_comp_kd=YAM_V1.grav_comp_kd,
                    coulomb_friction_nm=YAM_V1.coulomb_friction_nm,
                    directions=YAM_V1.directions, source="test")


def test_hardware_without_a_source_is_rejected():
    with pytest.raises(ValueError, match="need a source"):
        ArmHardware(name="bad", motors=YAM_V1.motors, kp=YAM_V1.kp, kd=YAM_V1.kd,
                    gravity_comp_factor=YAM_V1.gravity_comp_factor,
                    grav_comp_kd=YAM_V1.grav_comp_kd,
                    coulomb_friction_nm=YAM_V1.coulomb_friction_nm,
                    directions=YAM_V1.directions, source="")


def test_non_positive_gains_are_rejected():
    with pytest.raises(ValueError, match="gains"):
        ArmHardware(name="bad", motors=YAM_V1.motors, kp=(0.0,) * 6, kd=YAM_V1.kd,
                    gravity_comp_factor=YAM_V1.gravity_comp_factor,
                    grav_comp_kd=YAM_V1.grav_comp_kd,
                    coulomb_friction_nm=YAM_V1.coulomb_friction_nm,
                    directions=YAM_V1.directions, source="test")


def test_hardware_module_needs_no_physics_engine():
    import subprocess
    import sys
    code = ("import sys; import labgen.hardware;"
            "banned=('newton','warp','torch','isaaclab','pxr');"
            "print(','.join(m for m in banned if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".")
    assert out.returncode == 0, out.stderr
    assert not [m for m in out.stdout.strip().split(",") if m]
