"""The real robot's actuator and controller parameters, per joint, with sources.

The simulator's arm has to respond to a command the way the real arm does, and
that is decided almost entirely by numbers that do not live in the URDF: which
motor sits at each joint, what it can deliver, what gains the controller runs,
and how much friction the joint has. Until this module existed the sim used
numbers chosen here to make the arm hold still -- a position gain of 3000 on
every joint -- which is 37x stiffer than the real shoulder and 300x stiffer
than the real wrist. A sim that stiff tracks beautifully and tells you nothing
about the robot.

Every value below is copied from the i2rt code the REAL rig runs, not from the
newer clone the sim was first built against:

    i2rt 1.1.2, installed in pairlab/yam_teleop/.venv, which is the venv that
    deployment/quest_teleop.py drives the hardware from.
      robots/config/yam_v1.yml      motor list, kp, kd, gravity_comp_factor,
                                    grav_comp_kd, coulomb_friction
      motor_drivers/utils.py        MotorConstants per motor type (torque and
                                    velocity ranges) and _GEAR_RATIO

The ARM MODEL (masses, inertias, joint origins) is not here: it is identical
between i2rt 1.1.2 and 1.3.6 to within 0.002 mm of forward kinematics
(scripts/probe_model_versions.py), so the URDF the sim loads is the right one.

What is deliberately NOT here, because nobody has measured it:

* Rotor inertia, so joint armature. Armature is J_rotor * gear_ratio^2 and i2rt
  records the gear ratios (DM4310 10:1, DM4340 40:1) but not J_rotor.
* Continuous torque ratings. The torque limits below are the MIT-protocol
  encoding ranges, which Damiao sets to peak capability. A motor reaches them
  briefly, not indefinitely.
* The mounted gripper. Both deployment configs say `gripper_type: no_gripper`
  with the comment "set to what is mounted" -- it was never filled in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "MotorSpec",
    "ArmHardware",
    "DM4310",
    "DM4340",
    "YAM_V1",
    "I2RT_REAL_VERSION",
]

# The i2rt release the real hardware runs. Pinned so a later upgrade of the
# deployment venv is a visible change rather than a silent one.
I2RT_REAL_VERSION = "1.1.2"


@dataclass(frozen=True)
class MotorSpec:
    """One Damiao motor type, as the real driver encodes it."""

    name: str
    torque_max_nm: float        # MIT encoding range: PEAK, not continuous
    velocity_max_rad_s: float
    gear_ratio: float
    source: str


# motor_drivers/utils.py, MotorType.get_motor_constants and _GEAR_RATIO.
DM4310 = MotorSpec("DM4310", torque_max_nm=10.0, velocity_max_rad_s=30.0,
                   gear_ratio=10.0,
                   source=f"i2rt {I2RT_REAL_VERSION} motor_drivers/utils.py")
DM4340 = MotorSpec("DM4340", torque_max_nm=28.0, velocity_max_rad_s=10.0,
                   gear_ratio=40.0,
                   source=f"i2rt {I2RT_REAL_VERSION} motor_drivers/utils.py")


@dataclass(frozen=True)
class ArmHardware:
    """Per-joint actuator and controller parameters for one arm variant.

    The real controller is an MIT-mode PD running on each motor plus a
    gravity-compensation feedforward:

        tau = kp * (q_des - q) + kd * (qd_des - qd) + gravity_comp_factor * tau_g

    `gravity_comp_factor` is how much MORE gravity torque the real arm needs
    than its model predicts. It is 1.1-1.2 on joints 2-4. In simulation the
    plant IS the model, so the sim applies 1.0 -- exactly what i2rt's own
    SimRobot does (get_robot.py: sim_grav_comp = np.ones(...)). The hardware
    factor is kept here as evidence: the real arm is 10-20% heavier in effect
    on those joints than the URDF says, and a sim that ignores that will
    under-predict the load on the real shoulder.
    """

    name: str
    motors: tuple[MotorSpec, ...]
    kp: tuple[float, ...]                   # N.m/rad
    kd: tuple[float, ...]                   # N.m.s/rad
    gravity_comp_factor: tuple[float, ...]  # real arm only; sim uses 1.0
    grav_comp_kd: tuple[float, ...]         # damping in gravity-comp idle mode
    coulomb_friction_nm: tuple[float, ...]
    directions: tuple[int, ...]
    source: str
    armature_kg_m2: tuple[float, ...] | None = None   # unknown: no rotor inertia recorded

    def __post_init__(self) -> None:
        n = len(self.motors)
        for name in ("kp", "kd", "gravity_comp_factor", "grav_comp_kd",
                     "coulomb_friction_nm", "directions"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{self.name}: {name} has {len(getattr(self, name))} "
                                 f"entries for {n} motors")
        if any(k <= 0 for k in self.kp) or any(d < 0 for d in self.kd):
            raise ValueError(f"{self.name}: gains must be kp > 0, kd >= 0")
        if not (self.source or "").strip():
            raise ValueError(f"{self.name}: hardware parameters need a source")

    @property
    def n_joints(self) -> int:
        return len(self.motors)

    @property
    def torque_max_nm(self) -> np.ndarray:
        return np.array([m.torque_max_nm for m in self.motors])

    @property
    def velocity_max_rad_s(self) -> np.ndarray:
        return np.array([m.velocity_max_rad_s for m in self.motors])

    def headroom(self, tau_nm) -> np.ndarray:
        """Fraction of each joint's PEAK torque that `tau_nm` uses."""
        return np.abs(np.asarray(tau_nm, float)) / self.torque_max_nm

    @property
    def armature_known(self) -> bool:
        return self.armature_kg_m2 is not None


# robots/config/yam_v1.yml, verbatim.
YAM_V1 = ArmHardware(
    name="yam_v1",
    motors=(DM4340, DM4340, DM4340, DM4310, DM4310, DM4310),
    kp=(80.0, 80.0, 80.0, 10.0, 10.0, 10.0),
    kd=(5.0, 5.0, 5.0, 1.5, 1.5, 1.5),
    gravity_comp_factor=(1.0, 1.1, 1.1, 1.2, 1.0, 1.0),
    grav_comp_kd=(0.1, 0.1, 0.1, 0.3, 0.05, 0.05),
    coulomb_friction_nm=(0.3, 0.3, 0.3, 0.06, 0.06, 0.06),
    directions=(1, 1, 1, 1, 1, 1),
    source=f"i2rt {I2RT_REAL_VERSION} robots/config/yam_v1.yml "
           f"(pairlab/yam_teleop/.venv, the venv that drives the real rig)",
)
