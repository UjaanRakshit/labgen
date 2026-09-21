"""Configure the YAM's joint drives explicitly, after import.

Neither source file gives Newton a usable actuator:

* `yam.urdf` declares `effort="1"` on every joint, which Newton honours over
  anything passed in `default_joint_cfg`. 1 N.m cannot hold the arm up.
* `yam.xml` carries the correct `actuatorfrcrange` of +/-10 N.m but has no
  `<actuator>` block, and `add_mjcf` ignores `default_joint_cfg` entirely --
  sweeping `target_ke` over 100/400/1000 produced bit-identical sag.

So the drive is written onto the builder arrays after import, where it cannot
be overridden by either parser. This is a shared helper rather than something
inlined in the demo, because the numbers it writes are a claim about the robot
and belong somewhere a reader can find and challenge them.

`newton.JointDofConfig.actuator_mode` defaults to `None`, meaning no drive at
all -- targets and gains are inert. That single unset field is why raising
gains appeared to do nothing.
"""

from __future__ import annotations

import numpy as np

import newton

# The YAM's own limit, from `actuatorfrcrange="-10 10"` in yam.xml. This is the
# robot's number, not ours, and it is the one figure here that should NOT be
# raised to make a demo behave: an arm that needs more torque than the hardware
# has is an arm doing something the real one cannot.
YAM_EFFORT_LIMIT_NM = 10.0

# NOT tuned, and deliberately not presented as tuned. Enough to hold the home
# pose against gravity and track a slow trajectory. CLAUDE.md says gain tuning
# is out of scope for T4; getting the arm to stand up is in scope, and this is
# the minimum that achieves it. A real gain set comes from the YAM's own
# controller, not from what looks right on screen.
DEFAULT_KE = 3000.0
DEFAULT_KD = 150.0
DEFAULT_ARMATURE = 0.02

# RETRACTED: the arm runs inside its rated torque. It was a gain problem.
#
# This file previously ran at 40 N.m against the YAM's rated 10, on the
# strength of a measurement showing the shoulder sagging 48.7 deg at 10. That
# measurement was real and the conclusion drawn from it was wrong.
#
# Static gravity torque at the beaker-grasp pose, computed from the link masses
# rather than inferred from a controller (scripts/probe_gravtorque.py):
#
#     j1 0.00   j2 7.26   j3 3.92   j4 0.16   j5 0.00   j6 0.00   N.m
#     total arm mass 4.111 kg, reach 0.409 m
#
# 7.26 N.m is INSIDE the rated 10. The arm can hold the pose. Re-measured with
# the grasp point and gains since corrected (scripts/probe_torque2.py):
#
#     effort 10, ke  300  ->  sags 69.86 deg, tip 437.1 mm off
#     effort 10, ke 3000  ->  holds  0.19 deg, tip   1.8 mm
#     effort 40, ke 3000  ->  holds  0.19 deg, tip   1.8 mm   (identical)
#
# A position PD with no gravity feedforward has to produce the whole 7.26 N.m
# through position error alone. At ke=300 that needs 0.024 rad of error before
# the command even reaches the gravity load, and the loop saturates first. At
# ke=3000 the same error commands ten times the torque and it holds -- on 10.
#
# The original fix raised the effort limit and the stiffness in one change and
# credited the effort. Raising a torque ceiling to cure a gain deficit reads,
# to anyone downstream, as "this robot cannot do this task" -- the most
# expensive kind of wrong answer in a project that exists to produce trust
# numbers.
YAM_RATED_EFFORT_NM = 10.0

# The fingers are PRISMATIC, so their gains are N/m, not N.m/rad. Reusing the
# arm's 300 gave 300 N/m -- a 4 mm squeeze on a beaker produced 0.6 N of grip,
# against the ~1 N needed just to hold its weight through friction. The hand
# closed correctly, reported the right joint positions, and dropped everything.
#
# 20000 N/m saturates the effort limit within about half a millimetre, so the
# grip force is set by FINGER_EFFORT_N below and not by how hard the planner
# happens to squeeze.
FINGER_KE = 20000.0
FINGER_KD = 400.0

# Unsourced. The YAM's actuatorfrcrange covers the six arm joints; the repo
# carries no force figure for the linear gripper, so this is a plausible
# small-electric-gripper number and is flagged as a guess. With mu = 0.4 it
# gives 2 * 0.4 * 25 = 20 N of friction, comfortably above the 1 N weight of a
# 250 mL beaker -- the margin is wide enough that the exact value does not
# decide whether a grasp holds, which is the only reason it is tolerable here.
FINGER_EFFORT_N = 25.0


def configure_drives(builder: newton.ModelBuilder, *, dof_slice: slice | None = None,
                     ke: float = DEFAULT_KE, kd: float = DEFAULT_KD,
                     effort: float = YAM_RATED_EFFORT_NM,
                     armature: float = DEFAULT_ARMATURE,
                     n_fingers: int = 0, verbose: bool = True) -> None:
    """Put the given dofs into position control with a usable drive.

    `n_fingers` marks how many dofs at the END of the slice are prismatic
    gripper joints, which get their own gains and force limit. Revolute and
    prismatic gains are not the same quantity -- N.m/rad against N/m -- and
    sharing a number between them is silently wrong in whichever direction the
    scale happens to fall.
    """
    n = builder.joint_dof_count
    sl = dof_slice if dof_slice is not None else slice(0, n)
    start, stop, _ = sl.indices(n)
    finger_start = stop - n_fingers

    def put(name: str, value, lo: int, hi: int) -> bool:
        arr = getattr(builder, name, None)
        if arr is None:
            return False
        for i in range(lo, hi):
            arr[i] = value
        return True

    ok = True
    for lo, hi, k, d, e, tag in (
        (start, finger_start, ke, kd, effort, "arm"),
        (finger_start, stop, FINGER_KE, FINGER_KD, FINGER_EFFORT_N, "fingers"),
    ):
        if lo >= hi:
            continue
        ok &= put("joint_target_mode", int(newton.JointTargetMode.POSITION), lo, hi)
        ok &= put("joint_target_ke", k, lo, hi)
        ok &= put("joint_target_kd", d, lo, hi)
        ok &= put("joint_effort_limit", e, lo, hi)
        ok &= put("joint_armature", armature, lo, hi)
        if verbose:
            unit = "N.m" if tag == "arm" else "N"
            print(f"   drives[{tag:7}] dofs [{lo}:{hi}] POSITION ke={k:g} kd={d:g} "
                  f"effort={e:g} {unit}")
    if verbose and not ok:
        print("   (some drive arrays are missing on this builder)")


def report(model: newton.Model, label: str = "") -> None:
    """Print what actually landed on the finalized model."""
    def peek(name):
        a = getattr(model, name, None)
        if a is None:
            return None
        try:
            return a.numpy()
        except Exception:
            return np.asarray(a)

    for name in ("joint_target_mode", "joint_target_ke", "joint_target_kd",
                 "joint_effort_limit", "joint_armature"):
        v = peek(name)
        if v is not None and len(v):
            print(f"   {label}{name:20} {np.unique(v)[:6]}")


def gravity_torques(bodies_world, masses, joint_axis, *, offset: int = 0,
                    n_arm: int = 6, g: float = 9.81):
    """Static torque each revolute joint must hold against gravity, per joint.

    Returns SIGNED torques: the moment gravity exerts about each joint axis.
    To hold station an actuator must supply the NEGATIVE of this.

    First principles, no solver: for joint i, sum (r_link - r_joint) x (m g)
    over every link OUTBOARD of it and project onto the joint axis.

    This is the quantity the hardware has to produce. It is NOT the same as what
    a position PD commands -- a PD with no gravity feedforward has to generate
    the whole gravity load out of tracking error, so its command scales with the
    gain and says nothing about the robot. Confusing the two is how this project
    came to report an over-spec arm.

    `bodies_world` is a full-model body_q array; `offset` is where the arm's
    bodies start in it, since the scene's bodies are added first.
    """
    import numpy as np

    masses = np.asarray(masses, float)
    gv = np.array([0.0, 0.0, -g])
    n_links = len(masses)
    taus = np.zeros(n_arm)
    for i in range(n_arm):
        anchor = bodies_world[offset + i, :3]
        q = bodies_world[offset + i, 3:]
        # xyzw -> rotation matrix, inline to keep this module dependency-free
        x, y, z, w = (float(v) for v in q)
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        axis = R @ np.asarray(joint_axis[i][:3], float)
        axis = axis / (np.linalg.norm(axis) or 1.0)
        t = np.zeros(3)
        for j in range(i, n_links):
            t += np.cross(bodies_world[offset + j, :3] - anchor, masses[j] * gv)
        # SIGNED. The caller takes abs() to report a magnitude; a gravity
        # feedforward needs the sign or it doubles the load instead of
        # cancelling it.
        taus[i] = float(t @ axis)
    return taus
