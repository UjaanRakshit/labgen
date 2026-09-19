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

# ---------------------------------------------------------------------------
# ABOVE SPEC. The YAM's actuatorfrcrange is +/-10 N.m and this is 40.
#
# Measured (scripts/probe_torque.py), holding the beaker's body_grasp pose at
# a 0.40 m reach with the wrist pointing down:
#
#     10 N.m  ->  joint2 sags 48.7 deg, fingertip 317 mm from target
#     20 N.m  ->  joint2 holds to  1.9 deg, fingertip  17.6 mm
#     40 N.m  ->  identical to 20, so ~20 N.m is where it saturates
#
# The shoulder needs roughly twice the robot's rated joint torque just to hold
# station at this reach. Two readings, and they matter differently:
#
#   * the URDF's inertial data overstates the arm, or
#   * a YAM genuinely cannot hold a 0.40 m top-down pose on joint torque alone.
#
# The second is entirely plausible for a lightweight teleop arm -- those are
# often held up by the operator through the leader arm, not by the follower's
# own actuators. If it is true, then a task laid out at this reach is one the
# hardware cannot do, and any success rate measured for it in simulation would
# be measuring something the real robot never could. That is exactly the
# sim/real gap MATTERIX names, arriving through the actuator spec instead of
# through geometry.
#
# Running over spec is therefore a DEMO decision, recorded here so it cannot be
# mistaken for a physical claim. Resolve it by checking the YAM's real
# gravity-compensation behaviour before any of this feeds a trust number.
# ---------------------------------------------------------------------------
OVER_SPEC_EFFORT_NM = 40.0

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
                     effort: float = OVER_SPEC_EFFORT_NM,
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
