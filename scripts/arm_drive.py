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

# HISTORY OF THIS NUMBER -- three corrections, kept because each was believed.
#
# 1. This file once ran at 40 N.m "against the YAM's rated 10", from a real
#    measurement (the shoulder sagged 48.7 deg at 10) and a wrong conclusion.
#    Retracted: the sag was a missing gravity feedforward, not a torque ceiling.
#    Measured: effort 10/ke 3000 held to 0.19 deg; effort 40 changed nothing.
#
# 2. That retraction called 10 N.m "the robot's rating" and quoted static
#    gravity at the grasp pose as 7.26 N.m, "inside the rated 10". Both wrong.
#    The 7.26 came from gravity_torques putting each body's weight at its frame
#    ORIGIN instead of its centre of mass (link2's is 133 mm away, link3's
#    150 mm). With centre-of-mass lever arms it is 9.71 N.m at the grasp pose
#    and 10.60 N.m at the worst pose of the task plan.
#
# 3. And 10 N.m is not the rating of joints 1-3 at all. i2rt 1.1.2 -- the
#    version the real rig runs -- declares j1-3 DM4340 (28 N.m peak) and j4-6
#    DM4310 (10 N.m). The uniform 10 is i2rt's simplified MJCF. So the task
#    plan is OVER a 10 N.m limit and at 38% of the real shoulder motor.
#
# Per-joint real limits now live in labgen.hardware.YAM_V1, and
# configure_real_drives applies them. This constant is kept only as the
# simplified-MJCF value for the older callers that still use configure_drives;
# it is NOT the robot's rating and should not be quoted as one.
YAM_RATED_EFFORT_NM = 10.0   # i2rt MJCF simplification, NOT the j1-3 motor rating

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


def gravity_torques(bodies_world, masses, joint_axis, *, coms, offset: int = 0,
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

    `coms` is each arm body's centre of mass IN ITS OWN FRAME, and it is
    required. The first version of this function put each body's weight at its
    frame ORIGIN, and on this arm that is badly wrong: link2 (1.47 kg) has its
    centre of mass 133 mm from its origin and link3 (0.98 kg) 150 mm. Every
    torque it returned used the wrong lever arm. It went unnoticed for as long as
    the sim ran kp=3000, because a gain that stiff absorbs a bad feedforward; it
    surfaced the moment the sim ran the real robot's kp=80, as 2.4 deg of sag
    with the feedforward switched on. Required rather than defaulted so that the
    wrong answer is not available by omission.
    """
    import numpy as np

    masses = np.asarray(masses, float)
    coms = np.asarray(coms, float)
    if coms.shape != (len(masses), 3):
        raise ValueError(f"coms must be ({len(masses)}, 3), one body-frame centre "
                         f"of mass per body; got {coms.shape}")
    gv = np.array([0.0, 0.0, -g])

    def rot(qv):
        x, y, z, w = (float(v) for v in qv)
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    com_world = np.array([
        bodies_world[offset + j, :3] + rot(bodies_world[offset + j, 3:]) @ coms[j]
        for j in range(len(masses))])
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
            t += np.cross(com_world[j] - anchor, masses[j] * gv)
        # SIGNED. The caller takes abs() to report a magnitude; a gravity
        # feedforward needs the sign or it doubles the load instead of
        # cancelling it.
        taus[i] = float(t @ axis)
    return taus


def configure_real_drives(builder, hw, *, dof_slice: slice, n_fingers: int = 2,
                          verbose: bool = True, armature_fallback: float = 0.0) -> None:
    """Configure the arm dofs with the REAL robot's per-joint parameters.

    `hw` is a labgen.hardware.ArmHardware -- kp, kd, peak torque, velocity limit
    and Coulomb friction per joint, copied from the i2rt config the real rig
    runs. Replaces configure_drives' single kp/kd/effort for the whole arm,
    which was a stiffness chosen here to make the arm hold still (kp=3000
    everywhere: 37x the real shoulder, 300x the real wrist).

    Must be called AFTER the arm's builder has been merged, for the same reason
    configure_drives must: the slice is clamped to the builder's current dof
    count, and before the merge that clamp produces an empty range and applies
    nothing.

    Armature is not in the real config (no rotor inertia recorded), so it is set
    to `armature_fallback` and reported as such rather than presented as known.
    """
    n = builder.joint_dof_count
    start, stop, _ = dof_slice.indices(n)
    n_arm = stop - start - n_fingers
    if n_arm != hw.n_joints:
        raise ValueError(
            f"dof slice holds {n_arm} arm joints but {hw.name} describes "
            f"{hw.n_joints}. Configure after merging the arm's builder.")

    torque = hw.torque_max_nm
    vmax = hw.velocity_max_rad_s
    mode = int(newton.JointTargetMode.POSITION)
    for j in range(n_arm):
        i = start + j
        builder.joint_target_mode[i] = mode
        builder.joint_target_ke[i] = hw.kp[j]
        builder.joint_target_kd[i] = hw.kd[j]
        builder.joint_effort_limit[i] = torque[j]
        builder.joint_velocity_limit[i] = vmax[j]
        builder.joint_friction[i] = hw.coulomb_friction_nm[j]
        builder.joint_armature[i] = (hw.armature_kg_m2[j] if hw.armature_known
                                     else armature_fallback)

    # Fingers keep their own prismatic gains: the real gripper is not in the arm
    # config and is not even driven by the lab's teleop ("held, not driven").
    for i in range(start + n_arm, stop):
        builder.joint_target_mode[i] = mode
        builder.joint_target_ke[i] = FINGER_KE
        builder.joint_target_kd[i] = FINGER_KD
        builder.joint_effort_limit[i] = FINGER_EFFORT_N
        builder.joint_armature[i] = armature_fallback

    if verbose:
        print(f"   real drives [{hw.name}]: kp {list(hw.kp)}  kd {list(hw.kd)}")
        print(f"      peak torque {list(torque)} N.m   friction "
              f"{list(hw.coulomb_friction_nm)} N.m")
        if not hw.armature_known:
            print(f"      armature UNKNOWN (no rotor inertia recorded) -> "
                  f"{armature_fallback}")
