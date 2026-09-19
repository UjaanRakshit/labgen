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
DEFAULT_KE = 300.0
DEFAULT_KD = 30.0
DEFAULT_ARMATURE = 0.02


def configure_drives(builder: newton.ModelBuilder, *, dof_slice: slice | None = None,
                     ke: float = DEFAULT_KE, kd: float = DEFAULT_KD,
                     effort: float = YAM_EFFORT_LIMIT_NM,
                     armature: float = DEFAULT_ARMATURE, verbose: bool = True) -> None:
    """Put the given dofs into position control with a usable drive."""
    n = builder.joint_dof_count
    sl = dof_slice if dof_slice is not None else slice(0, n)

    def put(name: str, value) -> bool:
        arr = getattr(builder, name, None)
        if arr is None:
            return False
        for i in range(*sl.indices(n)):
            arr[i] = value
        return True

    applied = {
        "joint_target_mode": put("joint_target_mode", int(newton.JointTargetMode.POSITION)),
        "joint_target_ke": put("joint_target_ke", ke),
        "joint_target_kd": put("joint_target_kd", kd),
        "joint_effort_limit": put("joint_effort_limit", effort),
        "joint_armature": put("joint_armature", armature),
    }
    if verbose:
        missing = [k for k, ok in applied.items() if not ok]
        print(f"   drives: dofs [{sl.start}:{sl.stop}] mode=POSITION "
              f"ke={ke} kd={kd} effort={effort} N.m")
        if missing:
            print(f"   (builder has no {missing})")


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
