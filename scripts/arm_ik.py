"""Newton-backed forward kinematics for the YAM, against catalog keypoints.

The SOLVER moved to `labgen.control`; what is left here is the part that
genuinely needs a simulator, plus back-compatible wrappers for the probe
scripts. New code should import from labgen.control.

Why not a joint-space waveform: the first attempt at a demo swept the joints on
a sine, which drove the arm through the worktop and scattered the glassware --
joint2 and joint3 cannot go negative, so "small oscillation about zero" is not
small and is not about zero. A motion that means anything has to be expressed
in the task frame and solved back into joint space, subject to the robot's real
limits.

Why it targets keypoints: `catalog.py` already names the places on an object
that matter (`rim_grasp`, `pour_lip`, `fill_point`). Solving to a keypoint
rather than to a hand-typed offset is what makes a motion survive swapping a
250 mL beaker for a 500 mL one -- the same property the keypoints exist to give
task specs.

This is scripted motion. It is NOT a policy, and nothing here says anything
about whether a policy could do the task.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import newton

_PKG = Path("/mnt/c/Ujaan Docx/Research/labgen")
if _PKG.exists() and str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
from labgen.control import jacobian, lerp_path, solve_pose   # noqa: E402,F401

GRIPPER_BODY = -1        # last body in the YAM chain; verified against the
                         # README's documented home pose of (0.1106, 0, 0.1735)


class ArmFK:
    """Forward kinematics by asking the simulator, not by re-deriving it.

    Re-implementing the DH chain here would give a second source of truth for
    the arm's geometry that could drift from the one physics uses. Asking
    Newton is slower and always agrees with what will actually be simulated.
    """

    n_joints = 6                  # satisfies labgen.control.Kinematics

    def __init__(self, model: newton.Model, dof_slice: slice):
        self.model = model
        self.slice = dof_slice
        self.state = model.state()
        self._q = model.joint_q.numpy().copy()

    def gripper_pose(self, q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = self._q.copy()
        q[self.slice] = q_arm
        self.model.joint_q.assign(q.astype(np.float32))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        pose = self.state.body_q.numpy()[GRIPPER_BODY]
        return pose[:3].copy(), pose[3:].copy()

    def position(self, q_arm: np.ndarray) -> np.ndarray:
        return self.gripper_pose(q_arm)[0]

    def pose(self, q_arm: np.ndarray):
        """labgen.control.Kinematics: (point, approach axis).

        The axis here is the gripper MOUNT's +Z, which is not where the fingers
        close -- grasp.GraspFK is the one that solves to the pad centroids. This
        exists so the same solver can drive both.
        """
        pos, quat = self.gripper_pose(q_arm)
        x, y, z, w = quat
        return pos, np.array([2 * (x * z + y * w), 2 * (y * z - x * w),
                              1 - 2 * (x * x + y * y)])


def solve_ik(fk, target, lower, upper, seed=None, restarts: int = 8,
             tol_m: float = 0.005, iters: int = 120, rng=None):
    """Back-compatible wrapper over labgen.control.solve_pose, position only.

    Returns the old (q, residual_metres, converged) tuple. Position only:
    constraining orientation as well shrinks the reachable set sharply for this
    arm, and a real grasp orientation belongs with a real grasp -- which is
    what grasp.py does.
    """
    res = solve_pose(fk, target, lower, upper, seed=seed, restarts=restarts,
                     tol_m=tol_m, iters=iters, rng=rng)
    return res.q, res.position_error_m, res.ok
