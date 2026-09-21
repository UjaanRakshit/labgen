"""Grasp-aware kinematics for the YAM: fingertip frame, approach axis, fingers.

What the previous demo got wrong, and this fixes:

* It solved IK for the gripper MOUNT frame. The fingertips sit 57 mm further
  along the approach axis, so the hand consistently overshot whatever it was
  "reaching". Here the target is the midpoint between the two fingertip bodies,
  read straight out of forward kinematics rather than hardcoded as an offset.
* It constrained position only, so the hand arrived at the right place in an
  arbitrary orientation -- often sideways through the object. A grasp needs the
  approach axis pointed at the thing.
* It had no fingers at all. The MJCF is arm-only (6 dof); the URDF carries the
  two prismatic finger joints (8 dof), so the URDF is the source here even
  though its effort limits are placeholders -- arm_drive.configure_drives
  overwrites those with the YAM's real +/-10 N.m.

Finger calibration, measured from FK rather than guessed:
    q = 0.000  ->  fingertips 89.0 mm apart  (fully open)
    q = -0.047 ->  fingertips  5.0 mm apart  (fully closed)
and linear in between.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import newton

# labgen is the package; this directory is a bag of scripts that drive it.
_PKG = Path("/mnt/c/Ujaan Docx/Research/labgen")
if _PKG.exists() and str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
from labgen.control import (YAM_JAWS, YAM_N_ARM, YAM_N_FINGER,   # noqa: E402
                            GraspTooWide, solve_pose)

# The finger model, the jaw calibration and the IK solver all live in
# labgen.control now. This module keeps only what genuinely needs Newton: the
# forward kinematics of the grasp frame, read out of the simulator so it cannot
# drift from the geometry physics actually uses.
#
# The names below are re-exported so the ~20 probe scripts that import them
# keep working. New code should import from labgen.control directly.
N_ARM = YAM_N_ARM
N_FINGER = YAM_N_FINGER
MAX_GRASP_WIDTH = YAM_JAWS.max_gap_m
FINGER_CLOSED = YAM_JAWS.q_closed
FINGER_OPEN = YAM_JAWS.q_open
GAP_CLOSED = YAM_JAWS.gap_closed_m
GAP_OPEN = YAM_JAWS.gap_open_m

finger_q_for_gap = YAM_JAWS.q_for_gap


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Newton stores quaternions xyzw."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class GraspFK:
    """FK for the grasp frame: where the fingers actually close, and which way.

    Everything comes from the simulator's own forward kinematics, so it cannot
    drift from the geometry physics uses.
    """

    n_joints = YAM_N_ARM          # satisfies labgen.control.Kinematics

    def __init__(self, model: newton.Model, coord_slice: slice):
        self.model = model
        self.slice = coord_slice
        self.state = model.state()
        self._q = model.joint_q.numpy().copy()

        # Where the fingers actually are, in each tip body's own frame.
        #
        # The obvious grasp point -- the midpoint of the two tip BODY origins --
        # is wrong by 36 mm along the approach axis, because a body origin is a
        # joint location and has nothing to do with where the geometry sits.
        # The YAM's tip STL lives at z -219..-126 mm in its own file frame and
        # the URDF <origin> moves it again. Using the origins pinned every
        # carried object about 5 cm behind the pads, level with the wrist, which
        # is exactly what it looked like on screen.
        #
        # So: take each tip's collision mesh, push it through its shape
        # transform into body-local coordinates once, and keep the centroid.
        # Applying the body transform to that each call is cheap and exact.
        self._tip_local = self._tip_centroids()

    def _tip_centroids(self) -> list[tuple[int, np.ndarray]]:
        model = self.model
        shape_body = model.shape_body.numpy()
        shape_xf = model.shape_transform.numpy()
        out: dict[int, list[np.ndarray]] = {}
        for i in range(model.shape_count):
            bi = int(shape_body[i])
            if bi not in (model.body_count - 1, model.body_count - 2):
                continue
            geo = model.shape_source[i]
            if geo is None or not hasattr(geo, "vertices"):
                continue
            v = np.asarray(geo.vertices, dtype=float)
            xf = shape_xf[i]
            v = v @ quat_to_matrix(xf[3:]).T + xf[:3]
            out.setdefault(bi, []).append(v.mean(axis=0))
        if len(out) != 2:
            # No usable tip geometry (e.g. the arm was imported visual-only and
            # the shapes were dropped). Fall back to the body origins and say so
            # rather than silently grasping 36 mm off.
            print("   GraspFK: no tip geometry found; falling back to body "
                  "origins (grasp point will be ~36 mm behind the pads)")
            return []
        return [(bi, np.mean(c, axis=0)) for bi, c in sorted(out.items())]

    def _eval(self, q_arm: np.ndarray) -> np.ndarray:
        q = self._q.copy()
        full = np.zeros(N_ARM + N_FINGER)
        full[:len(q_arm)] = q_arm
        q[self.slice] = full
        self.model.joint_q.assign(q.astype(np.float32))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        return self.state.body_q.numpy()

    def _grasp_from(self, bodies: np.ndarray) -> np.ndarray:
        if not self._tip_local:
            return 0.5 * (bodies[-1, :3] + bodies[-2, :3])
        pts = [quat_to_matrix(bodies[bi, 3:]) @ c + bodies[bi, :3]
               for bi, c in self._tip_local]
        return 0.5 * (pts[0] + pts[1])

    def grasp_point(self, q_arm: np.ndarray) -> np.ndarray:
        """Where the fingers actually close -- the centroid of the pad geometry."""
        return self._grasp_from(self._eval(q_arm))

    def approach_axis(self, q_arm: np.ndarray) -> np.ndarray:
        """The gripper's +Z in world.

        The YAM README states the tool convention: +X right, +Y down, +Z
        forward, with +Z the approach axis.
        """
        bodies = self._eval(q_arm)
        # The gripper mount is the body just before the two tips.
        R = quat_to_matrix(bodies[-3, 3:])
        return R @ np.array([0.0, 0.0, 1.0])

    def pose(self, q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bodies = self._eval(q_arm)
        axis = quat_to_matrix(bodies[-3, 3:]) @ np.array([0.0, 0.0, 1.0])
        return self._grasp_from(bodies), axis

    def grasp_point_from_state(self, body_q: np.ndarray) -> np.ndarray:
        """Same thing, but from a simulated state rather than an FK evaluation."""
        return self._grasp_from(body_q)


def solve_grasp_ik(fk, target, lower, upper, approach=None, seed=None,
                   w_axis: float = 0.05, restarts: int = 10,
                   tol_m: float = 0.004, tol_deg: float = 12.0,
                   iters: int = 160, rng=None):
    """Back-compatible wrapper over labgen.control.solve_pose.

    Returns the old (q, position_error_m, axis_error_deg, converged) tuple.
    The solver itself, its damping schedule and the reason it differentiates the
    forward map rather than the residual are all in the package now, with tests
    that run against an analytic chain and need no GPU.
    """
    res = solve_pose(fk, target, lower[:N_ARM], upper[:N_ARM],
                     approach=approach, seed=seed, w_axis=w_axis,
                     restarts=restarts, tol_m=tol_m, tol_deg=tol_deg,
                     iters=iters, rng=rng)
    return res.q, res.position_error_m, res.axis_error_deg, res.ok


def with_fingers(q_arm: np.ndarray, finger: float) -> np.ndarray:
    """Full 8-dof configuration: six arm joints plus both fingers together."""
    out = np.zeros(N_ARM + N_FINGER)
    out[:N_ARM] = np.asarray(q_arm, float)[:N_ARM]
    out[N_ARM:] = finger
    return out
