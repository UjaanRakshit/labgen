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

import numpy as np

import newton

# Finger joints are the last two dofs of the URDF chain.
N_ARM = 6
N_FINGER = 2

# Finger calibration, measured from the JAW geometry (scripts/probe_gap2.py):
#
#     q =  0.000  ->   0.06 mm   jaws closed
#     q = -0.047  ->  94.06 mm   jaws fully open
#
# and linear to within the sampling resolution across the whole stroke.
#
# The previous model had this exactly backwards -- 89 mm at q=0 and 5 mm at
# q=-0.047 -- because it was read off the separation of the tip BODY ORIGINS.
# Those origins DO move together as q goes to -0.047, but each finger is a
# wedge that extends inward from its own origin, so the jaws do the opposite of
# what the origins do. Commanding "open" therefore shut the hand, and
# commanding a 70 mm grip drove the pads to roughly 12 mm and straight through
# the glass. Nothing about the numbers looked wrong; the render did.
#
# A second trap on the way: the minimum distance between the two finger meshes
# taken as a whole plateaus at 19 mm regardless of jaw position, because the
# brackets near the mount sit a fixed distance apart. Measuring that instead of
# the jaws says the gripper can only open 19 mm and that a beaker can never be
# picked up. Only the distal half of the finger is the jaw.
FINGER_CLOSED = 0.0
FINGER_OPEN = -0.047
GAP_CLOSED = 0.00006
GAP_OPEN = 0.09406

MAX_GRASP_WIDTH = GAP_OPEN


class GraspTooWide(ValueError):
    """The object is wider than the gripper can open."""


def finger_q_for_gap(gap_m: float) -> float:
    """Joint coordinate that opens the jaws to `gap_m`.

    Refuses rather than clamping. Silently clamping a too-wide request to the
    fully open position produces a hand that closes on nothing and a task that
    reports success while the object never moves -- which is the failure this
    whole exercise has been chasing.
    """
    if gap_m > MAX_GRASP_WIDTH:
        raise GraspTooWide(
            f"asked for a {gap_m * 1000:.1f} mm opening but the YAM's jaws max "
            f"out at {MAX_GRASP_WIDTH * 1000:.1f} mm. This object cannot be "
            f"grasped by this gripper -- it is a task feasibility question, not "
            f"something to clamp away."
        )
    frac = (gap_m - GAP_CLOSED) / (GAP_OPEN - GAP_CLOSED)
    return float(np.clip(FINGER_CLOSED + frac * (FINGER_OPEN - FINGER_CLOSED),
                         FINGER_OPEN, FINGER_CLOSED))


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


def forward(fk: GraspFK, q: np.ndarray, with_axis: bool, w_axis: float) -> np.ndarray:
    """The quantity being driven to a goal: grasp point, optionally + axis.

    Deliberately the FORWARD map, not the error. The Jacobian below is
    d(forward)/dq, and the LM step is J_pinv @ (goal - forward). Differentiating
    the error instead flips the sign of every step: each one then increases the
    cost, the line search rejects all of them, and the solver silently returns
    its seed. That produced identical 157 mm residuals across four different
    weightings -- the giveaway that nothing was moving at all.
    """
    point, axis = fk.pose(q)
    return np.concatenate([point, w_axis * axis]) if with_axis else point


def goal_vector(target: np.ndarray, approach: np.ndarray | None,
                w_axis: float) -> np.ndarray:
    return (np.concatenate([target, w_axis * approach])
            if approach is not None else np.asarray(target, float))


def solve_grasp_ik(fk: GraspFK, target: np.ndarray, lower: np.ndarray,
                   upper: np.ndarray, approach: np.ndarray | None = None,
                   seed: np.ndarray | None = None, w_axis: float = 0.05,
                   restarts: int = 10, tol_m: float = 0.004,
                   tol_deg: float = 12.0, iters: int = 160,
                   rng: np.random.Generator | None = None):
    """Damped least squares over position and approach direction.

    `w_axis` converts a unit-vector error into metres so both terms live in one
    residual. 0.05 means "a fully wrong approach direction costs about as much
    as being 10 cm away" -- position dominates, which is what you want: a grasp
    that is 5 mm off and 10 degrees rotated is recoverable, one that is 50 mm
    off is not.

    Central differences at 1e-4 rad. Newton's joint_q is float32, so a
    generic optimiser's 1e-8 step is below the representable difference and
    descends on rounding noise -- that cost this project a day.

    Returns (q, position_error_m, axis_error_deg, converged).
    """
    rng = rng or np.random.default_rng(0)
    target = np.asarray(target, float)
    if approach is not None:
        approach = np.asarray(approach, float)
        approach = approach / np.linalg.norm(approach)

    n = N_ARM
    lo, hi = lower[:n], upper[:n]
    eps = 1e-4

    def scores(q):
        point, axis = fk.pose(q)
        pos = float(np.linalg.norm(target - point))
        ang = float(np.degrees(np.arccos(np.clip(axis @ approach, -1, 1)))) \
            if approach is not None else 0.0
        return pos, ang

    best = (None, np.inf, 180.0)
    seeds = [seed[:n] if seed is not None else 0.5 * (lo + hi)]
    for _ in range(restarts - 1):
        seeds.append(rng.uniform(lo, hi))

    for s0 in seeds:
        use_axis = approach is not None
        goal = goal_vector(target, approach, w_axis)
        q = np.clip(np.asarray(s0, float)[:n], lo, hi)
        lam = 0.05
        r = goal - forward(fk, q, use_axis, w_axis)
        cost = float(r @ r)

        for _ in range(iters):
            rows = len(r)
            J = np.zeros((rows, n))
            for i in range(n):
                dq = np.zeros(n)
                dq[i] = eps
                J[:, i] = (forward(fk, q + dq, use_axis, w_axis)
                           - forward(fk, q - dq, use_axis, w_axis)) / (2 * eps)

            JJt = J @ J.T + (lam ** 2) * np.eye(rows)
            step = J.T @ np.linalg.solve(JJt, r)
            nrm = np.linalg.norm(step)
            if nrm > 0.3:
                step *= 0.3 / nrm

            q_new = np.clip(q + step, lo, hi)
            r_new = goal - forward(fk, q_new, use_axis, w_axis)
            cost_new = float(r_new @ r_new)
            if cost_new < cost:
                q, r, cost = q_new, r_new, cost_new
                lam = max(lam * 0.7, 1e-3)
            else:
                lam = min(lam * 2.0, 10.0)
                if lam >= 10.0:
                    break

        pos, ang = scores(q)
        if pos < best[1]:
            best = (q.copy(), pos, ang)
        if pos < tol_m and ang < tol_deg:
            break

    q, pos, ang = best
    return q, pos, ang, bool(pos < tol_m and ang < tol_deg)


def with_fingers(q_arm: np.ndarray, finger: float) -> np.ndarray:
    """Full 8-dof configuration: 6 arm joints plus both fingers together."""
    out = np.zeros(N_ARM + N_FINGER)
    out[:N_ARM] = q_arm[:N_ARM]
    out[N_ARM:] = finger
    return out
