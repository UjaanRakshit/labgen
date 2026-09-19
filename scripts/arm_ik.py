"""Numerical inverse kinematics for the YAM, against catalog keypoints.

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

import numpy as np
import newton

GRIPPER_BODY = -1        # last body in the YAM chain; verified against the
                         # README's documented home pose of (0.1106, 0, 0.1735)


class ArmFK:
    """Forward kinematics by asking the simulator, not by re-deriving it.

    Re-implementing the DH chain here would give a second source of truth for
    the arm's geometry that could drift from the one physics uses. Asking
    Newton is slower and always agrees with what will actually be simulated.
    """

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


def jacobian(fk: ArmFK, q: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """3xN position Jacobian by central differences.

    `eps` is 1e-4 rad, not the 1e-8 a generic optimiser would pick. Newton's
    `joint_q` is float32, so a 1e-8 perturbation is below the representable
    difference and the resulting "gradient" is pure rounding noise. That is why
    an off-the-shelf L-BFGS-B run converged to 283 mm on a target that random
    sampling reached within 43 mm -- it was descending on noise.
    """
    n = len(q)
    J = np.zeros((3, n))
    for i in range(n):
        dq = np.zeros(n)
        dq[i] = eps
        J[:, i] = (fk.position(q + dq) - fk.position(q - dq)) / (2.0 * eps)
    return J


def solve_ik(fk: ArmFK, target: np.ndarray, lower: np.ndarray, upper: np.ndarray,
             seed: np.ndarray | None = None, restarts: int = 8,
             tol_m: float = 0.005, iters: int = 120,
             rng: np.random.Generator | None = None):
    """Damped least squares IK for position, inside the joint limits.

    Levenberg-Marquardt on the position residual: dq = J^T (J J^T + lambda I)^-1 e,
    clamped to the limits each step. Damping keeps it stable near the singular
    configurations a 6-dof arm hits constantly, where a plain pseudo-inverse
    produces enormous joint steps.

    Position only. Constraining orientation as well shrinks the reachable set
    sharply for this arm -- joint2 and joint3 cannot go negative -- and a real
    grasp orientation belongs with a real grasp, which is later work.

    Returns (q, residual_metres, converged).
    """
    rng = rng or np.random.default_rng(0)
    target = np.asarray(target, dtype=float)
    best_q, best_err = None, np.inf

    seeds = [seed if seed is not None else 0.5 * (lower + upper)]
    for _ in range(restarts - 1):
        seeds.append(rng.uniform(lower, upper))

    for s0 in seeds:
        q = np.clip(np.asarray(s0, dtype=float), lower, upper)
        lam = 0.05
        err = np.linalg.norm(fk.position(q) - target)

        for _ in range(iters):
            e = target - fk.position(q)
            if np.linalg.norm(e) < tol_m:
                break
            J = jacobian(fk, q)
            JJt = J @ J.T + (lam ** 2) * np.eye(3)
            step = J.T @ np.linalg.solve(JJt, e)

            # Cap the per-iteration joint motion. Without it a near-singular
            # step can throw the arm across its whole range and the search
            # never settles.
            norm = np.linalg.norm(step)
            if norm > 0.3:
                step *= 0.3 / norm

            q_new = np.clip(q + step, lower, upper)
            err_new = np.linalg.norm(fk.position(q_new) - target)
            if err_new < err:
                q, err = q_new, err_new
                lam = max(lam * 0.7, 1e-3)      # trusting: take bigger steps
            else:
                lam = min(lam * 2.0, 10.0)      # cautious: shorten and retry
                if lam >= 10.0:
                    break

        if err < best_err:
            best_q, best_err = q.copy(), err
        if best_err < tol_m:
            break

    return best_q, float(best_err), bool(best_err < tol_m)


def lerp_path(waypoints: list[np.ndarray], durations: list[float], fps: int):
    """Joint-space interpolation with a smoothstep ease, one entry per frame.

    Joint-space rather than Cartesian: every sample is then guaranteed to sit
    inside the joint limits, which a straight line in task space is not.
    """
    out: list[np.ndarray] = []
    for a, b, secs in zip(waypoints, waypoints[1:], durations):
        n = max(int(secs * fps), 1)
        for i in range(n):
            u = (i + 1) / n
            s = u * u * (3.0 - 2.0 * u)      # smoothstep: zero velocity at ends
            out.append((1.0 - s) * a + s * b)
    return out
