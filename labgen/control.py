"""Scripted control: where the hand should go, and how to get it there.

This is the package home for what used to live in `scripts/arm_ik.py` and
`scripts/grasp.py`. The demos are now thin callers. The move matters because
control is about to be used by two things that are not demos -- teleop
recording and generated demonstrations -- and a controller that only exists
inside a demo script cannot be tested, versioned, or reused by either.

**It is scripted motion, not a policy.** Nothing here observes anything. It
solves a geometry problem against a known target and interpolates. CLAUDE.md
fences phase 1 at exactly that line.

Two structural decisions, both load-bearing:

* **No `newton` import.** Like `labgen.settle`, everything here is pure numpy
  and the kinematics arrive through the `Kinematics` protocol. Importing the
  controller must never require a GPU, or it stops being testable in CI -- and
  a solver that can only be exercised on the one machine with an RTX in it is a
  solver nobody checks. `tests/test_control.py` runs the whole thing against an
  analytic planar chain with closed-form answers.

* **Everything is per-arm.** The real rig is two YAMs. An `ArmSpec` names one
  arm's joints, limits and base placement, and nothing in this module assumes
  there is only one. The second arm's mounting offset is a measured physical
  quantity, so `ArmSpec` refuses to carry one without a source, the same way
  `CatalogItem` refuses an unsourced dimension.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

__all__ = [
    "ArmSpec",
    "JawModel",
    "Kinematics",
    "GraspTooWide",
    "UnsourcedPlacement",
    "require_sourced_placement",
    "IKResult",
    "jacobian",
    "solve_pose",
    "lerp_path",
    "Trajectory",
    "Waypoint",
    "YAM_JAWS",
    "FingerPad",
    "YAM_PAD",
    "YAM_N_ARM",
    "YAM_N_FINGER",
    "yam_arm_spec",
    "IK_EPS",
    "IK_MAX_STEP",
]

# Central-difference step for the Jacobian, in radians.
#
# 1e-4, not the 1e-8 a generic optimiser would pick. Newton's `joint_q` is
# float32, so a 1e-8 perturbation is below the representable difference and the
# resulting "gradient" is rounding noise. An off-the-shelf L-BFGS-B run
# converged to 283 mm on a target that random sampling reached within 43 mm --
# it was descending on noise. Kept here as a named constant so nobody
# "improves" it toward machine epsilon.
IK_EPS = 1e-4

# Cap on per-iteration joint motion, radians. Without it a near-singular step
# throws the arm across its whole range and the search never settles.
IK_MAX_STEP = 0.3


class GraspTooWide(ValueError):
    """The object is wider than the gripper can open.

    Raised rather than clamped. Silently clamping a too-wide request to the
    fully open position produces a hand that closes on nothing and a task that
    reports success while the object never moves.
    """


class UnsourcedPlacement(ValueError):
    """An arm was placed at an offset nobody measured."""


def require_sourced_placement(name: str, position, orientation_wxyz,
                              source: str) -> None:
    """Refuse a non-origin placement that nobody measured.

    Shared by `ArmSpec` (the controller's view of an arm) and
    `isaaclab_cfg.ArmSource` (the emitter's), because it is one rule about the
    real rig and having it in two places is how the two drift apart.

    A single-arm rig defines the frame, so the origin is not a claim and needs
    no source. A second arm's mounting offset is a physical quantity measured
    off the real bench; inventing it produces a scene that looks right and puts
    every bimanual reach in the wrong place.
    """
    at_origin = (tuple(float(v) for v in position) == (0.0, 0.0, 0.0)
                 and tuple(float(v) for v in orientation_wxyz) == (1.0, 0.0, 0.0, 0.0))
    if at_origin:
        return
    s = (source or "").strip()
    if not s or s.upper().startswith("TODO"):
        raise UnsourcedPlacement(
            f"{name} is placed at {tuple(position)} but its placement source is "
            f"{source!r}. A second arm's mounting offset is measured off the "
            f"real rig, not chosen. Supply the measurement, or leave the arm at "
            f"the origin."
        )


class Kinematics(Protocol):
    """Anything that can say where the grasp frame is for a configuration.

    Implemented against the simulator rather than a re-derived DH chain: a
    second source of truth for the arm's geometry can drift from the one
    physics actually uses. The protocol exists so the solver can also be driven
    by an analytic chain in tests, where the right answer is known.
    """

    n_joints: int

    def pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (grasp point in world, approach axis as a unit vector)."""
        ...


@dataclass(frozen=True)
class JawModel:
    """A parallel gripper's jaw opening as a function of joint coordinate.

    Calibrated from the JAW geometry, not from the finger body origins. The
    previous model had this exactly backwards -- each finger is a wedge
    extending inward from its own origin, so the jaws do the opposite of what
    the origins do, and commanding "open" shut the hand.

    A second trap worth recording: the minimum distance between the two finger
    meshes taken as a whole plateaus at 19 mm regardless of jaw position,
    because the brackets near the mount sit a fixed distance apart. Measuring
    that instead of the jaws says the gripper can open 19 mm and that a beaker
    can never be picked up. Only the distal half of the finger is the jaw.
    """

    q_closed: float
    q_open: float
    gap_closed_m: float
    gap_open_m: float
    source: str

    @property
    def max_gap_m(self) -> float:
        return self.gap_open_m

    def q_for_gap(self, gap_m: float) -> float:
        """Joint coordinate that opens the jaws to `gap_m`. Refuses if too wide."""
        if gap_m > self.gap_open_m:
            raise GraspTooWide(
                f"asked for a {gap_m * 1000:.1f} mm opening but these jaws max "
                f"out at {self.gap_open_m * 1000:.1f} mm. This object cannot be "
                f"grasped by this gripper -- that is a task feasibility "
                f"question, not something to clamp away."
            )
        span = self.gap_open_m - self.gap_closed_m
        frac = (gap_m - self.gap_closed_m) / span
        q = self.q_closed + frac * (self.q_open - self.q_closed)
        return float(np.clip(q, min(self.q_open, self.q_closed),
                             max(self.q_open, self.q_closed)))

    def gap_for_q(self, q: float) -> float:
        """The inverse, for reading a measured configuration back as a width."""
        frac = (q - self.q_closed) / (self.q_open - self.q_closed)
        return float(self.gap_closed_m + frac * (self.gap_open_m - self.gap_closed_m))


# Measured from the jaw geometry (scripts/probe_gap2.py), linear across the
# whole stroke to within the sampling resolution.
YAM_JAWS = JawModel(
    q_closed=0.0,
    q_open=-0.047,
    gap_closed_m=0.00006,
    gap_open_m=0.09406,
    source="measured from i2rt YAM URDF finger meshes, scripts/probe_gap2.py",
)


@dataclass(frozen=True)
class FingerPad:
    """The gripping face of one finger, as a box.

    This exists because deriving finger colliders from the visual mesh does not
    work. The fingertip is an L-shaped bracket, so its convex hull spans
    +/-36 mm and bridges the jaw to the mount. Two of those hulls start deeply
    overlapped with any object between them, and the solver resolves that by
    ejecting it -- measured at 9.7 metres of beaker displacement.

    The jaw itself is small. Measured off the URDF's own tip mesh by taking the
    slab within 8 mm of the face that looks at the other finger
    (scripts/probe_padbox.py): about 25.7 mm wide by 19.5 mm long.

    `verified` is False until someone puts calipers on the real pads, and it
    means the same thing here as on a CatalogItem: a number nobody measured on
    real hardware must not be presented as one that was.
    """

    width_m: float           # across the jaw face
    length_m: float          # along the finger
    thickness_m: float       # into the finger, away from the face
    centre_m: tuple[float, float, float]
    source: str
    verified: bool = False

    def __post_init__(self) -> None:
        for name in ("width_m", "length_m", "thickness_m"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not (self.source or "").strip():
            raise ValueError("a finger pad needs a source: where did these "
                             "dimensions come from?")
        if self.verified and self.source.strip().upper().startswith("TODO"):
            raise ValueError("verified=True needs a real source, not a TODO")

    @property
    def half_extents_m(self) -> tuple[float, float, float]:
        return (self.width_m / 2.0, self.thickness_m / 2.0, self.length_m / 2.0)

    def centre_for(self, side: int) -> tuple[float, float, float]:
        """Pad centre for one finger. `side` is +1 or -1; the pair is mirrored
        across the plane the jaws close on."""
        x, y, z = self.centre_m
        return (side * x, side * y, z)


# Derived from the mesh, NOT from calipers. Ujaan is measuring the real pads;
# until those numbers arrive this is flagged unverified everywhere it is used.
YAM_PAD = FingerPad(
    width_m=0.0257,
    length_m=0.0195,
    thickness_m=0.0079,
    centre_m=(0.0029, -0.0007, -0.0061),
    source="derived from the i2rt yam.urdf fingertip mesh, "
           "scripts/probe_padbox.py -- NOT caliper-verified",
    verified=False,
)


@dataclass
class ArmSpec:
    """One arm: its joint layout, its limits, and where its base sits.

    `base_position_m` and `base_orientation_wxyz` are in the SCENE's frame. For
    a single-arm rig the arm defines the frame and sits at the origin, which
    needs no measurement and is the default. A second arm does not: its offset
    from the first is a physical quantity on the real rig, and inventing it
    produces a scene that looks right and puts every bimanual reach in the
    wrong place.

    So a non-origin placement without a source raises. Same rule as
    `CatalogItem`: a confidently wrong number is worse than a missing one,
    because it produces output that looks correct.
    """

    name: str
    n_arm: int
    n_finger: int
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    jaws: JawModel
    base_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    placement_source: str = ""

    def __post_init__(self) -> None:
        self.joint_lower = np.asarray(self.joint_lower, dtype=float)
        self.joint_upper = np.asarray(self.joint_upper, dtype=float)
        if self.joint_lower.shape != self.joint_upper.shape:
            raise ValueError(
                f"{self.name}: {self.joint_lower.size} lower limits against "
                f"{self.joint_upper.size} upper limits")
        if self.joint_lower.size < self.n_arm:
            raise ValueError(
                f"{self.name}: {self.joint_lower.size} joint limits for an arm "
                f"with {self.n_arm} joints")
        if np.any(self.joint_lower > self.joint_upper):
            bad = int(np.argmax(self.joint_lower > self.joint_upper))
            raise ValueError(
                f"{self.name}: joint {bad + 1} has lower limit "
                f"{self.joint_lower[bad]} above upper {self.joint_upper[bad]}")
        require_sourced_placement(self.name, self.base_position_m,
                                  self.base_orientation_wxyz,
                                  self.placement_source)

    @property
    def at_origin(self) -> bool:
        return (tuple(self.base_position_m) == (0.0, 0.0, 0.0)
                and tuple(self.base_orientation_wxyz) == (1.0, 0.0, 0.0, 0.0))

    @property
    def n_dof(self) -> int:
        return self.n_arm + self.n_finger

    @property
    def arm_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self.joint_lower[:self.n_arm], self.joint_upper[:self.n_arm]

    def with_fingers(self, q_arm: np.ndarray, finger: float) -> np.ndarray:
        """Full configuration: the arm joints plus both fingers together."""
        out = np.zeros(self.n_dof)
        out[:self.n_arm] = np.asarray(q_arm, float)[:self.n_arm]
        out[self.n_arm:] = finger
        return out

    def clamp(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, float)
        n = min(q.size, self.joint_lower.size)
        out = q.copy()
        out[:n] = np.clip(q[:n], self.joint_lower[:n], self.joint_upper[:n])
        return out


# The YAM's own joint layout, from the URDF: six revolute joints then two
# prismatic finger joints. The arm-only MJCF has no fingers, which is why the
# URDF is the source for anything that grasps.
YAM_N_ARM = 6
YAM_N_FINGER = 2
YAM_LAYOUT_SOURCE = "i2rt yam.urdf, joints 1-6 revolute + joints 7-8 prismatic"


def yam_arm_spec(joint_lower, joint_upper, *, name: str = "yam",
                 base_position_m=(0.0, 0.0, 0.0),
                 base_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
                 placement_source: str = "") -> ArmSpec:
    """An ArmSpec for a YAM, with the limits read from the loaded URDF.

    The limits are NOT hardcoded here on purpose: they belong to whichever
    revision of the URDF is actually loaded, and a second copy of them in this
    file is a second thing to drift.
    """
    return ArmSpec(
        name=name, n_arm=YAM_N_ARM, n_finger=YAM_N_FINGER,
        joint_lower=joint_lower, joint_upper=joint_upper, jaws=YAM_JAWS,
        base_position_m=tuple(base_position_m),
        base_orientation_wxyz=tuple(base_orientation_wxyz),
        placement_source=placement_source,
    )


@dataclass
class IKResult:
    """What the solver found, and whether it is usable.

    `ok` is not "the solver finished" -- it is "the result is within both
    tolerances". A caller that treats a non-converged solve as a pose will
    command the arm somewhere it did not ask for.
    """

    q: np.ndarray
    position_error_m: float
    axis_error_deg: float
    ok: bool
    restarts_used: int = 0

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        return (f"IK {'ok' if self.ok else 'FAILED'}: "
                f"{self.position_error_m * 1000:.2f} mm, "
                f"{self.axis_error_deg:.2f} deg, "
                f"{self.restarts_used} restart(s)")


def _forward(kin: Kinematics, q: np.ndarray, use_axis: bool,
             w_axis: float) -> np.ndarray:
    """The quantity being driven to a goal: grasp point, optionally + axis.

    Deliberately the FORWARD map, not the error. The Jacobian below is
    d(forward)/dq and the step is J^T (J J^T + lam^2 I)^-1 @ (goal - forward).
    Differentiating the error instead flips the sign of every step: each one
    then increases the cost, the line search rejects all of them, and the
    solver silently returns its seed. That produced bit-identical 157 mm
    residuals across four different weightings -- the giveaway that nothing was
    moving at all.
    """
    point, axis = kin.pose(q)
    point = np.asarray(point, float)
    if not use_axis:
        return point
    return np.concatenate([point, w_axis * np.asarray(axis, float)])


def jacobian(kin: Kinematics, q: np.ndarray, *, use_axis: bool = False,
             w_axis: float = 0.05, eps: float = IK_EPS) -> np.ndarray:
    """Central-difference Jacobian of the forward map. See IK_EPS."""
    q = np.asarray(q, float)
    cols = q.size
    rows = 6 if use_axis else 3
    J = np.zeros((rows, cols))
    for i in range(cols):
        dq = np.zeros(cols)
        dq[i] = eps
        J[:, i] = (_forward(kin, q + dq, use_axis, w_axis)
                   - _forward(kin, q - dq, use_axis, w_axis)) / (2.0 * eps)
    return J


def solve_pose(kin: Kinematics, target, lower, upper, *,
               approach=None, seed=None, w_axis: float = 0.05,
               restarts: int = 10, tol_m: float = 0.004,
               tol_deg: float = 12.0, iters: int = 160,
               rng: np.random.Generator | None = None) -> IKResult:
    """Damped least squares over grasp position and approach direction.

    Levenberg-Marquardt, clamped to the joint limits every step. Damping keeps
    it stable near the singular configurations a 6-dof arm hits constantly,
    where a plain pseudo-inverse produces enormous joint steps.

    `w_axis` converts a unit-vector error into metres so both terms live in one
    residual. 0.05 means "a fully wrong approach direction costs about as much
    as being 10 cm away" -- position dominates, which is what you want: a grasp
    5 mm off and 10 degrees rotated is recoverable, one 50 mm off is not.

    Multi-start because the residual is not convex and a 6-dof arm has
    configurations the first seed cannot escape. The seeds are drawn from a
    seeded Generator, so a failure is reproducible.
    """
    rng = rng or np.random.default_rng(0)
    target = np.asarray(target, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    n = lower.size

    use_axis = approach is not None
    if use_axis:
        approach = np.asarray(approach, float)
        norm = np.linalg.norm(approach)
        if norm == 0.0:
            raise ValueError("approach axis is the zero vector")
        approach = approach / norm
        goal = np.concatenate([target, w_axis * approach])
    else:
        goal = target

    def scores(q):
        point, axis = kin.pose(q)
        pos = float(np.linalg.norm(target - np.asarray(point, float)))
        if not use_axis:
            return pos, 0.0
        axis = np.asarray(axis, float)
        cos = float(np.clip(axis @ approach, -1.0, 1.0))
        return pos, float(math.degrees(math.acos(cos)))

    seeds = [np.asarray(seed, float)[:n] if seed is not None
             else 0.5 * (lower + upper)]
    for _ in range(max(restarts - 1, 0)):
        seeds.append(rng.uniform(lower, upper))

    best_q, best_pos, best_ang, used = None, np.inf, 180.0, 0

    for attempt, s0 in enumerate(seeds, start=1):
        q = np.clip(np.asarray(s0, float)[:n], lower, upper)
        lam = 0.05
        r = goal - _forward(kin, q, use_axis, w_axis)
        cost = float(r @ r)

        for _ in range(iters):
            J = jacobian(kin, q, use_axis=use_axis, w_axis=w_axis)
            JJt = J @ J.T + (lam ** 2) * np.eye(J.shape[0])
            step = J.T @ np.linalg.solve(JJt, r)
            nrm = np.linalg.norm(step)
            if nrm > IK_MAX_STEP:
                step *= IK_MAX_STEP / nrm

            q_new = np.clip(q + step, lower, upper)
            r_new = goal - _forward(kin, q_new, use_axis, w_axis)
            cost_new = float(r_new @ r_new)
            if cost_new < cost:
                q, r, cost = q_new, r_new, cost_new
                lam = max(lam * 0.7, 1e-3)       # trusting: take bigger steps
            else:
                lam = min(lam * 2.0, 10.0)       # cautious: shorten and retry
                if lam >= 10.0:
                    break

        pos, ang = scores(q)
        if pos < best_pos:
            best_q, best_pos, best_ang, used = q.copy(), pos, ang, attempt
        if pos < tol_m and ang < tol_deg:
            break

    return IKResult(q=best_q, position_error_m=best_pos, axis_error_deg=best_ang,
                    ok=bool(best_pos < tol_m and best_ang < tol_deg),
                    restarts_used=used)


def lerp_path(waypoints: Sequence[np.ndarray], durations: Sequence[float],
              fps: int) -> list[np.ndarray]:
    """Joint-space interpolation with a smoothstep ease, one entry per frame.

    Joint-space rather than Cartesian: every sample is then guaranteed to sit
    inside the joint limits, which a straight line in task space is not.

    Smoothstep gives zero velocity at both ends of every segment. That is not
    cosmetic -- a step change in commanded velocity is a spike in commanded
    torque, and on an arm running at its real effort limit a spike is a
    saturation event.
    """
    if len(durations) != max(len(waypoints) - 1, 0):
        raise ValueError(
            f"{len(waypoints)} waypoints need {max(len(waypoints) - 1, 0)} "
            f"durations, got {len(durations)}")
    out: list[np.ndarray] = []
    for a, b, secs in zip(waypoints, waypoints[1:], durations):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        n = max(int(secs * fps), 1)
        for i in range(n):
            u = (i + 1) / n
            s = u * u * (3.0 - 2.0 * u)
            out.append((1.0 - s) * a + s * b)
    return out


@dataclass
class Waypoint:
    """One commanded configuration, with the label that explains why."""

    cfg: np.ndarray
    label: str = ""
    arm: str = ""


@dataclass
class Trajectory:
    """A sequence of commanded configurations for one arm, built waypoint by
    waypoint and materialised at a fixed frame rate.

    Holds the arm's name so a bimanual plan can be assembled from two of these
    without either one having to know about the other.
    """

    arm: str
    fps: int = 60
    steps: list[Waypoint] = field(default_factory=list)
    ok: bool = True

    @property
    def q(self) -> np.ndarray:
        """The configuration currently at the end of the trajectory."""
        if not self.steps:
            raise ValueError(f"{self.arm}: trajectory is empty, there is no "
                             f"current configuration to build from")
        return self.steps[-1].cfg

    def start(self, cfg: np.ndarray, label: str = "start") -> None:
        if self.steps:
            raise ValueError(f"{self.arm}: already started")
        self.steps.append(Waypoint(np.asarray(cfg, float), label, self.arm))

    def move_to(self, cfg: np.ndarray, secs: float, label: str = "") -> None:
        """Interpolate from the current configuration to `cfg` over `secs`."""
        frames = lerp_path([self.q, np.asarray(cfg, float)], [secs], self.fps)
        self.steps.extend(Waypoint(f, label, self.arm) for f in frames)

    def hold(self, secs: float, label: str = "hold") -> None:
        n = max(int(secs * self.fps), 1)
        self.steps.extend(Waypoint(self.q.copy(), label, self.arm)
                          for _ in range(n))

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def seconds(self) -> float:
        return len(self.steps) / self.fps


def pad_to_longest(trajectories: Sequence[Trajectory]) -> list[Trajectory]:
    """Hold each arm at its final pose until the longest one finishes.

    Bimanual plans are assembled per arm and then run on one clock. Without
    this the shorter arm's command array simply ends, and whatever reads it
    next gets an index error or, worse, a stale row.
    """
    if not trajectories:
        return []
    longest = max(len(t) for t in trajectories)
    for t in trajectories:
        if not t.steps:
            raise ValueError(f"{t.arm}: cannot pad an empty trajectory")
        while len(t) < longest:
            t.steps.append(Waypoint(t.q.copy(), "wait", t.arm))
    return list(trajectories)
