"""The settle test: step the scene with no actuation and see if it stays put.

CLAUDE.md calls this the cheapest high-value check in the project, and it is --
it catches interpenetration, bad mass, a collider that seals a vessel, and a
scene that was quietly assembled 8% wrong, all at once and without anyone
having to predict which of those went wrong.

It is also the only check here that needs a physics engine, so it is kept out
of `labgen.validate`: importing the validators must never drag in Newton, or
the geometric gates stop running in CI. The backend is resolved lazily and by
name, and a missing one is reported rather than silently skipped -- a settle
test that quietly does nothing is worse than no settle test, because the report
still says the scene is fine.

    from labgen.settle import settle, NewtonBackend
    result = settle("out/bench_5.usda", backend=NewtonBackend())
    print(result)
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

__all__ = [
    "SettleResult",
    "BodyMotion",
    "Backend",
    "NewtonBackend",
    "BackendUnavailable",
    "settle",
    "SETTLE_SECONDS",
    "SETTLE_TOLERANCE_M",
    "SETTLE_DT",
    "CONTACT_KE",
    "CONTACT_KD",
]

# CLAUDE.md: step 2 seconds with no actuation; nothing moves more than 2 mm.
SETTLE_SECONDS = 2.0
SETTLE_TOLERANCE_M = 0.002

# Contact stiffness and timestep, measured together (scripts/probe_ke_dt.py).
#
# These are one setting, not two. A penetration-based contact is stable only
# while the stiffness is small relative to what the timestep can integrate, so
# sweeping either alone finds an optimum that is an artifact of the other's
# fixed value. Both earlier one-dimensional findings were exactly that:
#
#     ke = 2500 (the import default), sweeping dt:
#         1/120 -> 4.88 mm   1/240 -> 8.76   1/960 -> 10.57
#         ...from which "1/240 is best, smaller is worse" -- true, and only
#         because the stiffness was far too low to resolve the contact at all.
#
#     dt = 1/240, sweeping ke:
#         2500 -> 8.76 mm    10000 -> 1.29    50000 -> 17.47
#         ...from which "stiffness helps then hurts" -- the rise at the end is
#         the timestep failing to integrate a stiffer contact, not a limit.
#
# Moved together the surface is monotone and the answer is unambiguous:
#
#     ke = 160000, dt = 1/960  ->  petri 0.09 mm
#                                  beaker -0.01, test tube 0.00,
#                                  erlenmeyer -0.01, cylinder -0.00
#
# WHY these values -- corrected. An earlier version of this comment justified
# ke=160000 as "glass on a steel benchtop is far stiffer than 2500 N/m". That
# was wrong, and it was a justification written after the numbers went green.
# ke is not a stiffness in N/m on this solver. Newton hands both values to
# MuJoCo through convert_solref(ke, kd, 1, 1):
#
#     timeconst = 2 / kd                 dampratio = kd / (2 * sqrt(ke))
#
# so the two constants below are really a constraint time constant of 2.5 ms
# and a damping ratio of exactly 1.0. The importer default (ke=2500, kd=100)
# was ALSO critically damped -- 100 = 2*sqrt(2500) -- so the thing that
# actually changed was the time constant, 20 ms -> 2.5 ms.
#
# Measured across timeconst at fixed critical damping and dt=1/1920
# (scripts/probe_solref.py):
#
#     timeconst   petri sink   beaker sink    sink / (g * tc^2)
#       20 ms      10.57 mm      1.565 mm      petri 2.69  beaker 0.40
#       10 ms       2.57         0.403               2.62         0.41
#        5 ms       0.61         0.086               2.49         0.35
#      2.5 ms       0.14         0.021               2.32         0.35
#
# Sink scales as timeconst squared -- the ratio column is roughly flat per
# vessel -- so the mechanism is the constraint time constant, and an 8x
# shorter one buys ~70x less sink. What did NOT hold is the tidy law
# delta = g * tc^2: the prefactor differs ~7x between vessels, set by contact
# geometry, so a new object's sink cannot be predicted from first principles
# to better than an order of magnitude. The gate stays an empirical
# measurement, not a derivation.
#
# Two constraints bound the choice. MuJoCo resolves a constraint only when
# timeconst >= 2*dt: at 2.5 ms, dt=1/240 (0.6x) sinks 0.57 mm and 1/480 (1.2x)
# 0.21 mm, against 0.09 mm at 1/960 (2.4x). And damping ratio is not neutral --
# at fixed 2.5 ms, dampratio 2.0 / 1.0 / 0.5 sinks 0.54 / 0.14 / 0.03 mm -- so
# ke does matter, through dampratio rather than as a stiffness. Critical is
# kept because underdamped contact rings.
#
# For sim-to-real: real glass on a real bench has essentially zero resting
# penetration, so every millimetre here is numerical. Shorter timeconst means
# less of it, at the cost of a smaller dt.
#
# Verified not to have bought a green gate with a broken vessel
# (scripts/probe_petri_cavity.py): a tube dropped into the petri dish now rests
# ON its floor at 7.89 mm rather than passing through it to 0.02 mm. The fix
# made the cavity MORE correct, not less. That check is not optional here --
# convex_decomposition also takes this gate to 0.01 mm, by sealing the dish.
SETTLE_DT = 1.0 / 960.0
CONTACT_KE = 160_000.0
CONTACT_KD = 800.0          # timeconst 2/kd = 2.5 ms; kd = 2*sqrt(ke) -> dampratio 1.0


class BackendUnavailable(RuntimeError):
    """No physics backend could be loaded. Reported, never swallowed."""


@dataclass
class BodyMotion:
    name: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    displacement_m: float
    peak_m: float

    @property
    def within(self) -> bool:
        return self.displacement_m <= SETTLE_TOLERANCE_M


@dataclass
class SettleResult:
    scene: str
    seconds: float
    tolerance_m: float
    bodies: list[BodyMotion] = field(default_factory=list)
    diverged_at: int | None = None
    backend: str = "?"

    @property
    def ok(self) -> bool:
        return self.diverged_at is None and all(b.within for b in self.bodies)

    @property
    def worst(self) -> BodyMotion | None:
        return max(self.bodies, key=lambda b: b.displacement_m, default=None)

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        head = (f"settle test: {self.scene}  ({self.seconds:.1f} s unactuated, "
                f"{self.backend}, dt={SETTLE_DT * 1000:.2f} ms)")
        if self.diverged_at is not None:
            return f"{head}\n  DIVERGED at step {self.diverged_at} -- the scene exploded"
        rows = [f"  {'body':14} {'dz (mm)':>9} {'moved (mm)':>11} {'peak (mm)':>10}  verdict"]
        for b in sorted(self.bodies, key=lambda x: -x.displacement_m):
            rows.append(f"  {b.name:14} {(b.end[2] - b.start[2]) * 1000:9.3f} "
                        f"{b.displacement_m * 1000:11.3f} {b.peak_m * 1000:10.3f}  "
                        f"{'ok' if b.within else 'MOVED'}")
        worst = self.worst
        rows.append(f"  worst {worst.displacement_m * 1000:.3f} mm against a "
                    f"{self.tolerance_m * 1000:.1f} mm tolerance -- "
                    f"{'PASS' if self.ok else 'FAIL'}" if worst else "  no free bodies")
        return "\n".join([head, *rows])


class Backend(Protocol):
    """Anything that can step a `.usda` and report where the bodies ended up."""

    name: str

    def run(self, usda: Path, seconds: float) -> tuple[list[BodyMotion], int | None]:
        ...


class NewtonBackend:
    """Newton / MuJoCo-Warp, as shipped with Isaac Lab 3.0.

    Imported inside `run` on purpose. Constructing a NewtonBackend on a machine
    without Newton is fine and fails only when you actually try to simulate,
    which keeps `labgen.settle` importable everywhere.
    """

    name = "newton"

    def __init__(self, iterations: int = 100, ls_iterations: int = 50):
        self.iterations = iterations
        self.ls_iterations = ls_iterations

    def run(self, usda: Path, seconds: float):
        try:
            newton = importlib.import_module("newton")
            np = importlib.import_module("numpy")
        except ImportError as exc:
            raise BackendUnavailable(
                f"newton is not importable here ({exc}). The settle test needs "
                f"Isaac Lab's environment; the geometric gates in "
                f"labgen.validate do not."
            ) from exc

        builder = newton.ModelBuilder()
        info = builder.add_usd(str(usda))

        # Contact stiffness is applied here rather than authored into the USD:
        # it is a property of how this solver integrates, not of the objects.
        # Baking it into the scene would make the asset carry one backend's
        # tuning around with it.
        for i in range(builder.shape_count):
            builder.shape_material_ke[i] = CONTACT_KE
            builder.shape_material_kd[i] = CONTACT_KD

        model = builder.finalize()
        paths = {v: k for k, v in info["path_body_map"].items()}

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        solver = newton.solvers.SolverMuJoCo(
            model, iterations=self.iterations, ls_iterations=self.ls_iterations)

        start = state_0.body_q.numpy().copy()
        peak = np.zeros(model.body_count)
        diverged = None

        for step in range(int(seconds / SETTLE_DT)):
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, control, contacts, SETTLE_DT)
            state_0, state_1 = state_1, state_0
            q = state_0.body_q.numpy()
            if not np.isfinite(q).all():
                diverged = step
                break
            peak = np.maximum(peak, np.linalg.norm(q[:, :3] - start[:, :3], axis=1))

        end = state_0.body_q.numpy()
        motions = []
        for i in range(model.body_count):
            name = paths.get(i, f"body{i}").replace("/World/", "")
            motions.append(BodyMotion(
                name=name,
                start=tuple(float(v) for v in start[i, :3]),
                end=tuple(float(v) for v in end[i, :3]),
                displacement_m=float(np.linalg.norm(end[i, :3] - start[i, :3])),
                peak_m=float(peak[i]),
            ))
        return motions, diverged


def settle(usda: str | Path, *, backend: Backend | None = None,
           seconds: float = SETTLE_SECONDS,
           tolerance_m: float = SETTLE_TOLERANCE_M) -> SettleResult:
    """Step `usda` unactuated and report what moved.

    Raises BackendUnavailable rather than returning a vacuous pass when no
    engine is present.
    """
    usda = Path(usda)
    if not usda.exists():
        raise FileNotFoundError(usda)
    backend = backend or NewtonBackend()

    motions, diverged = backend.run(usda, seconds)
    return SettleResult(scene=usda.stem, seconds=seconds, tolerance_m=tolerance_m,
                        bodies=motions, diverged_at=diverged,
                        backend=getattr(backend, "name", type(backend).__name__))
