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
]

# CLAUDE.md: step 2 seconds with no actuation; nothing moves more than 2 mm.
SETTLE_SECONDS = 2.0
SETTLE_TOLERANCE_M = 0.002

# 1/240 s. Measured, not conventional: on bench_arm the worst settle
# displacement is 1.25 mm at 1/240, 3.43 mm at 1/480 and 3.56 mm at 1/960
# (scripts/probe_dt.py). Smaller is not better here -- MuJoCo's soft-contact
# equilibrium is time-scaled, so shrinking the step changes where objects come
# to rest. Pinning it makes the tolerance mean something.
SETTLE_DT = 1.0 / 240.0


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
