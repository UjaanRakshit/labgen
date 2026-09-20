"""The settle test's plumbing, exercised without a physics engine.

The physics itself needs Isaac Lab and a GPU, so what is tested here is
everything around it: that a missing backend is reported rather than silently
passing, that the pass/fail arithmetic is right, and that the report says which
body moved and by how much.

The one property worth stating plainly: a settle test that quietly does nothing
is worse than no settle test at all, because the report still reads as a pass.
`test_missing_backend_raises` is the guard for that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from labgen.settle import (SETTLE_DT, SETTLE_SECONDS, SETTLE_TOLERANCE_M,
                           BackendUnavailable, BodyMotion, NewtonBackend,
                           SettleResult, settle)


class FakeBackend:
    """Returns a scripted outcome so the reporting can be tested anywhere."""

    name = "fake"

    def __init__(self, motions, diverged=None):
        self._motions, self._diverged = motions, diverged

    def run(self, usda, seconds):
        return self._motions, self._diverged


def motion(name, moved_m, peak_m=None):
    return BodyMotion(name=name, start=(0.0, 0.0, 0.0),
                      end=(0.0, 0.0, -moved_m), displacement_m=moved_m,
                      peak_m=peak_m if peak_m is not None else moved_m)


@pytest.fixture
def usda(tmp_path) -> Path:
    p = tmp_path / "scene.usda"
    p.write_text("#usda 1.0\n", encoding="utf-8")
    return p


# --- constants match CLAUDE.md -------------------------------------------

def test_defaults_are_what_claude_md_specifies():
    assert SETTLE_SECONDS == 2.0
    assert SETTLE_TOLERANCE_M == 0.002


def test_timestep_is_pinned():
    """1/240 is measured, not conventional.

    On bench_arm the worst settle displacement is 1.25 mm at 1/240, 3.43 mm at
    1/480 and 3.56 mm at 1/960. MuJoCo's soft-contact equilibrium is
    time-scaled, so a smaller step moves where things come to rest. If the
    timestep drifts the tolerance stops meaning anything.
    """
    assert SETTLE_DT == pytest.approx(1.0 / 240.0)


# --- pass / fail arithmetic ----------------------------------------------

def test_everything_still_is_a_pass(usda):
    result = settle(usda, backend=FakeBackend([motion("beaker", 0.0005),
                                               motion("tube", 0.0001)]))
    assert result.ok and bool(result)
    assert result.worst.name == "beaker"


def test_one_body_over_tolerance_fails_the_whole_scene(usda):
    result = settle(usda, backend=FakeBackend([motion("beaker", 0.0005),
                                               motion("petri", 0.0084)]))
    assert not result.ok
    assert result.worst.name == "petri"
    assert "MOVED" in str(result) and "FAIL" in str(result)


def test_exactly_at_tolerance_passes(usda):
    assert settle(usda, backend=FakeBackend([motion("x", SETTLE_TOLERANCE_M)])).ok


def test_just_over_tolerance_fails(usda):
    assert not settle(usda, backend=FakeBackend(
        [motion("x", SETTLE_TOLERANCE_M + 1e-9)])).ok


def test_divergence_is_a_failure_even_if_nothing_moved(usda):
    """A scene that explodes must not pass on the strength of its last frame."""
    result = settle(usda, backend=FakeBackend([motion("x", 0.0)], diverged=57))
    assert not result.ok
    assert "DIVERGED at step 57" in str(result)


# --- the report says what happened ---------------------------------------

def test_report_names_the_body_and_the_distance(usda):
    text = str(settle(usda, backend=FakeBackend([motion("petri", 0.0084, 0.0097)])))
    assert "petri" in text
    assert "8.400" in text        # displacement in mm
    assert "9.700" in text        # peak in mm
    assert "2.0 mm tolerance" in text


def test_report_sorts_worst_first(usda):
    text = str(settle(usda, backend=FakeBackend(
        [motion("small", 0.0001), motion("big", 0.009), motion("mid", 0.001)])))
    order = [line.split()[0] for line in text.splitlines()
             if line.startswith("  ") and line.split() and
             line.split()[0] in {"small", "big", "mid"}]
    assert order == ["big", "mid", "small"]


def test_empty_scene_does_not_crash_the_report(usda):
    assert "no free bodies" in str(settle(usda, backend=FakeBackend([])))


# --- the guard that matters ----------------------------------------------

def test_missing_backend_raises_rather_than_passing_vacuously(usda, monkeypatch):
    """A settle test that silently does nothing still reports a pass.

    That is strictly worse than not running one, so the absence of a physics
    engine has to be an error.
    """
    import importlib

    real = importlib.import_module

    def no_newton(name, *a, **kw):
        if name == "newton":
            raise ImportError("no module named 'newton'")
        return real(name, *a, **kw)

    monkeypatch.setattr(importlib, "import_module", no_newton)
    with pytest.raises(BackendUnavailable, match="Isaac Lab"):
        settle(usda, backend=NewtonBackend())


def test_constructing_the_newton_backend_needs_no_newton():
    """Importable and constructible everywhere; only `run` needs the engine."""
    assert NewtonBackend().name == "newton"


def test_a_missing_file_is_reported_before_any_backend_work(tmp_path):
    with pytest.raises(FileNotFoundError):
        settle(tmp_path / "nope.usda", backend=FakeBackend([]))


def test_backend_name_reaches_the_report(usda):
    assert "fake" in str(settle(usda, backend=FakeBackend([motion("x", 0.0)])))


# --- known open failure, pinned ------------------------------------------

def test_petri_dish_is_the_known_settle_failure():
    """Recorded so it cannot be quietly forgotten or quietly "fixed".

    petri_dish_100 sinks 8.4 mm in 2 s against a 2 mm gate, in both reference
    scenes. It is NOT an SDF resolution problem -- the sink is bit-identical
    across authored voxel sizes of 1.6/0.8/0.4 mm and with the SDF schema
    removed entirely -- so it is contact softness in the solver.

    The tempting fix, convex_decomposition, drops it to 0.01 mm and seals the
    dish: a tube dropped in then rests on the rim at z=23 mm instead of the
    floor at z=0.02 mm. check_colliders now rejects that, which is why this
    stays open rather than being made green.
    """
    from labgen.catalog import CATALOG
    assert CATALOG["petri_dish_100"].collider == "sdf"
