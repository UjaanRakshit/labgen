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

from labgen.settle import (CONTACT_KD, CONTACT_KE, SETTLE_DT, SETTLE_SECONDS,
                           SETTLE_TOLERANCE_M,
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


def test_contact_stiffness_and_timestep_are_pinned_together():
    """They are one setting, not two, and were measured as a grid.

    A penetration-based contact is stable only while the stiffness is small
    relative to what the timestep can integrate, so sweeping either alone finds
    an optimum that is an artifact of the other's fixed value. This project
    made that mistake in both directions before running the 2D sweep: at the
    default ke=2500 a smaller timestep looked strictly worse, and at dt=1/240
    more stiffness looked like it helped then hurt.

    Pinned as a pair so nobody "optimises" one of them in isolation and
    reintroduces a sink the tolerance will not catch.
    """
    assert SETTLE_DT == pytest.approx(1.0 / 960.0)
    assert CONTACT_KE == pytest.approx(160_000.0)
    assert CONTACT_KD == pytest.approx(800.0)


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

def test_the_petri_dish_fix_did_not_seal_the_vessel():
    """The settle test passes now. This guards HOW it was made to pass.

    Two routes took the petri dish from an 8.4 mm sink to under a millimetre.
    One of them seals the dish: switching it to convex_decomposition reaches
    0.01 mm while a tube dropped in comes to rest on the rim at z=23 mm instead
    of the floor. The other raises contact stiffness and shrinks the timestep,
    after which the same tube rests ON the dish floor at 7.89 mm -- physically
    more correct than the 0.02 mm it reached before, where it was passing
    through the floor entirely.

    The collider must therefore still be sdf. A green gate is not evidence on
    its own; it was only trustworthy here because the cavity was checked
    separately (scripts/probe_petri_cavity.py).
    """
    from labgen.catalog import CATALOG
    assert CATALOG["petri_dish_100"].collider == "sdf"


def test_sinking_was_never_a_single_object_problem():
    """Recorded because the first report framed it as "the petri dish fails".

    Measured across every catalog vessel on its own, the sink correlated with
    contact pressure at r = -0.76 -- NEGATIVELY, so the least-loaded object
    sank most, which is not what compliance does. All five beakers sank ~1.25 mm
    across a 13x mass range: a fixed artifact, not load-dependent penetration.
    The beaker was passing the 2 mm gate by 0.7 mm of luck while exhibiting the
    same defect as the object that failed it.

    The lesson is about the gate, not the dish: a tolerance that one object
    clears and another does not can be hiding a single shared failure.
    """
    assert SETTLE_TOLERANCE_M == 0.002
