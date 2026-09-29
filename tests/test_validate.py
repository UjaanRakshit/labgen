"""T3 acceptance: corrupt the reference scene five ways, get five right answers.

TASKS.md asks for exactly this -- float an object, overlap two, strip a mass,
hull a beaker, scale a beaker 20% up -- and for no false positives on the clean
scene. The last clause is the one that actually bites: a validator that fails
everything is as useless as one that passes everything, and it is much easier
to write.

Each corruption asserts on the SPECIFIC check that should fire and on the
absence of unrelated failures, so a check that fires for the wrong reason is
caught too.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from labgen.catalog import CATALOG
from labgen.types import SceneSpec
from labgen.validate import (CHECKS, Severity, YAM, check_colliders,
                             check_graspable, check_physics_completeness,
                             check_scale, check_support, validate)

BENCH_5 = "examples/bench_5.json"


@pytest.fixture
def scene() -> SceneSpec:
    return SceneSpec.read(BENCH_5)


@pytest.fixture
def catalog_guard():
    """Restore catalog entries after a test mutates one."""
    saved = dict(CATALOG)
    yield CATALOG
    CATALOG.clear()
    CATALOG.update(saved)


def failed_checks(report) -> set[str]:
    return {f.check for f in report.failures}


# --- the clean scene -----------------------------------------------------

def test_clean_scene_has_no_failures(scene):
    report = validate(scene)
    assert report.ok, f"clean scene reported failures:\n{report}"


def test_clean_scene_warnings_are_only_the_known_ones(scene):
    """Warnings are allowed, but they have to be explicable.

    Two are expected: nothing in the catalog is sourced yet, and the hotplate
    and petri dish are wider than the gripper's jaws. Anything else appearing
    here is a regression in a check, not in the scene.
    """
    kinds = {f.check for f in validate(scene).warnings}
    assert kinds <= {"provenance", "graspable"}, f"unexpected warnings: {kinds}"


def test_report_renders_the_measured_value(scene):
    """A failure has to say what it measured, or it is unactionable at 2am."""
    broken = dataclasses.replace(scene.by_id("beaker"),
                                 position=(scene.by_id("beaker").position[0],
                                           scene.by_id("beaker").position[1], 0.05))
    scene.objects = [broken if o.instance_id == "beaker" else o for o in scene.objects]
    text = str(validate(scene))
    assert "measured" in text and "limit" in text
    assert "mm" in text


# --- corruption 1: float an object ---------------------------------------

def test_floating_object_is_caught(scene):
    beaker = scene.by_id("beaker")
    scene.objects = [
        dataclasses.replace(beaker, position=(beaker.position[0], beaker.position[1], 0.05))
        if o.instance_id == "beaker" else o for o in scene.objects]

    report = validate(scene)
    assert not report.ok
    assert failed_checks(report) == {"support"}, str(report)
    bad = [f for f in report.failures if f.instance_id == "beaker"][0]
    assert bad.measured == pytest.approx(50.0, abs=0.1), "should report the 50 mm gap"


def test_a_one_millimetre_gap_is_tolerated(scene):
    """The tolerance is real and must not be accidentally zero."""
    beaker = scene.by_id("beaker")
    scene.objects = [
        dataclasses.replace(beaker, position=(beaker.position[0], beaker.position[1], 0.0009))
        if o.instance_id == "beaker" else o for o in scene.objects]
    assert validate(scene).ok


# --- corruption 2: overlap two -------------------------------------------

def test_overlapping_objects_are_reported(scene):
    beaker = scene.by_id("beaker")
    tube = scene.by_id("test_tube")
    scene.objects = [
        dataclasses.replace(tube, position=(beaker.position[0], beaker.position[1], 0.0))
        if o.instance_id == "test_tube" else o for o in scene.objects]

    report = validate(scene)
    kinds = {f.check for f in report.warnings}
    assert "interpenetration" in kinds, str(report)
    msg = " ".join(f.message for f in report.warnings if f.check == "interpenetration")
    assert "beaker" in msg and "test_tube" in msg


# --- corruption 3: strip a mass ------------------------------------------

def test_zero_mass_dynamic_body_is_caught(scene, catalog_guard):
    catalog_guard["beaker_250"] = dataclasses.replace(
        catalog_guard["beaker_250"], mass_kg=0.0)

    report = validate(scene)
    assert not report.ok
    assert failed_checks(report) == {"physics"}, str(report)
    assert "explodes" in " ".join(f.message for f in report.failures)


def test_a_static_fixture_may_have_zero_mass(scene):
    """bench_top ships mass 0 and is fixed. That must stay legal."""
    assert CATALOG["bench_top"].mass_kg == 0.0
    assert validate(scene).ok


# --- corruption 4: hull a beaker -----------------------------------------

def test_convex_hull_on_an_open_vessel_is_a_hard_failure(scene, catalog_guard):
    catalog_guard["beaker_250"] = dataclasses.replace(
        catalog_guard["beaker_250"], collider="convex_hull")

    report = validate(scene)
    assert not report.ok
    assert failed_checks(report) == {"collider"}, str(report)
    assert "seals the opening" in " ".join(f.message for f in report.failures)


def test_convex_decomposition_on_an_open_vessel_is_also_a_hard_failure(
        scene, catalog_guard):
    """Measured, not assumed: decomposition seals a shallow dish too.

    Switching petri_dish_100 to convex_decomposition made the settle test pass
    (8.00 mm sink -> 0.01 mm) while a tube dropped into it came to rest on the
    rim at z=23 mm instead of the floor at z=0.02 mm.
    """
    catalog_guard["petri_dish_100"] = dataclasses.replace(
        catalog_guard["petri_dish_100"], collider="convex_decomposition")

    report = validate(scene)
    assert not report.ok
    assert failed_checks(report) == {"collider"}, str(report)


def test_decomposition_is_still_fine_on_a_rack(catalog_guard):
    """The ban is scoped to open vessels; racks depend on decomposition."""
    assert CATALOG["test_tube_rack_6x12"].collider == "convex_decomposition"
    from labgen.validate import Report
    report = Report(scene="x")
    check_colliders(SceneSpec.read(BENCH_5), report, YAM)
    assert report.ok


# --- corruption 5: scale a beaker 20% up ---------------------------------

def test_scaling_the_catalog_does_NOT_trip_the_scale_check(scene, catalog_guard):
    """The honest version of TASKS.md's "scale a beaker 20% up".

    Inflating a catalog entry inflates the generated mesh by the same factor,
    so mesh-vs-catalog still agrees and the check passes. That is not a bug in
    the check -- it is the two-checks rule. "Is the mesh right for these
    numbers" and "are these numbers right" are different questions; the second
    one is `verified` / `source`, and no amount of geometry can answer it.

    Pinned as a test so nobody later "fixes" check_scale into a tautology that
    appears to catch catalog errors and cannot.
    """
    item = catalog_guard["beaker_250"]
    catalog_guard["beaker_250"] = dataclasses.replace(
        item, dims={**item.dims, "outer_d": item.dims["outer_d"] * 1.20,
                    "base_d": item.dims["base_d"] * 1.20})
    assert "scale" not in failed_checks(validate(scene))
    # but the provenance check still says the entry is unsourced
    assert any(f.check == "provenance" for f in validate(scene).warnings)


def test_a_mesh_that_disagrees_with_the_catalog_IS_caught(scene, monkeypatch):
    """Where check_scale has real teeth: a generator emitting the wrong size.

    Simulated by making the mesh builder return a 20% oversized beaker while
    the catalog keeps its true dimensions -- exactly the disagreement a
    generator regression, or a reconstructed asset, would produce.
    """
    import labgen.validate as V
    real_build = V.build

    def fat_build(item, *a, **kw):
        mesh = real_build(item, *a, **kw)
        if getattr(item, "key", None) == "beaker_250":
            mesh.vertices = mesh.vertices * np.array([1.20, 1.20, 1.0])
        return mesh

    monkeypatch.setattr(V, "build", fat_build)
    report = validate(scene)
    assert not report.ok
    assert "scale" in failed_checks(report), str(report)
    bad = [f for f in report.failures if f.check == "scale"][0]
    assert bad.instance_id == "beaker"
    assert bad.measured == pytest.approx(20.0, abs=0.5)


def test_a_four_percent_mesh_error_is_tolerated(scene, monkeypatch):
    """5% is the gate; 4% must pass or the tolerance is not what it says."""
    import labgen.validate as V
    real_build = V.build

    def slightly_fat(item, *a, **kw):
        mesh = real_build(item, *a, **kw)
        if getattr(item, "key", None) == "beaker_250":
            mesh.vertices = mesh.vertices * np.array([1.04, 1.04, 1.0])
        return mesh

    monkeypatch.setattr(V, "build", slightly_fat)
    assert "scale" not in failed_checks(validate(scene))


# --- the checks this project added ---------------------------------------

def test_object_wider_than_the_jaws_is_flagged(scene):
    warn = [f for f in validate(scene).warnings if f.check == "graspable"]
    ids = {f.instance_id for f in warn}
    assert ids == {"hotplate", "petri"}, f"expected the 160 mm and 100 mm items, got {ids}"


def test_a_scene_outside_the_workspace_warns():
    scene = SceneSpec.read(BENCH_5)
    beaker = scene.by_id("beaker")
    scene.objects = [dataclasses.replace(beaker, position=(0.0, 1.30, 0.0))
                     if o.instance_id == "beaker" else o for o in scene.objects]
    warn = [f for f in validate(scene).warnings if f.check == "reachability"]
    assert warn and warn[0].instance_id == "beaker"
    assert warn[0].measured == pytest.approx(1.30, abs=1e-6)


def test_video_derived_scene_without_a_marker_is_rejected(scene):
    """CLAUDE.md: monocular video has no metric scale."""
    scene.source["method"] = "video"
    scene.source["marker_config"] = ""
    report = validate(scene)
    assert not report.ok
    assert failed_checks(report) == {"provenance"}
    assert "metric scale" in " ".join(f.message for f in report.failures)


def test_hand_authored_scene_needs_no_marker(scene):
    assert scene.source["method"] == "hand_authored"
    assert "provenance" not in failed_checks(validate(scene))


def test_robot_spec_demands_a_source():
    from labgen.validate import RobotSpec
    with pytest.raises(ValueError, match="source"):
        RobotSpec(name="ghost", reach_min_m=0.1, reach_max_m=0.5,
                  jaw_opening_m=0.08, source="")


def test_every_check_runs_on_the_reference_scene(scene):
    """No check may silently do nothing -- that is how coverage rots."""
    for check in CHECKS:
        from labgen.validate import Report
        report = Report(scene=scene.name)
        check(scene, report, YAM)
        assert report.findings, f"{check.__name__} produced no findings at all"


def test_a_rotated_support_still_supports_what_sits_on_it():
    """A bench turned 90 deg: its bounds must rotate with it, or objects on the
    part that only exists after rotation read as floating."""
    import json
    from pathlib import Path
    from labgen.types import SceneSpec
    from labgen.validate import validate
    spec = SceneSpec.read(Path(__file__).resolve().parents[1] / "examples/bench_arm.json")
    bench = spec.by_id("bench")
    assert abs(bench.orientation_wxyz[0] - 1.0) > 1e-6, "fixture expects the rotated bench"
    report = validate(spec)
    assert not [f for f in report.findings if f.check == "support" and f.severity.name == "FAIL"]
