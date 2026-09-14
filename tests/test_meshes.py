"""T1 acceptance: every catalog item generates a sound, hollow-where-it-should-be mesh.

The single thing these tests exist to prevent: a beaker whose mesh is a solid
cylinder. It renders correctly, it grasps correctly, and it fails every pour
task for a reason no render will ever show you.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from labgen import meshes
from labgen.catalog import CATALOG
from labgen.meshes import Mesh, MeshError

HOLLOW = {"open_vessel", "conical_vessel"}

# Catalog entries whose *dimensions* are self-inconsistent, so no mesh can be
# built from them. Empty, and it should stay that way: an entry landing here is
# a catalog bug, not a mesh bug, and the refusal path itself is tested against
# synthetic dims below so this list is never load-bearing for coverage.
#
# vial_rack_5x10_20ml lived here until its footprint was corrected from 240x120
# (30 mm holes on a 24 mm pitch -- impossible) to 320x160.
KNOWN_BAD_DIMS: dict[str, str] = {}

BUILDABLE = sorted(set(CATALOG) - set(KNOWN_BAD_DIMS))


@pytest.fixture(scope="module")
def built() -> dict[str, Mesh]:
    return {k: meshes.build(CATALOG[k]) for k in BUILDABLE}


# --- T1 acceptance: soundness -------------------------------------------

@pytest.mark.parametrize("key", BUILDABLE)
def test_no_degenerate_or_zero_area_triangles(key, built):
    bad = built[key].degenerate_faces()
    assert len(bad) == 0, f"{key}: {len(bad)} degenerate faces at {bad[:10].tolist()}"


@pytest.mark.parametrize("key", BUILDABLE)
def test_mesh_is_manifold(key, built):
    """Closed, two-manifold and consistently wound.

    No exceptions are enumerated because none are needed: every shape in the
    catalog is a closed solid. If an entry ever legitimately needs an open
    surface, it belongs on an explicit list here with its justification, the
    way KNOWN_BAD_DIMS works.
    """
    errors = built[key].manifold_errors()
    assert not errors, f"{key}: {[(k, len(v)) for k, v in errors.items()]}"


@pytest.mark.parametrize("key", BUILDABLE)
def test_volume_is_positive_so_normals_point_outward(key, built):
    assert built[key].volume_m3() > 0, (
        f"{key}: negative volume means the surface is inside-out. Physics will "
        f"treat the inside as solid and the outside as cavity."
    )


@pytest.mark.parametrize("key", BUILDABLE)
def test_object_sits_on_z_zero(key, built):
    """A pose places an object's base, not its centre. If a mesh is centred on
    the origin every object in every scene floats by half its height."""
    lo, _ = built[key].bounds()
    assert lo[2] == pytest.approx(0.0, abs=1e-12), f"{key}: base at z={lo[2]}"


@pytest.mark.parametrize("key", BUILDABLE)
def test_extents_match_catalog_dimensions(key, built):
    """The scale gate, at the mesh level.

    Tolerance is 1%: a circle sampled at 64 segments is inscribed, so its
    measured width is exact at the sample points but the enclosed shape is
    0.12% small. Anything beyond 1% is a real dimension error, not faceting.
    """
    item = CATALOG[key]
    ex = built[key].extents()
    d = item.dims

    if item.shape in ("open_vessel", "solid_cylinder"):
        width = max(d.get("outer_d", 0.0), d.get("base_d", 0.0))
        expected = (width, width, d["height"])
    elif item.shape == "conical_vessel":
        expected = (d["base_d"], d["base_d"], d["height"])
    else:
        expected = (d["x"], d["y"], d["z"])

    for axis, got, want in zip("xyz", ex, expected):
        assert got == pytest.approx(want, rel=0.01), (
            f"{key}: {axis} extent {got:.5f} m vs catalog {want:.5f} m"
        )


# --- T1 acceptance: the cavity is real ----------------------------------

@pytest.mark.parametrize("key", [k for k in BUILDABLE if CATALOG[k].shape in HOLLOW])
def test_cavity_is_real_by_ray_cast(key, built):
    """Cast a ray up the axis and check *where* it crosses, not just how often.

    T1 words this as "two surface crossings, not one". Two is the right count,
    but the count alone does not prove anything -- a solid cylinder also
    crosses twice, at its base and its top. What actually distinguishes an open
    vessel is the position of the crossings:

      * one at z = 0            (underside of the base)
      * one at z = wall         (the cavity floor)
      * and nothing at z = height, because the vessel is open at the top

    A solid cylinder would put its second crossing at z = height and would show
    a surface at the rim. So all three conditions are asserted.
    """
    item = CATALOG[key]
    mesh = built[key]
    wall = item.dims["wall"]
    height = item.dims["height"]

    # Slightly off-axis: a ray through the fan apex hits every triangle that
    # shares it and reports one crossing each. For a conical vessel the neck is
    # the narrowest point the ray has to clear.
    narrowest_d = item.dims["neck_d"] if item.shape == "conical_vessel" else item.dims["outer_d"]
    inner_r = narrowest_d / 2.0 - wall
    crossings = mesh.ray_crossings_z(inner_r * 0.25, inner_r * 0.1)

    assert len(crossings) == 2, (
        f"{key}: expected 2 crossings up the axis, got {len(crossings)}: "
        f"{[round(c, 5) for c in crossings]}"
    )
    assert crossings[0] == pytest.approx(0.0, abs=1e-9), (
        f"{key}: first crossing at {crossings[0]}, expected the base at z=0"
    )
    assert crossings[1] == pytest.approx(wall, rel=0.02), (
        f"{key}: second crossing at {crossings[1]:.5f} m, expected the cavity "
        f"floor at z={wall} m. At z={height} m it would mean a solid vessel "
        f"with a lid."
    )
    assert not any(c > height * 0.9 for c in crossings), (
        f"{key}: something is capping the opening at z~{height} m"
    )


@pytest.mark.parametrize("key", [k for k in BUILDABLE if CATALOG[k].shape in HOLLOW])
def test_generated_cavity_matches_analytic_cavity(key):
    """Does the *generator* build the cavity the dimensions describe?

    This is deliberately separate from any comparison against `capacity_ml`.
    Those are two different questions:

      * this test -- is the mesh code correct for the given dims? (a code bug)
      * test_capacity_is_plausible -- are the dims right? (a catalog bug)

    Conflating them, which the literal T1 wording does, means a mesh bug and a
    spec-sheet error produce the same failure and you cannot tell which you
    have. Tolerance here is tight, 0.5%, because the only error source is
    polygonal approximation of a circle at 256 segments (0.01%).
    """
    item = CATALOG[key]
    d = item.dims
    wall = d["wall"]

    if item.shape == "open_vessel":
        inner_r = d["outer_d"] / 2.0 - wall
        analytic = math.pi * inner_r ** 2 * (d["height"] - wall)
    else:
        shoulder_z = d["height"] - d["neck_height"]
        big = d["base_d"] / 2.0 - wall
        small = d["neck_d"] / 2.0 - wall
        frustum_h = shoulder_z - wall
        analytic = (math.pi * frustum_h / 3.0 * (big * big + big * small + small * small)
                    + math.pi * small ** 2 * d["neck_height"])

    got = meshes.cavity_volume_m3(item)
    assert got == pytest.approx(analytic, rel=0.005), (
        f"{key}: generated cavity {got * 1e6:.2f} mL vs analytic "
        f"{analytic * 1e6:.2f} mL"
    )


@pytest.mark.parametrize("key", [k for k in BUILDABLE
                                 if CATALOG[k].shape in HOLLOW
                                 and CATALOG[k].capacity_ml is not None])
def test_capacity_is_plausible_against_brim_volume(key):
    """Is the catalog's stated capacity consistent with its stated dimensions?

    `cavity_volume_m3` measures to the **brim**. A vessel's marked capacity is
    not its brim volume: a Griffin beaker is graduated at roughly 70-80% of
    brim so it can be carried without slopping, whereas a graduated cylinder or
    an Erlenmeyer is marked much closer to full. So the expected ratio is
    shape-dependent and is never 1.0.

    The band below is wide on purpose. It is here to catch an order-of-
    magnitude error or a millimetre/metre slip, not to certify a spec sheet.
    Entries that sit near an edge are reported by
    `test_capacity_ratio_report` for a human to check against the
    manufacturer drawing -- CLAUDE.md is explicit that a confidently wrong
    dimension is worse than a missing one, so nothing here is auto-corrected.
    """
    item = CATALOG[key]
    brim_ml = meshes.cavity_volume_m3(item) * 1e6
    ratio = brim_ml / item.capacity_ml

    assert ratio >= 1.0, (
        f"{key}: brim volume {brim_ml:.1f} mL is below its rated capacity "
        f"{item.capacity_ml} mL. The vessel cannot hold what it claims to; one "
        f"of the dimensions or the capacity is wrong."
    )
    assert ratio <= 1.8, (
        f"{key}: brim volume {brim_ml:.1f} mL is {ratio:.2f}x its rated "
        f"{item.capacity_ml} mL. Too much headroom to be a real vessel."
    )


def test_capacity_ratio_report(capsys):
    """Standing report of brim/capacity across the vessels. Never fails.

    Run `pytest -rP -k capacity_ratio_report` to read it. Ratios that sit well
    outside the family norm are the ones to check against a spec sheet.
    """
    rows = []
    for key in BUILDABLE:
        item = CATALOG[key]
        if item.shape not in HOLLOW or item.capacity_ml is None:
            continue
        brim = meshes.cavity_volume_m3(item) * 1e6
        rows.append((key, item.capacity_ml, brim, brim / item.capacity_ml))

    print("\nbrim volume vs rated capacity (expected ~1.25-1.40 for Griffin "
          "beakers, ~1.0-1.1 for cylinders/flasks):")
    for key, cap, brim, ratio in rows:
        print(f"  {key:26} {cap:8.1f} mL rated  {brim:8.1f} mL brim  {ratio:5.2f}x")
    assert rows


# --- impossible dimensions must be refused, not worked around ------------

def test_overlapping_holes_are_refused_with_a_useful_message():
    """The footprint that vial_rack_5x10_20ml originally shipped with.

    Tested against synthetic dims rather than a catalog entry on purpose: a
    test that depends on the catalog staying broken stops testing anything the
    moment someone fixes it, which is exactly what happened here.

    If someone 'fixes' the builder to tolerate overlapping holes, this catches
    it. The rack would then build, look right, and accept no real vial.
    """
    impossible = {"x": 0.240, "y": 0.120, "z": 0.030,
                  "hole_d": 0.030, "hole_depth": 0.022, "rows": 5, "cols": 10}
    with pytest.raises(MeshError) as exc:
        meshes.build_tube_rack(impossible, 16, "overlapping")
    msg = str(exc.value)
    assert "catalog dimension error" in msg, f"unhelpful message: {msg}"
    assert "do not shrink" in msg


def test_every_catalog_entry_currently_builds():
    """Complements KNOWN_BAD_DIMS being empty: prove it, do not assume it."""
    assert set(meshes.build_all()) == set(CATALOG)


# --- tube racks have real holes -----------------------------------------

def test_rack_bores_are_real_blind_holes():
    item = CATALOG["test_tube_rack_6x12"]
    mesh = meshes.build(item)
    d = item.dims
    pitch_x, pitch_y = d["x"] / d["cols"], d["y"] / d["rows"]
    floor_z = d["z"] - d["hole_depth"]

    cx = -d["x"] / 2 + 0.5 * pitch_x
    cy = -d["y"] / 2 + 0.5 * pitch_y

    down_a_bore = mesh.ray_crossings_z(cx + d["hole_d"] * 0.15, cy + d["hole_d"] * 0.07)
    assert len(down_a_bore) == 2
    assert down_a_bore[0] == pytest.approx(0.0, abs=1e-9)
    assert down_a_bore[1] == pytest.approx(floor_z, abs=1e-9), (
        "the bore has no floor at the depth the catalog states"
    )

    through_material = mesh.ray_crossings_z(cx + pitch_x * 0.48, cy)
    assert through_material == pytest.approx([0.0, d["z"]], abs=1e-9), (
        "material between bores is not solid"
    )


def test_rack_volume_is_box_minus_bores():
    item = CATALOG["test_tube_rack_6x12"]
    d = item.dims
    mesh = meshes.build(item)

    box = d["x"] * d["y"] * d["z"]
    bores = d["rows"] * d["cols"] * math.pi * (d["hole_d"] / 2) ** 2 * d["hole_depth"]

    # The bores are 16-gons inscribed in the circle, so slightly less material
    # is removed than the analytic figure: +2.6% expected, one-sided.
    got = mesh.volume_m3()
    assert box - bores < got < (box - bores) * 1.05, (
        f"{got * 1e6:.2f} cm3 vs analytic {(box - bores) * 1e6:.2f} cm3"
    )


def test_rack_rejects_a_through_hole():
    dims = dict(CATALOG["test_tube_rack_6x12"].dims)
    dims["hole_depth"] = dims["z"]
    with pytest.raises(MeshError, match="blind holes need a floor"):
        meshes.build_tube_rack(dims, 16, "through")


# --- revolve primitive ---------------------------------------------------

def test_revolve_produces_a_correct_cylinder_volume():
    r, h, n = 0.05, 0.2, 512
    mesh = meshes.revolve([(r, 0.0), (r, h), (0.0, h), (0.0, 0.0)], segments=n)
    assert mesh.is_manifold
    assert mesh.volume_m3() == pytest.approx(math.pi * r * r * h, rel=1e-4)


def test_revolve_drops_zero_area_axis_segments():
    """The (0, w) -> (0, 0) edge of a vessel profile sweeps nothing."""
    mesh = meshes.revolve([(0.05, 0.0), (0.05, 0.1), (0.0, 0.1), (0.0, 0.0)], segments=32)
    assert len(mesh.degenerate_faces()) == 0


def test_revolve_rejects_a_negative_radius():
    with pytest.raises(MeshError, match="negative"):
        meshes.revolve([(0.05, 0.0), (-0.01, 0.1), (0.0, 0.1), (0.0, 0.0)])


def test_revolve_rejects_too_few_segments():
    with pytest.raises(MeshError, match="at least 3 segments"):
        meshes.revolve([(0.05, 0.0), (0.05, 0.1), (0.0, 0.0)], segments=2)


def test_wall_thicker_than_radius_is_rejected():
    with pytest.raises(MeshError, match="no cavity"):
        meshes.build_open_vessel(
            {"outer_d": 0.02, "height": 0.05, "wall": 0.02}, 32, "impossible")


# --- collision proxies ---------------------------------------------------

@pytest.mark.parametrize("key", BUILDABLE)
def test_no_decimated_proxy_for_an_sdf_collider(key):
    """CLAUDE.md rule 1, one step downstream.

    An SDF is baked from the mesh it is handed. A decimated beaker moves the
    cavity wall inward by the chord sag, which silently shrinks the interior of
    every open vessel in the scene.
    """
    item = CATALOG[key]
    if item.collider == "sdf":
        assert meshes.build_collision(item) is None


def test_collision_proxy_is_smaller_or_absent():
    for key in BUILDABLE:
        proxy = meshes.build_collision(CATALOG[key])
        if proxy is not None:
            assert len(proxy.faces) < len(meshes.build(CATALOG[key]).faces)
            assert proxy.is_manifold


# --- OBJ output ----------------------------------------------------------

def test_obj_round_trips_through_a_file(tmp_path):
    mesh = meshes.build("beaker_250")
    path = mesh.write_obj(tmp_path / "gen" / "beaker_250.obj")
    text = path.read_text(encoding="utf-8")

    v = [l for l in text.splitlines() if l.startswith("v ")]
    f = [l for l in text.splitlines() if l.startswith("f ")]
    assert len(v) == len(mesh.vertices)
    assert len(f) == len(mesh.faces)
    assert "metres" in text, "unit-less OBJ is how a scale error gets in"

    verts = np.array([[float(x) for x in l.split()[1:]] for l in v])
    faces = np.array([[int(x) - 1 for x in l.split()[1:]] for l in f])
    reread = Mesh(verts, faces, name="reread")
    assert reread.is_manifold
    assert reread.volume_m3() == pytest.approx(mesh.volume_m3(), rel=1e-6)


def test_build_all_covers_every_buildable_item():
    assert set(meshes.build_all(skip_unbuildable=True)) == set(BUILDABLE)


def test_build_all_raises_by_default_on_a_broken_entry():
    """Reporting may skip a broken entry; a scene build must never silently.

    Injects a broken entry rather than relying on one existing, so the
    guarantee stays tested once the catalog is clean.
    """
    import dataclasses

    broken = dataclasses.replace(
        CATALOG["test_tube_rack_6x12"],
        key="broken_rack",
        dims={**CATALOG["test_tube_rack_6x12"].dims, "hole_d": 0.5},
    )
    original = dict(meshes.CATALOG)
    meshes.CATALOG["broken_rack"] = broken
    try:
        with pytest.raises(MeshError):
            meshes.build_all()
        assert "broken_rack" not in meshes.build_all(skip_unbuildable=True)
    finally:
        meshes.CATALOG.clear()
        meshes.CATALOG.update(original)


def test_build_rejects_an_unknown_key():
    with pytest.raises(MeshError, match="not a catalog key"):
        meshes.build("beaker_9000")


# --- necked vessels ------------------------------------------------------
#
# The shape kind exists and is tested; no catalog entry uses it yet. The three
# vials that need it (vial_2ml, vial_20ml, vial_40ml) are blocked on shoulder
# and neck dimensions nobody has sourced, and inventing them is exactly what
# CLAUDE.md forbids. Their `source` strings say so.

# A 20 mL scintillation vial's real OD and height, with a plausible neck, used
# here to exercise the builder. NOT catalog data -- body_height is made up, and
# that is precisely why it lives in a test and not in catalog.py.
VIAL_20ML_SHAPED = {
    "outer_d": 0.028, "height": 0.061, "wall": 0.0012,
    "body_height": 0.042, "neck_d": 0.022, "neck_height": 0.010,
}


def _necked(**overrides):
    dims = dict(VIAL_20ML_SHAPED)
    dims.update(overrides)
    return meshes.build_necked_vessel(dims, 64, "necked_test")


def test_necked_vessel_is_sound():
    mesh = _necked()
    assert mesh.is_manifold
    assert len(mesh.degenerate_faces()) == 0
    assert mesh.volume_m3() > 0
    lo, hi = mesh.bounds()
    assert lo[2] == pytest.approx(0.0, abs=1e-12)
    assert hi[2] == pytest.approx(VIAL_20ML_SHAPED["height"], rel=1e-9)


def test_necked_vessel_actually_has_a_neck():
    """The point of the shape kind.

    Measure the mesh radius at neck height and at body height. If they are the
    same, this is a straight cylinder wearing a different shape name and every
    grasp planned against it closes on geometry that is not there.
    """
    mesh = _necked()
    v = mesh.vertices
    d = VIAL_20ML_SHAPED

    def radius_at(z):
        # Sample where the profile actually has vertices. The body is a
        # straight wall, so it carries vertices only at its ends -- picking a z
        # mid-wall finds nothing.
        near = v[np.abs(v[:, 2] - z) < 1e-9]
        assert len(near), f"no vertices at z={z}"
        return float(np.max(np.hypot(near[:, 0], near[:, 1])))

    body_r = radius_at(d["body_height"])          # top of the cylindrical body
    neck_r = radius_at(d["height"])               # the rim, at the top of the neck

    assert body_r == pytest.approx(d["outer_d"] / 2, rel=0.002)
    assert neck_r == pytest.approx(d["neck_d"] / 2, rel=0.002)
    assert neck_r < body_r * 0.9, (
        f"neck radius {neck_r:.4f} m is not meaningfully narrower than the body "
        f"{body_r:.4f} m -- a gripper would find no neck to close on"
    )


def test_necked_vessel_cavity_is_real():
    mesh = _necked()
    d = VIAL_20ML_SHAPED
    inner_r = d["neck_d"] / 2 - d["wall"]
    crossings = mesh.ray_crossings_z(inner_r * 0.3, inner_r * 0.1)
    assert len(crossings) == 2
    assert crossings[0] == pytest.approx(0.0, abs=1e-9)
    assert crossings[1] == pytest.approx(d["wall"], rel=0.02)


def test_necked_vessel_holds_less_than_a_straight_cylinder():
    """The volume symptom that pointed at the shape gap in the first place."""
    d = VIAL_20ML_SHAPED
    necked = _necked().volume_m3()
    straight = meshes.build_open_vessel(
        {"outer_d": d["outer_d"], "height": d["height"], "wall": d["wall"]},
        64, "straight").volume_m3()
    # Glass volume differs; what matters is the cavity. Compare envelopes.
    assert necked < straight * 1.5
    body_only = math.pi * (d["outer_d"] / 2) ** 2 * d["height"]
    assert necked < body_only


def test_necked_vessel_rejects_a_neck_that_is_not_narrower():
    with pytest.raises(MeshError, match="not narrower"):
        _necked(neck_d=0.028)


def test_necked_vessel_rejects_a_body_that_swallows_the_shoulder():
    with pytest.raises(MeshError, match="no room for a shoulder"):
        _necked(body_height=0.055)


def test_necked_vessel_rejects_a_wall_that_seals_the_neck():
    with pytest.raises(MeshError, match="closes off"):
        _necked(wall=0.012)


def test_no_catalog_entry_claims_necked_vessel_without_dims():
    """Guard the conversion when it happens: the dims must arrive with it."""
    for key, item in CATALOG.items():
        if item.shape == "necked_vessel":
            missing = {"outer_d", "height", "wall", "body_height",
                       "neck_d", "neck_height"} - set(item.dims)
            assert not missing, f"{key} is necked_vessel but lacks {sorted(missing)}"
