"""Invariants the catalog must hold before any other stage can trust it.

These are cheap, they run in milliseconds, and each one guards a rule from
CLAUDE.md that is easy to break by accident while adding an entry.
"""

from __future__ import annotations

import pytest

from labgen import catalog
from labgen.catalog import CATALOG, CatalogItem

OPEN_SHAPES = {"open_vessel", "conical_vessel"}

# Required dimension keys per shape family, from the `dims` contract in
# catalog.py. meshes.py (T1) reads exactly these.
REQUIRED_DIMS: dict[str, set[str]] = {
    "open_vessel": {"outer_d", "height", "wall", "base_d"},
    "necked_vessel": {"outer_d", "height", "wall", "body_height", "neck_d", "neck_height"},
    "conical_vessel": {"base_d", "neck_d", "height", "neck_height", "wall"},
    "solid_cylinder": {"outer_d", "height"},
    "box": {"x", "y", "z"},
    "tube_rack": {"x", "y", "z", "hole_d", "hole_depth", "rows", "cols"},
}


def test_catalog_is_populated():
    assert len(CATALOG) > 0


@pytest.mark.parametrize("key", sorted(CATALOG))
def test_key_matches_registration(key: str):
    assert CATALOG[key].key == key


@pytest.mark.parametrize("key", sorted(CATALOG))
def test_open_vessels_never_use_a_convex_hull(key: str):
    """CLAUDE.md hard rule 1.

    A convex hull over a beaker seals the opening. Nothing can be placed
    inside it, and a policy appears to succeed by resting an object on a lid
    that does not exist. This is the single most damaging thing that can be
    wrong in the catalog, because the resulting scene looks correct.
    """
    item = CATALOG[key]
    if item.shape in OPEN_SHAPES:
        assert item.collider == "sdf", (
            f"{key} is an open vessel with collider={item.collider!r}. "
            f"Open vessels must use SDF; a hull seals the opening."
        )


@pytest.mark.parametrize("key", sorted(CATALOG))
def test_dimensions_are_si_metres_and_plausible(key: str):
    """Catches the millimetre/metre slip, which is the other silent killer."""
    item = CATALOG[key]
    for name, value in item.dims.items():
        if name in ("rows", "cols"):
            assert float(value).is_integer() and value >= 1, f"{key}.{name}={value}"
            continue
        assert value > 0, f"{key}.{name}={value} must be positive"
        # Nothing on a lab bench is under 0.1 mm or over 3 m. A value of 70
        # (mm entered as metres) or 0.00007 (um) lands outside this.
        assert 1e-4 < value < 3.0, (
            f"{key}.{name}={value} is outside the plausible range for a bench "
            f"object in metres. Millimetres entered as metres?"
        )


@pytest.mark.parametrize("key", sorted(CATALOG))
def test_required_dims_present_for_shape(key: str):
    item = CATALOG[key]
    required = REQUIRED_DIMS[item.shape]
    missing = required - set(item.dims)
    assert not missing, f"{key} ({item.shape}) is missing dims {sorted(missing)}"


@pytest.mark.parametrize("key", sorted(CATALOG))
def test_wall_thinner_than_radius(key: str):
    """A wall thicker than the radius means the cavity is inverted."""
    item = CATALOG[key]
    if "wall" not in item.dims:
        return
    inner_radius = item.dims.get("outer_d", item.dims.get("base_d", 0.0)) / 2 - item.dims["wall"]
    assert inner_radius > 0, (
        f"{key}: wall={item.dims['wall']} leaves no interior cavity. "
        f"A vessel with no cavity is a solid cylinder and will fail every pour task."
    )


@pytest.mark.parametrize("key", sorted(CATALOG))
def test_mass_is_explicit_and_nonnegative(key: str):
    """CLAUDE.md hard rule 4: no silent defaults for physics properties."""
    item = CATALOG[key]
    assert item.mass_kg >= 0.0, f"{key}.mass_kg={item.mass_kg}"
    if item.mass_kg == 0.0:
        # Only legal for static fixtures, which carry no rigid body at all.
        assert item.key == "bench_top" or "bench" in item.key, (
            f"{key} has mass 0 but is not a static fixture. A dynamic body with "
            f"zero mass is a physics explosion waiting to happen."
        )


@pytest.mark.parametrize("key", sorted(CATALOG))
def test_has_base_center_keypoint(key: str):
    """Support and settle checks (T3) place objects by their base."""
    item = CATALOG[key]
    assert "base_center" in item.keypoints or "surface_center" in item.keypoints, (
        f"{key} has no base_center keypoint; the support validator has no "
        f"defined contact point to measure against."
    )


@pytest.mark.parametrize("key", sorted(CATALOG))
def test_capacity_is_consistent_with_geometry(key: str):
    """A stated capacity that the geometry cannot hold means one is wrong.

    Loose bound only -- graduations sit well below the rim, and a Griffin
    beaker's marked capacity is nominal. T1 tightens this to 10% against the
    *generated cavity*; here we only catch order-of-magnitude errors.
    """
    item = CATALOG[key]
    if item.capacity_ml is None or item.shape not in OPEN_SHAPES:
        return
    d = item.dims.get("outer_d") or item.dims.get("base_d")
    h = item.dims["height"]
    brim_ml = 3.14159265 * (d / 2) ** 2 * h * 1e6  # m^3 -> mL
    assert item.capacity_ml <= brim_ml, (
        f"{key} claims {item.capacity_ml} mL but its outer envelope is only "
        f"{brim_ml:.1f} mL. One of the dimensions or the capacity is wrong."
    )
    assert item.capacity_ml >= 0.25 * brim_ml, (
        f"{key} claims {item.capacity_ml} mL in a {brim_ml:.1f} mL envelope, "
        f"which is implausibly empty. Check the spec sheet."
    )


# --- provenance ----------------------------------------------------------

def test_verified_requires_a_real_source():
    """`verified` means sourced, not confident.

    Those are different claims and only the first is worth anything: a flag
    that tracks confidence is set by the same process that produced the wrong
    number, so it marks bad entries as good. That is precisely how the seed
    catalog ended up with five wrong dimensions all flagged verified=True.
    """
    from labgen.catalog import GLASS

    with pytest.raises(ValueError, match="verified=True requires"):
        CatalogItem(key="ghost", display_name="unsourced", shape="box",
                    collider="box", material=GLASS, mass_kg=1.0,
                    dims={"x": 0.1, "y": 0.1, "z": 0.1}, verified=True)


def test_a_todo_does_not_count_as_a_source():
    """A TODO is the absence of a source, written down."""
    from labgen.catalog import GLASS

    with pytest.raises(ValueError, match="verified=True requires"):
        CatalogItem(key="ghost", display_name="unsourced", shape="box",
                    collider="box", material=GLASS, mass_kg=1.0,
                    dims={"x": 0.1, "y": 0.1, "z": 0.1},
                    source="TODO: look this up", verified=True)


def test_a_real_source_permits_verified():
    from labgen.catalog import GLASS

    item = CatalogItem(key="real", display_name="sourced", shape="box",
                       collider="box", material=GLASS, mass_kg=1.0,
                       dims={"x": 0.1, "y": 0.1, "z": 0.1},
                       source="Corning 1000-250, catalog p.42", verified=True)
    assert item.verified


@pytest.mark.parametrize("key", sorted(CATALOG))
def test_every_entry_says_where_its_numbers_came_from(key: str):
    """Even an unverified entry must name what needs checking."""
    assert CATALOG[key].source.strip(), (
        f"{key} has no `source`. An entry with no provenance cannot be "
        f"audited, and it is indistinguishable from one that was measured."
    )


def test_unverified_items_are_visible():
    """Not a failure -- a standing inventory of what still needs a spec sheet.

    CLAUDE.md: a confidently wrong dimension is worse than a missing entry.
    This test never fails; it exists so `pytest -rP` prints the list.
    """
    unverified = sorted(k for k, v in CATALOG.items() if not v.verified)
    print(f"\nunverified catalog entries ({len(unverified)}): {unverified}")
    assert isinstance(unverified, list)


# --- resolution ----------------------------------------------------------

def test_resolve_exact_key():
    assert catalog.resolve("beaker_250").key == "beaker_250"


def test_resolve_alias_and_display_name():
    assert catalog.resolve("250ml beaker").key == "beaker_250"
    assert catalog.resolve("250 mL Griffin beaker").key == "beaker_250"


def test_resolve_is_case_and_space_insensitive():
    assert catalog.resolve("  GLASS BEAKER  ").key == "beaker_250"


def test_resolve_returns_none_rather_than_guessing():
    """T6 depends on this: a miss must be a miss, not the nearest match."""
    assert catalog.resolve("rotary evaporator") is None
    assert catalog.resolve("") is None


def test_aliases_do_not_collide():
    """Two items claiming the same alias means one silently shadows the other."""
    seen: dict[str, str] = {}
    for key, item in CATALOG.items():
        for alias in item.aliases:
            a = alias.lower()
            assert a not in seen, (
                f"alias {alias!r} claimed by both {seen[a]!r} and {key!r}; "
                f"resolution is order-dependent and therefore wrong."
            )
            seen[a] = key


def test_search_suggests_on_a_miss():
    assert "beaker_250" in catalog.search("beaker")


def test_to_dict_is_json_ready():
    import json

    item: CatalogItem = CATALOG["beaker_250"]
    blob = json.dumps(item.to_dict())
    assert '"glass"' in blob


# --- census --------------------------------------------------------------

def test_collider_census_is_pinned():
    """Pin the shape of the catalog so nothing drops out of iteration silently.

    A catalog entry that quietly stops being enumerated -- by a filter that
    excludes it, a dict comprehension that swallows an exception, a shape kind
    with no builder -- does not fail anywhere. It just stops being in the
    scene, and the first place you notice is a validator in T3 reporting a
    clean bench that is missing an object.

    Update these numbers deliberately when the catalog changes. Do not adjust
    them to make a run go green.
    """
    from collections import Counter

    by_collider = Counter(i.collider for i in CATALOG.values())
    assert dict(by_collider) == {"sdf": 12, "box": 3, "convex_decomposition": 2}
    assert sum(by_collider.values()) == len(CATALOG) == 17


def test_every_item_has_a_builder_and_is_reachable():
    """Every key round-trips through resolution and has a known shape family."""
    for key, item in CATALOG.items():
        assert item.shape in REQUIRED_DIMS, f"{key}: shape {item.shape!r} has no dims contract"
        assert catalog.resolve(key) is item, f"{key} is not reachable by its own key"
