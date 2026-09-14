"""Catalog of standard chemistry lab objects.

The point of this module: a chem lab is almost entirely catalog parts. Rather
than reconstructing a beaker from video (which fails on glass, and gives you no
scale and no colliders), we identify which catalog part it is and instantiate a
parametric template that already has correct dimensions, mass and collision
strategy.

All dimensions in metres, masses in kilograms. Sources are manufacturer spec
sheets for the common borosilicate / lab-supply lines. Anything marked
`unverified` should be checked against the physical object before it is used
for a trust-bearing evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

# Collision strategy per Moonlake's framing: the proxy must be tight enough that
# a bad policy fails for the right reason. An open vessel wrapped in a convex
# hull is sealed shut and nothing can ever be placed inside it, so open vessels
# must use SDF.
ColliderKind = Literal["sdf", "convex_hull", "convex_decomposition", "box", "cylinder"]

# Shape families the mesh generator knows how to build.
ShapeKind = Literal["open_vessel", "conical_vessel", "solid_cylinder", "box", "tube_rack"]


@dataclass(frozen=True)
class PhysicsMaterial:
    name: str
    static_friction: float
    dynamic_friction: float
    restitution: float
    note: str = ""


# Friction numbers are the usual starting values for rigid-body manipulation
# sim, not measured coefficients. Glass-on-steel in reality is nearer 0.2 but
# that makes grasps slip in a way real compliant fingers do not, so the standard
# practice is to run a little high. Flagged because this is exactly the kind of
# unexamined default that quietly moves a success rate.
GLASS = PhysicsMaterial("glass", 0.45, 0.40, 0.05, "tuned for grasp stability, not measured")
PLASTIC = PhysicsMaterial("plastic", 0.55, 0.50, 0.10, "polypropylene labware")
STEEL = PhysicsMaterial("steel", 0.35, 0.30, 0.15, "instrument housings, bench tops")
RUBBER = PhysicsMaterial("rubber", 0.90, 0.85, 0.30, "instrument feet, mats")


@dataclass(frozen=True)
class CatalogItem:
    """A parametric template for one standard lab object."""

    key: str
    display_name: str
    shape: ShapeKind
    collider: ColliderKind
    material: PhysicsMaterial
    mass_kg: float

    # Generic dimension bag. Which keys matter depends on `shape`:
    #   open_vessel      -> outer_d, height, wall, base_d (optional taper)
    #   conical_vessel   -> base_d, neck_d, height, neck_height, wall
    #   solid_cylinder   -> outer_d, height
    #   box              -> x, y, z
    #   tube_rack        -> x, y, z, hole_d, hole_depth, rows, cols
    dims: dict[str, float] = field(default_factory=dict)

    # Named frames on the asset. Task specs reference these instead of raw
    # offsets, so a success predicate survives swapping a 250 mL beaker for a
    # 500 mL one.
    keypoints: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    capacity_ml: float | None = None
    verified: bool = False
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["material"] = self.material.name
        return d


def _beaker(key, name, cap_ml, outer_d, height, mass, aliases=()):
    """Griffin low-form beaker. Rim keypoint is where a gripper closes."""
    return CatalogItem(
        key=key,
        display_name=name,
        shape="open_vessel",
        collider="sdf",
        material=GLASS,
        mass_kg=mass,
        dims={"outer_d": outer_d, "height": height, "wall": 0.0015, "base_d": outer_d},
        keypoints={
            "rim_grasp": (0.0, 0.0, height - 0.012),
            "body_grasp": (0.0, 0.0, height * 0.55),
            "pour_lip": (outer_d / 2, 0.0, height),
            "base_center": (0.0, 0.0, 0.0),
            "fill_point": (0.0, 0.0, height * 0.25),
        },
        capacity_ml=cap_ml,
        verified=True,
        aliases=aliases,
    )


def _vial(key, name, cap_ml, outer_d, height, mass, aliases=()):
    return CatalogItem(
        key=key,
        display_name=name,
        shape="open_vessel",
        collider="sdf",
        material=GLASS,
        mass_kg=mass,
        dims={"outer_d": outer_d, "height": height, "wall": 0.0012, "base_d": outer_d},
        keypoints={
            "neck_grasp": (0.0, 0.0, height - 0.008),
            "body_grasp": (0.0, 0.0, height * 0.5),
            "base_center": (0.0, 0.0, 0.0),
            "fill_point": (0.0, 0.0, height * 0.3),
        },
        capacity_ml=cap_ml,
        verified=True,
        aliases=aliases,
    )


CATALOG: dict[str, CatalogItem] = {}


def _register(item: CatalogItem) -> None:
    CATALOG[item.key] = item


# --- Beakers (Griffin low form, borosilicate 3.3) -------------------------
_register(_beaker("beaker_50", "50 mL Griffin beaker", 50, 0.038, 0.055, 0.025))
_register(_beaker("beaker_100", "100 mL Griffin beaker", 100, 0.050, 0.070, 0.055))
_register(_beaker("beaker_250", "250 mL Griffin beaker", 250, 0.070, 0.095, 0.100,
                  aliases=("beaker", "250ml beaker", "glass beaker")))
_register(_beaker("beaker_500", "500 mL Griffin beaker", 500, 0.085, 0.120, 0.180))
_register(_beaker("beaker_1000", "1 L Griffin beaker", 1000, 0.105, 0.145, 0.320))

# --- Vials ----------------------------------------------------------------
_register(_vial("vial_2ml", "2 mL HPLC vial", 2, 0.012, 0.032, 0.004,
                aliases=("hplc vial", "autosampler vial")))
_register(_vial("vial_20ml", "20 mL scintillation vial", 20, 0.028, 0.061, 0.013,
                aliases=("vial", "scintillation vial")))
_register(_vial("vial_40ml", "40 mL EPA vial", 40, 0.028, 0.095, 0.020))

# --- Erlenmeyer -----------------------------------------------------------
_register(CatalogItem(
    key="erlenmeyer_250",
    display_name="250 mL Erlenmeyer flask",
    shape="conical_vessel",
    collider="sdf",
    material=GLASS,
    mass_kg=0.135,
    dims={"base_d": 0.085, "neck_d": 0.022, "height": 0.140,
          "neck_height": 0.028, "wall": 0.0015},
    keypoints={
        "neck_grasp": (0.0, 0.0, 0.130),
        "base_center": (0.0, 0.0, 0.0),
        "pour_lip": (0.011, 0.0, 0.140),
        "fill_point": (0.0, 0.0, 0.030),
    },
    capacity_ml=250,
    verified=True,
    aliases=("erlenmeyer", "conical flask", "flask"),
))

# --- Graduated cylinder ---------------------------------------------------
_register(CatalogItem(
    key="graduated_cylinder_100",
    display_name="100 mL graduated cylinder",
    shape="open_vessel",
    collider="sdf",
    material=GLASS,
    mass_kg=0.150,
    dims={"outer_d": 0.026, "height": 0.250, "wall": 0.0015, "base_d": 0.065},
    keypoints={
        "body_grasp": (0.0, 0.0, 0.150),
        "rim_grasp": (0.0, 0.0, 0.240),
        "base_center": (0.0, 0.0, 0.0),
        "fill_point": (0.0, 0.0, 0.060),
    },
    capacity_ml=100,
    verified=True,
    aliases=("graduated cylinder", "measuring cylinder"),
))

# --- Test tube ------------------------------------------------------------
_register(CatalogItem(
    key="test_tube_16x100",
    display_name="16 x 100 mm test tube",
    shape="open_vessel",
    collider="sdf",
    material=GLASS,
    mass_kg=0.010,
    dims={"outer_d": 0.016, "height": 0.100, "wall": 0.001, "base_d": 0.016},
    keypoints={
        "neck_grasp": (0.0, 0.0, 0.090),
        "base_center": (0.0, 0.0, 0.0),
        "fill_point": (0.0, 0.0, 0.030),
    },
    capacity_ml=14,
    verified=True,
    aliases=("test tube",),
))

# --- Racks ----------------------------------------------------------------
_register(CatalogItem(
    key="vial_rack_5x10_20ml",
    display_name="50-position 20 mL vial rack",
    shape="tube_rack",
    collider="convex_decomposition",
    material=PLASTIC,
    mass_kg=0.210,
    dims={"x": 0.240, "y": 0.120, "z": 0.030,
          "hole_d": 0.030, "hole_depth": 0.022, "rows": 5, "cols": 10},
    keypoints={"base_center": (0.0, 0.0, 0.0)},
    verified=False,
    aliases=("vial rack", "rack"),
))

_register(CatalogItem(
    key="test_tube_rack_6x12",
    display_name="72-position test tube rack",
    shape="tube_rack",
    collider="convex_decomposition",
    material=PLASTIC,
    mass_kg=0.180,
    dims={"x": 0.230, "y": 0.110, "z": 0.028,
          "hole_d": 0.018, "hole_depth": 0.020, "rows": 6, "cols": 12},
    keypoints={"base_center": (0.0, 0.0, 0.0)},
    verified=False,
    aliases=("test tube rack",),
))

# --- Instruments ----------------------------------------------------------
_register(CatalogItem(
    key="hotplate_stirrer",
    display_name="IKA RCT-class hotplate stirrer",
    shape="box",
    collider="box",
    material=STEEL,
    mass_kg=2.60,
    dims={"x": 0.160, "y": 0.270, "z": 0.105},
    keypoints={
        "plate_center": (0.0, -0.045, 0.105),
        "base_center": (0.0, 0.0, 0.0),
    },
    verified=False,
    aliases=("hotplate", "hot plate", "stirrer", "ika"),
))

_register(CatalogItem(
    key="analytical_balance",
    display_name="Analytical balance",
    shape="box",
    collider="box",
    material=STEEL,
    mass_kg=4.50,
    dims={"x": 0.210, "y": 0.320, "z": 0.090},
    keypoints={
        "pan_center": (0.0, -0.060, 0.090),
        "base_center": (0.0, 0.0, 0.0),
    },
    verified=False,
    aliases=("balance", "scale", "weighing scale"),
))

_register(CatalogItem(
    key="petri_dish_100",
    display_name="100 mm petri dish",
    shape="open_vessel",
    collider="sdf",
    material=PLASTIC,
    mass_kg=0.016,
    dims={"outer_d": 0.100, "height": 0.015, "wall": 0.0012, "base_d": 0.100},
    keypoints={"rim_grasp": (0.0, 0.0, 0.012), "base_center": (0.0, 0.0, 0.0)},
    verified=True,
    aliases=("petri dish", "petri"),
))

# --- Fixtures -------------------------------------------------------------
_register(CatalogItem(
    key="bench_top",
    display_name="Lab bench top",
    shape="box",
    collider="box",
    material=STEEL,
    mass_kg=0.0,  # static
    dims={"x": 1.500, "y": 0.750, "z": 0.040},
    keypoints={"surface_center": (0.0, 0.0, 0.040)},
    verified=False,
    aliases=("bench", "table", "benchtop", "tabletop"),
))


_ALIAS_INDEX: dict[str, str] = {}
for _k, _item in CATALOG.items():
    _ALIAS_INDEX[_k.lower()] = _k
    _ALIAS_INDEX[_item.display_name.lower()] = _k
    for _a in _item.aliases:
        _ALIAS_INDEX[_a.lower()] = _k


def resolve(name: str) -> CatalogItem | None:
    """Map a free-text label (from a VLM, or a human) onto a catalog item."""
    key = _ALIAS_INDEX.get(name.strip().lower())
    return CATALOG[key] if key else None


def search(name: str) -> list[str]:
    """Loose substring match, for suggesting alternatives on a miss."""
    q = name.strip().lower()
    return sorted({k for alias, k in _ALIAS_INDEX.items() if q in alias or alias in q})
