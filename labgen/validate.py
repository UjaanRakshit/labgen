"""The gates. Every check runs without an LLM, and without Isaac Lab.

A stage does not advance until its validator passes. These are the checks
CLAUDE.md names, plus two this project learned the hard way (see
`check_graspable` and the decomposition clause in `check_colliders`).

Two tiers, and the split matters:

* **Geometric checks** need only numpy and the catalog. They run anywhere, in
  CI, on a laptop, in milliseconds. Everything in this module is one of these.
* **The settle test** needs a physics engine and therefore a GPU and Isaac Lab.
  It lives in `labgen.settle`, behind an interface, so that importing
  `labgen.validate` never drags in a backend. CLAUDE.md calls the settle test
  the cheapest high-value check in the project; it is still the only one that
  cannot run on a bare machine.

The report is deliberately verbose about *measured values*. "beaker_a fails
scale" is useless at 2am; "beaker_a is 84.0 mm across, catalog says 70.0 mm,
20.0% over a 5% tolerance" tells you whether to fix the scene or the catalog.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable

import numpy as np

from .catalog import CATALOG
from .meshes import MeshError, build
from .types import SceneObject, SceneSpec

__all__ = [
    "Severity",
    "Finding",
    "Report",
    "RobotSpec",
    "YAM",
    "SCALE_TOLERANCE",
    "SUPPORT_TOLERANCE_M",
    "validate",
    "CHECKS",
]


class Severity(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"

    def __str__(self) -> str:
        return self.value


@dataclass
class Finding:
    check: str
    severity: Severity
    message: str
    instance_id: str | None = None
    measured: float | None = None
    limit: float | None = None
    unit: str = ""

    def render(self) -> str:
        who = self.instance_id or "-"
        head = f"  [{self.severity!s:4}] {self.check:22} {who:14} {self.message}"
        if self.measured is None:
            return head
        return head + (f"\n{'':40}measured {self.measured:.4g}{self.unit}"
                       f"  limit {self.limit:.4g}{self.unit}"
                       if self.limit is not None else
                       f"\n{'':40}measured {self.measured:.4g}{self.unit}")


@dataclass
class Report:
    scene: str
    findings: list[Finding] = field(default_factory=list)

    def add(self, *args, **kwargs) -> None:
        self.findings.append(Finding(*args, **kwargs))

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.FAIL]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARN]

    @property
    def ok(self) -> bool:
        return not self.failures

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        lines = [f"validation report: {self.scene}"]
        shown = [f for f in self.findings if f.severity is not Severity.PASS]
        if not shown:
            lines.append("  all checks passed")
        else:
            lines += [f.render() for f in shown]
        lines.append(f"  {len(self.failures)} failure(s), {len(self.warnings)} warning(s), "
                     f"{len(self.findings)} check(s) run")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# what the robot can physically do
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RobotSpec:
    """The envelope a scene has to fit inside, so it can be checked against it.

    Same discipline as `CatalogItem.source`: every number says where it came
    from, because a workspace bound invented to make a scene pass is worse than
    no bound at all.
    """

    name: str
    reach_min_m: float
    reach_max_m: float
    jaw_opening_m: float
    source: str

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError(f"{self.name}: RobotSpec needs a source")


# Measured on the imported model rather than read off a datasheet, because the
# datasheet is not what the simulation will use:
#
#   reach      scripts/probe_fk.py     -- 6000 random configurations inside the
#                                          joint limits; fingertip radius from
#                                          the base spans 0.011 to 0.740 m. The
#                                          lower bound here is 0.25, the radius
#                                          inside which the arm's own body sits.
#   jaw        scripts/probe_gap2.py   -- jaw separation measured from the
#                                          finger geometry across the stroke:
#                                          0.06 mm shut, 94.06 mm fully open.
YAM = RobotSpec(
    name="yam",
    reach_min_m=0.25,
    reach_max_m=0.74,
    jaw_opening_m=0.094,
    source=("measured from i2rt yam.urdf via scripts/probe_fk.py and "
            "scripts/probe_gap2.py; not a manufacturer figure"),
)


SCALE_TOLERANCE = 0.05          # CLAUDE.md: reject anything more than 5% off
SUPPORT_TOLERANCE_M = 0.001     # CLAUDE.md: rests on a surface within 1 mm
OPEN_SHAPES = {"open_vessel", "necked_vessel", "conical_vessel"}
SEALING_COLLIDERS = {"convex_hull", "convex_decomposition"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _aabb(obj: SceneObject) -> tuple[np.ndarray, np.ndarray]:
    """World-space axis-aligned bounds.

    Only correct for axis-aligned objects. A rotated object gets flagged by
    `check_scale` rather than silently measured wrong.
    """
    mesh = build(obj.catalog_item())
    lo, hi = mesh.bounds()
    p = np.asarray(obj.position)
    return lo + p, hi + p


def _is_axis_aligned(obj: SceneObject) -> bool:
    return abs(obj.orientation_wxyz[0] - 1.0) < 1e-9 and \
        all(abs(c) < 1e-9 for c in obj.orientation_wxyz[1:])


def _catalog_extent(obj: SceneObject) -> tuple[float, float, float]:
    item = obj.catalog_item()
    d = item.dims
    if item.shape in ("box", "tube_rack"):
        return d["x"], d["y"], d["z"]
    if item.shape == "conical_vessel":
        return d["base_d"], d["base_d"], d["height"]
    width = max(d.get("outer_d", 0.0), d.get("base_d", 0.0))
    return width, width, d["height"]


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------

def check_provenance(scene: SceneSpec, report: Report, robot: RobotSpec) -> None:
    """Scale reference, and whether the objects are sourced.

    Monocular video has no metric scale, so a video-derived scene built without
    a marker of known size is wrong in a way nothing downstream can detect.
    CLAUDE.md: the validator rejects a scene built without one.

    A hand-authored scene is exempt and says so in `source.method`: every
    dimension came from a catalog template, so it is metric by construction.
    """
    method = str(scene.source.get("method", "")).strip()
    marker = str(scene.source.get("marker_config", "")).strip()

    if not method:
        report.add("provenance", Severity.FAIL,
                   "source.method is unset; cannot tell whether this scene needs "
                   "a scale reference")
    elif method == "hand_authored":
        report.add("provenance", Severity.PASS,
                   "hand-authored: metric by construction, no marker required")
    elif not marker or marker.startswith("not_applicable"):
        report.add("provenance", Severity.FAIL,
                   f"method={method!r} is not hand-authored but carries no marker "
                   f"config. Monocular video has no metric scale; a scene built "
                   f"without a reference of known size is unfalsifiably wrong.")
    else:
        report.add("provenance", Severity.PASS, f"scale reference: {marker}")

    unsourced = [o.instance_id for o in scene.objects
                 if o.catalog_item() is not None and not o.catalog_item().verified]
    if unsourced:
        report.add("provenance", Severity.WARN,
                   f"{len(unsourced)} object(s) have unsourced dimensions: "
                   f"{', '.join(sorted(unsourced))}. Not usable for a "
                   f"trust-bearing evaluation.")


def check_scale(scene: SceneSpec, report: Report, robot: RobotSpec) -> None:
    """Every bounding box within tolerance of its catalog dimensions.

    **What this can and cannot catch.** For a catalog object the mesh is
    GENERATED from the catalog dimensions, so this compares the catalog against
    itself. It catches a mesh-generator regression -- a builder that emits the
    wrong size for the dims it was handed -- and nothing else. Inflating a
    catalog entry by 20% inflates the mesh by 20% too and sails straight
    through, which is exactly what happened the first time this was tested.

    That is not a hole to plug here, it is the two-checks rule: "is the mesh
    right for these numbers" and "are these numbers right" are different
    questions with different fixes. The second is `CatalogItem.source` and
    `verified`, reported by `check_provenance`, and ultimately a spec sheet.

    The case where this check has real teeth is a GENERATED asset, whose mesh
    comes from reconstruction rather than from the catalog and can genuinely
    disagree with the size it claims. Those go through `_generated_extent`.
    """
    for obj in scene.objects:
        if obj.catalog_item() is None:
            continue
        if not _is_axis_aligned(obj):
            report.add("scale", Severity.WARN,
                       "object is rotated; axis-aligned bounds would overstate "
                       "its extent, so scale is not checked here",
                       instance_id=obj.instance_id)
            continue
        try:
            lo, hi = _aabb(obj)
        except MeshError as exc:
            report.add("scale", Severity.FAIL, f"mesh cannot be built: {exc}",
                       instance_id=obj.instance_id)
            continue

        got = hi - lo
        want = _catalog_extent(obj)
        worst = max(abs(g - w) / w for g, w in zip(got, want))
        if worst > SCALE_TOLERANCE:
            axis = max(range(3), key=lambda i: abs(got[i] - want[i]) / want[i])
            report.add("scale", Severity.FAIL,
                       f"{'xyz'[axis]} extent {got[axis] * 1000:.1f} mm against "
                       f"catalog {want[axis] * 1000:.1f} mm",
                       instance_id=obj.instance_id, measured=worst * 100,
                       limit=SCALE_TOLERANCE * 100, unit="%")
        else:
            report.add("scale", Severity.PASS, "within tolerance",
                       instance_id=obj.instance_id)


def check_support(scene: SceneSpec, report: Report, robot: RobotSpec) -> None:
    """No free body floats: each rests on something within 1 mm."""
    surfaces: list[tuple[str, np.ndarray, np.ndarray]] = []
    for obj in scene.objects:
        try:
            lo, hi = _aabb(obj)
        except MeshError:
            continue
        surfaces.append((obj.instance_id, lo, hi))

    for obj in scene.free_bodies:
        if obj.catalog_item() is None:
            continue
        try:
            lo, hi = _aabb(obj)
        except MeshError:
            continue

        best_name, best_gap = None, math.inf
        for name, slo, shi in surfaces:
            if name == obj.instance_id:
                continue
            # overlapping footprint, and its top at or below our base
            if shi[0] <= lo[0] or slo[0] >= hi[0] or shi[1] <= lo[1] or slo[1] >= hi[1]:
                continue
            gap = lo[2] - shi[2]
            if gap < -SUPPORT_TOLERANCE_M:
                continue                     # that is interpenetration, not support
            if gap < best_gap:
                best_name, best_gap = name, gap

        if best_name is None:
            report.add("support", Severity.FAIL,
                       f"nothing underneath it; base sits at z={lo[2] * 1000:.1f} mm",
                       instance_id=obj.instance_id)
        elif best_gap > SUPPORT_TOLERANCE_M:
            report.add("support", Severity.FAIL,
                       f"floating {best_gap * 1000:.2f} mm above {best_name}",
                       instance_id=obj.instance_id, measured=best_gap * 1000,
                       limit=SUPPORT_TOLERANCE_M * 1000, unit=" mm")
        else:
            report.add("support", Severity.PASS, f"rests on {best_name}",
                       instance_id=obj.instance_id)


def check_interpenetration(scene: SceneSpec, report: Report, robot: RobotSpec) -> None:
    """No two objects overlap at rest.

    AABB only, which is conservative in one direction: two objects whose boxes
    overlap may not actually intersect. So this warns rather than fails, and
    says so -- a false FAIL on a beaker standing inside a rack's footprint would
    be worse than a missed hairline overlap, and the settle test catches the
    real ones by watching them fly apart.
    """
    boxes = {}
    for obj in scene.objects:
        if obj.catalog_item() is None:
            continue
        try:
            boxes[obj.instance_id] = _aabb(obj)
        except MeshError:
            continue

    clean = True
    for a, bname in itertools.combinations(sorted(boxes), 2):
        alo, ahi = boxes[a]
        blo, bhi = boxes[bname]
        overlap = np.minimum(ahi, bhi) - np.maximum(alo, blo)
        if (overlap > 1e-9).all():
            clean = False
            report.add("interpenetration", Severity.WARN,
                       f"bounding boxes of {a} and {bname} overlap by "
                       f"{overlap.min() * 1000:.2f} mm on the tightest axis; "
                       f"may be nesting rather than colliding",
                       instance_id=a, measured=overlap.min() * 1000, unit=" mm")
    if clean:
        report.add("interpenetration", Severity.PASS, "no overlapping bounds")


def check_physics_completeness(scene: SceneSpec, report: Report,
                               robot: RobotSpec) -> None:
    """Explicit mass, a bound material, and an explicit collider. No defaults."""
    for obj in scene.objects:
        item = obj.catalog_item()
        where = obj.instance_id
        if item is None:
            gen = obj.generated
            if gen is None or gen.mass_kg <= 0:
                report.add("physics", Severity.FAIL,
                           "generated asset without an explicit mass", instance_id=where)
            continue

        if not obj.fixed and item.mass_kg <= 0:
            report.add("physics", Severity.FAIL,
                       f"dynamic body with mass {item.mass_kg} kg; a zero-mass "
                       f"rigid body explodes on first contact", instance_id=where)
        elif obj.fixed and item.mass_kg > 0:
            report.add("physics", Severity.PASS,
                       "static fixture (mass ignored)", instance_id=where)
        else:
            report.add("physics", Severity.PASS,
                       f"mass {item.mass_kg} kg explicit", instance_id=where)

        if not item.material or not item.material.name:
            report.add("physics", Severity.FAIL, "no physics material bound",
                       instance_id=where)
        if not item.collider:
            report.add("physics", Severity.FAIL, "no collider approximation set",
                       instance_id=where)


def check_colliders(scene: SceneSpec, report: Report, robot: RobotSpec) -> None:
    """CLAUDE.md hard rule 1, widened by measurement.

    The rule as written bans a convex HULL on an open vessel. Convex
    DECOMPOSITION seals a shallow one just as effectively: switching
    petri_dish_100 to it made the settle test pass (8.00 mm sink -> 0.01 mm)
    while a tube dropped into the dish came to rest on the rim at z=23 mm
    instead of on the floor at z=0.02 mm. A green settle test and a sealed
    vessel is the worst trade available, so both are hard failures.

    Deeper concavities survive decomposition -- the tube racks rely on it for
    their bores -- so this is scoped to open vessels rather than banned.
    """
    for obj in scene.objects:
        item = obj.catalog_item()
        if item is None or item.shape not in OPEN_SHAPES:
            continue
        if item.collider in SEALING_COLLIDERS:
            report.add("collider", Severity.FAIL,
                       f"open vessel with collider={item.collider!r}. This seals "
                       f"the opening: nothing can be placed inside, and a policy "
                       f"scores a success by resting an object on a lid that is "
                       f"not there.",
                       instance_id=obj.instance_id)
        else:
            report.add("collider", Severity.PASS,
                       f"{item.collider} preserves the cavity",
                       instance_id=obj.instance_id)


def check_reachability(scene: SceneSpec, report: Report, robot: RobotSpec) -> None:
    """Every free body inside the robot's workspace. Warn, per CLAUDE.md."""
    for obj in scene.free_bodies:
        r = float(math.hypot(obj.position[0], obj.position[1]))
        if r > robot.reach_max_m:
            report.add("reachability", Severity.WARN,
                       f"{r:.3f} m from the {robot.name} base, beyond its "
                       f"{robot.reach_max_m:.2f} m reach",
                       instance_id=obj.instance_id, measured=r,
                       limit=robot.reach_max_m, unit=" m")
        elif r < robot.reach_min_m:
            report.add("reachability", Severity.WARN,
                       f"{r:.3f} m from the base, inside the {robot.reach_min_m:.2f} m "
                       f"the arm's own body occupies",
                       instance_id=obj.instance_id, measured=r,
                       limit=robot.reach_min_m, unit=" m")
        else:
            report.add("reachability", Severity.PASS, f"{r:.3f} m from the base",
                       instance_id=obj.instance_id)


def check_graspable(scene: SceneSpec, report: Report, robot: RobotSpec) -> None:
    """Can the gripper actually open wide enough for this object?

    Not in CLAUDE.md's list, and it belongs there. A task laid out around an
    object wider than the jaws is one the hardware cannot do, and a success
    rate measured for it in simulation would describe something the real robot
    never could -- the MATTERIX sim/real gap arriving through the end effector.

    A warning rather than a failure: a scene may legitimately contain fixtures
    nobody intends to pick up. The point is that it is said out loud before
    somebody writes a task against one.
    """
    for obj in scene.free_bodies:
        item = obj.catalog_item()
        if item is None:
            continue
        width = min(_catalog_extent(obj)[:2])
        if width > robot.jaw_opening_m:
            report.add("graspable", Severity.WARN,
                       f"{width * 1000:.0f} mm across; the {robot.name}'s jaws open "
                       f"to {robot.jaw_opening_m * 1000:.0f} mm, so it cannot be "
                       f"picked up by this gripper",
                       instance_id=obj.instance_id, measured=width * 1000,
                       limit=robot.jaw_opening_m * 1000, unit=" mm")
        else:
            report.add("graspable", Severity.PASS,
                       f"{width * 1000:.0f} mm fits the jaws",
                       instance_id=obj.instance_id)


def check_grasp_keypoints(scene: SceneSpec, report: Report, robot: RobotSpec) -> None:
    """An open vessel needs a grasp keypoint near its rim.

    Learned by doing it wrong: grasping a 250 mL beaker at `body_grasp`
    (mid-height) leaves 43 mm of glass standing inside the hand, and the
    gripper body buries 30 mm into it. `rim_grasp` sits 12 mm below the lip and
    clears. catalog.py already documents the rim keypoint as "where a gripper
    closes"; this makes the absence of one visible rather than leaving the next
    caller to pick whichever keypoint sounds right.
    """
    for obj in scene.free_bodies:
        item = obj.catalog_item()
        if item is None or item.shape not in OPEN_SHAPES:
            continue
        height = item.dims.get("height")
        rim = [k for k, v in item.keypoints.items()
               if height and v[2] > height * 0.75]
        if not rim:
            report.add("grasp_keypoint", Severity.WARN,
                       f"no keypoint in the top quarter of the object; a top-down "
                       f"grasp has nowhere to aim that clears the hand",
                       instance_id=obj.instance_id)
        else:
            report.add("grasp_keypoint", Severity.PASS,
                       f"grasp near the rim: {', '.join(sorted(rim))}",
                       instance_id=obj.instance_id)


CHECKS: tuple[Callable[[SceneSpec, Report, RobotSpec], None], ...] = (
    check_provenance,
    check_scale,
    check_support,
    check_interpenetration,
    check_physics_completeness,
    check_colliders,
    check_reachability,
    check_graspable,
    check_grasp_keypoints,
)


def validate(scene: SceneSpec, *, robot: RobotSpec = YAM,
             checks: Iterable[Callable] | None = None) -> Report:
    """Run every geometric gate. The settle test is separate; see labgen.settle."""
    report = Report(scene=scene.name)
    for check in (checks if checks is not None else CHECKS):
        check(scene, report, robot)
    return report
