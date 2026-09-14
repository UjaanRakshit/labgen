"""SceneSpec -> .usda with UsdPhysics schemas.

Text USD, written directly. `pxr` is not required and is not imported: the
whole point of keeping labgen core import-clean is that the validators run in
CI without a GPU or an Isaac install, and a USD emitter that needed USD to run
would undo that. Text `.usda` is a stable, specified format, and a human can
read the diff when a scene changes -- which matters, because `SceneSpec` is
meant to be hand-edited.

What every object gets:

* `UsdGeomXform` at its pose, with a `UsdGeomMesh` child
* `PhysicsRigidBodyAPI` if dynamic, nothing if it is a static fixture
* `PhysicsMassAPI` with an explicit `mass`, never density-inferred
* `PhysicsCollisionAPI` + `PhysicsMeshCollisionAPI` with `approximation` set
  explicitly from the catalog's collider choice
* a bound `PhysicsMaterialAPI` from the catalog material
* keypoints as custom attributes, so a task spec can say `rim_grasp` instead of
  an offset and survive a 250 mL beaker being swapped for a 500 mL one

Stage metadata is `metersPerUnit = 1`, `upAxis = "Z"`, `defaultPrim = "World"`.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from .catalog import PhysicsMaterial
from .meshes import Mesh, build, build_collision
from .types import SceneObject, SceneSpec, SpecError

__all__ = ["UsdaError", "emit", "write_scene", "MATERIAL_SCOPE", "APPROXIMATION"]


class UsdaError(SpecError):
    """A SceneSpec cannot be expressed as valid USD physics."""


MATERIAL_SCOPE = "/World/PhysicsMaterials"

# labgen's collider vocabulary -> the token UsdPhysicsMeshCollisionAPI expects.
#
# If a backend ever rejects the `sdf` token, the correct fallback is `none`
# (use the triangles as they are), never `convexHull`. See HULL_IS_A_LIE.
APPROXIMATION: dict[str, str] = {
    # "none" means "collide against the triangles as authored". It is what an
    # SDF collider wants, because the SDF is baked FROM those triangles.
    #
    # There is no `sdf` approximation token, and writing one does nothing. This
    # emitter used to, and the result was silent: Newton's
    # approximation_to_remeshing_method map has no "sdf" key, so the lookup
    # returned None, every vessel came back with `sdf = None`, and the scene
    # simulated against raw triangle soup while the stage looked correct in
    # every structural check. SDF is requested by applying an API schema
    # instead -- see SDF_API below.
    "sdf": "none",
    "convex_hull": "convexHull",
    "convex_decomposition": "convexDecomposition",
    "box": "boundingCube",
}

# Newton's canonical opt-in: "Applying NewtonSDFCollisionAPI is the canonical
# signal that SDF generation is configured for this shape" (newton's own
# import_usd.py). It must be paired with approximation="none" -- Newton warns
# if a non-none approximation is co-authored with it.
#
# This is a Newton-specific schema, and CLAUDE.md says not to put backend
# specifics in labgen core. The rule is about *Python APIs*: a USD stage has to
# express SDF somehow, and there is no backend-neutral way to say it. The
# arrangement here is the safe one -- a backend that does not know this schema
# ignores it and falls back to approximation="none", i.e. the full triangle
# mesh. That is slower but geometrically correct. It never degrades to a convex
# hull, which would seal every open vessel in the catalog.
SDF_API = "NewtonSDFCollisionAPI"

# How many voxels we insist on across the thinnest wall in the object.
#
# The default is newton:sdfMaxResolution = 64 over the longest axis, which for
# test_tube_16x100 is 100 mm / 64 = 1.56 mm voxels against a 1.0 mm wall: the
# wall is thinner than a single voxel and the cavity does not survive
# rasterisation. Deriving the voxel size from the wall instead of accepting a
# resolution makes the setting a property of the object rather than of the
# default, which is the whole premise of the catalog.
SDF_VOXELS_ACROSS_WALL = 3.0

# A dense SDF is (extent / voxel)^3 samples. Refusing to author a grid bigger
# than this is a guard against a thin-walled, large-extent object silently
# asking for gigabytes.
SDF_MAX_VOXELS = 40_000_000

# CLAUDE.md hard rule 1, restated at the one place it could actually be
# violated. A convex hull over a beaker seals the opening: nothing can be put
# inside it, and a policy scores a success by resting an object on a lid that
# does not exist.
HULL_IS_A_LIE = (
    "refusing to write a convex-hull collider for an open vessel. A hull seals "
    "the opening, so nothing can ever be placed inside, and a policy scores a "
    "success by resting an object on a lid that is not there."
)

OPEN_SHAPES = {"open_vessel", "necked_vessel", "conical_vessel"}

PAD = "    "


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def _fmt(value: float) -> str:
    """Float that round-trips and stays readable in a diff."""
    return f"{value:.9g}"


def _vec3(v) -> str:
    return f"({_fmt(v[0])}, {_fmt(v[1])}, {_fmt(v[2])})"


def _quat(q) -> str:
    """A USD `quatd` literal: a FLAT 4-tuple, real part first.

    Not `(w, (x, y, z))`. That nested form is how the C++ Gf constructor reads,
    and it is what this emitter wrote until a real `pxr` parse rejected it with
    "Tuple nesting too deep". The whole file failed to load, and the project's
    own structural tests did not notice, because they were written against the
    same wrong assumption as the emitter.
    """
    return f"({_fmt(q[0])}, {_fmt(q[1])}, {_fmt(q[2])}, {_fmt(q[3])})"


def _escape(text: str) -> str:
    """Escape for a USD double-quoted string literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _safe(name: str) -> str:
    """A USD prim name: alphanumeric and underscore, not starting with a digit."""
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
    if not out or out[0].isdigit():
        out = f"_{out}"
    return out


def _indent(lines: list[str], depth: int) -> list[str]:
    return [(PAD * depth + ln) if ln.strip() else "" for ln in lines]


def _mesh_points(mesh: Mesh) -> str:
    return ", ".join(_vec3(v) for v in mesh.vertices)


def _material_prim(material: PhysicsMaterial) -> str:
    return f"{MATERIAL_SCOPE}/{_safe(material.name)}"


def _approximation_for(obj: SceneObject) -> str:
    collider = obj.collider()
    item = obj.catalog_item()
    shape = item.shape if item else None

    if collider == "convex_hull" and shape in OPEN_SHAPES:
        raise UsdaError(f"{obj.instance_id} ({item.key if item else '?'}): {HULL_IS_A_LIE}")

    if collider == "cylinder":
        # USD has no cylinder token for a mesh collider. boundingSphere is not
        # a cylinder, and substituting it would be wrong in a way that is hard
        # to see in a render, so refuse rather than approximate quietly.
        raise UsdaError(
            f"{obj.instance_id}: collider='cylinder' has no faithful "
            f"UsdPhysicsMeshCollisionAPI token. Use a UsdGeomCylinder prim with "
            f"a primitive collider, or change the catalog entry to 'sdf'."
        )

    try:
        return APPROXIMATION[collider]
    except KeyError:
        raise UsdaError(f"{obj.instance_id}: unknown collider {collider!r}") from None


def sdf_voxel_size(item, mesh: Mesh) -> float:
    """Voxel size for an SDF collider, derived from the object's own wall.

    Returns metres. Sized so `SDF_VOXELS_ACROSS_WALL` voxels span the thinnest
    wall, because an SDF coarser than the wall closes the cavity: the inside
    and outside surfaces land in the same voxel and the vessel rasterises
    solid. That is the same failure as a convex hull, arrived at numerically
    instead of structurally, and it is just as invisible in a render.
    """
    wall = item.dims.get("wall")
    if wall is None:
        raise UsdaError(
            f"{item.key}: SDF collider on a shape with no `wall` dimension, so "
            f"there is no thickness to resolve against. Set one, or choose a "
            f"different collider -- do not fall back to the default resolution."
        )

    voxel = wall / SDF_VOXELS_ACROSS_WALL
    extent = mesh.extents()
    voxels = float((extent / voxel).prod())
    if voxels > SDF_MAX_VOXELS:
        raise UsdaError(
            f"{item.key}: resolving a {wall * 1000:.1f} mm wall at "
            f"{SDF_VOXELS_ACROSS_WALL:g} voxels across needs a "
            f"{voxel * 1000:.3f} mm grid over a "
            f"{extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f} m extent, "
            f"which is {voxels / 1e6:.0f}M voxels. Refusing to author it. "
            f"Either the wall or the extent is wrong."
        )
    return voxel


# --------------------------------------------------------------------------
# emission
#
# Built line by line at an explicit depth, rather than by nesting
# textwrap.dedent blocks. USD does not care about whitespace, but this file is
# meant to be read in a diff when a scene changes by hand, and nested dedent
# produces output that is technically valid and practically unreadable.
# --------------------------------------------------------------------------

def _emit_material(m: PhysicsMaterial) -> list[str]:
    out = [
        f'def Material "{_safe(m.name)}" (',
        '    prepend apiSchemas = ["PhysicsMaterialAPI"]',
        ")",
        "{",
    ]
    if m.note:
        out.append(f"    # {m.note}")
    out += [
        f"    float physics:staticFriction = {_fmt(m.static_friction)}",
        f"    float physics:dynamicFriction = {_fmt(m.dynamic_friction)}",
        f"    float physics:restitution = {_fmt(m.restitution)}",
        "}",
    ]
    return out


def _emit_mesh_prim(name: str, mesh: Mesh, purpose: str = "default",
                    collider: tuple[str, PhysicsMaterial] | None = None,
                    sdf_voxel: float | None = None) -> list[str]:
    """A UsdGeomMesh, optionally carrying the collision schemas.

    The collision APIs belong **here**, on the Gprim, not on the parent Xform.
    UsdPhysicsCollisionAPI applies to a UsdGeomGprim; applied to an Xform it is
    silently not a collider. Newton says so out loud --

        CollisionAPI applied to an unknown UsdGeomGPrim type, prim /World/beaker

    -- and then builds a shape from the child mesh using its own defaults. The
    effect is that `physics:approximation` is read by nobody: every vessel came
    back with `sdf = None` and friction 1.0 instead of glass's 0.45, while the
    stage looked completely correct in every structural check.

    PhysicsRigidBodyAPI and PhysicsMassAPI stay on the Xform, which is right --
    the body is the Xform, the collider is the geometry under it.
    """
    counts = ", ".join("3" for _ in mesh.faces)
    indices = ", ".join(str(int(i)) for f in mesh.faces for i in f)

    header = [f'def Mesh "{name}"']
    if collider is not None:
        schemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"]
        if sdf_voxel is not None:
            schemas.append(SDF_API)
        header += [
            "(",
            f'    prepend apiSchemas = [{", ".join(chr(34) + s + chr(34) for s in schemas)}]',
            ")",
        ]
    out = header + [
        "{",
        f"    point3f[] points = [{_mesh_points(mesh)}]",
        f"    int[] faceVertexCounts = [{counts}]",
        f"    int[] faceVertexIndices = [{indices}]",
        '    uniform token subdivisionScheme = "none"',
    ]
    if purpose != "default":
        out.append(f'    uniform token purpose = "{purpose}"')
    if collider is not None:
        approximation, material = collider
        out += [
            "",
            f'    uniform token physics:approximation = "{approximation}"',
            "    bool physics:collisionEnabled = 1",
            # MaterialBindingAPI is applied above. Without it USD warns
            # "Found material bindings ... but MaterialBindingAPI is not
            # applied" and the binding is ignored, which is how a bench of
            # glassware ended up simulating at Newton's default friction.
            f"    rel material:binding:physics = <{_material_prim(material)}>",
        ]
        if sdf_voxel is not None:
            out += [
                "",
                f"    # SDF sized to this object's own wall, not to the default",
                f"    # resolution. An SDF coarser than the wall rasterises the",
                f"    # vessel solid and seals the cavity.",
                f"    float newton:sdfTargetVoxelSize = {_fmt(sdf_voxel)}",
            ]
    out.append("}")
    return out


def _emit_keypoints(obj: SceneObject) -> list[str]:
    item = obj.catalog_item()
    if not item or not item.keypoints:
        return []
    out = [
        "# Keypoints in the object's own frame, metres. Task specs reference",
        "# these by name, so a predicate survives swapping a 250 mL beaker for",
        "# a 500 mL one.",
    ]
    for kp_name, offset in sorted(item.keypoints.items()):
        out.append(f"custom point3f keypoints:{_safe(kp_name)} = {_vec3(offset)}")
    return out


def _emit_object(obj: SceneObject) -> list[str]:
    item = obj.catalog_item()
    name = _safe(obj.instance_id)

    if item is None:
        gen = obj.generated
        assert gen is not None                      # enforced by SceneObject
        raise UsdaError(
            f"{obj.instance_id}: mesh-generated assets are not emitted yet. "
            f"T1 builds catalog templates; a generated asset needs its OBJ "
            f"loaded from {gen.mesh_path!r} and referenced, which is T6 work."
        )

    mesh = build(item)
    collision = build_collision(item)
    approximation = _approximation_for(obj)
    mass = obj.mass_kg()

    # The Xform is the *body*; the Mesh under it is the *collider*. Collision
    # and material-binding schemas go on the Mesh (see _emit_mesh_prim). A
    # static fixture is a collider with no body, so its Xform carries nothing.
    schemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"] if not obj.fixed else []
    schema_list = ", ".join('"' + s + '"' for s in schemas)

    sourced = "yes" if item.verified else "NO -- dimensions unverified"
    out = [f'def Xform "{name}"']
    if schemas:
        out += ["(", f"    prepend apiSchemas = [{schema_list}]", ")"]
    out += [
        "{",
        f"    # {item.display_name}",
        f"    # provenance={obj.provenance}  confidence={obj.confidence:.2f}  sourced={sourced}",
    ]
    if obj.note:
        for line in textwrap.wrap(obj.note, 72):
            out.append(f"    # {line}")

    out += [
        "",
        "    # Pose in the robot base frame, metres.",
        f"    double3 xformOp:translate = {_vec3(obj.position)}",
        f"    quatd xformOp:orient = {_quat(obj.orientation_wxyz)}",
        '    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]',
        "",
    ]

    if obj.fixed:
        out.append("    # Static fixture: collider only, no rigid body, no mass.")
    else:
        if mass <= 0.0:
            raise UsdaError(
                f"{obj.instance_id}: dynamic body with mass {mass} kg. A "
                f"zero-mass rigid body explodes on the first contact. "
                f"CLAUDE.md rule 4: no silent defaults for physics properties."
            )
        out += [
            "    # Mass is explicit, never density-inferred (CLAUDE.md rule 4).",
            f"    float physics:mass = {_fmt(mass)}",
            "    bool physics:rigidBodyEnabled = 1",
            "    bool physics:kinematicEnabled = 0",
        ]

    voxel = sdf_voxel_size(item, mesh) if obj.collider() == "sdf" else None

    out.append("")
    out += _indent(_emit_keypoints(obj), 1)
    out.append("")
    out += _indent(
        _emit_mesh_prim("geom", mesh, collider=(approximation, item.material),
                        sdf_voxel=voxel), 1)
    if collision is not None:
        out.append("")
        out += _indent(_emit_mesh_prim("collision", collision, purpose="guide"), 1)
    out.append("}")
    return out


def emit(scene: SceneSpec) -> str:
    """Render a SceneSpec as a text `.usda` stage."""
    if not scene.objects:
        raise UsdaError(f"scene {scene.name!r} has no objects")

    materials: dict[str, PhysicsMaterial] = {}
    for obj in scene.objects:
        item = obj.catalog_item()
        if item is not None:
            materials.setdefault(item.material.name, item.material)

    unsourced = sorted(
        o.instance_id for o in scene.objects
        if (it := o.catalog_item()) is not None and not it.verified
    )
    unsourced_line = ", ".join(unsourced) if unsourced else "none"

    lines: list[str] = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        f'    doc = """Generated by labgen from SceneSpec {scene.name!r}.',
        "",
        "Units are metres and kilograms. Z-up, right-handed. Do not rescale on",
        "import: every dimension here is SI and came from a catalog template,",
        "not from a reconstruction.",
        "",
        f"Object poses are in the robot base frame {scene.robot_base_frame!r},",
        "not a reconstruction frame and not an arbitrary world origin.",
        "",
        f"Objects whose dimensions are not sourced: {unsourced_line}.",
        "An unsourced object has no spec sheet behind its size. Do not use this",
        "scene for a trust-bearing evaluation until that list reads 'none'.",
        '"""',
        ")",
        "",
        'def Xform "World"',
        "{",
        '    def Scope "Provenance"',
        "    {",
    ]

    for key, value in sorted(scene.source.items()):
        attr = _safe(key)
        if isinstance(value, bool):
            lines.append(f"        bool source:{attr} = {int(value)}")
        elif isinstance(value, (int, float)):
            lines.append(f"        double source:{attr} = {_fmt(float(value))}")
        elif isinstance(value, str):
            lines.append(f'        string source:{attr} = "{_escape(value)}"')
        # Nested structures (a full marker config) stay in the JSON SceneSpec,
        # which is the authoritative artifact. USD carries the flat provenance a
        # human reads with the stage open in front of them.

    lines += [
        f'        string source:robot_base_frame = "{_escape(scene.robot_base_frame)}"',
        f'        string source:scene_name = "{_escape(scene.name)}"',
        "    }",
        "",
        '    def Scope "PhysicsMaterials"',
        "    {",
    ]
    for m in materials.values():
        lines += _indent(_emit_material(m), 2)
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    lines += ["    }", ""]

    for obj in scene.objects:
        lines += _indent(_emit_object(obj), 1)
        lines.append("")

    lines.append("}")
    return "\n".join(lines) + "\n"


def write_scene(scene: SceneSpec, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(emit(scene), encoding="utf-8")
    return path
