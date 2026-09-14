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
    "sdf": "sdf",
    "convex_hull": "convexHull",
    "convex_decomposition": "convexDecomposition",
    "box": "boundingCube",
}

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


def _emit_mesh_prim(name: str, mesh: Mesh, purpose: str = "default") -> list[str]:
    counts = ", ".join("3" for _ in mesh.faces)
    indices = ", ".join(str(int(i)) for f in mesh.faces for i in f)
    out = [
        f'def Mesh "{name}"',
        "{",
        f"    point3f[] points = [{_mesh_points(mesh)}]",
        f"    int[] faceVertexCounts = [{counts}]",
        f"    int[] faceVertexIndices = [{indices}]",
        '    uniform token subdivisionScheme = "none"',
    ]
    if purpose != "default":
        out.append(f'    uniform token purpose = "{purpose}"')
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

    schemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    if not obj.fixed:
        schemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"] + schemas
    schema_list = ", ".join('"' + s + '"' for s in schemas)

    sourced = "yes" if item.verified else "NO -- dimensions unverified"
    out = [
        f'def Xform "{name}" (',
        f"    prepend apiSchemas = [{schema_list}]",
        ")",
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

    out += [
        "",
        f'    uniform token physics:approximation = "{approximation}"',
        "    bool physics:collisionEnabled = 1",
        f"    rel material:binding:physics = <{_material_prim(item.material)}>",
        "",
    ]
    out += _indent(_emit_keypoints(obj), 1)
    out.append("")
    out += _indent(_emit_mesh_prim("geom", mesh), 1)
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
