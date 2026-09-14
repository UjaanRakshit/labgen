"""T2 acceptance: SceneSpec -> .usda with correct UsdPhysics schemas.

**What these tests do not prove.** The T2 acceptance text ends with "it opens
in Isaac Lab without error". That cannot be checked here: Isaac Lab is Linux
only and this suite is built to run without a GPU, without Isaac, and without
`pxr`. So what follows is a structural check of the emitted stage -- balanced
scopes, correct metadata, and every physics invariant CLAUDE.md names -- and
the load test remains genuinely outstanding. It is logged in README.md under
known gaps. Do not read a green run here as "the scene loads".
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from labgen import usda
from labgen.catalog import CATALOG
from labgen.types import SceneObject, SceneSpec
from labgen.usda import UsdaError

BENCH_5 = "examples/bench_5.json"


# --------------------------------------------------------------------------
# a minimal .usda structural reader
#
# Deliberately not `pxr`. If this suite imported USD to check USD, it would
# stop running in the environment it exists to protect -- CI with no GPU.
# --------------------------------------------------------------------------

@dataclasses.dataclass
class Prim:
    type_name: str
    name: str
    path: str
    depth: int
    attrs: dict[str, str]
    schemas: list[str]


DEF_RE = re.compile(r'^\s*def\s+(\w+)\s+"([^"]+)"')
ATTR_RE = re.compile(r'^\s*(?:custom\s+)?(?:uniform\s+)?([\w\[\]]+)\s+([\w:]+)\s*=\s*(.+?)\s*$')
SCHEMA_RE = re.compile(r'prepend\s+apiSchemas\s*=\s*\[(.*?)\]')


def parse_usda(text: str) -> tuple[dict[str, str], list[Prim]]:
    """Return (stage metadata, prims). Good enough to assert structure on."""
    lines = text.splitlines()

    # Strip triple-quoted blocks so a doc string cannot be mistaken for syntax.
    stripped: list[str] = []
    in_doc = False
    for ln in lines:
        ticks = ln.count('"""')
        if in_doc:
            if ticks:
                in_doc = False
            continue
        if ticks == 1:
            in_doc = True
            stripped.append(ln.split('"""')[0])
            continue
        stripped.append(ln)

    meta: dict[str, str] = {}
    prims: list[Prim] = []
    stack: list[str] = []
    pending: Prim | None = None
    depth = 0
    in_stage_meta = False

    for ln in stripped:
        bare = ln.strip()
        if not bare or bare.startswith("#usda"):
            continue

        if depth == 0 and bare == "(" and not prims:
            in_stage_meta = True
            continue
        if in_stage_meta:
            if bare == ")":
                in_stage_meta = False
                continue
            if "=" in bare:
                k, v = bare.split("=", 1)
                meta[k.strip()] = v.strip()
            continue

        m = DEF_RE.match(ln)
        if m:
            type_name, name = m.groups()
            path = "/".join(["", *stack, name]) if stack else f"/{name}"
            pending = Prim(type_name, name, path, depth, {}, [])
            prims.append(pending)
            continue

        if pending is not None and (s := SCHEMA_RE.search(bare)):
            pending.schemas = re.findall(r'"([^"]+)"', s.group(1))
            continue

        if bare.startswith("{"):
            if pending is not None:
                stack.append(pending.name)
                pending = None
            depth += 1
            continue

        if bare.startswith("}"):
            depth -= 1
            if stack:
                stack.pop()
            continue

        if prims and "=" in bare and not bare.startswith((")", "(")):
            a = ATTR_RE.match(ln)
            if a:
                _type, attr, value = a.groups()
                owner = next((p for p in reversed(prims)
                              if p.path == "/".join(["", *stack])), None)
                if owner is not None:
                    owner.attrs[attr] = value

    assert depth == 0, f"unbalanced braces: ended at depth {depth}"
    return meta, prims


@pytest.fixture(scope="module")
def scene() -> SceneSpec:
    return SceneSpec.read(BENCH_5)


@pytest.fixture(scope="module")
def emitted(scene) -> str:
    return usda.emit(scene)


@pytest.fixture(scope="module")
def parsed(emitted):
    return parse_usda(emitted)


def _prim(prims, path) -> Prim:
    hit = [p for p in prims if p.path == path]
    assert hit, f"no prim at {path}; have {[p.path for p in prims]}"
    return hit[0]


# --- stage metadata ------------------------------------------------------

def test_stage_metadata_is_si_and_z_up(parsed):
    """CLAUDE.md rule 2. A stage that says metersPerUnit = 0.01 makes every
    dimension in the catalog wrong by 100x, and it looks fine until something
    is measured."""
    meta, _ = parsed
    assert meta["metersPerUnit"] == "1"
    assert meta["upAxis"] == '"Z"'
    assert meta["defaultPrim"] == '"World"'


def test_braces_balance(emitted):
    parse_usda(emitted)          # asserts internally
    assert emitted.count("{") == emitted.count("}")


def test_output_is_deterministic(scene):
    """A scene that re-emits differently makes every diff unreadable, which
    defeats the point of hand-editable SceneSpec."""
    assert usda.emit(scene) == usda.emit(scene)


def test_doc_names_the_robot_base_frame(emitted, scene):
    assert scene.robot_base_frame in emitted
    assert "robot base frame" in emitted


def test_doc_lists_unsourced_objects(emitted):
    """Every object in bench_5 is currently unsourced, and the stage says so
    rather than presenting the scene as trustworthy."""
    assert "Objects whose dimensions are not sourced:" in emitted
    assert "trust-bearing evaluation" in emitted


def test_provenance_scope_carries_the_git_sha(parsed):
    _, prims = parsed
    prov = _prim(prims, "/World/Provenance")
    assert "source:git_sha" in prov.attrs
    assert re.fullmatch(r'"[0-9a-f]{40}"', prov.attrs["source:git_sha"])
    assert prov.attrs["source:robot_base_frame"] == '"yam_base_link"'


# --- object structure ----------------------------------------------------

EXPECTED = {
    "bench": ("bench_top", True, "boundingCube", "steel"),
    "hotplate": ("hotplate_stirrer", False, "boundingCube", "steel"),
    "beaker": ("beaker_250", False, "sdf", "glass"),
    "test_tube": ("test_tube_16x100", False, "sdf", "glass"),
    "petri": ("petri_dish_100", False, "sdf", "plastic"),
}


def test_every_object_becomes_an_xform(parsed):
    _, prims = parsed
    for name in EXPECTED:
        assert _prim(prims, f"/World/{name}").type_name == "Xform"


def test_every_object_has_a_mesh_child(parsed):
    _, prims = parsed
    for name in EXPECTED:
        geom = _prim(prims, f"/World/{name}/geom")
        assert geom.type_name == "Mesh"
        assert geom.attrs["points"].startswith("[(")
        assert geom.attrs["faceVertexCounts"].startswith("[3")


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_collider_approximation_is_explicit_and_correct(name, parsed):
    """CLAUDE.md: every collider has an explicit approximation set."""
    _, prims = parsed
    _, _, approximation, _ = EXPECTED[name]
    prim = _prim(prims, f"/World/{name}")
    assert prim.attrs["physics:approximation"] == f'"{approximation}"'
    assert prim.attrs["physics:collisionEnabled"] == "1"
    assert "PhysicsCollisionAPI" in prim.schemas
    assert "PhysicsMeshCollisionAPI" in prim.schemas


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_dynamic_bodies_have_explicit_mass_and_statics_have_none(name, parsed):
    """CLAUDE.md rule 4: no silent defaults, and no density inference."""
    _, prims = parsed
    catalog_key, fixed, _, _ = EXPECTED[name]
    prim = _prim(prims, f"/World/{name}")

    if fixed:
        assert "PhysicsRigidBodyAPI" not in prim.schemas
        assert "physics:mass" not in prim.attrs, (
            f"{name} is a static fixture but carries a mass"
        )
    else:
        assert "PhysicsRigidBodyAPI" in prim.schemas
        assert "PhysicsMassAPI" in prim.schemas
        assert float(prim.attrs["physics:mass"]) == pytest.approx(
            CATALOG[catalog_key].mass_kg)
        assert prim.attrs["physics:rigidBodyEnabled"] == "1"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_object_binds_a_physics_material(name, parsed):
    _, prims = parsed
    _, _, _, material = EXPECTED[name]
    prim = _prim(prims, f"/World/{name}")
    assert prim.attrs["material:binding:physics"] == f"</World/PhysicsMaterials/{material}>"
    _prim(prims, f"/World/PhysicsMaterials/{material}")


def test_materials_carry_friction_and_restitution(parsed):
    _, prims = parsed
    glass = _prim(prims, "/World/PhysicsMaterials/glass")
    assert glass.schemas == ["PhysicsMaterialAPI"]
    assert float(glass.attrs["physics:staticFriction"]) == pytest.approx(0.45)
    assert float(glass.attrs["physics:dynamicFriction"]) == pytest.approx(0.40)
    assert float(glass.attrs["physics:restitution"]) == pytest.approx(0.05)


def test_materials_are_emitted_once_each(emitted):
    """Three objects bind glass; there must still be one glass prim."""
    assert emitted.count('def Material "glass"') == 1


# --- poses ---------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_pose_matches_the_scene_spec(name, parsed, scene):
    _, prims = parsed
    obj = scene.by_id(name)
    prim = _prim(prims, f"/World/{name}")
    got = [float(v) for v in re.findall(r"-?[\d.eE+-]+", prim.attrs["xformOp:translate"])]
    assert got == pytest.approx(list(obj.position))
    assert prim.attrs["xformOpOrder"] == '["xformOp:translate", "xformOp:orient"]'


def test_orientation_is_a_flat_quat_literal(parsed):
    """A USD `quatd` literal is a FLAT 4-tuple, real part first.

    This test previously asserted "(1, (0, 0, 0))" -- the nested form -- and
    passed, while the emitted file did not parse in USD at all. The structural
    reader in this module was written against the same wrong assumption as the
    emitter, so the two agreed with each other and both were wrong. That is the
    precise reason `test_usda_real_pxr.py` exists.
    """
    _, prims = parsed
    assert _prim(prims, "/World/beaker").attrs["xformOp:orient"] == "(1, 0, 0, 0)"


def test_bench_is_sunk_so_its_top_lands_on_the_robot_base_plane(scene):
    bench = scene.by_id("bench")
    assert bench.position[2] == pytest.approx(-CATALOG["bench_top"].dims["z"])


# --- keypoints -----------------------------------------------------------

def test_keypoints_are_written_as_custom_attributes(parsed):
    _, prims = parsed
    beaker = _prim(prims, "/World/beaker")
    for kp in ("rim_grasp", "body_grasp", "pour_lip", "base_center", "fill_point"):
        assert f"keypoints:{kp}" in beaker.attrs, f"beaker is missing keypoint {kp}"


def test_a_non_origin_keypoint_survives_emission(parsed):
    """The hotplate's plate_center is at (0, -0.045, 0.105), not the origin --
    the case that catches an emitter writing keypoints in the wrong frame."""
    _, prims = parsed
    value = _prim(prims, "/World/hotplate").attrs["keypoints:plate_center"]
    got = [float(v) for v in re.findall(r"-?[\d.eE+-]+", value)]
    assert got == pytest.approx([0.0, -0.045, 0.105])


# --- refusals ------------------------------------------------------------

def test_a_convex_hull_on_an_open_vessel_is_a_hard_failure():
    """CLAUDE.md hard rule 1, at the last point it could be violated.

    Guards against someone 'optimising' a beaker's collider in the catalog: the
    emitter refuses rather than sealing the vessel.
    """
    hulled = dataclasses.replace(CATALOG["beaker_250"], collider="convex_hull")
    original = CATALOG["beaker_250"]
    CATALOG["beaker_250"] = hulled
    try:
        scene = SceneSpec.read(BENCH_5)
        with pytest.raises(UsdaError, match="hull seals the opening"):
            usda.emit(scene)
    finally:
        CATALOG["beaker_250"] = original


def test_a_zero_mass_dynamic_body_is_refused():
    massless = dataclasses.replace(CATALOG["beaker_250"], mass_kg=0.0)
    original = CATALOG["beaker_250"]
    CATALOG["beaker_250"] = massless
    try:
        scene = SceneSpec.read(BENCH_5)
        with pytest.raises(UsdaError, match="zero-mass rigid body"):
            usda.emit(scene)
    finally:
        CATALOG["beaker_250"] = original


def test_an_empty_scene_is_refused():
    with pytest.raises(UsdaError, match="no objects"):
        usda.emit(SceneSpec(name="empty", robot_base_frame="yam_base_link", objects=[]))


def test_a_generated_asset_is_refused_with_a_pointer_to_the_task():
    """Not silently skipped. An object that vanishes from the stage is the
    failure mode the collider census exists to catch elsewhere."""
    from labgen.types import GeneratedAsset

    scene = SceneSpec(
        name="gen", robot_base_frame="yam_base_link",
        objects=[SceneObject(
            instance_id="clamp", catalog_key=None, position=(0.0, 0.3, 0.0),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0), fixed=False, confidence=0.4,
            provenance="generated",
            generated=GeneratedAsset(mesh_path="a.obj", mass_kg=0.05,
                                     collider="convex_decomposition", material="steel"),
        )],
    )
    with pytest.raises(UsdaError, match="T6"):
        usda.emit(scene)


def test_cylinder_collider_is_refused_rather_than_approximated():
    """There is no faithful mesh-collider token for a cylinder. Substituting a
    bounding sphere would be wrong in a way no render shows."""
    cyl = dataclasses.replace(CATALOG["hotplate_stirrer"], collider="cylinder")
    original = CATALOG["hotplate_stirrer"]
    CATALOG["hotplate_stirrer"] = cyl
    try:
        with pytest.raises(UsdaError, match="no faithful"):
            usda.emit(SceneSpec.read(BENCH_5))
    finally:
        CATALOG["hotplate_stirrer"] = original


# --- the example scene ---------------------------------------------------

def test_bench_5_has_five_objects_and_covers_both_buildable_collider_paths(scene):
    assert len(scene) == 5
    colliders = {o.collider() for o in scene.objects}
    assert colliders == {"box", "sdf"}, (
        "bench_5 is chosen for collider coverage; convex_decomposition is a "
        "known gap until a rack footprint is sourced"
    )


def test_bench_5_writes_to_disk(scene, tmp_path):
    path = usda.write_scene(scene, tmp_path / "nested" / "bench_5.usda")
    assert path.exists()
    meta, prims = parse_usda(path.read_text(encoding="utf-8"))
    assert meta["metersPerUnit"] == "1"
    assert len([p for p in prims if p.depth == 1]) == 7   # 2 scopes + 5 objects


# --- provenance integrity ------------------------------------------------

def test_recorded_git_sha_resolves_in_this_repo(scene):
    """A scene records the commit it was built at. That must still exist.

    Not hypothetical: rewriting this repo's history to correct the commit
    authorship changed every SHA, and bench_5.json was left pointing at a
    commit that no longer existed. A provenance record that names a
    non-existent commit is worse than no record -- it looks traceable and is
    not, which is the same failure shape as an unsourced dimension flagged
    `verified`.

    Skips outside a git checkout so a source tarball still tests clean.
    """
    import shutil
    import subprocess

    sha = scene.source.get("git_sha")
    assert sha, "bench_5 records no git_sha"
    assert re.fullmatch(r"[0-9a-f]{40}", sha), f"malformed git_sha {sha!r}"

    if shutil.which("git") is None:
        pytest.skip("git not available")
    probe = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("not a git checkout")

    found = subprocess.run(["git", "cat-file", "-t", sha],
                           capture_output=True, text=True)
    assert found.returncode == 0 and found.stdout.strip() == "commit", (
        f"bench_5.json records git_sha {sha}, which does not resolve to a "
        f"commit in this repository. History rewritten? Update the scene's "
        f"provenance to the commit it actually corresponds to."
    )
