"""Validate the emitted stage against a real USD implementation.

`tests/test_usda.py` parses the emitted `.usda` with a reader written in this
repo. That is a useful smoke test and it is *not* validation: a reader and an
emitter written by the same hand against the same wrong assumption agree with
each other perfectly. That is not a hypothetical -- the emitter wrote `quatd`
as a nested `(w, (x, y, z))` tuple, the repo's own reader happily accepted it,
396 tests passed, and no USD implementation on earth would open the file.

So this module uses actual `pxr`, from the standalone `usd-core` wheel. No
Isaac, no GPU, no Omniverse -- `usd-core` is the reference implementation
packaged for pip, and it installs anywhere in seconds.

    uv venv --python 3.12 .venv-usd
    VIRTUAL_ENV=.venv-usd uv pip install usd-core
    .venv-usd/Scripts/python -m pytest tests/test_usda_real_pxr.py

It is a separate venv on purpose (CLAUDE.md rule 5). These tests skip when
`pxr` is absent so the main suite still runs on a bare machine, but CI must run
them -- a skip here is the gap that let a non-parsing file through.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "pxr",
    reason="needs the standalone usd-core wheel; see this module's docstring",
)

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from labgen import usda  # noqa: E402
from labgen.catalog import CATALOG  # noqa: E402
from labgen.meshes import build  # noqa: E402
from labgen.types import SceneSpec  # noqa: E402

BENCH_5 = "examples/bench_5.json"

EXPECTED = {
    "bench": ("bench_top", True, "boundingCube"),
    "hotplate": ("hotplate_stirrer", False, "boundingCube"),
    "beaker": ("beaker_250", False, "sdf"),
    "test_tube": ("test_tube_16x100", False, "sdf"),
    "petri": ("petri_dish_100", False, "sdf"),
}


@pytest.fixture(scope="module")
def stage(tmp_path_factory):
    """Open the emitted stage, failing on ANY USD diagnostic.

    `Usd.Stage.Open` reports malformed attributes through the Tf error system
    and can still hand back a usable stage, so a bare `is not None` check is
    not enough -- the quaternion bug produced exactly that: errors raised,
    stage returned. The mark keeps every diagnostic fatal.
    """
    from pxr import Tf

    path = tmp_path_factory.mktemp("usd") / "bench_5.usda"
    usda.write_scene(SceneSpec.read(BENCH_5), path)

    mark = Tf.Error.Mark()
    mark.SetMark()
    opened = Usd.Stage.Open(str(path))
    errors = [str(e.commentary).strip() for e in mark.GetErrors()]
    assert not errors, "USD reported diagnostics while opening the stage:\n  " + \
                       "\n  ".join(errors[:10])
    assert opened is not None
    return opened


# --- the stage parses ----------------------------------------------------

def test_stage_opens_without_a_single_usd_diagnostic(stage):
    assert stage.GetDefaultPrim().GetPath() == "/World"


def test_stage_units_are_si_and_z_up(stage):
    """Read back through USD's own accessors, not by grepping the text."""
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z


# --- physics schemas actually apply --------------------------------------

@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_collision_schemas_resolve(name, stage):
    prim = stage.GetPrimAtPath(f"/World/{name}")
    assert prim.IsValid(), f"/World/{name} is not a valid prim"
    assert UsdPhysics.CollisionAPI(prim), "PhysicsCollisionAPI did not apply"
    assert UsdPhysics.MeshCollisionAPI(prim), "PhysicsMeshCollisionAPI did not apply"

    attr = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
    assert attr.IsValid()
    assert attr.Get() == EXPECTED[name][2]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_rigid_bodies_and_masses_resolve(name, stage):
    catalog_key, fixed, _ = EXPECTED[name]
    prim = stage.GetPrimAtPath(f"/World/{name}")

    if fixed:
        assert not UsdPhysics.RigidBodyAPI(prim), (
            f"{name} is a static fixture but carries PhysicsRigidBodyAPI"
        )
        assert not prim.GetAttribute("physics:mass").IsValid()
    else:
        assert UsdPhysics.RigidBodyAPI(prim)
        mass_api = UsdPhysics.MassAPI(prim)
        assert mass_api
        got = mass_api.GetMassAttr().Get()
        # float32 in USD vs float64 in the catalog.
        assert got == pytest.approx(CATALOG[catalog_key].mass_kg, rel=1e-6)
        assert got > 0.0


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_physics_material_binding_resolves_to_a_real_prim(name, stage):
    prim = stage.GetPrimAtPath(f"/World/{name}")
    rel = prim.GetRelationship("material:binding:physics")
    assert rel.IsValid()
    targets = rel.GetTargets()
    assert len(targets) == 1
    bound = stage.GetPrimAtPath(targets[0])
    assert bound.IsValid(), f"{name} binds {targets[0]}, which does not exist"
    assert UsdPhysics.MaterialAPI(bound), "bound prim is not a physics material"


def test_material_friction_values_round_trip(stage):
    glass = stage.GetPrimAtPath("/World/PhysicsMaterials/glass")
    api = UsdPhysics.MaterialAPI(glass)
    assert api.GetStaticFrictionAttr().Get() == pytest.approx(0.45)
    assert api.GetDynamicFrictionAttr().Get() == pytest.approx(0.40)
    assert api.GetRestitutionAttr().Get() == pytest.approx(0.05)


# --- transforms compose correctly ----------------------------------------

@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_pose_resolves_through_usd_xform_composition(name, stage):
    """Compute the transform the way a consumer will, not by reading the text.

    This is what catches a malformed `quatd`: the literal has to parse AND
    compose into a matrix.
    """
    scene = SceneSpec.read(BENCH_5)
    obj = scene.by_id(name)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(f"/World/{name}"))
    matrix = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    assert tuple(matrix.ExtractTranslation()) == pytest.approx(obj.position, abs=1e-9)

    rotation = matrix.ExtractRotationQuat()
    got = (rotation.GetReal(), *rotation.GetImaginary())
    assert got == pytest.approx(obj.orientation_wxyz, abs=1e-9)


def test_a_rotated_object_composes_to_the_right_matrix(tmp_path):
    """Identity quaternions hide quaternion bugs. Rotate 90 degrees about Z and
    check a known point lands where it should."""
    import math

    scene = SceneSpec.read(BENCH_5)
    beaker = scene.by_id("beaker")
    h = math.sqrt(0.5)
    beaker.orientation_wxyz = (h, 0.0, 0.0, h)

    path = usda.write_scene(scene, tmp_path / "rotated.usda")
    rotated = Usd.Stage.Open(str(path))
    matrix = UsdGeom.Xformable(
        rotated.GetPrimAtPath("/World/beaker")
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # +X in the object frame must map to +Y in the world frame.
    local_x = Gf.Vec3d(0.01, 0.0, 0.0)
    world = matrix.TransformDir(local_x)
    assert tuple(world) == pytest.approx((0.0, 0.01, 0.0), abs=1e-9)


# --- geometry survives the round trip ------------------------------------

@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_mesh_topology_matches_what_the_generator_built(name, stage):
    catalog_key, _, _ = EXPECTED[name]
    source = build(CATALOG[catalog_key])
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/World/{name}/geom"))
    assert mesh, f"/World/{name}/geom is not a UsdGeomMesh"

    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()

    assert len(points) == len(source.vertices)
    assert len(counts) == len(source.faces)
    assert len(indices) == len(source.faces) * 3
    assert set(counts) == {3}, "every face must be a triangle"
    assert max(indices) < len(points), "face index out of range"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_world_space_size_matches_the_catalog(name, stage):
    """The scale gate, measured by USD rather than by us.

    CLAUDE.md's scale-sanity validator rejects anything more than 5% off. This
    asserts far tighter, because at this stage there is no estimation involved
    -- any error is a bug, not a measurement.
    """
    catalog_key, _, _ = EXPECTED[name]
    dims = CATALOG[catalog_key].dims
    if CATALOG[catalog_key].shape == "box":
        expected = (dims["x"], dims["y"], dims["z"])
    else:
        width = max(dims.get("outer_d", 0.0), dims.get("base_d", 0.0))
        expected = (width, width, dims["height"])

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    size = cache.ComputeWorldBound(
        stage.GetPrimAtPath(f"/World/{name}")).ComputeAlignedRange().GetSize()
    assert tuple(size) == pytest.approx(expected, rel=1e-6)


def test_every_free_body_rests_on_the_bench_top(stage):
    """Support, measured in world space through USD.

    CLAUDE.md's support validator allows 1 mm. Here the tolerance is 1 um,
    because bench_5 is hand-authored and every object was placed deliberately.
    """
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bench_top = cache.ComputeWorldBound(
        stage.GetPrimAtPath("/World/bench")).ComputeAlignedRange().GetMax()[2]

    for name, (_, fixed, _) in EXPECTED.items():
        if fixed:
            continue
        base = cache.ComputeWorldBound(
            stage.GetPrimAtPath(f"/World/{name}")).ComputeAlignedRange().GetMin()[2]
        assert base == pytest.approx(bench_top, abs=1e-6), (
            f"{name} rests at z={base:.6f} but the bench top is at "
            f"z={bench_top:.6f} -- it floats or it is sunk"
        )


def test_keypoints_survive_as_typed_attributes(stage):
    prim = stage.GetPrimAtPath("/World/hotplate")
    attr = prim.GetAttribute("keypoints:plate_center")
    assert attr.IsValid()
    assert attr.GetTypeName() == "point3f"
    assert tuple(attr.Get()) == pytest.approx((0.0, -0.045, 0.105))


# --- a finding, recorded as a test ---------------------------------------

def test_sdf_is_not_a_standard_usd_approximation_token(stage):
    """`sdf` is a PhysX/Isaac extension, not part of UsdPhysics.

    UsdPhysics allows [none, convexDecomposition, convexHull, boundingSphere,
    boundingCube, meshSimplification]. `usd-core` accepts `sdf` because
    allowedTokens is advisory, but a strict consumer need not.

    This test documents the risk rather than asserting a fix, because only a
    real Isaac load can say whether Isaac honours the token or wants
    PhysxSDFMeshCollisionAPI instead. If it turns out to want the API, the
    correct change is approximation="none" plus that schema -- NEVER
    convexHull, which would seal every open vessel in the catalog.
    """
    attr = UsdPhysics.MeshCollisionAPI(
        stage.GetPrimAtPath("/World/beaker")).GetApproximationAttr()
    allowed = attr.GetMetadata("allowedTokens")
    assert "sdf" not in allowed, (
        "sdf is now a standard UsdPhysics token -- delete this test and the "
        "corresponding known-gap note in README.md"
    )
    assert attr.Get() == "sdf", "the emitter still relies on the extension token"
