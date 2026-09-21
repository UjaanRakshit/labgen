"""T4: the emitted Isaac Lab scene config.

These run on a bare machine and check what can be checked without Isaac: that
the output is valid Python, that the conventions are right, and that nothing
backend-specific leaked in.

**They do not prove Isaac Lab accepts it.** That needs Isaac Lab, and
`scripts/verify_cfg.py` does it by importing the generated module and
instantiating the class inside that environment. Run it there before trusting
a change to the emitter -- this project has already shipped one artifact that
its own reader loved and no real implementation would open.
"""

from __future__ import annotations

import ast
import math

import pytest

from labgen.isaaclab_cfg import ArmSource, YAM_ARM, emit_scene_cfg, write_scene_cfg
from labgen.types import SceneSpec

BENCH = "examples/bench_arm.json"


@pytest.fixture
def scene() -> SceneSpec:
    return SceneSpec.read(BENCH)


@pytest.fixture
def source(scene) -> str:
    return emit_scene_cfg(scene, usda_path="/tmp/bench_arm.usda")


def parsed(src: str) -> ast.Module:
    return ast.parse(src)


# --- it is real Python ---------------------------------------------------

def test_output_is_valid_python(source):
    parsed(source)


def test_output_is_deterministic(scene):
    a = emit_scene_cfg(scene, usda_path="/tmp/x.usda")
    b = emit_scene_cfg(scene, usda_path="/tmp/x.usda")
    assert a == b


def test_defines_a_scene_cfg_subclass(source):
    tree = parsed(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    assert len(classes) == 1
    cls = classes[0]
    assert cls.name == "BenchArmSceneCfg"
    assert [b.id for b in cls.bases if isinstance(b, ast.Name)] == ["InteractiveSceneCfg"]


def test_writes_to_disk(scene, tmp_path):
    p = write_scene_cfg(scene, tmp_path / "gen" / "cfg.py", usda_path="/tmp/x.usda")
    assert p.exists()
    parsed(p.read_text(encoding="utf-8"))


# --- conventions that are easy to get silently wrong ---------------------

def test_quaternions_are_emitted_xyzw_not_wxyz(scene):
    """Isaac Lab 3.0 stores (x, y, z, w); SceneSpec stores (w, x, y, z).

    Checked against AssetBaseCfg.InitialStateCfg, whose identity default is
    (0, 0, 0, 1). An identity object must therefore emit rot=(0, 0, 0, 1), and
    a 90 degree rotation about Z must put the sqrt(1/2) terms in slots 2 and 3.
    """
    h = math.sqrt(0.5)
    beaker = scene.by_id("beaker")
    beaker.orientation_wxyz = (h, 0.0, 0.0, h)          # wxyz
    src = emit_scene_cfg(scene, usda_path="/tmp/x.usda")

    assert "rot=(0, 0, 0, 1)" in src, "identity must be xyzw"
    assert f"rot=(0, 0, {h:.6g}, {h:.6g})" in src, "z and w carry the rotation"
    assert f"rot=({h:.6g}, 0, 0, {h:.6g})" not in src, "that would be wxyz"


def test_objects_are_not_respawned(source):
    """The scene USD is spawned once; objects attach to prims inside it.

    Spawning each object separately would duplicate the geometry and discard
    the colliders, masses and materials already authored in the USD.
    """
    assert source.count("UsdFileCfg(usd_path=SCENE_USD)") == 1
    # every per-object entry disables spawning
    assert source.count("spawn=None") == 5


def test_static_and_dynamic_objects_get_different_classes(source, scene):
    """A fixture is an AssetBaseCfg; a free body is a RigidObjectCfg."""
    assert "bench: AssetBaseCfg" in source
    for name in ("beaker", "test_tube", "petri", "hotplate"):
        assert f"{name}: RigidObjectCfg" in source
    assert "bench: RigidObjectCfg" not in source


def test_every_scene_object_appears(source, scene):
    for obj in scene.objects:
        assert f"{obj.instance_id}: " in source


def test_positions_match_the_scene_spec(source, scene):
    for obj in scene.objects:
        x, y, z = obj.position
        needle = f"pos=({x:.6g}, {y:.6g}, {z:.6g})"
        assert needle in source, f"{obj.instance_id} pose {needle} missing"


# --- backend agnosticism -------------------------------------------------

def test_generated_config_imports_only_isaac_labs_abstract_layer(source):
    """CLAUDE.md: the scene generator must not care which engine is underneath.

    Checked on the IMPORTS via the AST, not by substring. The first version of
    this test scanned the raw text and failed on the module docstring's own
    sentence about running on "PhysX or Newton" -- prose is not a dependency,
    and a check that cannot tell the difference will be silenced rather than
    fixed.
    """
    modules = set()
    for node in ast.walk(parsed(source)):
        if isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)

    assert modules, "the generated config imports nothing at all"
    for m in modules:
        root = m.split(".")[0]
        assert root in {"isaaclab", "__future__"}, (
            f"generated config imports {m!r}; only Isaac Lab's abstract layer "
            f"is allowed, so the scene runs on either backend")

    # and no attribute access into a backend namespace
    attrs = {f"{n.value.id}.{n.attr}" for n in ast.walk(parsed(source))
             if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    assert not {a for a in attrs
                if a.split(".")[0] in {"physx", "newton", "omni", "pxr"}}


def test_labgen_still_imports_without_isaaclab():
    """The emitter writes Isaac Lab code; it must never import Isaac Lab.

    Run in a CLEAN interpreter. Inspecting this process's sys.modules tests the
    wrong thing: another test module in the same session imports `pxr`, so the
    check failed on a module this one never touched. What matters is what
    importing the emitter pulls in by itself.
    """
    import subprocess
    import sys

    code = (
        "import sys; import labgen.isaaclab_cfg;"
        "banned=('isaaclab','omni','pxr','newton','warp','torch');"
        "print(','.join(m for m in banned if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=".")
    assert out.returncode == 0, out.stderr
    leaked = [m for m in out.stdout.strip().split(",") if m]
    assert not leaked, f"importing the emitter dragged in {leaked}"


# --- the arm -------------------------------------------------------------

def test_arm_is_spawned_from_mjcf_directly(source):
    """TASKS.md asks USD or MJCF. Isaac Lab 3.0 ships MjcfFileCfg, so: MJCF."""
    assert "MjcfFileCfg" in source
    assert "asset_path=" in source


def test_arm_collision_comes_from_visuals_and_is_decomposed(source):
    """Both settings are load-bearing and both were measured.

    The YAM ships zero <collision> elements, so without collision_from_visuals
    the arm touches nothing (0 contacts out of 109). And the converter's
    "Convex Hull" default makes each L-shaped fingertip a solid block spanning
    +/-36 mm, which swallows a 70 mm beaker.
    """
    assert "collision_from_visuals=True" in source
    assert 'collision_type="Convex Decomposition"' in source
    assert '"Convex Hull"' not in source.replace('# NOT the "Convex Hull" default', "")


def test_arm_is_fixed_to_the_bench(source):
    assert "fix_base=True" in source


def test_effort_limit_is_the_robots_own_rating(source):
    """The arm runs inside its spec, and the file explains why it can.

    An earlier version ran at 40 N.m against the rated 10, from a real
    measurement (the shoulder sagged at 10) and a wrong conclusion. Static
    gravity at that pose is 7.26 N.m; the sag was the position gain, not the
    ceiling. Pinned because "this robot cannot do this task" is the most
    expensive kind of wrong answer here.
    """
    assert "joint_effort_limit=10.0" in source
    assert "joint_effort_limit=40.0" not in source
    assert "7.26 N.m" in source, "the gravity figure behind the decision belongs in the file"


def test_file_warns_that_the_actuator_does_not_compensate_gravity(source):
    """Holding the pose and executing a trajectory are different questions.

    Both were answered the wrong way once. The effort limit was raised because
    the arm sagged (it was the gain), and then the trajectory failure was blamed
    on acceleration torque (it was not -- slowing 4x moved peak error 100.3 to
    89.9 deg). Gravity peaks at 8.06 N.m of the rated 10 over the plan, so a PD
    with no feedforward idles at 81% saturation. With the feedforward, the same
    10 N.m and the same gains track to 3.95 deg.

    ImplicitActuatorCfg is a PD. Anyone driving a trajectory through this
    generated config inherits the problem, so the file has to say so.
    """
    assert "8.06" in source, "the gravity figure over the plan belongs in the file"
    assert "does not compensate gravity" in source
    assert "3.95 deg" in source, "and the number showing the feedforward fixes it"


def test_arm_can_be_omitted(scene):
    src = emit_scene_cfg(scene, usda_path="/tmp/x.usda", arm=None)
    parsed(src)
    assert "ArticulationCfg(" not in src
    assert "robot:" not in src


def test_unknown_arm_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown arm source kind"):
        ArmSource("x.foo", kind="collada")


def test_urdf_arm_uses_the_urdf_spawner(scene):
    src = emit_scene_cfg(scene, usda_path="/tmp/x.usda",
                         arm=ArmSource("yam.urdf", kind="urdf"))
    parsed(src)
    assert "UrdfFileCfg" in src and "MjcfFileCfg" not in src


# --- provenance carried into the generated file --------------------------

def test_generated_file_names_the_robot_base_frame(source, scene):
    assert scene.robot_base_frame in source


def test_generated_file_lists_unsourced_objects(source):
    assert "not sourced" in source
    assert "trust-bearing evaluation" in source


def test_generated_file_says_not_to_hand_edit_it(source):
    assert "Do not edit by hand" in source
    assert "SceneSpec JSON" in source
