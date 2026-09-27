"""Build the YAM (arm + linear gripper) as a USD articulation Isaac Lab can spawn.

Why this exists: Isaac Lab's UrdfFileCfg / MjcfFileCfg spawners call Isaac
Sim's importer, which is not in the kit-less Newton install, and CLAUDE.md
rule 5 keeps new packages out of that environment. So the conversion runs
offline in its own venv (/home/ujaan/isaac/.venv-usdconv, with NVIDIA's
standalone `urdf-usd-converter`) and Isaac Lab spawns the result with a plain
UsdFileCfg.

The i2rt URDF has visuals and no <collision> at all. This adds:
  * links: the visual mesh as the collider, convex hull
  * joint8 mimics joint7: one motor drives both jaws on the real gripper
  * fingertips: NOT the tip mesh (its hull is an L-bracket spanning +/-36 mm
    that swallows a beaker and ejects it) but the pad boxes of
    labgen.control.YAM_PAD -- the ones the grasp acceptance ran on.
    UNVERIFIED until calipers.

Usage (WSL):
    /home/ujaan/isaac/.venv-usdconv/bin/python make_yam_usd.py [out_dir]
"""
from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

def _labgen_repo() -> str:
    """The labgen checkout: $LABGEN_REPO, else the repo this script sits in, else
    this project's WSL path (scripts are synced out of the repo on that machine)."""
    import os
    from pathlib import Path as _P
    here = _P(__file__).resolve().parents[1]
    return os.environ.get("LABGEN_REPO") or (str(here) if (here / "labgen" / "__init__.py").is_file()
                                             else "/mnt/c/Ujaan Docx/Research/labgen")


sys.path.insert(0, _labgen_repo())
from labgen.control import YAM_PAD  # noqa: E402

import os  # noqa: E402
# i2rt's yam.urdf (arm + linear gripper) and NVIDIA's standalone converter, in its
# own venv. Override per machine with YAM_URDF / URDF_USD_CONVERTER.
URDF = Path(os.environ.get("YAM_URDF", "/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"))
CONVERTER = os.environ.get("URDF_USD_CONVERTER", "/home/ujaan/isaac/.venv-usdconv/bin/urdf_usd_converter")
TIPS = {"tip_left": +1, "tip_right": -1}      # side convention of FingerPad.centre_for


def patched_urdf(out: Path) -> Path:
    tree = ET.parse(URDF)
    root = tree.getroot()
    for link in root.findall("link"):
        vis = link.find("visual")
        if vis is None:
            continue
        mesh = vis.find("geometry/mesh")
        if mesh is not None and not Path(mesh.get("filename")).is_absolute():
            mesh.set("filename", str((URDF.parent / mesh.get("filename")).resolve()))
        name = link.get("name")
        col = ET.SubElement(link, "collision")
        if name in TIPS:
            cx, cy, cz = YAM_PAD.centre_for(TIPS[name])
            hx, hy, hz = YAM_PAD.half_extents_m
            ET.SubElement(col, "origin", xyz=f"{cx} {cy} {cz}", rpy="0 0 0")
            g = ET.SubElement(col, "geometry")
            ET.SubElement(g, "box", size=f"{2*hx} {2*hy} {2*hz}")
        else:
            o = vis.find("origin")
            if o is not None:
                col.append(o)
            col.append(vis.find("geometry"))
    # Couple the fingers. The real linear gripper drives both jaws from ONE
    # motor through a rack; yam.urdf leaves joint7 and joint8 independent, so a
    # held beaker pushes both fingers sideways together (measured: 8.8 mm while
    # lifting). Same coordinate convention on both (open -0.047, closed 0).
    for j in root.findall("joint"):
        if j.get("name") == "joint8" and j.find("mimic") is None:
            ET.SubElement(j, "mimic", joint="joint7", multiplier="1", offset="0")
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out)
    return out


def set_collision_approximation(usd_path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    # The converter pins the base to the WORLD origin with a fixed joint. Two
    # arms spawned from this file would then both sit at the origin whatever
    # their init_state says. Isaac Lab fixes the root itself where the arm is
    # placed (ArticulationRootPropertiesCfg(fix_root_link=True)), so drop it.
    for prim in list(stage.Traverse()):
        b0 = prim.GetRelationship("physics:body0").GetTargets() if prim.IsA(UsdPhysics.FixedJoint) else None
        # body0 empty, or a plain Xform (the converter uses the asset root /yam): world.
        if b0 is not None and (not b0 or not stage.GetPrimAtPath(b0[0]).HasAPI(UsdPhysics.RigidBodyAPI)):
            print(f"removed world-anchoring {prim.GetPath()}")
            stage.RemovePrim(prim.GetPath())
    n = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexHull")
            n += 1
    stage.GetRootLayer().Save()
    print(f"convexHull on {n} mesh collider(s)")


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else Path(_labgen_repo()) / "out/yam")
    urdf = patched_urdf(out_dir / "yam_labgen.urdf")
    subprocess.run([CONVERTER, "--no-layer-structure", "--no-physics-scene", str(urdf),
                    str(out_dir / "usd")], check=True)
    usds = sorted((out_dir / "usd").glob("*.usd*"))
    if not usds:
        raise SystemExit("converter produced no USD")
    for u in usds:
        set_collision_approximation(u)
    print("wrote", *usds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
