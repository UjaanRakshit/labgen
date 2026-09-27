"""Load Isaac-LabBench-YAM-IK-Rel-v0, settle it, and pick the beaker with robot0.

Drives the env only through its 14-d relative-IK action -- the same interface
COBALT's phone teleop sends -- so a pass here means the task works for teleop,
not just that the scene loads.

    python run_bench_task.py [--headless] [--no-pick]
"""
from __future__ import annotations

import argparse
import math
import sys

def _labgen_repo() -> str:
    """The labgen checkout: $LABGEN_REPO, else the repo this script sits in, else
    this project's WSL path (scripts are synced out of the repo on that machine)."""
    import os
    from pathlib import Path as _P
    here = _P(__file__).resolve().parents[1]
    return os.environ.get("LABGEN_REPO") or (str(here) if (here / "labgen" / "__init__.py").is_file()
                                             else "/mnt/c/Ujaan Docx/Research/labgen")


sys.path.insert(0, _labgen_repo())

import torch  # noqa: E402
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--no-pick", action="store_true")
parser.add_argument("--iters", type=int, default=0, help="override solver iterations")
parser.add_argument("--ls-iters", type=int, default=0)
parser.add_argument("--substeps", type=int, default=0)
parser.add_argument("--kd", type=float, default=0.0, help="contact kd; ke = (kd/2)^2")
parser.add_argument("--coll-dec", type=int, default=-1)
parser.add_argument("--dt", type=float, default=0.0)
parser.add_argument("--decimation", type=int, default=0)
add_launcher_args(parser)
args = parser.parse_args()

import gymnasium as gym  # noqa: E402

import labgen_tasks  # noqa: E402,F401
from labgen_tasks.bench_yam import BenchYamIkRelEnvCfg  # noqa: E402
from labgen.catalog import CATALOG  # noqa: E402
from labgen.types import SceneSpec  # noqa: E402
from labgen_tasks.bench_yam import SCENE_JSON  # noqa: E402

TASK = "Isaac-LabBench-YAM-IK-Rel-v0"
OBJECTS = ("beaker", "test_tube", "petri", "hotplate")


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return torch.tensor([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main() -> int:
    cfg = BenchYamIkRelEnvCfg()
    if args.iters:
        cfg.sim.physics.solver_cfg.iterations = args.iters
    if args.ls_iters:
        cfg.sim.physics.solver_cfg.ls_iterations = args.ls_iters
    if args.dt:
        cfg.sim.dt = args.dt
    if args.decimation:
        cfg.decimation = args.decimation
    if args.substeps:
        cfg.sim.physics.num_substeps = args.substeps
    if args.kd:
        cfg.sim.physics.default_shape_cfg.kd = args.kd
        cfg.sim.physics.default_shape_cfg.ke = (args.kd / 2) ** 2
    if args.coll_dec >= 0:
        cfg.sim.physics.collision_decimation = args.coll_dec
    p = cfg.sim.physics
    print(f"PHYSICS substeps {p.num_substeps} (substep {cfg.sim.dt/p.num_substeps*1000:.2f} ms) "
          f"ke {p.default_shape_cfg.ke:.0f} kd {p.default_shape_cfg.kd:.0f} "
          f"collision_decimation {p.collision_decimation}")
    with launch_simulation(cfg, args):
        env = gym.make(TASK, cfg=cfg)
        u = env.unwrapped
        print(f"action space {env.action_space.shape}  terms {u.action_manager.active_terms}")
        obs, _ = env.reset()
        pol = obs["policy"]
        steps_per_s = round(1.0 / (cfg.sim.dt * cfg.decimation))

        def pose(name):
            return pol[f"{name}_pose"][0, :3].clone()

        def act(d0=(0, 0, 0), r0=(0, 0, 0), g0=1.0):
            a = torch.zeros(env.action_space.shape, device=u.device)
            a[0, 0:3] = torch.tensor(d0, device=u.device) / cfg.actions.robot0_arm.scale
            a[0, 3:6] = torch.tensor(r0, device=u.device) / cfg.actions.robot0_arm.scale
            a[0, 6] = g0
            a[0, 13] = 1.0          # robot1 gripper open
            return a

        # ---- settle: 2 s of hold, nothing should move --------------------
        start = {n: pose(n) for n in OBJECTS}
        eef0 = [pol["robot0_eef_pos"][0].clone(), pol["robot1_eef_pos"][0].clone()]
        print(f"robot1 base quat (wxyz) {pol['robot1_base_ori'][0].tolist()}  "
              f"-- expect ~(0,0,0,1) for the 180 deg yaw")
        import time
        t0 = time.perf_counter()
        for _ in range(2 * steps_per_s):
            obs, *_ = env.step(act())
            pol = obs["policy"]
        per = (time.perf_counter() - t0) / (2 * steps_per_s) * 1000
        print(f"STEP TIME {per:.1f} ms per env step (real time needs <= {1000/steps_per_s:.0f})")
        print("SETTLE (2 s, zero action):")
        worst = 0.0
        for n in OBJECTS:
            d = float((pose(n) - start[n]).norm()) * 1000
            worst = max(worst, d)
            print(f"   {n:10s} moved {d:7.3f} mm   z={float(pose(n)[2])*1000:7.2f} mm")
        for i in (0, 1):
            d = float((pol[f'robot{i}_eef_pos'][0] - eef0[i]).norm()) * 1000
            print(f"   robot{i} grasp point drifted {d:.3f} mm")
        print(f"   settle {'PASS' if worst <= 2.0 else 'FAIL'} (worst {worst:.3f} mm, gate 2 mm)")

        if args.no_pick:
            env.close()
            return 0

        # ---- pick the beaker with robot0 ---------------------------------
        spec = SceneSpec.read(SCENE_JSON)
        b = spec.by_id("beaker")
        rim_z = b.position[2] + CATALOG[b.catalog_key].keypoints["rim_grasp"][2]
        bx, by = float(pose("beaker")[0]), float(pose("beaker")[1])
        down = torch.tensor([0.0, 0.0, -1.0])

        def servo(target, grip, seconds, vmax=0.25, tol=None):
            """Move the grasp point toward `target` with the gripper pointing down."""
            nonlocal obs, pol
            for _ in range(int(seconds * steps_per_s)):
                p = pol["robot0_eef_pos"][0].cpu()
                R = quat_wxyz_to_R(pol["robot0_eef_quat"][0].cpu())
                err = torch.tensor(target) - p
                step = err.clamp(-vmax / steps_per_s, vmax / steps_per_s)
                z = R[:, 2]
                axis = torch.linalg.cross(z, down)
                ang = math.atan2(float(axis.norm()), float(z @ down))
                rot = (axis / axis.norm() * min(ang, 0.1)) if axis.norm() > 1e-6 else torch.zeros(3)
                obs, *_ = env.step(act(step.tolist(), rot.tolist(), grip))
                pol = obs["policy"]
                if tol is not None and float(err.norm()) < tol and ang < math.radians(2):
                    break
            return float((torch.tensor(target) - pol["robot0_eef_pos"][0].cpu()).norm())

        def show(tag):
            e = pol["robot0_eef_pos"][0].tolist(); bp = pose("beaker").tolist()
            j = pol["robot0_joint_pos"][0].tolist()
            print(f"   [{tag:6s}] eef {[round(v*1000,1) for v in e]} mm  beaker {[round(v*1000,1) for v in bp]} mm"
                  f"  fingers {[round(v*1000,2) for v in j[6:8]]} mm")

        print(f"   rim grasp z = {rim_z*1000:.1f} mm")
        above = [bx, by, rim_z + 0.08]
        at = [bx, by, rim_z]
        e1 = servo(above, 1.0, 6.0, tol=0.003)
        e2 = servo(at, 1.0, 4.0, vmax=0.08, tol=0.002)
        print(f"PICK: reach above err {e1*1000:.1f} mm, at rim err {e2*1000:.1f} mm")
        show("at")
        art = u.scene["robot0"]
        fj = [art.joint_names.index("joint7"), art.joint_names.index("joint8")]
        for k in range(steps_per_s):                        # close in place, logged
            obs, *_ = env.step(act((0, 0, 0), (0, 0, 0), -1.0))
            pol = obs["policy"]
            if k < 12 or k % 5 == 0:
                q = art.data.joint_pos.torch[0, fj].tolist()
                f = art.data.applied_torque.torch[0, fj].tolist()
                bp = pose("beaker")
                print(f"      close t={k/steps_per_s:4.2f}s fingers {[round(v*1000,2) for v in q]} mm "
                      f"force {[round(v,1) for v in f]} N  beaker xy "
                      f"{[round(v*1000,1) for v in bp[:2].tolist()]} mm")
        show("closed")
        z_before = float(pose("beaker")[2])
        lift = [bx, by, rim_z + 0.10]
        servo(lift, -1.0, 8.0, vmax=0.08, tol=0.003)
        show("lifted")
        rel0 = pose("beaker") - pol["robot0_eef_pos"][0]
        servo(lift, -1.0, 2.0, vmax=0.08)                  # hold 2 s
        rel1 = pose("beaker") - pol["robot0_eef_pos"][0]
        lifted = float(pose("beaker")[2]) - z_before
        slip = float((rel1 - rel0).norm())
        ok = lifted >= 0.09 and slip <= 0.005
        print(f"   lifted {lifted*1000:.1f} mm (need 90), slip over 2 s hold {slip*1000:.2f} mm "
              f"(need <=5) -> {'PASS' if ok else 'FAIL'}")

        # ---- robot1 answers its own half of the action, in its own base frame ----
        # +x in robot1's base frame is -x in the world (it is yawed 180 deg).
        p1 = pol["robot1_eef_pos"][0].clone()
        for _ in range(3 * steps_per_s):
            a = act((0, 0, 0), (0, 0, 0), -1.0)            # robot0 keeps holding
            a[0, 7:10] = torch.tensor([0.004, 0.0, 0.004], device=u.device) / cfg.actions.robot1_arm.scale
            obs, *_ = env.step(a)
            pol = obs["policy"]
        d = (pol["robot1_eef_pos"][0] - p1).tolist()
        print(f"ROBOT1: commanded +x,+z in its base frame; grasp point moved world "
              f"{[round(v*1000,1) for v in d]} mm (expect x<0, z>0, y~0)")
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
