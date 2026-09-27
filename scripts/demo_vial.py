"""Scripted demo of the vial task, recorded as a Mimic source dataset.

Picks the vial, sets it on the hotplate, lets go, picks it up again and puts it
back on the bench -- driven only through the task's 14-d relative-IK action,
the interface a teleop operator uses. Proves the task is doable in the sim and
gives MimicGen a source demo to test generation with before real demos exist.
It is NOT a substitute for human demos: every motion is a straight line.

    python demo_vial.py [--out demos/vial_scripted.hdf5]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")

import torch  # noqa: E402
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/home/ujaan/isaac/labgen/demos/vial_scripted.hdf5")
add_launcher_args(ap)
args = ap.parse_args()

import gymnasium as gym  # noqa: E402

import labgen_tasks  # noqa: E402,F401
from labgen_tasks.recorders import MimicAnnotatedRecorderManagerCfg  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402
from labgen_tasks import vial_hotplate as V  # noqa: E402

# The Mimic variant, so the demo is recorded already annotated for MimicGen
# (labgen_tasks.recorders); the task itself is identical.
TASK = "Isaac-LabBench-YAM-VialHotplate-IK-Rel-Mimic-v0"
DOWN = torch.tensor([0.0, 0.0, -1.0])
CARRY_Z = 0.20            # grasp-point height in transit: vial base clears the 105 mm plate


def wxyz_to_R(q):
    w, x, y, z = q
    return torch.tensor([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main() -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = V.VialHotplateMimicEnvCfg()
    cfg.recorders = MimicAnnotatedRecorderManagerCfg()
    cfg.recorders.dataset_export_dir_path = str(out.parent)
    cfg.recorders.dataset_filename = out.stem
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    with launch_simulation(cfg, args):
        env = gym.make(TASK, cfg=cfg)
        u = env.unwrapped
        hz = round(1.0 / (cfg.sim.dt * cfg.decimation))
        scale = cfg.actions.robot0_arm.scale
        obs, _ = env.reset()
        state = {"obs": obs, "done": False, "steps": 0}

        def pol():
            return state["obs"]["policy"]

        def signals():
            t = state["obs"]["subtask_terms"]
            return {k: int(t[k][0].item()) for k in ("grasp_1", "place_1", "grasp_2")}

        def xyz(name):
            return pol()[f"{name}_pose"][0, :3].cpu()

        def servo(target, grip, seconds, vmax=0.15, tol=0.002, dwell=False):
            """Move toward target; `dwell` holds for the full time (jaw open/close)."""
            for _ in range(int(seconds * hz)):
                if state["done"]:
                    return 0.0
                p = pol()["robot0_eef_pos"][0].cpu()
                R = wxyz_to_R(pol()["robot0_eef_quat"][0].cpu().tolist())
                err = torch.tensor(target, dtype=torch.float32) - p
                step = err.clamp(-vmax / hz, vmax / hz)
                z = R[:, 2]
                axis = torch.linalg.cross(z, DOWN)
                ang = math.atan2(float(axis.norm()), float(z @ DOWN))
                rot = axis / axis.norm() * min(ang, 0.1) if axis.norm() > 1e-6 else torch.zeros(3)
                a = torch.zeros(env.action_space.shape, device=u.device)
                a[0, 0:3] = step.to(u.device) / scale
                a[0, 3:6] = rot.to(u.device) / scale
                a[0, 6] = grip
                a[0, 13] = 1.0
                obs, _, term, trunc, _ = env.step(a)
                state["steps"] += 1
                if bool(term[0]) or bool(trunc[0]):
                    state["done"] = True           # success fired; the env has reset and exported
                    return 0.0
                state["obs"] = obs
                if not dwell and float(err.norm()) < tol and ang < math.radians(2):
                    break
            return float((torch.tensor(target) - pol()["robot0_eef_pos"][0].cpu()).norm())

        def log(tag):
            v = xyz("vial")
            print(f"   [{tag:10s}] step {state['steps']:4d}  vial {[round(float(c)*1000, 1) for c in v]} mm  "
                  f"jaw {float(pol()['robot0_joint_pos'][0, 6])*1000:6.1f} mm  signals {signals()}")

        OPEN, CLOSE = 1.0, -1.0
        g = V.VIAL_GRASP_Z
        v0 = xyz("vial")
        hp = xyz("hotplate")
        plate = hp + torch.tensor(V.HOT.keypoints["plate_center"][:2] + (0.0,))
        print(f"vial at {[round(float(c), 3) for c in v0]}  plate centre {[round(float(c), 3) for c in plate]}")
        log("start")

        # 1. grasp the vial
        servo([v0[0], v0[1], CARRY_Z], OPEN, 6)
        servo([v0[0], v0[1], v0[2] + g], OPEN, 5, vmax=0.06)
        servo([v0[0], v0[1], v0[2] + g], CLOSE, 1.2, vmax=0.0, dwell=True)
        log("grasped")
        # 2. carry it over the plate and set it down, let go, back off
        servo([v0[0], v0[1], CARRY_Z], CLOSE, 4, vmax=0.08)
        servo([plate[0], plate[1], CARRY_Z], CLOSE, 6, vmax=0.10)
        servo([plate[0], plate[1], V.PLATE_Z + g + 0.002], CLOSE, 6, vmax=0.04)
        servo([plate[0], plate[1], V.PLATE_Z + g + 0.002], OPEN, 1.2, vmax=0.0, dwell=True)
        log("on plate")
        servo([plate[0], plate[1], CARRY_Z], OPEN, 4, vmax=0.08)
        # 3. grasp it again where it actually stands
        v1 = xyz("vial")
        servo([v1[0], v1[1], v1[2] + g], OPEN, 5, vmax=0.06)
        servo([v1[0], v1[1], v1[2] + g], CLOSE, 1.2, vmax=0.0, dwell=True)
        log("regrasped")
        # 4. take it off the plate, back to where it started
        servo([v1[0], v1[1], CARRY_Z], CLOSE, 4, vmax=0.08)
        servo([v0[0], v0[1], CARRY_Z], CLOSE, 6, vmax=0.10)
        servo([v0[0], v0[1], g + 0.002], CLOSE, 9, vmax=0.04)   # 167 mm at 4 cm/s plus lag
        if not state["done"]:
            log("on bench")
        servo([v0[0], v0[1], g + 0.002], OPEN, 1.5, vmax=0.0, dwell=True)
        servo([v0[0], v0[1], CARRY_Z], OPEN, 3, vmax=0.08)
        print(f"SUCCESS -> episode ended and exported to {out}" if state["done"]
              else "NO SUCCESS: the episode did not satisfy task_success")
        env.close()
    return 0 if state["done"] else 1


if __name__ == "__main__":
    sys.exit(main())
