"""Replay a recorded demo open-loop, the way Isaac Lab Mimic's annotator does,
and report where it departs from what was recorded.

MimicGen's auto-annotation replays each demo's actions from its recorded
initial state and requires the replay to succeed. This measures, step by step,
how far the replayed vial and jaw drift from the recording, so a failed
annotation says WHY instead of just "the final task was not completed".

    python replay_check.py demos/x.hdf5 [--deterministic]
"""
from __future__ import annotations

import argparse
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

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--demo", default="demo_0")
ap.add_argument("--det", action="store_true", help="NewtonCfg.deterministic_mode='run_to_run'")
ap.add_argument("--no-sim-reset", action="store_true", help="skip env.sim.reset() (the annotator calls it)")
add_launcher_args(ap)
args = ap.parse_args()

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import labgen_tasks  # noqa: E402,F401
from labgen_tasks import vial_hotplate as V  # noqa: E402

TASK = "Isaac-LabBench-YAM-VialHotplate-IK-Rel-v0"


def load(group):
    if isinstance(group, h5py.Dataset):
        return torch.as_tensor(group[()])
    return {k: load(v) for k, v in group.items()}


def main() -> int:
    with h5py.File(args.dataset) as f:
        d = f["data"][args.demo]
        actions = torch.as_tensor(d["actions"][()])
        init = load(d["initial_state"])
        rec_vial = d["states/rigid_object/vial/root_pose"][()]
        rec_jaw = d["states/articulation/robot0/joint_position"][()][:, 6]
    cfg = V.VialHotplateEnvCfg()
    cfg.terminations.success = None
    cfg.terminations.time_out = None
    if args.det:
        cfg.sim.physics.deterministic_mode = "run_to_run"
        cfg.sim.physics.solver_cfg.disable_sensors = True
    with launch_simulation(cfg, args):
        env = gym.make(TASK, cfg=cfg)
        u = env.unwrapped
        env.reset()
        if not args.no_sim_reset:
            u.sim.reset()
            from labgen_tasks.envs import restore_actuator_gains
            restore_actuator_gains(u)
        u.reset_to(init, None, is_relative=True)
        art = u.scene["robot0"]
        names = list(art.joint_names)
        for attr in ("joint_stiffness", "joint_damping", "joint_effort_limits", "joint_pos_limits"):
            v = getattr(art.data, attr, None)
            if v is not None:
                t = v.torch if hasattr(v, "torch") else v
                print(f"   {attr}: joint2 {t[0, names.index('joint2')].tolist()}  "
                      f"joint7 {t[0, names.index('joint7')].tolist()}  joint8 {t[0, names.index('joint8')].tolist()}")
        first = None
        worst = 0.0
        for i, a in enumerate(actions):
            obs, *_ = env.step(a.reshape(1, -1).to(u.device))
            if i + 1 >= len(rec_vial):
                break
            vp = obs["policy"]["vial_pose"][0, :3].cpu().numpy()
            # states[i] is recorded BEFORE action i, so compare step i's result to states[i+1]
            err = float(np.linalg.norm(vp - rec_vial[i + 1, :3]))
            worst = max(worst, err)
            if 194 <= i <= 200:
                term = u.action_manager.get_term("robot0_gripper")
                print(f"   dbg step {i}: action[6]={float(a[6]):+.2f} processed "
                      f"{term.processed_actions[0].tolist()} target {term._target[0].tolist()} "
                      f"jaw {float(obs['policy']['robot0_joint_pos'][0, 6])*1000:.2f}")
            if first is None and err > 1e-3:
                jaw = float(obs["policy"]["robot0_joint_pos"][0, 6])
                first = i
                print(f"first >1 mm vial departure at step {i}: {err*1000:.2f} mm "
                      f"(jaw replay {jaw*1000:.2f} vs recorded {rec_jaw[i+1]*1000:.2f} mm)")
            if i % 100 == 0:
                print(f"   step {i:4d} vial err {err*1000:8.3f} mm  signals "
                      f"{[int(obs['subtask_terms'][k][0].item()) for k in ('grasp_1','place_1','grasp_2')]}")
        ok = bool(V.task_success(u)[0])
        print(f"replayed {len(actions)} actions: worst vial error {worst*1000:.2f} mm, "
              f"success at end: {ok}")
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
