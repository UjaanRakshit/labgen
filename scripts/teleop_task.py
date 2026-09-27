"""Phone teleop of Isaac-LabBench-YAM-IK-Rel-v0, recording demos for MimicGen.

    python teleop_task.py [--stream] [--out demos/bench.hdf5] [--seconds N]

The execution model is COBALT's -- a phone drives a relative-IK Isaac Lab task
at 20 Hz and demos are recorded as HDF5 -- built from our own parts:

  input      labgen.devices.TcpPoseSource from scripts/teleop_touch.py (any
             phone) or teleop_bridge.py (WebXR), unchanged
  anchoring  labgen.devices.RelativeRetargeter, one per arm; hand "left"
             drives robot0, "right" robot1
  action     each tick: (retargeted target - MEASURED grasp point), in that
             arm's base frame, clipped to MAX_SPEED -- so the loop closes on
             where the hand actually is and the soft real gains cannot make
             the error accumulate
  view       Newton's viewer, headless, JPEG frames pushed back to the phone
             page (--stream), as in teleop_sim.py
  recording  Isaac Lab's own recorder (ActionStateRecorderManagerCfg), the
             format isaaclab_mimic consumes. SAVE DEMO on the phone exports the
             episode and resets; DISCARD drops it and resets.

Every dataset is stamped with what is unverified in the setup that made it.
"""
from __future__ import annotations

import argparse
import io
import socket
import sys
import time
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

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--host", default="AUTO")
ap.add_argument("--port", type=int, default=9871)
ap.add_argument("--stream", action="store_true", help="push JPEG frames to the phone page")
ap.add_argument("--frame-port", type=int, default=9872)
ap.add_argument("--view-width", type=int, default=960)
ap.add_argument("--view-height", type=int, default=540)
ap.add_argument("--scale", type=float, default=1.0, help="device-to-hand motion scale")
ap.add_argument("--out", default="demos/bench.hdf5")
ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = never)")
ap.add_argument("--task", default="bench", choices=("bench", "vial"),
                help="bench: free play. vial: pick the vial, set it on the hotplate, take "
                     "it off -- recorded already annotated for MimicGen, auto-saved on success")
add_launcher_args(ap)
args = ap.parse_args()

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import labgen_tasks  # noqa: E402,F401
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402
from labgen.devices import RelativeRetargeter, TcpPoseSource, Workspace  # noqa: E402
from labgen_tasks import bench_yam as B  # noqa: E402

TASKS = {"bench": "Isaac-LabBench-YAM-IK-Rel-v0",
         "vial": "Isaac-LabBench-YAM-VialHotplate-IK-Rel-Mimic-v0"}
TASK = TASKS[args.task]
MAX_SPEED_M_S = 0.25            # grasp point speed cap per tick
MAX_TURN_RAD = 0.10             # per-tick orientation correction cap
DOWN = np.array([0.0, 0.0, -1.0])
LOOK_AT = np.array([0.05, 0.35, 0.12])
EYE = LOOK_AT + np.array([1.05, 0.05, 0.70])
# Teleop convenience, not a physical claim: keep the grasp point over the bench
# and above its surface so a slip of the thumb cannot drive a finger into it.
BENCH_BOX = Workspace(lower_m=(-0.45, -0.05, 0.005), upper_m=(0.45, 0.75, 0.45))
HANDS = {"left": 0, "right": 1}
UNVERIFIED = [
    "finger pads (size, friction): measured off the URDF mesh, not calipered",
    f"grip force cap {B.GRIP_FORCE_N} N and finger PD gains: controller choices",
    f"robot1 placement {B.ROBOT1_POS}: PLACEHOLDER, rig not measured",
    "scene object dimensions: see the SceneSpec's unsourced list",
    "gravity compensation 1.0 (real controller uses 1.1-1.2 on j2-j4)",
]


def wxyz_to_R(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def screen_frame() -> np.ndarray:
    """Device axes aligned to the camera: 'drag up the screen' = away from viewer."""
    fwd = LOOK_AT - EYE
    fwd[2] = 0.0
    fwd /= np.linalg.norm(fwd)
    right = np.array([fwd[1], -fwd[0], 0.0])
    return np.column_stack([right, fwd, np.array([0.0, 0.0, 1.0])])


def aim(viewer) -> None:
    """Point the camera (from teleop_sim.aim). Without it the viewer opens
    looking at nothing, which is indistinguishable from a broken renderer."""
    d = LOOK_AT - EYE
    viewer.set_camera(tuple(float(v) for v in EYE),
                      float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1])))),
                      float(np.degrees(np.arctan2(d[1], d[0]))))


def point_down(R_hand: np.ndarray) -> np.ndarray:
    """World-frame rotation vector turning the gripper's approach axis to -z."""
    z = R_hand[:, 2]
    axis = np.cross(z, DOWN)
    s = float(np.linalg.norm(axis))
    ang = float(np.arctan2(s, float(z @ DOWN)))
    return np.zeros(3) if s < 1e-9 else axis / s * min(ang, MAX_TURN_RAD)


def stamp(path: Path, n_saved: int) -> None:
    import h5py
    with h5py.File(path, "a") as f:
        f.attrs["labgen_task"] = TASK
        f.attrs["labgen_unverified"] = "\n".join(UNVERIFIED)
        f.attrs["labgen_demos_saved"] = n_saved


def main() -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.task == "vial":
        from labgen_tasks import vial_hotplate as V
        from labgen_tasks.recorders import MimicAnnotatedRecorderManagerCfg
        cfg = V.VialHotplateMimicEnvCfg()
        # Success ends the episode and the recorder exports it: completing the
        # task IS the save. SAVE DEMO / DISCARD still work as overrides.
        recorder = MimicAnnotatedRecorderManagerCfg()
    else:
        cfg = B.BenchYamIkRelEnvCfg()
        recorder = ActionStateRecorderManagerCfg()
    cfg.terminations.time_out = None             # the operator ends episodes
    if args.stream:
        # Off by default kit-less: the arms' link meshes arrive only as
        # invisible colliders, and the streamed view showed a bench with no
        # robot on it.
        cfg.sim.physics.load_visual_shapes = True
    cfg.recorders = recorder
    cfg.recorders.dataset_export_dir_path = str(out.parent)
    cfg.recorders.dataset_filename = out.stem
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    if args.host == "AUTO":
        # Under WSL2 the phone bridge runs on the Windows host, reached at the
        # default gateway. On native Linux the gateway is the router -- the
        # bridge is on this machine.
        wsl = "microsoft" in Path("/proc/version").read_text().lower() if Path("/proc/version").exists() else False
        host = TcpPoseSource.wsl_default_gateway() if wsl else "127.0.0.1"
    else:
        host = args.host

    with launch_simulation(cfg, args):
        env = gym.make(TASK, cfg=cfg)
        u = env.unwrapped
        hz = 1.0 / (cfg.sim.dt * cfg.decimation)
        scale = cfg.actions.robot0_arm.scale
        obs, _ = env.reset()
        pol = obs["policy"]
        # Connect only once the sim is up: a phone driving during the ~30 s
        # build would be commanding an arm that does not exist yet.
        print(f"connecting to the phone bridge at {host}:{args.port} ...")
        source = TcpPoseSource(host, args.port)
        print("connected.")

        frame = screen_frame()
        arms = [dict(ret=RelativeRetargeter(position_scale=args.scale, use_orientation=False,
                                            frame=frame, workspace=BENCH_BOX),
                     target=None, grip_closed=False, grip=0.0) for _ in range(2)]

        viewer = sock = None
        if args.stream:
            from newton.viewer import ViewerGL
            from PIL import Image
            from isaaclab_newton.physics.newton_manager import NewtonManager as NM
            viewer = ViewerGL(width=args.view_width, height=args.view_height,
                              headless=True, vsync=False)
            viewer.set_model(NM._model)
            sock = socket.create_connection((host, args.frame_port), timeout=10.0)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"streaming frames to {host}:{args.frame_port}")

        last_cmd_seq, saved, discarded, ticks = None, 0, 0, 0
        started = last_report = time.perf_counter()
        next_tick = started
        step_ms, window = 0.0, 0
        print(f"ready at {hz:.0f} Hz. ENGAGE on the phone to drive (LEFT = robot0, "
              f"RIGHT = robot1); SAVE DEMO / DISCARD end the episode.")
        try:
            while not (args.seconds and time.perf_counter() - started > args.seconds):
                ev = source.poll_latest()
                if source.closed:
                    print("phone bridge closed the connection")
                    break

                reset = None
                if ev is not None:
                    if last_cmd_seq is None:
                        last_cmd_seq = ev.command_seq      # ignore presses from before we joined
                    elif ev.command_seq != last_cmd_seq:
                        last_cmd_seq = ev.command_seq
                        reset = ev.command
                    i = HANDS.get(ev.hand, 0)
                    arm = arms[i]
                    hand_now = pol[f"robot{i}_eef_pos"][0].cpu().numpy()
                    tgt = arm["ret"].update(ev, hand_now)
                    arm["target"] = None if not arm["ret"].engaged else (
                        tgt.position_m if tgt is not None else arm["target"])
                    arm["grip_closed"] = ev.grip > 0.5
                    arm["grip"] = float(min(max(ev.grip, 0.0), 1.0))

                if reset in ("save", "discard"):
                    rm = u.recorder_manager
                    if reset == "save":
                        rm.record_pre_reset([0], force_export_or_skip=False)
                        rm.set_success_to_episodes([0], torch.tensor([[True]], device=u.device))
                        rm.export_episodes([0])
                        saved += 1
                        print(f"SAVED demo {saved} ({ticks} steps) -> {out}")
                    else:
                        discarded += 1
                        print(f"discarded episode ({ticks} steps)")
                    rm.reset()
                    obs, _ = env.reset()
                    pol = obs["policy"]
                    for arm in arms:
                        arm["ret"].release()
                        arm["target"], arm["grip_closed"], arm["grip"] = None, False, 0.0
                    ticks = 0
                    continue

                a = torch.zeros(env.action_space.shape, device=u.device)
                for i, arm in enumerate(arms):
                    Rb = wxyz_to_R(pol[f"robot{i}_base_ori"][0].cpu().numpy())
                    if arm["target"] is not None:
                        p = pol[f"robot{i}_eef_pos"][0].cpu().numpy()
                        Rh = wxyz_to_R(pol[f"robot{i}_eef_quat"][0].cpu().numpy())
                        d = np.clip(np.asarray(arm["target"]) - p, -MAX_SPEED_M_S / hz, MAX_SPEED_M_S / hz)
                        a[0, 7 * i:7 * i + 3] = torch.as_tensor(Rb.T @ d / scale)
                        a[0, 7 * i + 3:7 * i + 6] = torch.as_tensor(Rb.T @ point_down(Rh) / scale)
                    # Proportional, like the lab's trigger: slider 0 = open (+1), 1 = closed (-1).
                    a[0, 7 * i + 6] = 1.0 - 2.0 * arm["grip"]

                t0 = time.perf_counter()
                obs, _, term, trunc, _ = env.step(a)
                if bool(term[0]) or bool(trunc[0]):
                    # The task's success check fired: the env has already
                    # exported the episode and reset itself.
                    saved += 1
                    print(f"SUCCESS -> demo {saved} saved automatically ({ticks} steps) -> {out}")
                    pol = obs["policy"]
                    for arm in arms:
                        arm["ret"].release()
                        arm["target"], arm["grip_closed"], arm["grip"] = None, False, 0.0
                    ticks = 0
                    continue
                step_ms += (time.perf_counter() - t0) * 1000
                window += 1
                pol = obs["policy"]
                ticks += 1

                if viewer is not None:
                    from isaaclab_newton.physics.newton_manager import NewtonManager as NM
                    aim(viewer)
                    viewer.begin_frame(ticks / hz)
                    viewer.log_state(NM._state_0)
                    viewer.end_frame()
                    f = viewer.get_frame()
                    if f is not None:
                        arr = f.numpy() if hasattr(f, "numpy") else np.asarray(f)
                        if arr.dtype != np.uint8:
                            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                        buf = io.BytesIO()
                        Image.fromarray(arr[:, :, :3]).save(buf, format="JPEG", quality=70)
                        jpeg = buf.getvalue()
                        try:
                            sock.sendall(len(jpeg).to_bytes(4, "big") + jpeg)
                        except OSError:
                            print("frame stream closed")
                            viewer = None

                now = time.perf_counter()
                if now - last_report > 5.0:
                    live = [f"robot{i}{' ENGAGED' if a_['target'] is not None else ''}"
                            f"{' GRIP' if a_['grip_closed'] else ''}" for i, a_ in enumerate(arms)]
                    print(f"   step {step_ms / max(window, 1):5.1f} ms (budget {1000 / hz:.0f}) | "
                          f"{' | '.join(live)} | saved {saved} discarded {discarded}", flush=True)
                    last_report, step_ms, window = now, 0.0, 0
                next_tick += 1.0 / hz
                sleep = next_tick - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.perf_counter()
        except KeyboardInterrupt:
            print("\nstopped by operator")
        finally:
            source.close()
            env.close()
    if saved and out.exists():
        stamp(out, saved)
    print(f"done: {saved} demo(s) saved to {out}, {discarded} discarded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
