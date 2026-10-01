"""Phone teleop of Isaac-LabBench-YAM-IK-Rel-v0, recording demos for MimicGen.

    python teleop_task.py [--stream] [--out demos/bench.hdf5] [--seconds N]

The execution model is COBALT's -- a phone drives a relative-IK Isaac Lab task
at 20 Hz and demos are recorded as HDF5 -- built from our own parts:

  input      labgen.devices.TcpPoseSource from scripts/teleop_touch.py (any
             phone) or teleop_bridge.py (WebXR), unchanged
  anchoring  labgen.devices.RelativeRetargeter, one per arm; hand "right"
             drives robot0 (the right arm), "left" robot1
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
import threading
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
ap.add_argument("--view-width", type=int, default=640)
ap.add_argument("--view-height", type=int, default=360)
ap.add_argument("--scale", type=float, default=1.0, help="device-to-hand motion scale")
ap.add_argument("--out", default="demos/bench.hdf5")
ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = never)")
ap.add_argument("--input", default="touch", choices=("touch", "cobalt"),
                help="touch: scripts/teleop_touch.py (any phone, drag pad). cobalt: the COBALT "
                     "phone app, unchanged, via scripts/cobalt_app_bridge.py -- 6-DoF, COBALT's mapping")
ap.add_argument("--rotation", action="store_true",
                help="cobalt input: also track the phone's rotation (6-DoF, as COBALT). Default: "
                     "the phone moves the gripper's POSITION and the gripper stays pointing down")
ap.add_argument("--trace", help="write a per-tick trace (.npz) of phone motion, target and gripper")
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
MAX_SPEED_M_S = 0.50            # fastest the target may move, m/s (the phone can be faster)
# Integral trim on the Cartesian target. The real gains (kp 80) and the fitted
# joint friction leave the gripper 5-7 mm short of a held target (axis test:
# lead exact, gripper off by that much). An operator closes that loop by eye; this
# does it for them. Only while nearly settled (no windup during moves), bounded,
# reset on every engage. A teleop controller choice; the arm model is unchanged.
# It integrates only once the target has stayed within TRIM_STILL_BAND_M for
# TRIM_STILL_S: integrating the lag DURING a move wound it up and overshot every
# stop by 5-6 mm. Measured with it (axis test, scale 3): holds within 0.3 mm
# horizontally, 2.2 mm vertically 1 s after the stop.
TRIM_STILL_S = 0.15
TRIM_STILL_BAND_M = 0.003
TRIM_TAU_S = 0.25
TRIM_NEAR_M = 0.015
TRIM_MAX_M = 0.012
LEAD_M = 0.08                   # COBALT input: how far the target may run ahead of the arm
                                # (0.04 dropped 50 of 120 mm on a fast 0.3 s flick)
# Teleop convenience, not a physical claim: keep the target in a shell the arm can
# reach, so it is never driven into its own base or past full extension, where
# the IK folds the arm in ways the operator cannot predict.
REACH_MIN_M, REACH_MAX_M = 0.18, 0.60      # horizontal distance from that arm's base
Z_MAX_M = 0.50
# Where each gripper is parked, pointing down, before the operator takes over:
# relative to its arm's base. The rig's reset pose puts the grasp point 0.45 m
# out and 0.33 m up -- nearly stretched for a pointing-down gripper. Measured
# from there (axis test through the app protocol): a 120 mm forward command
# moved the gripper 14 mm forward and 50 mm DOWN, "up" pulled it 60-100 mm back,
# and it held 8 mm off target. A sim convenience; the rig's reset pose is unchanged.
READY_OFFSET_M = np.array([0.30, 0.0, 0.20])
READY_TOL_M = 0.002             # park this close (with the trim) before handing over
LEAD_RAD = 0.35
MAX_TURN_RAD_S = 2.0            # orientation correction cap, rad/s -- PER SECOND: a per-tick cap
                                # of 0.1 rad became 10 rad/s at 100 Hz and shook the soft wrist
MAX_TURN_RAD = MAX_TURN_RAD_S / 20.0   # set per tick from the control rate in main()
DOWN = np.array([0.0, 0.0, -1.0])
# Behind and above the two arms, looking forward (+x) -- where the operator stands,
# so 'drag up the screen' is 'away from me' and the left arm is on the left.
LOOK_AT = np.array([0.35, 0.325, 0.05])
EYE = np.array([-0.75, 0.325, 0.85])
# Teleop convenience, not a physical claim: keep the grasp point over the bench
# and above its surface so a slip of the thumb cannot drive a finger into it.
BENCH_BOX = Workspace(lower_m=(-0.45, -0.05, 0.005), upper_m=(0.45, 0.75, 0.45))
HANDS = {"right": 0, "left": 1}    # robot0 = right arm, robot1 = left arm (measured layout)
UNVERIFIED = [
    "finger pads (size, friction): measured off the URDF mesh, not calipered",
    f"grip force cap {B.GRIP_FORCE_N} N and finger PD gains: controller choices",
    f"object layout on the bench: hand-authored PLACEHOLDER (arm layout measured 2026-09-28)",
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


def aim(viewer, eye=None, look=None) -> None:
    """Point the camera (from teleop_sim.aim). Without it the viewer opens
    looking at nothing, which is indistinguishable from a broken renderer."""
    eye = EYE if eye is None else np.asarray(eye, float)
    look = LOOK_AT if look is None else np.asarray(look, float)
    d = look - eye
    viewer.set_camera(tuple(float(v) for v in eye),
                      float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1])))),
                      float(np.degrees(np.arctan2(d[1], d[0]))))


def track_rotation(R_target: np.ndarray, R_hand: np.ndarray) -> np.ndarray:
    """World-frame rotation vector from the gripper's orientation toward the target's, capped."""
    R = R_target @ R_hand.T
    c = (np.trace(R) - 1.0) / 2.0
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    s = float(np.linalg.norm(v))
    ang = float(np.arctan2(s, c))
    if s < 1e-9:
        return np.zeros(3)
    return v / s * min(ang, MAX_TURN_RAD)


def _slerp_toward(R0: np.ndarray, R1: np.ndarray, k: float) -> np.ndarray:
    """The rotation k of the way from R0 to R1 (0 <= k <= 1), about the shortest axis."""
    R = R1 @ R0.T
    c = (np.trace(R) - 1.0) / 2.0
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return R0.copy()
    ang = float(np.arctan2(s, c)) * k
    ax = v / s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K) @ R0


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


class RenderWorker(threading.Thread):
    """Renders the sim view off the control loop.

    One 640x360 render costs ~44 ms here (WSL GL plus readback), more than a
    40 Hz control tick. Inline, it held the loop at 9-17 Hz. This thread owns
    its own viewer and state copy, takes the newest body poses whenever the loop
    publishes them, and renders the overview and a gripper close-up alternately,
    side by side -- as fast as it can, never blocking control.
    """

    def __init__(self, model, sock, size, eye, look):
        super().__init__(daemon=True)
        self.model, self.sock, self.size = model, sock, size
        self.eye, self.look = eye, look
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._snap = None
        self.frames = 0
        self.alive = True

    def submit(self, body_q: np.ndarray, grip_world: np.ndarray) -> None:
        with self._lock:
            self._snap = (body_q, grip_world)
        self._new.set()

    def run(self) -> None:
        from newton.viewer import ViewerGL
        from PIL import Image
        viewer = ViewerGL(width=self.size[0], height=self.size[1], headless=True, vsync=False)
        viewer.set_model(self.model)
        state = self.model.state()
        panes, k, t = [None, None], 0, 0.0
        while self.alive:
            if not self._new.wait(timeout=1.0):
                continue
            self._new.clear()
            with self._lock:
                bq, g = self._snap
            state.body_q.assign(bq)
            eye, look = ((self.eye, self.look) if k == 0 else
                         (g + np.array([-0.30, 0.0, 0.22]), g + np.array([0.08, 0.0, -0.04])))
            aim(viewer, eye, look)
            t += 0.05
            viewer.begin_frame(t)
            viewer.log_state(state)
            viewer.end_frame()
            f = viewer.get_frame()
            if f is None:
                continue
            arr = f.numpy() if hasattr(f, "numpy") else np.asarray(f)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            panes[k] = arr[:, :, :3]
            k ^= 1
            if panes[0] is None or panes[1] is None:
                continue
            buf = io.BytesIO()
            Image.fromarray(np.concatenate(panes, axis=1)).save(buf, format="JPEG", quality=70)
            jpeg = buf.getvalue()
            try:
                self.sock.sendall(len(jpeg).to_bytes(4, "big") + jpeg)
                self.frames += 1
            except OSError:
                print("frame stream closed", flush=True)
                return


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
        global MAX_TURN_RAD
        MAX_TURN_RAD = MAX_TURN_RAD_S / hz
        scale = cfg.actions.robot0_arm.scale
        obs, _ = env.reset()
        pol = obs["policy"]
        # Connect only once the sim is up: a phone driving during the ~30 s
        # build would be commanding an arm that does not exist yet.
        print(f"connecting to the phone bridge at {host}:{args.port} ...")
        source = TcpPoseSource(host, args.port)
        print("connected.")

        if args.input == "cobalt":
            # The bridge already applies COBALT's YAM mapping (robot frame, /1.5):
            # no screen frame, and the phone's rotation is tracked, as in COBALT.
            ret_kw = dict(position_scale=args.scale, use_orientation=True, frame=None)
        else:
            ret_kw = dict(position_scale=args.scale, use_orientation=False, frame=screen_frame())
        arms = [dict(ret=RelativeRetargeter(workspace=BENCH_BOX, **ret_kw),
                     target=None, target_R=None, grip_closed=False, grip=0.0,
                     # COBALT input: phone motion since the last tick, applied to the
                     # arm where it IS (see below), never integrated into a target.
                     dev_p=None, dev_R=None, d_p=np.zeros(3), d_R=np.eye(3),
                     lead_p=None, lead_R=None) for _ in range(2)]

        renderer = None
        if args.stream:
            from isaaclab_newton.physics.newton_manager import NewtonManager as NM
            sock = socket.create_connection((host, args.frame_port), timeout=10.0)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            renderer = RenderWorker(NM._model, sock, (args.view_width, args.view_height), EYE, LOOK_AT)
            renderer.start()
            print(f"streaming frames to {host}:{args.frame_port} (render thread)")

        def go_ready(obs, max_s=6.0):
            """Park both grippers at READY_OFFSET_M (pointing down) before handing over."""
            pol_ = obs["policy"]
            sps, trims, k_ = [None, None], [np.zeros(3), np.zeros(3)], 0
            for k_ in range(int(max_s * hz)):
                a_ = torch.zeros(env.action_space.shape, device=u.device)
                worst = 0.0
                for i_ in range(2):
                    base_ = np.asarray(B.ROBOT1_POS if i_ == 1 else (0.0, 0.0, 0.0), float)
                    p_ = pol_[f"robot{i_}_eef_pos"][0].cpu().numpy()
                    Rh_ = wxyz_to_R(pol_[f"robot{i_}_eef_quat"][0].cpu().numpy())
                    Rb_ = wxyz_to_R(pol_[f"robot{i_}_base_ori"][0].cpu().numpy())
                    e_ = base_ + READY_OFFSET_M - p_
                    worst = max(worst, float(np.linalg.norm(e_)))
                    if sps[i_] is None:
                        sps[i_] = p_.copy()
                    g_ = base_ + READY_OFFSET_M - sps[i_]
                    gn = float(np.linalg.norm(g_))
                    sps[i_] = sps[i_] + (g_ if gn <= 0.25 / hz else g_ * (0.25 / hz / gn))
                    if gn <= 0.25 / hz and float(np.linalg.norm(e_)) < TRIM_NEAR_M:   # arrived: trim
                        trims[i_] = trims[i_] + e_ / (TRIM_TAU_S * hz)
                        tn_ = float(np.linalg.norm(trims[i_]))
                        if tn_ > TRIM_MAX_M:
                            trims[i_] = trims[i_] * (TRIM_MAX_M / tn_)
                    d_ = sps[i_] - p_ + trims[i_]
                    a_[0, 7 * i_:7 * i_ + 3] = torch.as_tensor(Rb_.T @ d_ / scale)
                    a_[0, 7 * i_ + 3:7 * i_ + 6] = torch.as_tensor(Rb_.T @ point_down(Rh_) / scale)
                    a_[0, 7 * i_ + 6] = 1.0
                obs, *_ = env.step(a_)
                pol_ = obs["policy"]
                if worst < READY_TOL_M:
                    break
            print(f"   grippers parked at the ready pose ({worst*1000:.1f} mm, {(k_ + 1) / hz:.1f} s)", flush=True)
            return obs

        obs = go_ready(obs)
        pol = obs["policy"]
        last_cmd_seq, saved, discarded, ticks = None, 0, 0, 0
        started = last_report = time.perf_counter()
        next_tick = started
        step_ms, window = 0.0, 0
        trace, last_tick_t, period_sum, period_n = [], None, 0.0, 0
        render_ms = 0.0
        print(f"ready at {hz:.0f} Hz. ENGAGE on the phone to drive (RIGHT = right arm/robot0, "
              f"LEFT = left arm/robot1); SAVE DEMO / DISCARD end the episode.")
        try:
            while not (args.seconds and time.perf_counter() - started > args.seconds):
                events = source.poll_latest_by_hand()
                if source.closed:
                    print("phone bridge closed the connection")
                    break

                reset = None
                for ev in events.values():
                    if args.input == "cobalt":
                        # Delta control, as COBALT: accumulate the phone's motion since
                        # the previous tick. The bridge's position is cumulative, so the
                        # newest event's difference covers every sample since then.
                        arm_c = arms[HANDS.get(ev.hand, 0)]
                        if ev.engaged:
                            if arm_c["dev_p"] is not None:
                                arm_c["d_p"] = arm_c["d_p"] + (ev.position - arm_c["dev_p"])
                                arm_c["d_R"] = (ev.rotation @ arm_c["dev_R"].T) @ arm_c["d_R"]
                            arm_c["dev_p"], arm_c["dev_R"] = ev.position.copy(), ev.rotation.copy()
                        else:
                            arm_c["dev_p"] = arm_c["dev_R"] = None
                    if last_cmd_seq is None:
                        last_cmd_seq = ev.command_seq      # ignore presses from before we joined
                    elif ev.command_seq != last_cmd_seq:
                        last_cmd_seq = ev.command_seq
                        reset = ev.command
                    i = HANDS.get(ev.hand, 0)
                    arm = arms[i]
                    hand_now = pol[f"robot{i}_eef_pos"][0].cpu().numpy()
                    hand_R = wxyz_to_R(pol[f"robot{i}_eef_quat"][0].cpu().numpy())
                    tgt = arm["ret"].update(ev, hand_now, hand_R)
                    if not arm["ret"].engaged:
                        arm["target"], arm["target_R"] = None, None
                    elif tgt is not None:
                        arm["target"], arm["target_R"] = tgt.position_m, tgt.rotation
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
                    obs = go_ready(obs)
                    pol = obs["policy"]
                    for arm in arms:
                        arm["ret"].release()
                        arm["target"], arm["target_R"], arm["grip_closed"], arm["grip"] = None, None, False, 0.0
                    ticks = 0
                    continue

                a = torch.zeros(env.action_space.shape, device=u.device)
                trace_dp = arms[0]["d_p"].copy()
                for i, arm in enumerate(arms):
                    Rb = wxyz_to_R(pol[f"robot{i}_base_ori"][0].cpu().numpy())
                    if args.input == "cobalt":
                        # The phone's motion moves a target that may lead the arm by at
                        # most LEAD_M / LEAD_RAD; beyond that it is dropped. Pure
                        # integration ran 1.8 m out of reach (arm lunged or went dead);
                        # pure per-tick deltas lost what the soft arm could not do in
                        # one tick (9.7 cm commanded, 7.8 cm done: "slow and vague").
                        p = pol[f"robot{i}_eef_pos"][0].cpu().numpy()
                        Rh = wxyz_to_R(pol[f"robot{i}_eef_quat"][0].cpu().numpy())
                        if arm["dev_p"] is None:
                            arm["lead_p"] = arm["lead_R"] = None
                            arm["trim"], arm["still"], arm["anchor"] = np.zeros(3), 0, None
                            arm["backlog"] = np.zeros(3)
                        else:
                            if arm["lead_p"] is None:
                                arm["lead_p"], arm["lead_R"] = p.copy(), Rh.copy()
                            # Phone motion goes into a backlog that feeds the target at
                            # up to MAX_SPEED; the rest carries to the next tick. The phone
                            # samples at 20 Hz against a 40 Hz loop, so its motion arrives
                            # as 2-tick jumps: clipping without carry-over dropped 45 of
                            # 120 mm on a fast flick. Backlog bounded by LEAD_M (no windup).
                            arm["backlog"] = arm.get("backlog", np.zeros(3)) + arm["d_p"]
                            bn = float(np.linalg.norm(arm["backlog"]))
                            if bn > LEAD_M:
                                arm["backlog"] = arm["backlog"] * (LEAD_M / bn)
                                bn = LEAD_M
                            cap = MAX_SPEED_M_S / hz
                            step_p = arm["backlog"] if bn <= cap else arm["backlog"] * (cap / bn)
                            arm["backlog"] = arm["backlog"] - step_p
                            lp = arm["lead_p"] + step_p
                            lp[2] = min(max(lp[2], BENCH_BOX.lower_m[2]), Z_MAX_M)   # never into the bench
                            base = np.asarray(B.ROBOT1_POS if i == 1 else (0.0, 0.0, 0.0), float)
                            h = lp[:2] - base[:2]
                            r = float(np.linalg.norm(h))
                            if r > 1e-9 and not (REACH_MIN_M <= r <= REACH_MAX_M):
                                lp[:2] = base[:2] + h * (min(max(r, REACH_MIN_M), REACH_MAX_M) / r)
                            off = lp - p
                            n = float(np.linalg.norm(off))
                            arm["lead_p"] = p + off * (LEAD_M / n) if n > LEAD_M else lp
                            # Still = the target stayed within TRIM_STILL_BAND_M (a held
                            # phone jitters; an exact-zero test would never fire).
                            anc = arm.get("anchor")
                            if anc is None or float(np.linalg.norm(arm["lead_p"] - anc)) > TRIM_STILL_BAND_M:
                                arm["anchor"], arm["still"] = arm["lead_p"].copy(), 0
                            else:
                                arm["still"] = arm.get("still", 0) + 1
                            lR = arm["d_R"] @ arm["lead_R"]
                            ang = float(np.arccos(np.clip((np.trace(lR @ Rh.T) - 1) / 2, -1, 1)))
                            if ang > LEAD_RAD:
                                lR = _slerp_toward(Rh, lR, LEAD_RAD / ang)
                            arm["lead_R"] = lR
                            # The full gap to the lead (bounded by LEAD_M), not a per-tick
                            # step: a vmax/hz clip stalls the soft arm at high control rates.
                            d = arm["lead_p"] - p
                            trim = arm.get("trim", np.zeros(3))
                            if (float(np.linalg.norm(d)) < TRIM_NEAR_M
                                    and arm.get("still", 0) >= TRIM_STILL_S * hz):
                                trim = trim + d / (TRIM_TAU_S * hz)
                                tn = float(np.linalg.norm(trim))
                                if tn > TRIM_MAX_M:
                                    trim = trim * (TRIM_MAX_M / tn)
                            arm["trim"] = trim
                            d = d + trim
                            a[0, 7 * i:7 * i + 3] = torch.as_tensor(Rb.T @ d / scale)
                            rot = track_rotation(lR, Rh) if args.rotation else point_down(Rh)
                            a[0, 7 * i + 3:7 * i + 6] = torch.as_tensor(Rb.T @ rot / scale)
                        arm["d_p"], arm["d_R"] = np.zeros(3), np.eye(3)
                    elif arm["target"] is not None:
                        p = pol[f"robot{i}_eef_pos"][0].cpu().numpy()
                        Rh = wxyz_to_R(pol[f"robot{i}_eef_quat"][0].cpu().numpy())
                        d = np.clip(np.asarray(arm["target"]) - p, -MAX_SPEED_M_S / hz, MAX_SPEED_M_S / hz)
                        a[0, 7 * i:7 * i + 3] = torch.as_tensor(Rb.T @ d / scale)
                        rot = (track_rotation(arm["target_R"], Rh) if arm["target_R"] is not None
                               and args.input == "cobalt" else point_down(Rh))
                        a[0, 7 * i + 3:7 * i + 6] = torch.as_tensor(Rb.T @ rot / scale)
                    # Proportional, like the lab's trigger: slider 0 = open (+1), 1 = closed (-1).
                    a[0, 7 * i + 6] = 1.0 - 2.0 * arm["grip"]

                if args.trace:
                    trace.append(np.concatenate([[time.perf_counter() - started,
                                                  float(arms[0]["dev_p"] is not None)],
                                                 pol["robot0_eef_pos"][0].cpu().numpy(),
                                                 arms[0]["lead_p"] if arms[0].get("lead_p") is not None
                                                 else np.full(3, np.nan), trace_dp]))
                now_t = time.perf_counter()
                if last_tick_t is not None:
                    period_sum += now_t - last_tick_t
                    period_n += 1
                last_tick_t = now_t
                t0 = time.perf_counter()
                obs, _, term, trunc, _ = env.step(a)
                if bool(term[0]) or bool(trunc[0]):
                    # The env has reset itself. Only a termination is the success
                    # check (time_out is off above and the recorder exports
                    # successes only); a truncation saved nothing, so say so.
                    if bool(term[0]):
                        saved += 1
                        print(f"SUCCESS -> demo {saved} saved automatically ({ticks} steps) -> {out}")
                        try:
                            source.send({"event": "success"})     # the COBALT app is told 'complete'
                        except OSError:
                            pass
                    else:
                        discarded += 1
                        print(f"episode truncated after {ticks} steps -> not saved")
                    obs = go_ready(obs)
                    pol = obs["policy"]
                    for arm in arms:
                        arm["ret"].release()
                        arm["target"], arm["target_R"], arm["grip_closed"], arm["grip"] = None, None, False, 0.0
                    ticks = 0
                    continue
                step_ms += (time.perf_counter() - t0) * 1000
                window += 1
                pol = obs["policy"]
                ticks += 1

                # Video: hand the newest poses to the render thread (cheap); it
                # renders at its own pace and never blocks this loop.
                if renderer is not None and ticks % 2 == 0:
                    from isaaclab_newton.physics.newton_manager import NewtonManager as NM
                    t_r = time.perf_counter()
                    g = pol["robot0_eef_pos"][0].cpu().numpy() + u.scene.env_origins[0].cpu().numpy()
                    renderer.submit(NM._state_0.body_q.numpy().copy(), g)
                    render_ms += (time.perf_counter() - t_r) * 1000

                now = time.perf_counter()
                if now - last_report > 5.0:
                    live = []
                    for i, a_ in enumerate(arms):
                        tag = f"robot{i}"
                        e_now = pol[f"robot{i}_eef_pos"][0].cpu().numpy() * 1000
                        tag += f" eef ({e_now[0]:.0f},{e_now[1]:.0f},{e_now[2]:.0f}) mm"
                        if a_.get("dev_p") is not None:
                            tag += " ENGAGED"
                        if a_["target"] is not None and args.input != "cobalt":
                            e = np.asarray(a_["target"]) - pol[f"robot{i}_eef_pos"][0].cpu().numpy()
                            tag += f" ENGAGED lag {np.linalg.norm(e)*1000:.0f} mm"
                        live.append(tag + (" GRIP" if a_["grip_closed"] else ""))
                    rate = period_n / period_sum if period_sum else 0.0
                    period_sum, period_n = 0.0, 0
                    print(f"   loop {rate:4.1f} Hz | step {step_ms / max(window, 1):5.1f} ms + render "
                          f"{render_ms / max(window, 1):5.1f} ms (budget {1000 / hz:.0f}) | video "
                          f"{(renderer.frames if renderer else 0)} frames | "
                          f"{' | '.join(live)} | saved {saved} discarded {discarded}", flush=True)
                    last_report, step_ms, window, render_ms = now, 0.0, 0, 0.0
                next_tick += 1.0 / hz
                sleep = next_tick - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.perf_counter()
        except KeyboardInterrupt:
            print("\nstopped by operator")
        finally:
            if renderer is not None:
                renderer.alive = False
            source.close()
            env.close()
            if args.trace and trace:
                np.savez(args.trace, trace=np.array(trace),
                         columns="t,engaged,eef_x,eef_y,eef_z,lead_x,lead_y,lead_z,dp_x,dp_y,dp_z")
                print(f"trace: {len(trace)} ticks -> {args.trace}")
    if saved and out.exists():
        stamp(out, saved)
    print(f"done: {saved} demo(s) saved to {out}, {discarded} discarded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
