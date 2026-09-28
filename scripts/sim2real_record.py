"""Run the sim-to-real protocol through i2rt and log it. SIM BY DEFAULT.

Run this in the lab's teleop venv, the one that drives the real rig:

    cd $TELEOP                            # the lab's arm-driving repo and venv
    .venv/bin/python ~/labgen/scripts/sim2real_record.py --out i2rt_sim.npz      # i2rt's own sim
    .venv/bin/python ~/labgen/scripts/sim2real_record.py --backend real \
        --teleop-config deployment/config.yaml \
        --i-have-cleared-the-workspace --out real_right.npz                     # THE REAL ARM

For a real arm the channel and gripper come from that arm's teleop config. The
adapter serial comes from the config or the systemd .link that permanently names
the CAN interface. The measured gripper endpoints come from the teleop's saved
calibration file, so opening the arm cannot auto-calibrate the gripper. The
recorder takes a per-channel lock and checks that teleop is stopped; the current
Jetson teleop checkout does not participate in that lock.

then compare either log with the Newton sim's (scripts/sim2real_newton.py) using
scripts/sim2real_compare.py.

WHAT THE i2rt SIM IS, AND IS NOT. i2rt's SimRobot does not simulate dynamics
under position control: command_joint_pos TELEPORTS the joints to the target
(sim_robot.py: "CONTROL mode uses teleport") with physics and gravity
compensation switched off. Its log therefore shows perfect, instant tracking.
That makes it useful for exactly two things -- proving this script's plumbing
end to end, and giving the zero-lag baseline -- and useless as a dynamics
reference. The same applies to the lab teleop's `backend: sim`. Real dynamics
come only from `--backend real`.

SAFETY, because one backend is a real arm:
  * nothing touches the CAN bus without --backend real,
    --i-have-cleared-the-workspace AND --teleop-config (adapter-serial check +
    bus lock, and verifies that no teleop process is running)
  * the arm is opened WITH its gripper (config gripper_type, linear_4310 on the
    rig) and in zero-gravity mode, as the teleop opens it. The first version
    used NO_GRIPPER: i2rt's gravity compensation would then leave out a
    ~0.55 kg gripper and the arm would not behave as it does under teleop
  * the arm is brought to the rig's recorded reset pose by an eased 4 s move
    before anything else happens (our own, so it is identical on both
    backends; i2rt's move_joints exists only on the real motor chain)
  * excitations are 0.10 rad, eased, one joint at a time (labgen.sim2real
    refuses anything over 0.25 rad)
  * any measured joint speed over 3.0 rad/s -- the lab teleop's own parking
    threshold (deployment/config.yaml) -- stops the protocol and holds the
    arm where it is
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labgen.sim2real import (CONTROL_HZ, RESET_POSE_RAD, Log, current_pose_protocol,
                             default_protocol)  # noqa: E402

MAX_JOINT_SPEED = 3.0      # rad/s -- the lab teleop's parking threshold
REAL_MAX_START_DELTA_RAD = 0.15  # refuse a long automatic move to reset


def expected_adapter_serial(cfg: dict, channel: str,
                            link_dir: Path = Path("/etc/systemd/network")) -> str:
    """Require a configured USB serial or a persistent systemd link for this CAN name."""
    explicit = str(cfg.get("adapter_serial", "")).strip()
    if explicit:
        if cfg.get("mapping_verified") is not True:
            raise ValueError(f"{channel}: adapter_serial requires mapping_verified=true")
        return explicit
    if "mapping_verified" in cfg and cfg["mapping_verified"] is not True:
        raise ValueError(f"{channel}: mapping_verified is not true")
    matches = []
    for path in link_dir.glob("*.link"):
        lines = [line.strip() for line in path.read_text().splitlines()]
        if f"Name={channel}" not in lines or "Driver=gs_usb" not in lines:
            continue
        ids = [line.removeprefix("Property=ID_SERIAL=")
               for line in lines if line.startswith("Property=ID_SERIAL=")]
        if len(ids) == 1:
            matches.append(ids[0].rsplit("_", 1)[-1])
    if len(matches) != 1 or not matches[0]:
        raise ValueError(f"{channel}: need exactly one serial-pinned systemd .link "
                         "or robot.adapter_serial in the teleop config")
    return matches[0]


def saved_gripper_limits(cfg: dict, config_path: Path, channel: str,
                         gripper: str) -> np.ndarray:
    """Use teleop's measured endpoints; absence must never start auto-calibration."""
    value = cfg.get("gripper_limits_path")
    if not value:
        raise ValueError(f"{channel}: gripper_limits_path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.resolve().parent / path
    data = json.loads(path.read_text())
    if data.get("channel") != channel or data.get("gripper_type") != gripper:
        raise ValueError(f"{path}: gripper limits do not match {channel}/{gripper}")
    limits = np.asarray(data.get("gripper_limits_rad"), dtype=float)
    if limits.shape != (2,) or not np.all(np.isfinite(limits)) or limits[0] == limits[1]:
        raise ValueError(f"{path}: expected two distinct finite gripper endpoints")
    return limits


def start_pose_error(start, goal=RESET_POSE_RAD) -> float:
    """Largest joint move needed to reach the protocol's known starting pose."""
    q = np.asarray(start, dtype=float).reshape(-1)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError("arm start pose must be six finite joint angles")
    return float(np.max(np.abs(q - np.asarray(goal, dtype=float))))


def require_joint_margin(commands, limits, margin_rad: float = 0.05) -> None:
    """Refuse a protocol that approaches the driver's reported joint limits."""
    q = np.asarray(commands, dtype=float)
    bounds = np.asarray(limits, dtype=float)
    if q.ndim != 2 or q.shape[1] != 6 or bounds.shape[0] < 6 or bounds.shape[1:] != (2,):
        raise ValueError("joint commands or driver limits have an unexpected shape")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(bounds[:6])):
        raise ValueError("joint commands and driver limits must be finite")
    for j in range(6):
        low, high = bounds[j]
        actual_low, actual_high = float(q[:, j].min()), float(q[:, j].max())
        if actual_low < low + margin_rad or actual_high > high - margin_rad:
            raise ValueError(f"j{j + 1} commands [{actual_low:.3f}, {actual_high:.3f}] "
                             f"approach driver limits [{low:.3f}, {high:.3f}] "
                             f"within {margin_rad:.3f} rad")


def wait_for_guided_reset(robot, goal: np.ndarray) -> bool:
    """Read angles while a person hand-guides under gravity compensation.

    Only the exact READY command with hands clear and a close measured pose lets
    the recorder proceed. Nothing in this loop commands a joint target.
    """
    print("Guide the supported arm near reset under gravity compensation.")
    print("Target joint angles (rad): " + np.array2string(goal, precision=3))
    print("When all joints are within 0.150 rad, remove your hands and tell the "
          "operator to type READY. Type ABORT to close without recording.")
    while True:
        current = np.asarray(robot.get_observations()["joint_pos"], float)[:6]
        delta = start_pose_error(current, goal)
        print("current " + np.array2string(current, precision=3) +
              f"  largest difference {delta:.3f} rad", flush=True)
        if select.select([sys.stdin], [], [], 0.5)[0]:
            line = sys.stdin.readline()
            if not line or line.strip().upper() == "ABORT":
                print("guided reset aborted; no trajectory commanded")
                return False
            if line.strip().upper() != "READY":
                print("type READY to continue or ABORT to close")
                continue
            current = np.asarray(robot.get_observations()["joint_pos"], float)[:6]
            delta = start_pose_error(current, goal)
            if delta <= REAL_MAX_START_DELTA_RAD:
                print(f"ready at {delta:.3f} rad from reset; starting eased move")
                return True
            print(f"still {delta:.3f} rad from reset; no motion commanded")


def claim_bus(cfg: dict):
    """Verify the adapter identity and take the recorder's per-channel bus lock.

    The current Jetson teleop checkout has no matching lock. The operator must
    stop it before recording; refuse when its process is still present.
    """
    import fcntl
    channel = str(cfg.get("channel", "can0"))
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = proc.read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        busy = (b"deployment.quest_teleop", b"quest_teleop.py",
                b"i2rt.motor_drivers.dm_driver", b"ping_motors", b"candump")
        if any(tag in arg for arg in argv for tag in busy):
            raise SystemExit("another teleop/CAN process is running; stop it before recording")
    try:
        expected = expected_adapter_serial(cfg, channel)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    device = Path("/sys/class/net") / channel / "device"
    actual = None
    if device.exists():
        for parent in [device.resolve(), *device.resolve().parents]:
            serial = parent / "serial"
            if serial.is_file():
                actual = serial.read_text().strip()
                break
    if not expected or actual != expected:
        raise SystemExit(f"{channel}: USB serial {actual!r} does not match configured {expected!r}")
    lock_dir = Path(f"/tmp/yam-teleop-{os.getuid()}")
    lock_dir.mkdir(mode=0o700, exist_ok=True)
    lock = (lock_dir / hashlib.sha256(channel.encode()).hexdigest()[:16]).open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise SystemExit(f"{channel} is in use (is the teleop running?) -- stop it first")
    return lock


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("sim", "real"), default="sim")
    ap.add_argument("--channel", default="can0", help="sim backend only; real takes it from --teleop-config")
    ap.add_argument("--gripper", default="linear_4310", help="sim backend only; real takes it from --teleop-config")
    ap.add_argument("--teleop-config", help="yam_vr_teleop per-arm config (deployment/config.yaml or "
                                            "config_left.yaml); REQUIRED for --backend real")
    ap.add_argument("--amplitude", type=float, default=0.10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-have-cleared-the-workspace", action="store_true",
                    help="required for --backend real: nothing within reach of the arm")
    ap.add_argument("--guide-to-reset", action="store_true",
                    help="real only: hold gravity compensation and print angles until "
                         "the operator sends READY near the verified reset pose")
    ap.add_argument("--start-at-current", action="store_true",
                    help="real only: small eased steps and returns around the measured "
                         "starting pose; no move to reset and no shoulder sine")
    args = ap.parse_args()

    if args.guide_to_reset and args.backend != "real":
        ap.error("--guide-to-reset requires --backend real")
    if args.start_at_current and args.backend != "real":
        ap.error("--start-at-current requires --backend real")
    if args.start_at_current and args.guide_to_reset:
        ap.error("choose either --start-at-current or --guide-to-reset")

    if args.backend == "real" and not args.i_have_cleared_the_workspace:
        print("refusing to move the real arm without --i-have-cleared-the-workspace")
        return 2

    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import ArmType, GripperType

    bus_lock = None
    channel, gripper = args.channel, args.gripper
    if args.backend == "real":
        if not args.teleop_config:
            print("refusing to move the real arm without --teleop-config (adapter check + bus lock)")
            return 2
        import yaml
        config_path = Path(args.teleop_config)
        rcfg = yaml.safe_load(config_path.read_text())["robot"]
        channel, gripper = str(rcfg["channel"]), str(rcfg.get("gripper_type", "no_gripper"))
        try:
            limits = saved_gripper_limits(rcfg, config_path, channel, gripper)
        except (ValueError, OSError, KeyError) as exc:
            raise SystemExit(f"refusing real arm: {exc}") from exc
        bus_lock = claim_bus(rcfg)
        print(f"teleop config {args.teleop_config}: {channel}, gripper {gripper}, "
              f"adapter {expected_adapter_serial(rcfg, channel)} verified, "
              "saved gripper limits loaded, recorder bus locked")
        robot = get_yam_robot(channel=channel, arm_type=ArmType.YAM,
                              gripper_type=GripperType[gripper.upper()], zero_gravity_mode=True,
                              gripper_limits_override=limits)
    else:
        robot = get_yam_robot(channel=channel, arm_type=ArmType.YAM,
                              gripper_type=GripperType[gripper.upper()], sim=True)
    source = ("real-" + channel if args.backend == "real"
              else "i2rt-sim (KINEMATIC: teleports, no dynamics)")
    # With a gripper, i2rt's joint vector is 7 long: the arm command must carry
    # the gripper's target too (held where it started), as the teleop does.
    q_start = np.asarray(robot.get_joint_pos(), float).reshape(-1)
    grip_hold = float(q_start[6]) if q_start.size == 7 else None

    def command(q6):
        q6 = np.asarray(q6, float)
        robot.command_joint_pos(q6 if grip_hold is None else np.append(q6, grip_hold))
    print(f"robot: {type(robot).__name__}  source={source}")

    try:
        protocol = (current_pose_protocol(q_start[:6], args.amplitude)
                    if args.start_at_current else default_protocol(args.amplitude))
        q_cmd, seg = protocol.commands()
        if args.backend == "real":
            require_joint_margin(q_cmd, robot.get_robot_info()["joint_limits"])
    except BaseException:
        robot.close()
        if bus_lock is not None:
            bus_lock.close()
        raise
    n = len(q_cmd)
    q = np.full((n, 6), np.nan)
    qd = np.full((n, 6), np.nan)
    eff = np.full((n, 6), np.nan)
    t = np.full(n, np.nan)

    try:
        if args.start_at_current:
            print("starting from the measured pose; no move to reset or shoulder sine")
            now = np.asarray(robot.get_observations()["joint_pos"], float)[:6]
            drift = float(np.max(np.abs(now - np.asarray(protocol.base))))
            if drift > 0.05:
                print(f"refusing: the arm moved {drift:.3f} rad since the start pose "
                      "was measured; no trajectory commanded")
                return 2
        else:
            print("moving to the rig's reset pose over 4 s ...")
        # Our own eased move, identical on both backends. i2rt's move_joints exists
        # only on the real motor chain, and a move that differs between sim and
        # real would put the two logs at different starting states.
        start = np.asarray(robot.get_observations()["joint_pos"], float)[:6]
        goal = np.asarray(RESET_POSE_RAD, float)
        if args.backend == "real" and not args.start_at_current:
            delta = start_pose_error(start, goal)
            if delta > REAL_MAX_START_DELTA_RAD:
                if not args.guide_to_reset:
                    print(f"refusing an automatic move to reset: largest joint change "
                          f"{delta:.3f} rad exceeds {REAL_MAX_START_DELTA_RAD:.3f} rad. "
                          "Position the arm near the verified reset pose under supervision.")
                    return 2
                if not wait_for_guided_reset(robot, goal):
                    return 2
                start = np.asarray(robot.get_observations()["joint_pos"], float)[:6]
        if not args.start_at_current:
            steps = int(4.0 * CONTROL_HZ)
            for i in range(1, steps + 1):
                u = i / steps
                command(start + (goal - start) * (u * u * (3 - 2 * u)))
                time.sleep(1.0 / CONTROL_HZ)
            time.sleep(0.5)

        period = 1.0 / CONTROL_HZ
        t0 = time.perf_counter()
        next_tick = t0
        aborted = None
        for k in range(n):
            if k == 0 or seg[k] != seg[k - 1]:
                print(f"segment {seg[k] + 1}/{len(protocol.segments)}: "
                      f"{protocol.segments[seg[k]].label}", flush=True)
            command(q_cmd[k])
            obs = robot.get_observations()
            t[k] = time.perf_counter() - t0
            q[k] = np.asarray(obs["joint_pos"], float)[:6]
            qd[k] = np.asarray(obs["joint_vel"], float)[:6]
            if "joint_eff" in obs:
                eff[k] = np.asarray(obs["joint_eff"], float)[:6]
            if np.abs(qd[k]).max() > MAX_JOINT_SPEED:
                aborted = (f"joint speed {np.abs(qd[k]).max():.2f} rad/s over "
                           f"{MAX_JOINT_SPEED} at tick {k}")
                break
            if np.abs(q[k] - q_cmd[k]).max() > 0.25:
                aborted = (f"joint tracking error {np.abs(q[k] - q_cmd[k]).max():.3f} rad "
                           f"over 0.250 rad at tick {k}")
                break
            next_tick += period
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)

        if aborted:
            print(f"ABORTED: {aborted} -- holding position")
            hold = np.asarray(robot.get_observations()["joint_pos"], float)[:6]
            for _ in range(int(0.5 * CONTROL_HZ)):
                command(hold)
                time.sleep(1.0 / CONTROL_HZ)
            return 3

        jitter = np.diff(t) - 1.0 / CONTROL_HZ
        print(f"ran {n} ticks in {t[-1]:.1f} s; tick jitter p99 "
              f"{np.percentile(np.abs(jitter), 99)*1000:.2f} ms")
        Log(t=np.arange(n) / CONTROL_HZ, q_cmd=q_cmd, q=q, qd=qd, eff=eff,
            segment=seg, source=source, protocol_json=protocol.to_json()).save(args.out)
        print(f"wrote {args.out}")
    finally:
        robot.close()
        if bus_lock is not None:
            bus_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
