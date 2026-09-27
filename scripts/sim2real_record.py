"""Run the sim-to-real protocol through i2rt and log it. SIM BY DEFAULT.

Run this in the lab's teleop venv, the one that drives the real rig:

    cd ~/yam_vr_teleop                     # the lab's arm-driving repo and venv
    .venv/bin/python ~/labgen/scripts/sim2real_record.py --out i2rt_sim.npz      # i2rt's own sim
    .venv/bin/python ~/labgen/scripts/sim2real_record.py --backend real \
        --teleop-config deployment/config.yaml \
        --i-have-cleared-the-workspace --out real_right.npz                     # THE REAL ARM

For a real arm the channel, adapter serial and gripper come from that arm's
yam_vr_teleop config, and this script takes the SAME bus lock the teleop takes:
it refuses a channel whose USB adapter serial does not match the config, and
refuses to run while the teleop (or anything else) holds that bus.

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
    bus lock, exactly as yam_vr_teleop's deployment/robot.py does)
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
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labgen.sim2real import CONTROL_HZ, RESET_POSE_RAD, Log, default_protocol  # noqa: E402

MAX_JOINT_SPEED = 3.0      # rad/s -- the lab teleop's parking threshold


def claim_bus(cfg: dict):
    """yam_vr_teleop's deployment/robot.py _claim_bus, reproduced exactly.

    Same checks (mapping_verified, adapter serial read from sysfs) and the SAME
    lock file, so this script and the teleop can never drive one bus at once.
    """
    import fcntl
    channel = str(cfg.get("channel", "can0"))
    if cfg.get("mapping_verified") is not True:
        raise SystemExit(f"{channel}: mapping_verified is not true in the teleop config")
    expected = str(cfg.get("adapter_serial", ""))
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
    args = ap.parse_args()

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
        rcfg = yaml.safe_load(Path(args.teleop_config).read_text())["robot"]
        channel, gripper = str(rcfg["channel"]), str(rcfg.get("gripper_type", "no_gripper"))
        bus_lock = claim_bus(rcfg)
        print(f"teleop config {args.teleop_config}: {channel}, gripper {gripper}, "
              f"adapter {rcfg.get('adapter_serial')} verified, bus locked")
        robot = get_yam_robot(channel=channel, arm_type=ArmType.YAM,
                              gripper_type=GripperType[gripper.upper()], zero_gravity_mode=True)
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

    protocol = default_protocol(args.amplitude)
    q_cmd, seg = protocol.commands()
    n = len(q_cmd)
    q = np.full((n, 6), np.nan)
    qd = np.full((n, 6), np.nan)
    eff = np.full((n, 6), np.nan)
    t = np.full(n, np.nan)

    try:
        print("moving to the rig's reset pose over 4 s ...")
        # Our own eased move, identical on both backends. i2rt's move_joints exists
        # only on the real motor chain, and a move that differs between sim and
        # real would put the two logs at different starting states.
        start = np.asarray(robot.get_observations()["joint_pos"], float)[:6]
        goal = np.asarray(RESET_POSE_RAD, float)
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
