"""Run the sim-to-real protocol through i2rt and log it. SIM BY DEFAULT.

Run this in the lab's teleop venv, the one that drives the real rig:

    cd "C:/Ujaan Docx/Research/pairlab/yam_teleop"
    .venv/Scripts/python "C:/Ujaan Docx/Research/labgen/scripts/sim2real_record.py" \\
        --out i2rt_sim.npz                                   # i2rt's own sim
    .venv/Scripts/python ".../sim2real_record.py" --backend real --channel can0 \\
        --i-have-cleared-the-workspace --out real_can0.npz   # THE REAL ARM

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
  * nothing touches the CAN bus without both --backend real and
    --i-have-cleared-the-workspace
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
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labgen.sim2real import CONTROL_HZ, RESET_POSE_RAD, Log, default_protocol  # noqa: E402

MAX_JOINT_SPEED = 3.0      # rad/s -- the lab teleop's parking threshold


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("sim", "real"), default="sim")
    ap.add_argument("--channel", default="can0")
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

    robot = get_yam_robot(channel=args.channel, arm_type=ArmType.YAM,
                          gripper_type=GripperType.NO_GRIPPER,
                          sim=(args.backend == "sim"))
    source = ("real-" + args.channel if args.backend == "real"
              else "i2rt-sim (KINEMATIC: teleports, no dynamics)")
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
            robot.command_joint_pos(start + (goal - start) * (u * u * (3 - 2 * u)))
            time.sleep(1.0 / CONTROL_HZ)
        time.sleep(0.5)

        period = 1.0 / CONTROL_HZ
        t0 = time.perf_counter()
        next_tick = t0
        aborted = None
        for k in range(n):
            robot.command_joint_pos(q_cmd[k])
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
                robot.command_joint_pos(hold)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
