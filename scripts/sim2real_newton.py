"""Run the sim-to-real protocol through the Newton sim and log it.

The dynamic half of the comparison. Same protocol, same 100 Hz command rate,
same log format as scripts/sim2real_record.py, so the two logs line up tick for
tick and labgen.sim2real.compare can set them against each other.

The simulated arm is as close to the real one as the recorded data allows:
  * actuators  labgen.hardware.YAM_V1 -- the per-joint kp/kd, peak torque and
               Coulomb friction from the i2rt 1.1.2 config the rig runs
  * control    MIT-style PD on the commanded target plus gravity compensation,
               which is what the real controller does (use_gravity_comp=True)
  * gravity    centre-of-mass lever arms, factor 1.0 -- the sim plant IS the
               model; the hardware's 1.1-1.2 factor corrects model error on the
               real arm and is kept in YAM_V1 as evidence, not applied here
  * state      grasp.set_state, so the arm genuinely starts where it is told

Run in the Isaac Lab environment:

    python sim2real_newton.py /path/to/yam.urdf --out newton_sim.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_drive import configure_real_drives, gravity_torques              # noqa: E402
from grasp import GraspFK, N_ARM, set_state, with_fingers                  # noqa: E402
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.control import YAM_JAWS                                        # noqa: E402
from labgen.hardware import YAM_V1                                          # noqa: E402
from labgen.sim2real import CONTROL_HZ, RESET_POSE_RAD, Log, default_protocol  # noqa: E402

PHYS_DT = 1.0 / 1000.0          # 10 physics steps per 100 Hz control tick


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("urdf", type=Path)
    ap.add_argument("--out", required=True)
    ap.add_argument("--amplitude", type=float, default=0.10)
    ap.add_argument("--no-friction", action="store_true",
                    help="drop Coulomb friction, to see how much of the result it explains")
    args = ap.parse_args()

    b = newton.ModelBuilder()
    arm = newton.ModelBuilder()
    arm.add_urdf(str(args.urdf), xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()),
                 floating=False, enable_self_collisions=False,
                 collapse_fixed_joints=True, up_axis=newton.Axis.Z,
                 parse_visuals_as_colliders=False)
    b.add_builder(arm)
    dofs = slice(0, b.joint_dof_count)
    configure_real_drives(b, YAM_V1, dof_slice=dofs, n_fingers=2, verbose=True)
    if args.no_friction:
        for i in range(N_ARM):
            b.joint_friction[i] = 0.0
    model = b.finalize()
    coords = slice(0, model.joint_coord_count)
    fk = GraspFK(model, coords)
    masses = np.asarray(arm.body_mass, float)
    axes = np.asarray(arm.joint_axis, float)
    coms = np.asarray(arm.body_com, float)
    off = model.body_count - len(masses)

    protocol = default_protocol(args.amplitude)
    q_cmd, seg = protocol.commands()
    n = len(q_cmd)

    s0, s1 = model.state(), model.state()
    control = model.control()
    q0 = with_fingers(np.asarray(RESET_POSE_RAD), YAM_JAWS.q_closed)
    set_state(model, s0, q0)
    set_state(model, s1, q0)
    solver = newton.solvers.SolverMuJoCo(model, iterations=20, ls_iterations=20)

    kp = np.asarray(YAM_V1.kp)
    kd = np.asarray(YAM_V1.kd)
    tmax = YAM_V1.torque_max_nm
    sub = int(round(1.0 / CONTROL_HZ / PHYS_DT))
    targets = control.joint_target_q.numpy().copy()
    ff = control.joint_f.numpy().copy()

    q = np.zeros((n, 6))
    qd = np.zeros((n, 6))
    eff = np.zeros((n, 6))
    for k in range(n):
        targets[coords] = with_fingers(q_cmd[k], YAM_JAWS.q_closed)
        control.joint_target_q.assign(targets)
        g = -gravity_torques(fk._eval(q_cmd[k]), masses, axes, coms=coms,
                             offset=off, n_arm=N_ARM)
        ff[:N_ARM] = g
        control.joint_f.assign(ff.astype(np.float32))
        for _ in range(sub):
            solver.step(s0, s1, control, None, PHYS_DT)
            s0, s1 = s1, s0
        jq = s0.joint_q.numpy()[:N_ARM]
        jqd = s0.joint_qd.numpy()[:N_ARM]
        q[k], qd[k] = jq, jqd
        # What the motor delivers: saturated PD plus the feedforward -- the
        # quantity the real robot reports as joint_eff.
        eff[k] = np.clip(kp * (q_cmd[k] - jq) - kd * jqd, -tmax, tmax) + g

    source = "newton-sim" + (" (no friction)" if args.no_friction else "")
    Log(t=np.arange(n) / CONTROL_HZ, q_cmd=q_cmd, q=q, qd=qd, eff=eff,
        segment=seg, source=source, protocol_json=protocol.to_json()).save(args.out)
    print(f"wrote {args.out}: {n} ticks, worst |q - q_cmd| "
          f"{np.degrees(np.abs(q - q_cmd).max()):.3f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
