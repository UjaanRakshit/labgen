"""How does the simulated arm respond, with the REAL controller gains?

The sim used kp=3000 on every joint, chosen here so the arm would hold still.
The real YAM runs kp 80 on the shoulder and 10 on the wrist and leans on gravity
compensation to hold. With gains that soft, holding depends entirely on the
feedforward being right, so it is measured rather than assumed.

Two experiments per gain set:

  hold   command the ready pose, apply gravity feedforward, step 3 s, report
         the worst joint error.  (Does the arm stay where it is put?)
  step   from the ready pose, command +0.20 rad on one joint and record the
         response: rise time (10-90%), overshoot, settling time (2% band),
         steady-state error.  (How does it move?)

The step response is also the thing the sim-to-real harness asks the REAL arm
to reproduce: same command, same joint, logged, compared.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_drive import (configure_drives, configure_real_drives,          # noqa: E402
                       gravity_torques)
from grasp import set_state, GraspFK, N_ARM, with_fingers                           # noqa: E402
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.control import YAM_JAWS                                      # noqa: E402
from labgen.hardware import YAM_V1                                        # noqa: E402

URDF = "/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"
DT = 1.0 / 1000.0
# The real rig's recorded reset pose (deployment/config.yaml, from
# balancing_act/yam_home.json) -- a pose the real arm actually holds.
READY = np.array([0.0, 1.5886, 0.9016, 1.0007, 0.6594, 0.0])


def build(real: bool):
    b = newton.ModelBuilder()
    arm = newton.ModelBuilder()
    arm.add_urdf(URDF, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                 floating=False, enable_self_collisions=False,
                 collapse_fixed_joints=True, up_axis=newton.Axis.Z,
                 parse_visuals_as_colliders=False)
    b.add_builder(arm)
    dofs = slice(0, b.joint_dof_count)
    if real:
        configure_real_drives(b, YAM_V1, dof_slice=dofs, n_fingers=2, verbose=False)
    else:
        configure_drives(b, dof_slice=dofs, n_fingers=2, ke=3000.0, kd=150.0,
                         effort=10.0, verbose=False)
    model = b.finalize()
    return model, arm, dofs, slice(0, model.joint_coord_count)


def run(real: bool, target: np.ndarray, seconds: float, ff: bool = True,
        origin_lever_arms: bool = False):
    model, arm, dofs, coords = build(real)
    fk = GraspFK(model, coords)
    masses = np.asarray(arm.body_mass, float)
    axes = np.asarray(arm.joint_axis, float)
    coms = np.asarray(arm.body_com, float)
    if origin_lever_arms:
        # coms = 0 puts every body's weight at its frame origin: exactly
        # the bug gravity_torques used to have.
        coms = np.zeros_like(coms)
    off = model.body_count - len(masses)

    s0, s1 = model.state(), model.state()
    control = model.control()
    q0 = model.joint_q.numpy().copy()
    q0[coords] = with_fingers(READY, YAM_JAWS.q_closed)
    set_state(model, s0, q0)
    set_state(model, s1, q0)

    tq = control.joint_target_q.numpy().copy()
    tq[coords] = with_fingers(target, YAM_JAWS.q_closed)
    control.joint_target_q.assign(tq)

    if ff:
        f = control.joint_f.numpy().copy()
        # Feedforward for the TARGET pose, as the real controller computes it.
        # In sim the plant is the model, so factor 1.0 -- i2rt's own SimRobot
        # does the same; the hardware factor corrects model error on the real arm.
        f[:N_ARM] = -gravity_torques(fk._eval(target), masses, axes, coms=coms,
                                     offset=off, n_arm=N_ARM)
        control.joint_f.assign(f.astype(np.float32))

    solver = newton.solvers.SolverMuJoCo(model, iterations=20, ls_iterations=20)
    trace = []
    for i in range(int(seconds / DT)):
        solver.step(s0, s1, control, None, DT)
        s0, s1 = s1, s0
        trace.append(s0.joint_q.numpy()[coords][:N_ARM].copy())
    return np.array(trace)


def step_metrics(trace, joint, start, goal):
    y = trace[:, joint]
    d = goal - start
    if abs(d) < 1e-9:
        return None
    frac = (y - start) / d
    t = np.arange(len(y)) * DT
    try:
        t10 = t[np.argmax(frac >= 0.1)]
        t90 = t[np.argmax(frac >= 0.9)]
        rise = t90 - t10 if frac.max() >= 0.9 else float("nan")
    except ValueError:
        rise = float("nan")
    overshoot = max(0.0, (frac.max() - 1.0) * 100)
    outside = np.where(np.abs(frac - 1.0) > 0.02)[0]
    settle = t[outside[-1]] if len(outside) else 0.0
    sse = abs(y[-1] - goal)
    return rise, overshoot, settle, np.degrees(sse)


def main() -> int:
    print("HOLD: command the real rig's recorded reset pose, 3 s")
    print(f"{'gains':22} {'feedforward':>12} {'worst joint error':>18} {'per joint (deg)'}")
    for real, label in ((False, "old: kp 3000 all"), (True, "REAL yam_v1")):
        for ff in (False, True):
            tr = run(real, READY, 3.0, ff=ff)
            err = np.degrees(np.abs(tr[-1] - READY))
            print(f"{label:22} {str(ff):>12} {err.max():15.3f} deg   "
                  + " ".join(f"{e:6.2f}" for e in err))

    print()
    print("CENTRE OF MASS vs FRAME ORIGIN as the gravity lever arm (real gains, 3 s hold)")
    for origin, label in ((True, "frame origin (the old bug)"), (False, "centre of mass (fixed)")):
        tr = run(True, READY, 3.0, ff=True, origin_lever_arms=origin)
        err = np.degrees(np.abs(tr[-1] - READY))
        print(f"   {label:28} worst {err.max():7.3f} deg   "
              + " ".join(f"{e:6.2f}" for e in err))

    print()
    print("STEP: +0.20 rad on one joint from the reset pose, 2 s, feedforward on")
    print(f"{'gains':22} {'joint':>6} {'rise 10-90':>11} {'overshoot':>10} "
          f"{'settle 2%':>10} {'steady err':>11}")
    for real, label in ((False, "old: kp 3000 all"), (True, "REAL yam_v1")):
        for joint in (1, 3, 5):
            target = READY.copy()
            target[joint] += 0.20
            tr = run(real, target, 2.0, ff=True)
            m = step_metrics(tr, joint, READY[joint], target[joint])
            rise, ov, st, sse = m
            print(f"{label:22} {'j'+str(joint+1):>6} {rise*1000:8.1f} ms {ov:8.1f} % "
                  f"{st*1000:7.0f} ms {sse:9.3f} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
