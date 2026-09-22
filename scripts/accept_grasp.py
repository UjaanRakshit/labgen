"""Item D acceptance: does a CONTACT grasp lift and hold the 250 mL beaker?

The criterion, as set:
    lift the 250 mL beaker 10 cm and hold it 2 s, succeed in at least 9 of 10
    repeat runs of the SAME configuration, never launch the object, and report
    the result across a range of grip forces.

Everything that decides this is the real robot's where it is known and flagged
where it is not:

  arm drives    labgen.hardware.YAM_V1 (i2rt 1.1.2, what the rig runs)
  finger pads   authored boxes, labgen.control.YAM_PAD -- UNVERIFIED, measured
                off the URDF mesh (an older linear_4310 tip revision), not with
                calipers
  pad friction  the catalog's glass, mu_s 0.45 -- UNVERIFIED, "tuned for grasp
                stability, not measured"
  contact       labgen.settle's CONTACT_KE / CONTACT_KD at dt 1/960, which
                satisfies MuJoCo's timeconst >= 2*dt -- not teleop's soft,
                real-time defaults
  initial state grasp.set_state, not eval_fk alone
  friction      elliptic cone, impratio 100, pad torsion 5 mm (condim 4): without
                them MuJoCo's soft friction creeps 1.85 mm/s and the pads have
                no twist resistance -- see SOLVER_OPTS. Torsion UNVERIFIED.

Grip force is swept rather than asserted. The real linear_4310's force map
(i2rt: F = tau * motor_stroke / gripper_stroke = tau * 6.57 / 0.096) puts its
theoretical ceiling near 684 N at the DM4310's 10 N.m peak; the controller's
force limiter keeps real grasps far below that, and the right operating force is
what this sweep is for. The minimum to hold the beaker by friction alone is
2 * mu * F >= m * g, i.e. about 1.1 N per finger.

"Launched" is judged two ways, because either alone misses a case: peak object
speed above LAUNCH_SPEED, or the beaker ending up far from the hand.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teleop_sim as T                                                     # noqa: E402
from arm_drive import gravity_torques                                      # noqa: E402
from grasp import GraspFK, N_ARM, PAD_TORSION_M, set_state, with_fingers   # noqa: E402
from grasp import quat_to_matrix as q2m                                    # noqa: E402
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.catalog import CATALOG                                         # noqa: E402
from labgen.control import YAM_JAWS, YAM_PAD, lerp_path, solve_pose        # noqa: E402
from labgen.register import rotation_error_deg                             # noqa: E402
from labgen.settle import CONTACT_KD, CONTACT_KE                           # noqa: E402
from labgen.types import SceneSpec                                         # noqa: E402

URDF = "/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"
SCENE_JSON = "/mnt/c/Ujaan Docx/Research/labgen/examples/bench_arm.json"

DT = 1.0 / 960.0
LIFT_M = 0.10
HOLD_S = 2.0
LIFT_PASS_M = 0.090        # at least 90 of the 100 mm
SLIP_PASS_M = 0.005        # the rim may slide no more than 5 mm against the pads during the hold
# Slip is measured AT THE GRIP: the beaker material point that sat at the grasp
# point when the hold began, against the grasp point. The first version measured
# the beaker's origin -- its base, 83 mm below the pads -- so a beaker PIVOTING in
# a grip that was not sliding at all read as slip: 4.1 mm at every force, a
# 2.8 deg swing. Pivot is now its own number (tilt), and the old base drift is
# still printed so the change of metric is visible, not silent.
LAUNCH_SPEED = 1.0         # m/s: far above anything a slow lift produces
LAUNCH_DIST_M = 0.05       # beaker more than 5 cm from the hand at the end
CLOSE_GAP_M = 0.040
import os as _os
# MuJoCo friction is a SOFT constraint: under a sustained tangential load it
# creeps. Measured at 25 N/finger (22x the beaker's weight in friction capacity):
# the rim slides against the pads at a steady 1.85 mm/s through the whole hold,
# linear from 0.09 mm at 50 ms to 3.72 mm at 2 s -- no transient, pure creep,
# which no real rubber-on-glass grip does. Creep over the 2 s hold at 25 N:
#     pyramidal (default)       3.72 mm      elliptic, impratio 10    0.21 mm
#     elliptic                  1.75 mm      elliptic, impratio 100   0.02 mm
# Newton's SolverMuJoCo does not expose MuJoCo's noslip_iterations, so the
# elliptic cone with a high impratio is the remedy. Override with --cone /
# --impratio; at 50-100 N it logs "linesearch iterations limit reached" but
# every trial still passed, and none diverged.
SOLVER_OPTS: dict = {"cone": "elliptic", "impratio": 100.0}
DIAG = _os.environ.get("LABGEN_DIAG", "") not in ("", "0")        # command the jaws well inside the 70 mm beaker so the PD squeezes


def plan(fk, lo, hi, beaker_xy, grasp_z):
    """Joint waypoints: above, at the rim, lifted. Down approach throughout."""
    above = np.array([beaker_xy[0], beaker_xy[1], grasp_z + 0.06])
    at = np.array([beaker_xy[0], beaker_xy[1], grasp_z])
    lifted = np.array([beaker_xy[0], beaker_xy[1], grasp_z + LIFT_M])
    out = []
    seed = None
    for label, p in (("above", above), ("grasp", at), ("lift", lifted)):
        r = solve_pose(fk, p, lo, hi, approach=T.DOWN, seed=seed)
        if not r.ok:
            raise RuntimeError(f"cannot reach {label} {p}: {r}")
        out.append(r.q)
        seed = r.q
    return out


def jaw_gap(inst, bq):
    """The TRUE jaw opening: world distance between the two pad faces.

    Not one finger's joint coordinate through the jaw model. In yam.urdf the two
    finger joints are not coupled (i2rt's current linear_4310 couples them with
    an equality constraint), so an object pushed sideways inside a closed grip
    moves both fingers the same way, and reading one finger reports that shift
    as the jaw prying open -- 17.4 mm at 50 N, which is not physical for a
    100 g beaker.
    """
    from grasp import quat_to_matrix as q2m  # noqa: F811 (module-level too)
    a, b = sorted(inst.fk._tip_ids)
    pa = bq[a, :3] + q2m(bq[a, 3:]) @ np.asarray(YAM_PAD.centre_for(1))
    pb = bq[b, :3] + q2m(bq[b, 3:]) @ np.asarray(YAM_PAD.centre_for(-1))
    return float(np.linalg.norm(pa - pb)) - YAM_PAD.thickness_m


def schedule(inst, qs, finger_open, finger_closed):
    """Every physics step of one pick: (phase, arm config, finger, feedforward).

    Built once and replayed for every repeat. The feedforward needs an FK
    evaluation per configuration, and doing that inside the physics loop made a
    single trial take 32 s -- for numbers that are identical on every repeat.
    """
    q_above, q_grasp, q_lift = qs
    hz = int(round(1 / DT))
    phases = [
        ("settle", [q_above] * int(0.4 * hz), finger_open),
        ("descend", lerp_path([q_above, q_grasp], [0.8], hz), finger_open),
        ("close", [q_grasp] * int(0.6 * hz), finger_closed),
        ("lift", lerp_path([q_grasp, q_lift], [1.0], hz), finger_closed),
        ("hold", [q_lift] * int(HOLD_S * hz), finger_closed),
    ]
    cache = {}
    out = []
    for name, cfgs, finger in phases:
        for q in cfgs:
            key = tuple(np.round(q, 9))
            if key not in cache:
                cache[key] = -gravity_torques(inst.fk._eval(q), inst.masses, inst.axes,
                                              coms=inst.coms, offset=inst.body_offset,
                                              n_arm=N_ARM)
            out.append((name, q, finger, cache[key]))
    return out


def trial(model, inst, beaker_body, qs, finger_open, finger_closed, sched=None):
    """One pick. Returns a dict of measurements."""
    q_above, q_grasp, q_lift = qs
    s0, s1 = model.state(), model.state()
    control = model.control()
    q0 = model.joint_q.numpy().copy()
    q0[inst.coords] = with_fingers(q_above, finger_open)
    set_state(model, s0, q0)
    set_state(model, s1, q0)

    solver = newton.solvers.SolverMuJoCo(model, iterations=30, ls_iterations=30,
                                         nconmax=T.NCONMAX, njmax=T.NJMAX, **SOLVER_OPTS)
    targets = control.joint_target_q.numpy().copy()
    ff = control.joint_f.numpy().copy()

    beaker_z0 = None
    peak_speed = 0.0
    hold_rel = []
    hold_pose = []             # (beaker pos, beaker R, grasp point, hand R) per hold tick
    grip_gap = {}
    prev = None
    last_phase = None
    for name, q_arm, finger, ffv in sched:
        targets[inst.coords] = with_fingers(q_arm, finger)
        control.joint_target_q.assign(targets)
        ff[inst.dofs.start: inst.dofs.start + N_ARM] = ffv
        control.joint_f.assign(ff.astype(np.float32))

        contacts = model.collide(s0)
        solver.step(s0, s1, control, contacts, DT)
        s0, s1 = s1, s0

        bq = s0.body_q.numpy()
        if not np.isfinite(bq).all():
            return dict(ok=False, diverged=True, phase=name)
        bpos = bq[beaker_body, :3].copy()
        if beaker_z0 is None:
            beaker_z0 = bpos[2]
        if prev is not None:
            peak_speed = max(peak_speed, float(np.linalg.norm(bpos - prev)) / DT)
        prev = bpos
        if name == "hold":
            g = inst.fk.grasp_point_from_state(bq)
            hold_rel.append(bpos - g)
            hold_pose.append((bpos, q2m(bq[beaker_body, 3:]), g,
                              q2m(bq[min(inst.fk._tip_ids), 3:])))
        if name != last_phase and last_phase is not None:
            grip_gap[last_phase] = jaw_gap(inst, bq)
        last_phase = name
    grip_gap[last_phase] = jaw_gap(inst, s0.body_q.numpy())

    bq = s0.body_q.numpy()
    bpos = bq[beaker_body, :3]
    hand = inst.fk.grasp_point_from_state(bq)
    lifted = float(bpos[2] - beaker_z0)
    rel = np.array(hold_rel)
    base_drift = float(np.linalg.norm(rel - rel[0], axis=1).max()) if len(rel) else float("nan")
    slip, tilt = float("nan"), float("nan")
    if hold_pose:
        b0, Rb0, g0, Rh0 = hold_pose[0]
        p_local = Rb0.T @ (g0 - b0)            # the beaker point under the pads at hold start
        rel0 = Rh0.T @ Rb0                     # beaker orientation in the hand frame
        series = [float(np.linalg.norm(b + Rb @ p_local - g)) for b, Rb, g, _ in hold_pose]
        slip = max(series)
        if DIAG:
            hz = int(round(1 / DT))
            marks = [int(t * hz) for t in (0.05, 0.1, 0.25, 0.5, 1.0, 1.5)] + [len(series) - 1]
            print("      slip@grip over hold (mm) at t=0.05,0.1,0.25,0.5,1,1.5,2 s: "
                  + " ".join(f"{series[min(m, len(series)-1)]*1000:.2f}" for m in marks))
        tilt = max(rotation_error_deg(Rh.T @ Rb, rel0) for _, Rb, _, Rh in hold_pose)
    # "Launched" is the object leaving the hand fast, or ending far from where
    # the hand was holding it. An earlier version compared the beaker's BASE to
    # the hand -- 83 mm apart by design, since the grasp is at the rim -- and so
    # flagged every trial, including ones where the beaker was simply left on
    # the bench.
    drift = float(np.linalg.norm(rel[-1] - rel[0])) if len(rel) else float("nan")
    launched = peak_speed > LAUNCH_SPEED or (lifted > 0.02 and drift > LAUNCH_DIST_M)
    left_behind = lifted < 0.010
    pried = grip_gap.get("hold", np.nan) - grip_gap.get("close", np.nan)
    ok = (lifted >= LIFT_PASS_M) and (slip <= SLIP_PASS_M) and not launched
    ok_old = (lifted >= LIFT_PASS_M) and (base_drift <= SLIP_PASS_M) and not launched
    return dict(ok=ok, ok_old=ok_old, diverged=False, lifted_m=lifted, slip_m=slip,
                tilt_deg=tilt, base_drift_m=base_drift,
                peak_speed=peak_speed, launched=launched, left_behind=left_behind,
                grip_close_m=grip_gap.get("close", np.nan), pried_m=pried)


def main() -> int:
    argv = list(sys.argv)
    torsion = PAD_TORSION_M                  # "--pad-torsion-mm off" for Newton's condim 3
    if "--pad-torsion-mm" in argv:
        i = argv.index("--pad-torsion-mm")
        torsion = None if argv[i + 1] == "off" else float(argv[i + 1]) / 1000.0
        del argv[i:i + 2]
    for flag, key, conv in (("--cone", "cone", str), ("--impratio", "impratio", float)):
        if flag in argv:
            i = argv.index(flag)
            SOLVER_OPTS[key] = conv(argv[i + 1])
            del argv[i:i + 2]
    sys.argv = argv
    forces = [float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2
                                 else "2,5,10,25,50")]
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    scene = SceneSpec.read(SCENE_JSON)
    beaker = scene.by_id("beaker")
    item = CATALOG[beaker.catalog_key]
    grasp_z = beaker.position[2] + item.keypoints["rim_grasp"][2]
    beaker_xy = np.asarray(beaker.position[:2], float)
    beaker_index = [o.instance_id for o in scene.free_bodies].index("beaker")

    probe = newton.ModelBuilder()
    probe.add_urdf(URDF, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()),
                   floating=False, enable_self_collisions=False,
                   collapse_fixed_joints=True, up_axis=newton.Axis.Z,
                   parse_visuals_as_colliders=False)
    lo_all = np.asarray(probe.joint_limit_lower, float)
    hi_all = np.asarray(probe.joint_limit_upper, float)
    rig = T.default_rig(False, lo_all, hi_all)

    finger_open = YAM_JAWS.q_for_gap(YAM_JAWS.max_gap_m)
    finger_closed = YAM_JAWS.q_for_gap(CLOSE_GAP_M)

    print(f"beaker_250: {item.mass_kg*1000:.0f} g, rim grasp at z={grasp_z*1000:.0f} mm; "
          f"min grip to hold by friction ~{item.mass_kg*9.81/(2*0.45):.1f} N/finger")
    print(f"contact ke={CONTACT_KE:g} kd={CONTACT_KD:g} dt=1/{1/DT:.0f}; "
          f"{repeats} repeats per force; pass = lift>={LIFT_PASS_M*1000:.0f} mm, "
          f"slip at grip<={SLIP_PASS_M*1000:.0f} mm, not launched")
    print(f"solver options: {SOLVER_OPTS or 'MuJoCo defaults (pyramidal cone, impratio 1)'}")
    print("pad torsion: " + ("OFF (Newton default condim 3: sliding friction only)" if torsion is None
                             else f"{torsion*1000:.1f} mm, condim 4 (UNVERIFIED)"))
    print()
    print(f"{'force/finger':>12} {'pass':>6} {'old':>5} {'lifted mm':>16} {'slip@grip mm':>14} "
          f"{'tilt deg':>12} {'base mm':>12} {'peak speed':>14} {'launched':>9} {'behind':>6}"
          f"   grip at close / pried open in hold")

    summary = []
    for F in forces:
        model, arms = T.build(Path(sys.argv[1]), Path(URDF), rig, collide=True,
                              contact=(CONTACT_KE, CONTACT_KD), finger_force_n=F,
                              pad_torsion_m=torsion)
        inst = arms[0]
        inst.fk = GraspFK(model, inst.coords, body_offset=inst.body_offset,
                          n_bodies=len(inst.masses), pad=YAM_PAD)
        qs = plan(inst.fk, inst.lo, inst.hi, beaker_xy, grasp_z)
        sched = schedule(inst, qs, finger_open, finger_closed)
        results = []
        t0 = time.time()
        for _ in range(repeats):
            results.append(trial(model, inst, beaker_index, qs, finger_open,
                                 finger_closed, sched=sched))
        ok = sum(r["ok"] for r in results)
        good = [r for r in results if not r["diverged"]]
        lifted = np.array([r["lifted_m"] for r in good]) * 1000
        slip = np.array([r["slip_m"] for r in good]) * 1000
        tilt = np.array([r["tilt_deg"] for r in good])
        base = np.array([r["base_drift_m"] for r in good]) * 1000
        ok_old = sum(r.get("ok_old", False) for r in results)
        speed = np.array([r["peak_speed"] for r in good])
        launched = sum(r.get("launched", False) for r in good)
        behind = sum(r.get("left_behind", False) for r in good)
        diverged = sum(r["diverged"] for r in results)
        grip = np.array([r["grip_close_m"] for r in good]) * 1000
        pried = np.array([r["pried_m"] for r in good]) * 1000
        print(f"{F:10.0f} N {ok:3d}/{repeats:<2d} {ok_old:3d}   "
              f"{lifted.min():6.1f}-{lifted.max():6.1f} "
              f"{slip.min():5.1f}-{slip.max():5.1f}   "
              f"{tilt.min():4.1f}-{tilt.max():4.1f}   "
              f"{base.min():4.1f}-{base.max():4.1f}  "
              f"{speed.min():5.2f}-{speed.max():5.2f} m/s {launched:6d} "
              f"{behind:6d}   jaw {grip.mean():5.1f}mm pried {pried.mean():+5.1f}mm"
              f"{f'  diverged {diverged}' if diverged else ''}"
              f"   ({time.time()-t0:.0f} s)", flush=True)
        summary.append((F, ok, repeats, launched))

    print()
    passing = [F for F, ok, n, l in summary if ok >= 0.9 * n and l == 0]
    print("ACCEPTANCE (>= 9/10, zero launches): "
          + (f"PASS at {passing} N per finger" if passing else "FAIL at every force tested"))
    print("Caveat: finger pads, pad friction" + ("" if torsion is None else " and pad torsion")
          + " are UNVERIFIED; this result is conditional on them until calipers and a "
          "friction measurement replace them. 'old' = pass count under the superseded "
          "base-drift metric.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
