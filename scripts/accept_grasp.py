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
from grasp import GraspFK, N_ARM, set_state, with_fingers                  # noqa: E402
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.catalog import CATALOG                                         # noqa: E402
from labgen.control import YAM_JAWS, YAM_PAD, lerp_path, solve_pose        # noqa: E402
from labgen.settle import CONTACT_KD, CONTACT_KE                           # noqa: E402
from labgen.types import SceneSpec                                         # noqa: E402

URDF = "/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"
SCENE_JSON = "/mnt/c/Ujaan Docx/Research/labgen/examples/bench_arm.json"

DT = 1.0 / 960.0
LIFT_M = 0.10
HOLD_S = 2.0
LIFT_PASS_M = 0.090        # at least 90 of the 100 mm
SLIP_PASS_M = 0.005        # beaker may move no more than 5 mm relative to the hand during the hold
LAUNCH_SPEED = 1.0         # m/s: far above anything a slow lift produces
LAUNCH_DIST_M = 0.05       # beaker more than 5 cm from the hand at the end
CLOSE_GAP_M = 0.040
import os as _os
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
    from grasp import quat_to_matrix as q2m
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
                                         nconmax=T.NCONMAX, njmax=T.NJMAX)
    targets = control.joint_target_q.numpy().copy()
    ff = control.joint_f.numpy().copy()

    beaker_z0 = None
    peak_speed = 0.0
    hold_rel = []
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
            hold_rel.append(bpos - inst.fk.grasp_point_from_state(bq))
        if name != last_phase and last_phase is not None:
            grip_gap[last_phase] = jaw_gap(inst, bq)
        last_phase = name
    grip_gap[last_phase] = jaw_gap(inst, s0.body_q.numpy())

    bq = s0.body_q.numpy()
    bpos = bq[beaker_body, :3]
    hand = inst.fk.grasp_point_from_state(bq)
    lifted = float(bpos[2] - beaker_z0)
    rel = np.array(hold_rel)
    slip = float(np.linalg.norm(rel - rel[0], axis=1).max()) if len(rel) else float("nan")
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
    return dict(ok=ok, diverged=False, lifted_m=lifted, slip_m=slip,
                peak_speed=peak_speed, launched=launched, left_behind=left_behind,
                grip_close_m=grip_gap.get("close", np.nan), pried_m=pried)


def main() -> int:
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
          f"slip<={SLIP_PASS_M*1000:.0f} mm, not launched")
    print()
    print(f"{'force/finger':>12} {'pass':>6} {'lifted mm':>16} {'slip mm':>14} "
          f"{'peak speed':>14} {'launched':>9} {'behind':>6}   grip at close / pried open in hold")

    summary = []
    for F in forces:
        model, arms = T.build(Path(sys.argv[1]), Path(URDF), rig, collide=True,
                              contact=(CONTACT_KE, CONTACT_KD), finger_force_n=F)
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
        speed = np.array([r["peak_speed"] for r in good])
        launched = sum(r.get("launched", False) for r in good)
        behind = sum(r.get("left_behind", False) for r in good)
        diverged = sum(r["diverged"] for r in results)
        grip = np.array([r["grip_close_m"] for r in good]) * 1000
        pried = np.array([r["pried_m"] for r in good]) * 1000
        print(f"{F:10.0f} N {ok:3d}/{repeats:<2d} "
              f"{lifted.min():6.1f}-{lifted.max():6.1f} "
              f"{slip.min():5.1f}-{slip.max():5.1f} "
              f"{speed.min():5.2f}-{speed.max():5.2f} m/s {launched:6d} "
              f"{behind:6d}   jaw {grip.mean():5.1f}mm pried {pried.mean():+5.1f}mm"
              f"{f'  diverged {diverged}' if diverged else ''}"
              f"   ({time.time()-t0:.0f} s)", flush=True)
        summary.append((F, ok, repeats, launched))

    print()
    passing = [F for F, ok, n, l in summary if ok >= 0.9 * n and l == 0]
    print("ACCEPTANCE (>= 9/10, zero launches): "
          + (f"PASS at {passing} N per finger" if passing else "FAIL at every force tested"))
    print("Caveat: finger pads and pad friction are UNVERIFIED; this result is "
          "conditional on them until calipers and a friction measurement replace them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
