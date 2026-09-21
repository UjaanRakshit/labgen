"""Why does the two-arm scene diverge while sitting still?

It held its ready pose, reported 16 Hz for several minutes, and then blew up
with no input at all. An idle scene that explodes is either contact that should
not be there or a drive fighting itself, and the way to tell them apart is to
remove one at a time.

Reports time-to-divergence and which body moved, for: no collision at all,
collision at the placeholder separation, and collision at wider separations.
"""
import sys
from pathlib import Path

import numpy as np

import newton

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teleop_sim as T                                                    # noqa: E402
from grasp import GraspFK, N_ARM, with_fingers                            # noqa: E402
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.control import (YAM_JAWS, BimanualRig, solve_pose,            # noqa: E402
                            yam_arm_spec)

URDF = "/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"
SECONDS = 25.0


def run(sep_m, collide, scene):
    import warp as wp
    probe = newton.ModelBuilder()
    probe.add_urdf(URDF, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0),
                                            wp.quat_identity()),
                   floating=False, enable_self_collisions=False,
                   collapse_fixed_joints=True, up_axis=newton.Axis.Z,
                   parse_visuals_as_colliders=False)
    lo_all = np.asarray(probe.joint_limit_lower, float)
    hi_all = np.asarray(probe.joint_limit_upper, float)

    rig = BimanualRig([
        yam_arm_spec(lo_all, hi_all, name="arm_left"),
        yam_arm_spec(lo_all, hi_all, name="arm_right",
                     base_position_m=(sep_m, 0.0, 0.0),
                     placement_source="PLACEHOLDER stability sweep"),
    ])
    model, arms = T.build(Path(scene), Path(URDF), rig, collide=collide)
    solver = newton.solvers.SolverMuJoCo(model, iterations=10, ls_iterations=20,
                                         nconmax=T.NCONMAX, njmax=T.NJMAX)

    s0, s1 = model.state(), model.state()
    control = model.control()
    q_all = model.joint_q.numpy().copy()
    for inst in arms:
        inst.fk = GraspFK(model, inst.coords, body_offset=inst.body_offset,
                          n_bodies=len(inst.masses))
        res = solve_pose(inst.fk, inst.ready, inst.lo, inst.hi, approach=T.DOWN)
        inst.q_cmd = res.q
        q_all[inst.coords] = with_fingers(res.q, YAM_JAWS.q_open)
    model.joint_q.assign(q_all.astype(np.float32))
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    newton.eval_fk(model, model.joint_q, model.joint_qd, s1)

    targets = control.joint_target_q.numpy().copy()
    for inst in arms:
        targets[inst.coords] = with_fingers(inst.q_cmd, YAM_JAWS.q_open)
    control.joint_target_q.assign(targets)

    start = s0.body_q.numpy()[:, :3].copy()
    peak_contacts = 0
    steps = int(SECONDS / T.DT)
    for i in range(steps):
        contacts = model.collide(s0)
        n = 0
        if hasattr(contacts, "rigid_contact_count"):
            n = int(contacts.rigid_contact_count.numpy()[0])
        peak_contacts = max(peak_contacts, n)
        solver.step(s0, s1, control, contacts, T.DT)
        s0, s1 = s1, s0
        q = s0.body_q.numpy()
        if not np.isfinite(q).all():
            return i * T.DT, peak_contacts, -1, 0.0
    end = s0.body_q.numpy()[:, :3]
    moved = np.linalg.norm(end - start, axis=1)
    return None, peak_contacts, int(moved.argmax()), float(moved.max()) * 1000


def main() -> int:
    scene = sys.argv[1]
    print(f"{'separation':>11} {'collide':>8} {'diverged at':>12} "
          f"{'peak contacts':>14} {'worst body move':>16}")
    for collide, sep in ((False, 0.44), (True, 0.44), (True, 0.60), (True, 0.80)):
        t, contacts, worst, moved = run(sep, collide, scene)
        when = f"{t:.2f} s" if t is not None else f"no ({SECONDS:.0f} s)"
        detail = "n/a" if t is not None else f"body {worst}: {moved:.2f} mm"
        print(f"{sep*1000:9.0f} mm {str(collide):>8} {when:>12} "
              f"{contacts:14d} {detail:>16}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
