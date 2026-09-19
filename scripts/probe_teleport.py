"""Does writing state.joint_q between steps actually move a free body?

If SolverMuJoCo keeps its own qpos and only syncs one way, then every
"kinematic carry" write is discarded: the object keeps falling in the solver's
world while the state says it is in the gripper, and at release the solver's
value wins. That would explain a beaker released 255 mm below the worktop.
"""
import sys, numpy as np, newton
b = newton.ModelBuilder(); r = b.add_usd("bench_arm.usda"); m = b.finalize()
paths = {v:k for k,v in r["path_body_map"].items()}
bk = [i for i in range(m.body_count) if paths[i].endswith("beaker")][0]
s0,s1 = m.state(), m.state(); c = m.control()
solver = newton.solvers.SolverMuJoCo(m, iterations=60)
print(f"beaker is body {bk}; joint_q slots [{7*bk}:{7*bk+7}]")
for step in range(6):
    # teleport it 100 mm up, every step
    jq = s0.joint_q.numpy(); jq[7*bk+2] = 0.10
    s0.joint_q.assign(jq.astype(np.float32))
    jqd = s0.joint_qd.numpy(); jqd[6*bk:6*bk+6] = 0.0
    s0.joint_qd.assign(jqd.astype(np.float32))
    newton.eval_fk(m, s0.joint_q, s0.joint_qd, s0)
    before = s0.body_q.numpy()[bk,2]
    ct = m.collide(s0); solver.step(s0,s1,c,ct,1/240); s0,s1 = s1,s0
    after = s0.body_q.numpy()[bk,2]
    print(f"  step {step}: wrote z=100.0 -> body_q says {before*1000:7.2f} mm"
          f" -> after solver.step {after*1000:7.2f} mm")
print("\nif 'after' collapses back toward 0, the solver is ignoring the write")
