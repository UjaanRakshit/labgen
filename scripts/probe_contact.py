import sys, numpy as np, newton
def run(ke, kd, label):
    b = newton.ModelBuilder(); r = b.add_usd("bench_arm.usda")
    if ke is not None:
        for i in range(b.shape_count):
            b.shape_material_ke[i] = ke
            b.shape_material_kd[i] = kd
    m = b.finalize()
    paths = {v:k for k,v in r["path_body_map"].items()}
    s0,s1 = m.state(), m.state(); c = m.control()
    solver = newton.solvers.SolverMuJoCo(m, iterations=100, ls_iterations=50)
    q0 = s0.body_q.numpy().copy()
    for _ in range(480):
        ct = m.collide(s0); solver.step(s0,s1,c,ct,1/240); s0,s1 = s1,s0
        if not np.isfinite(s0.body_q.numpy()).all(): print(f"{label}: DIVERGED"); return
    q1 = s0.body_q.numpy()
    out = []
    for i in range(m.body_count):
        n = paths.get(i,'?').replace('/World/','')
        out.append(f"{n}={(q1[i,2]-q0[i,2])*1000:+.2f}")
    print(f"{label:22} " + "  ".join(out))
run(None, None, "default")
for ke in (1e4, 1e5, 1e6):
    run(ke, ke/25, f"ke={ke:.0e}")
