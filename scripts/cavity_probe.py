import sys, numpy as np, newton
USDA = sys.argv[1]
b = newton.ModelBuilder(); r = b.add_usd(USDA); m = b.finalize()
paths = {v: k for k, v in r["path_body_map"].items()}
s0, s1 = m.state(), m.state(); c = m.control()
start = s0.body_q.numpy().copy()
solver = newton.solvers.SolverMuJoCo(m)
dt = 1/240
for _ in range(int(3.0*240)):
    contacts = m.collide(s0)
    solver.step(s0, s1, c, contacts, dt)
    s0, s1 = s1, s0
    if not np.isfinite(s0.body_q.numpy()).all():
        print("DIVERGED"); sys.exit(1)
end = s0.body_q.numpy()
print(f"{'body':10} {'start z':>9} {'end z':>9}")
print("-"*32)
mz = None
for i in range(m.body_count):
    n = paths.get(i,f"b{i}").replace("/World/","")
    print(f"{n:10} {start[i,2]:9.5f} {end[i,2]:9.5f}")
    if n == "marble": mz = end[i,2]
print()
print("beaker cavity floor = 0.00150 m ; beaker rim = 0.09500 m")
if mz is None:
    print("no marble")
elif mz < 0.010:
    print(f"marble rests at {mz:.5f} m -> INSIDE the cavity. Cavity is REAL.")
elif mz > 0.060:
    print(f"marble rests at {mz:.5f} m -> ON TOP. The vessel is SEALED.")
    print("   CLAUDE.md hard rule 1 is being violated by the solver, not the asset.")
else:
    print(f"marble rests at {mz:.5f} m -> ambiguous; inspect.")
