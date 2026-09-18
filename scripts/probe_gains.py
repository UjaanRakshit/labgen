import inspect, newton
print("SolverMuJoCo:", inspect.signature(newton.solvers.SolverMuJoCo.__init__))
b = newton.ModelBuilder()
print("\ndefault_joint_cfg:", type(getattr(b, "default_joint_cfg", None)).__name__)
cfg = getattr(b, "default_joint_cfg", None)
if cfg: print("  fields:", [f for f in dir(cfg) if not f.startswith('_')])
print("\nbuilder joint arrays:", [a for a in dir(b) if a.startswith("joint_") and "target" in a or a.startswith("joint_dof")][:15])
