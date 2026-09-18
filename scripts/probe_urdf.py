import inspect, newton
print("ModelBuilder urdf/mjcf entry points:",
      [a for a in dir(newton.ModelBuilder) if "urdf" in a.lower() or "mjcf" in a.lower()])
for name in ("add_urdf", "add_mjcf"):
    fn = getattr(newton.ModelBuilder, name, None)
    if fn: print(f"\n{name}{inspect.signature(fn)}")
print("\nnewton.Transform:", getattr(newton, "Transform", None))
print("transform helpers:", [a for a in dir(newton) if "ransform" in a or a in ("quat_identity","Axis")][:10])
