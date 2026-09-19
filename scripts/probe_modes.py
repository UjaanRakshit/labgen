import newton, inspect
cfg = newton.ModelBuilder.JointDofConfig
print("JointDofConfig signature:")
print(" ", inspect.signature(cfg.__init__))
print("\ndefault actuator_mode:", getattr(cfg(), "actuator_mode", "?"))
for name in dir(newton):
    if "Mode" in name or "Axis" == name:
        obj = getattr(newton, name)
        members = [m for m in dir(obj) if m.isupper() or (m[:1].isupper() and not m.startswith("_"))]
        print(f"newton.{name}: {members[:12]}")
