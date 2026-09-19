import inspect, newton
C = newton.ModelBuilder.MeshApproximationConfig
print("MeshApproximationConfig:", inspect.signature(C.__init__))
print("fields:", [f for f in dir(C()) if not f.startswith('_')])
S = newton.ModelBuilder.ShapeConfig
print("\nShapeConfig:", inspect.signature(S.__init__))
b = newton.ModelBuilder()
print("\ndefault_shape_cfg fields:", [f for f in dir(b.default_shape_cfg) if not f.startswith('_')])
print("builder mesh approx attr:", [a for a in dir(b) if 'approx' in a.lower()])
