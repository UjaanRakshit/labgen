import newton
print("ShapeFlags:", {a: int(getattr(newton.ShapeFlags,a)) for a in dir(newton.ShapeFlags) if a.isupper()} if hasattr(newton,'ShapeFlags') else 'n/a')
for n in dir(newton):
    if 'Flag' in n: print(" ", n, [a for a in dir(getattr(newton,n)) if a.isupper()])
