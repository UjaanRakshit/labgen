import newton, importlib
print("newton", newton.__version__)
for m in ("newton.viewer", "newton.utils.render"):
    try:
        mod = importlib.import_module(m)
        print(f"\n{m}: {[a for a in dir(mod) if not a.startswith('_')][:30]}")
    except Exception as e:
        print(f"\n{m}: {type(e).__name__} {e}")
