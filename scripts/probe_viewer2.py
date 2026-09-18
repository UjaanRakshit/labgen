import inspect
from newton.viewer import ViewerGL, ViewerFile, ViewerUSD, ViewerRTX, ViewerBase
for cls in (ViewerGL, ViewerFile, ViewerUSD, ViewerRTX):
    try:
        sig = inspect.signature(cls.__init__)
        print(f"{cls.__name__}{sig}")
    except Exception as e:
        print(f"{cls.__name__}: {e}")
    meth = [m for m in dir(cls) if not m.startswith('_')]
    print(f"   methods: {meth}\n")
print("ViewerBase methods:", [m for m in dir(ViewerBase) if not m.startswith('_')])
