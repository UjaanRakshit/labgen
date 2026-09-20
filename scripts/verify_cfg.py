"""Import the generated scene config inside Isaac Lab and instantiate it.

The point of this file is that labgen does NOT get to mark its own homework.
The emitter writes Python; the only meaningful check is whether Isaac Lab's own
classes accept it. Parsing it with `ast` proves nothing beyond syntax, and this
project has already shipped one artifact that its own reader loved and no real
implementation would open.
"""
import importlib.util, sys, traceback
from pathlib import Path

CFG = Path(sys.argv[1] if len(sys.argv) > 1 else "bench_arm_cfg.py")

spec = importlib.util.spec_from_file_location("generated_cfg", CFG)
mod = importlib.util.module_from_spec(spec)
# Register BEFORE executing: @configclass wraps @dataclass, and dataclasses
# resolves field types via sys.modules[cls.__module__]. A module that is not
# registered there makes that lookup return None and the decorator blow up --
# a harness bug that looks exactly like a bad generated config.
sys.modules["generated_cfg"] = mod
try:
    spec.loader.exec_module(mod)
except Exception:
    print("IMPORT FAILED:")
    traceback.print_exc()
    sys.exit(1)
print(f"imported {CFG.name}")

# Pick the class DEFINED here, not one it imported. `InteractiveSceneCfg` is
# in this module's namespace too and also ends in "SceneCfg"; selecting by name
# grabbed the base class, instantiated that, and printed VERIFIED while testing
# nothing the generator produced.
from isaaclab.scene import InteractiveSceneCfg as _Base
candidates = [v for v in vars(mod).values()
              if isinstance(v, type) and getattr(v, "__module__", "") == "generated_cfg"
              and issubclass(v, _Base) and v is not _Base]
if not candidates:
    print("no scene config class defined in the generated module")
    sys.exit(1)
cls = candidates[0]
print(f"found {cls.__name__} (subclass of InteractiveSceneCfg: {issubclass(cls, _Base)})")

cfg = cls(num_envs=1, env_spacing=2.0)
print("instantiated OK")

entities = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
for name, v in entities.items():
    kind = type(v).__name__
    extra = ""
    if hasattr(v, "init_state") and hasattr(v.init_state, "pos"):
        extra = f" pos={tuple(round(c,4) for c in v.init_state.pos)} rot={v.init_state.rot}"
    if hasattr(v, "prim_path"):
        print(f"   {name:12} {kind:18} {v.prim_path}{extra}")
    else:
        print(f"   {name:12} {kind}")

# The arm's actuators must have actually been parsed into cfg objects.
robot = getattr(cfg, "robot", None)
if robot is not None:
    print(f"\n   robot spawn: {type(robot.spawn).__name__} -> {robot.spawn.asset_path}")
    for aname, act in robot.actuators.items():
        print(f"   actuator {aname!r}: joints={act.joint_names_expr} "
              f"stiffness={act.stiffness} damping={act.damping} "
              f"effort={act.joint_effort_limit}")
print("\nVERIFIED: Isaac Lab accepts the generated config")
