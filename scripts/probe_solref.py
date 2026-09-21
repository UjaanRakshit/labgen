"""Is the contact fix physically justified, or just the numbers that went green?

The honest history: the values were found by sweeping (ke, dt) and taking the
corner where sink went to zero. No mechanism was derived at the time, and the
README justified ke=160000 as "glass on a steel benchtop is far stiffer than
the importer's 2500 N/m". That justification is WRONG. Newton hands ke and kd
to MuJoCo through convert_solref(ke, kd, 1, 1):

    timeconst = 2 / kd
    dampratio = kd / (2 * sqrt(ke))

so ke is not a stiffness in N/m at all -- it only sets the damping ratio. Both
the old and the new configuration are critically damped (dampratio = 1.000),
because the importer default kd=100 already equals 2*sqrt(2500). The only thing
that actually changed is the constraint time constant: 20 ms -> 2.5 ms.

If that is the mechanism, it makes a prediction. MuJoCo's soft constraint is a
critically damped second order system with natural frequency 1/timeconst, so an
object resting under gravity settles at a steady-state penetration of

    delta = g * timeconst^2

which is INDEPENDENT OF MASS -- and mass independence is exactly the signature
that was measured and found confusing (all five beakers sank ~1.25 mm across a
13x mass range, and sink anti-correlated with contact pressure at r = -0.76).

This sweeps timeconst at fixed critical damping, with a timestep small enough
that MuJoCo's timeconst >= 2*dt requirement holds everywhere, and checks the
measured sink against g*timeconst^2. If the quadratic holds, the choice was
principled after the fact. If it does not, the story is wrong and the values
are just the ones that went green.
"""
import math
import sys

import numpy as np

sys.path.insert(0, "/home/ujaan/isaac/labgen")
from probe_sink import scene_for, settle                                 # noqa: E402

G = 9.81
DT = 1.0 / 1920          # 0.52 ms; timeconst=1.25 ms still clears 2*dt
# Two vessels, not four: the prediction is that penetration is MASS
# INDEPENDENT, so the extreme pair tests it better than a crowd. The petri
# dish is 16 g on a wide base (20 Pa); beaker_1000 is 320 g (363 Pa).
VESSELS = ["petri_dish_100", "beaker_1000"]


def main() -> int:
    print(f"dt = 1/{1/DT:.0f} s ({DT*1000:.3f} ms), critical damping throughout")
    print(f"{'timeconst':>10} {'kd':>7} {'ke':>9} {'tc/dt':>6} {'predicted':>10}  measured (mm)")
    print(f"{'ms':>10} {'':>7} {'':>9} {'':>6} {'g*tc^2 mm':>10}  "
          + "  ".join(f"{v.split('_')[0][:7]:>8}" for v in VESSELS))

    rows = []
    for kd in (100.0, 200.0, 400.0, 800.0):
        tc = 2.0 / kd
        ke = (kd / 2.0) ** 2          # keeps dampratio = kd/(2*sqrt(ke)) = 1
        pred = G * tc * tc * 1000.0
        got = [abs(settle(scene_for(v), dt=DT, ke=ke, kd=kd)) * 1000 for v in VESSELS]
        rows.append((tc, pred, got))
        print(f"{tc*1000:10.3f} {kd:7.0f} {ke:9.0f} {tc/DT:6.1f} {pred:10.3f}  "
              + "  ".join(f"{g:8.3f}" for g in got))

    print()
    print("Does the quadratic hold? ratio of measured to g*timeconst^2:")
    for i, v in enumerate(VESSELS):
        ratios = [r[2][i] / r[1] for r in rows if r[1] > 0]
        print(f"   {v:22} " + "  ".join(f"{x:6.2f}" for x in ratios))

    # Damping ratio should NOT matter much if timeconst is what sets penetration.
    print()
    print("same timeconst (2.5 ms), different damping ratios:")
    for ke, label in ((40000.0, "dampratio 2.0 (overdamped)"),
                      (160000.0, "dampratio 1.0 (critical)"),
                      (640000.0, "dampratio 0.5 (underdamped)")):
        got = abs(settle(scene_for("petri_dish_100"), dt=DT, ke=ke, kd=800.0)) * 1000
        print(f"   ke={ke:8.0f}  {label:28} petri sink {got:6.3f} mm")

    # And the requirement that forces the timestep down.
    print()
    print("timeconst = 2.5 ms against timesteps that do and do not satisfy tc >= 2*dt:")
    for dt in (1/240, 1/480, 1/960, 1/1920):
        got = abs(settle(scene_for("petri_dish_100"), dt=dt, ke=160000.0, kd=800.0)) * 1000
        ok = "ok" if 0.0025 >= 2 * dt else "VIOLATES tc >= 2*dt"
        print(f"   dt=1/{1/dt:.0f}  tc/dt={0.0025/dt:5.2f}  petri sink {got:7.3f} mm   {ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
