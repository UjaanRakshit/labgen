# labgen

Turns video of a real chemistry bench into a scene that loads in Isaac Lab with
the right objects, at the right size, in the right places, **in the robot's
coordinate frame**.

This is phase 1 of a larger project. Phase 1 ends at "the scene is correct".
It contains no policy loading, no rollouts, no success predicates and no
scoring, and it should not grow any. See `CLAUDE.md`.

## Lab runbook

For whoever is at the lab — a person, or a Claude session on a lab machine —
with none of this project's history. Three jobs, on up to three machines:

| where | job | needs |
| --- | --- | --- |
| **arm laptop** — Linux, YAM teleop checkout, GS_USB CAN adapters | record the real arm's response (sim-to-real) | teleop's own environment (i2rt, pyyaml) + this repo |
| **Isaac Lab machine** — Ubuntu or WSL2, NVIDIA GPU | the sim: tasks, phone teleop, MimicGen | Isaac Lab `develop` @ `a8b4da3c2`, kit-less Newton |
| anywhere | turn sim demos into the lab's dataset format | numpy, h5py |

**Do not `pip install` into the Isaac Lab or yam_vr_teleop environments** (CLAUDE.md
rule 5). Everything extra here lives in its own venv.

### 1. Real arm: sim-to-real recording (arm laptop)

Non-negotiable: a person at the e-stop, the workspace clear, and the teleop
**stopped** — the recorder checks for a running teleop/CAN process, takes its
own per-channel lock, and checks the CAN adapter's USB serial against the
config or the host's persistent systemd link. This Jetson teleop checkout does
not share the lock, so the operator must keep teleop stopped. The
`--start-at-current` protocol uses 0.10 rad eased steps and eased returns, one
joint at a time, from the measured pose; it makes no move to the distant reset
pose and omits the shoulder sine. It checks driver joint limits, and aborts and
holds if any joint exceeds 3 rad/s or tracking error exceeds 0.25 rad. Support
the arm when its driver closes and torque drops. The older reset-pose protocol
remains available without `--start-at-current` and refuses an automatic reset
move over 0.15 rad on any joint.

```bash
git clone https://github.com/UjaanRakshit/labgen ~/labgen || git -C ~/labgen pull
set -o pipefail
DAY=~/labgen/data/lab/$(date +%F); mkdir -p "$DAY"; cp -n ~/labgen/data/lab/TEMPLATE.md "$DAY/README.md"
# find the checkout; the Jetson uses /home/pair/pair-mse/yam_teleop
TELEOP=$(dirname "$(dirname "$(find ~ -maxdepth 4 -path '*/deployment/quest_teleop.py' 2>/dev/null | head -1)")")
cd "$TELEOP" && echo "teleop checkout: $TELEOP"
TPY="$TELEOP/.conda-env/bin/python"; [ -x "$TPY" ] || TPY="$TELEOP/.venv/bin/python"
"$TPY" -m pip show i2rt | tee "$DAY/i2rt_version.txt"      # the sim's gains come from i2rt 1.1.2
"$TPY" -m deployment.preflight --config deployment/config.yaml     --second-config deployment/config_left.yaml | tee "$DAY/preflight.txt"      # nothing energizes; must pass before a real run
"$TPY" ~/labgen/scripts/sim2real_record.py --out "$DAY/i2rt_sim.npz" | tee "$DAY/dry_run.txt"
# ONLY after the owner confirms: person at the e-stop, workspace clear, teleop stopped
"$TPY" ~/labgen/scripts/sim2real_record.py --backend real --teleop-config deployment/config.yaml     --i-have-cleared-the-workspace --start-at-current --out "$DAY/real_right.npz" | tee "$DAY/real_right.txt"
"$TPY" ~/labgen/scripts/sim2real_record.py --backend real --teleop-config deployment/config_left.yaml     --i-have-cleared-the-workspace --start-at-current --out "$DAY/real_left.npz" | tee "$DAY/real_left.txt"
```

Each current-pose run is ~20 s. Its measured starting angles are stored in the
`.npz`, so the old `data/sim2real/newton_prediction.npz` is **not comparable**.
On the Isaac Lab machine, with these logs copied into the same `DAY` folder,
run Newton once per arm using the log's exact protocol and start pose:

```bash
PY=~/IsaacLab/.venv/bin/python
YAM_URDF=~/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf
for arm in right left; do
  "$PY" ~/labgen/scripts/sim2real_newton.py "$YAM_URDF" --protocol-log "$DAY/real_$arm.npz" --out "$DAY/newton_$arm.npz"
  "$PY" ~/labgen/scripts/sim2real_compare.py "$DAY/real_$arm.npz" "$DAY/newton_$arm.npz" | tee "$DAY/compare_$arm.txt"
done
```

### Lab session checklist

In priority order. Record every result in `data/lab/<YYYY-MM-DD>/`, starting
from `data/lab/TEMPLATE.md` (copy it to `data/lab/<date>/README.md`), then hand
it back:

```bash
git checkout -b lab/<YYYY-MM-DD>
git add data/lab/<YYYY-MM-DD> && git commit -m "Lab results <date>" && git push -u origin lab/<YYYY-MM-DD>
```

If the machine cannot push, copy the folder to a USB stick instead. Do not
merge to `main`.

1. **i2rt version** on the arm laptop (`pip show i2rt` in the teleop venv).
2. **Real-arm recordings and their comparison** (section 1 writes everything,
   console output included, into `$DAY`). Summarise in the folder's README
   whether each run completed and anything that looked wrong.
3. **Measurements**, with calipers and a tape: the fields in `TEMPLATE.md`
   (finger pads, which gripper, second arm's base offset and heading). Write
   down the tool and how each was measured. Do NOT edit `labgen/control.py`
   or `labgen/hardware.py`; the numbers are reviewed before they go in.
4. **Vial demos** on the phone (section 2), if a sim machine is there: five or
   more that complete the task; copy `demos/vial_real.hdf5` into the folder.

### 2. The sim (Isaac Lab machine)

One-time assets. The URDF→USD converter goes in **its own** venv (Isaac Lab's
URDF/MJCF spawners need Isaac Sim, which the kit-less install does not have):

```bash
uv venv -p 3.12 ~/.venv-usdconv && VIRTUAL_ENV=~/.venv-usdconv uv pip install urdf-usd-converter numpy
git clone https://github.com/i2rt-robotics/i2rt ~/i2rt                     # yam.urdf
YAM_URDF=~/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf \
URDF_USD_CONVERTER=~/.venv-usdconv/bin/urdf_usd_converter \
  ~/.venv-usdconv/bin/python scripts/make_yam_usd.py        # -> out/yam/usd/yam.usdc (colliders, pads, coupled fingers)
python scripts/make_bench_vial.py                           # -> examples/bench_vial.json, out/bench_vial.usda
python -c "from labgen.types import SceneSpec; from labgen import usda; \
usda.write_scene(SceneSpec.read('examples/bench_arm.json'), 'out/bench_arm.usda')"
```

Then, with `PY=~/IsaacLab/.venv/bin/python` (set `ISAACLAB_DIR` if Isaac Lab is
not at `/home/ujaan/isaac/IsaacLab`):

```bash
$PY scripts/run_bench_task.py      # expect: settle PASS, pick PASS (lift ~99 mm, slip < 1 mm)
$PY scripts/demo_vial.py           # scripted vial task; expect SUCCESS -> demos/vial_scripted.hdf5
```

**Phone teleop.** The phone page is stdlib Python and must run where the phone
can reach it — on native Linux, the same machine; under WSL2, on the Windows side:

```bash
python scripts/teleop_touch.py                               # prints the URL to open on the phone
$PY scripts/teleop_task.py --task vial --stream --out demos/vial_real.hdf5
```

Choose RIGHT on the phone (the right arm, robot0, which does the vial task), ENGAGE, drag to move, slider
to grip. Completing the task (vial onto the hotplate, released, back onto the
bench) saves the demo and resets; SAVE DEMO / DISCARD override. Demos are recorded
already annotated for MimicGen.

**MimicGen**, from five or more demos:

```bash
$PY scripts/mimic.py generate --task Isaac-LabBench-YAM-VialHotplate-IK-Rel-Mimic-v0 \
    --input_file demos/vial_real.hdf5 --output_file demos/vial_gen.hdf5 \
    --generation_num_trials 50 --num_envs 1
```

Expect roughly half the attempts to succeed; only successes are written.

### 3. Sim demos in the lab's dataset format

```bash
python scripts/export_yam_vr.py demos/vial_gen.hdf5 out/yam_vr/
cd "$TELEOP" && .venv/bin/python -m deployment.export_demos ~/labgen/out/yam_vr/ --lerobot out/sim_dataset
```

**Do not pass `--fps` above 20** for sim data: yam_vr_teleop's exporter only
resamples down, and asked for more it relabels the frames without resampling
(a 20 Hz demo exported "at 30 fps" is 1.5× time-compressed, with no warning).

### What the sim reproduces from the real stack, and what it does not

Reproduced, from the code that drives the real arms: i2rt 1.1.2's per-joint
gains and torque limits; gravity compensation through the actuators; the
teleop's command stage — joint targets within 0.25 rad of measured and slewed at
≤ 1.5 rad/s, proportional gripper slewed at one full range per second
(yam_vr_teleop `config.yaml` `safety`, `quest_teleop.py:971`); the rig's reset pose.

Different: 20 Hz control (real 100 Hz); Isaac Lab's damped-least-squares
relative IK (real: mink QP on an absolute target); no One Euro filter.

**Unverified**, and every recorded dataset says so in its attributes: finger pad
size and friction (URDF mesh, not calipers), finger PD gains and the 25 N grip
cap, the bench object layout (PLACEHOLDER; the two arms' layout is measured), the vial's neck and the
hotplate's dimensions (catalog TODOs), and gravity-comp factor (real 1.1–1.2 on
j2–j4, sim 1.0). No real-arm log has been compared yet.

## The design decision everything else follows from

Do not reconstruct objects from video. Identify them and fit a parametric
template.

A chemistry lab is almost entirely catalog parts with published dimensions. A
250 mL Griffin beaker is 70 mm across and 95 mm tall whether or not a mesh
generator agrees. Photogrammetry and mesh generation both fail on transparent
and thin-walled objects — which is most of a chem lab — and neither gives you
metric scale or collision geometry.

So: recognise the object, look it up, instantiate the template, estimate its
pose. Mesh generation is a fallback for genuinely non-catalog items and is
marked low-confidence in the output.

## Why the fussiness about scale and frames

MATTERIX¹ reports four causes for their real-world sim failures: pose
estimation error, camera calibration, object frame definition, and sim/real
asset size mismatch. All four are registration and scale problems. None are
modelling problems — one failure was caused purely by the simulated table being
a different size from the real one.

That is the whole reason this package refuses to guess. Monocular video has no
metric scale, so a scene built without a marker of known size is rejected
rather than estimated. Quaternions that are not unit norm are an error rather
than silently renormalised. Masses are never inferred from a density. A scene
that is quietly 8% wrong is worse than a scene that refuses to build, because
this pipeline feeds a system that will eventually produce trust numbers.

## Layout

| module            | contract                                          | task |
| ----------------- | ------------------------------------------------- | ---- |
| `catalog.py`      | parametric lab object templates (dims, mass, collider) | — |
| `types.py`        | the artifacts stages pass between each other      | T0 ✅ |
| `meshes.py`       | `CatalogItem` → watertight triangle mesh (OBJ)    | T1 ✅ |
| `usda.py`         | `SceneSpec` → `.usda` with UsdPhysics schemas     | T2 ✅ |
| `validate.py`     | the geometric gates                               | T3 ✅ |
| `settle.py`       | the settle test, behind a backend interface       | T3 ✅ |
| `isaaclab_cfg.py` | `SceneSpec` → `InteractiveSceneCfg` (emitted as text) | T4 ✅ |
| `control.py`      | scripted IK, jaws, finger pads, bimanual rig (pure numpy) | A ✅ |
| `hardware.py`     | the real robot's per-joint actuators, from the rig's own i2rt | — |
| `devices.py`      | pose-streaming devices → hand targets (phone, Quest)   | — |
| `sim2real.py`     | protocol, log format and comparison for sim vs real    | — |
| `register.py`     | reconstruction frame → robot base frame           | T5 |
| `identify.py`     | video → `[ObjectObservation]` (VLM)               | T6 |
| `fit.py`          | `ObjectObservation` → `SceneObject`               | T6 |
| `cli.py`          | `labgen build <video> --markers <cfg> -o out/`    | T7 |

Each stage takes a typed artifact and emits a typed artifact, and each has a
validator that runs **without an LLM in the loop**. A stage does not advance
until its validator passes.

`SceneSpec` is JSON and is meant to be edited by hand when the pipeline gets
something wrong. That round-trip is a feature, not a debugging escape hatch, so
loading is strict — an unknown key is an error rather than a shrug.

## Two checks, never one

Any check that compares generated geometry against a catalog number is split in
two, and new ones must be too:

1. **generated vs analytic** — is the mesh code correct for the given
   dimensions? Tight tolerance; the only error source is polygonal
   approximation.
2. **catalog number vs physical plausibility** — are the dimensions themselves
   right? A loose, shape-dependent band.

"We have a bug" and "the seed data is bad" have completely different fixes, and
a conflated check reports the same failure for both. Splitting T1's
cavity-volume criterion this way is what showed the mesh code was correct and
five catalog dimensions were not.

## Provenance: `verified` means sourced

`CatalogItem.verified` means a named spec sheet, part number, or measurement
exists — recorded in `CatalogItem.source`. It does **not** mean the author felt
confident, and `verified=True` without a source raises. A `source` starting with
`TODO` is the absence of provenance, written down.

A flag that tracks confidence is set by the same process that produces the wrong
number, so it marks bad entries as good. The seed catalog shipped five wrong
dimensions all flagged `verified=True`. Every entry is currently `verified=False`
with a `source` naming what needs checking.

## The gates

`labgen.validate` runs nine checks with numpy and the catalog alone — no GPU,
no Isaac, milliseconds:

`provenance` · `scale` · `support` · `interpenetration` · `physics` ·
`collider` · `reachability` · `graspable` · `grasp_keypoint`

The last two are not in CLAUDE.md's list and belong there. `graspable` compares
each object against the gripper's measured 94 mm jaw stroke; `grasp_keypoint`
wants a keypoint near the rim of an open vessel, because grasping a beaker at
mid-height buries the hand 30 mm into the glass. Both were learned by doing it
wrong.

Every finding carries the measured value and the limit. "beaker fails scale" is
useless at 2am; "x extent 84.0 mm against catalog 70.0 mm, 20% over a 5%
tolerance" tells you whether to fix the scene or the catalog.

**What `scale` can and cannot catch.** For a catalog object the mesh is
*generated from* the catalog, so the check compares the catalog against itself:
it catches a generator regression and nothing else. Inflating a catalog entry
by 20% inflates the mesh by 20% and sails straight through. That is the
two-checks rule, not a hole — "is the mesh right for these numbers" and "are
these numbers right" have different fixes, and the second one is `source` /
`verified` and ultimately a spec sheet. Pinned as a test so nobody converts it
into a tautology that appears to catch catalog errors.

`labgen.settle` is separate because it needs a physics engine. A missing
backend **raises**; it never returns a vacuous pass. A settle test that quietly
does nothing is worse than none, because the report still reads green.

## Isaac Lab config

`labgen.isaaclab_cfg` emits the scene config as Python **source**, the same way
`usda.py` emits USD text — the emitter never imports `isaaclab`, so core stays
CI-clean. Verified by `scripts/verify_cfg.py`, which imports the generated
module inside Isaac Lab and instantiates the class. Parsing it with `ast` here
proves only syntax; this project has already shipped one artifact its own
reader loved and no real implementation would open.

Answering TASKS.md's open T4 question: **Isaac Lab 3.0 takes MJCF and URDF
directly** via `MjcfFileCfg` / `UrdfFileCfg`, which run the converter for you.
There is no converter to write.

Three settings in the generated file are load-bearing and all three were
measured, not chosen:

- `collision_from_visuals=True` — the YAM ships **zero** `<collision>`
  elements, so without this the arm renders perfectly and touches nothing
  (0 contacts out of 109).
- `collision_type="Convex Decomposition"` — *not* the converter's `"Convex
  Hull"` default, which turns each L-shaped fingertip into a block spanning
  ±36 mm and swallows a 70 mm beaker.
- **Actuators are the real robot's, per joint**, copied from the i2rt 1.1.2
  config the rig runs (`labgen.hardware.YAM_V1`): j1–3 DM4340 (28 N·m peak,
  kp 80, kd 5), j4–6 DM4310 (10 N·m, kp 10, kd 1.5), emitted as two actuator
  groups. Isaac Lab accepts the result (`scripts/verify_cfg.py`).

  This replaced a history of three wrong answers about the arm's torque, each of
  which was believed at the time:
  1. It once ran at 40 N·m "against the rated 10" because the shoulder sagged at
     10. The sag was a missing gravity feedforward, not a ceiling.
  2. The retraction then called 10 N·m "the robot's rating" and quoted gravity at
     the grasp pose as 7.26 N·m. That figure put each link's weight at its frame
     origin instead of its centre of mass — 133 mm off for link2, 150 mm for
     link3. Corrected: **9.71 N·m** at the grasp pose, **10.60 N·m** at the
     worst pose of the task plan.
  3. And 10 N·m was never the shoulder's rating. It is i2rt's simplified MJCF;
     the DM4340 peaks at 28. So the plan is over a 10 N·m limit and at 38% of
     the real motor.

  The centre-of-mass error hid behind the old kp=3000, which absorbs a bad
  feedforward. It surfaced the moment the sim ran the real kp=80, as 2.4° of
  sag with the feedforward on; corrected, 0.03°.

  Not emitted: joint friction (real: 0.3 / 0.06 N·m Coulomb), because Isaac
  Lab's `friction` field is a torque on one backend and a coefficient on
  another; and armature, because no one has recorded the rotor inertia.

Rotations are emitted `(x, y, z, w)`. Isaac Lab 3.0 changed from the older
wxyz; `SceneSpec` stores wxyz and they are reordered.

## Sim-to-real

The simulated arm is built to respond the way the real one does, and the claim is
checkable rather than asserted.

**Sourced from what the rig actually runs.** The lab's YAMs are driven from
`pairlab/yam_teleop` with **i2rt 1.1.2**; the sim was first built on a newer 1.3.6
clone. Between them the YAM URDF differs by 407 lines, including a redefined
wrist frame, but world-frame forward kinematics of both agree to **0.002 mm**
(`scripts/probe_model_versions.py`) — the arm is identical, only the gripper link
frame was relabelled. So **exchange joint angles between sim and real, never
end-effector poses.**

**Real actuators** (`labgen.hardware.YAM_V1`, from i2rt 1.1.2's `yam_v1.yml` and
motor driver): j1–3 DM4340 at 28 N·m peak, j4–6 DM4310 at 10 N·m, kp
80/10, kd 5/1.5, Coulomb friction 0.3/0.06 N·m, plus the controller's gravity
compensation. The sim previously used kp=3000 on every joint — chosen to make
the arm hold still — which hid a broken gravity feedforward: it put each link's
weight at its frame origin rather than its centre of mass (133 mm off for
link2). With the real gains that surfaced as 2.38° of sag; fixed, 0.000°.

**The harness.** One gentle protocol — 0.10 rad eased steps per joint from the
rig's recorded reset pose, then a slow shoulder sine, at the teleop's 100 Hz — run
on both sides and compared:

```
# on the arm laptop, in yam_vr_teleop's venv -- see the Lab runbook for the full procedure
python scripts/sim2real_record.py --backend real --teleop-config deployment/config.yaml \
    --i-have-cleared-the-workspace --out real_right.npz
# in the Isaac Lab env
python scripts/sim2real_newton.py yam.urdf --out newton_sim.npz
python scripts/sim2real_compare.py real_can0.npz newton_sim.npz
```

What the Newton sim predicts the real arm will do (rise 10–90% / overshoot /
settle 2%): j1 110 ms/5.4%/830 ms, j2 140 ms/9.6%/970 ms, j3 130 ms/1.8%/200 ms,
j4 480 ms/0%/1490 ms, j5 450 ms/0%/1100 ms, j6 390 ms/0%/720 ms. **No real-arm log
exists yet**, so these are predictions, not validated numbers.

**i2rt's own sim is not a dynamics reference.** `SimRobot.command_joint_pos`
teleports the joints ("CONTROL mode uses teleport") with physics off, so it
tracks perfectly and instantly. The lab teleop's `backend: sim` runs exactly
that, so nothing tested through it says anything about lag, overshoot or torque
limits.

**Contact grasp (item D).** The finger pads were first authored on the wrong
surface — the bracket by the mount, not the fingertips — so the jaws closed on
nothing. Re-measured fingertip-first in the world frame, the pad separation is
the jaw gap + 3.9 mm across the whole stroke, and the grasp point is now the
midpoint of the pads rather than the tip-mesh centroid (30 mm off). The pads and
the pad friction are both **unverified**: calipers and a friction measurement
replace them.

## Known gaps

Things that are **not** covered, recorded so a later stage does not assume they
are:

- **`bench_5` does not exercise `convex_decomposition`.** It covers `box`
  (static and dynamic) and `sdf` (normal, small-feature, and extreme aspect
  ratio). Both racks build, but both have unsourced footprints, so neither
  belongs in the reference scene yet. T3's collider checks are untested against
  a decomposed collider.
- **The settle test now passes, and the first diagnosis of why it failed was
  wrong.** It was reported as "the petri dish fails" — 8.43 mm of sink against
  CLAUDE.md's 2 mm gate, with the other three bodies passing. Measuring every
  catalog vessel on its own instead of just the ones in the scene showed one
  shared defect, not one bad object: sink correlated with contact pressure at
  **r = -0.76**, so the *least*-loaded vessel sank most, which is the opposite
  of what compliance does. All five beakers sank ~1.25 mm across a 13x mass
  range. The beaker was clearing the gate by 0.7 mm of luck while exhibiting the
  same failure as the dish.

  The cause was the contact time constant, and it is bounded by the timestep.
  Sweeping `ke` and `dt` separately found optima that were artifacts of the
  other's fixed value; moved together, `ke=160000, kd=800, dt=1/960` puts every
  vessel under 0.1 mm (`bench_5` worst 0.598 mm, `bench_arm` worst 0.306 mm).

  **An earlier version of this paragraph justified that as "glass on steel is
  far stiffer than the importer's 2500 N/m". That was wrong.** `ke` is not a
  stiffness on this solver: Newton passes both values to MuJoCo via
  `convert_solref`, giving `timeconst = 2/kd` and `dampratio = kd/(2√ke)`. So
  the shipped values mean a 2.5 ms time constant at exactly critical damping —
  and the importer default was *also* critical, so what changed was the time
  constant, 20 ms → 2.5 ms. Measured (`scripts/probe_solref.py`), sink scales
  as timeconst² (roughly 70× less for 8× shorter), but the tidy law
  `sink = g·tc²` does not hold: the prefactor varies ~7× between vessels with
  contact geometry. MuJoCo also needs `timeconst ≥ 2·dt`, which is what forces
  `dt` down to 1/960.

  `convex_decomposition` reaches the same green number by **sealing the dish** —
  a tube dropped in rests on the rim at 23 mm. The stiffness fix makes the
  cavity more correct instead: the same tube now rests ON the dish floor at
  7.89 mm, where before it passed through to 0.02 mm. The cavity probe is what
  distinguishes those two, and a green gate was not evidence without it.
- **The SDF collider currently changes nothing under `SolverMuJoCo`.**
  `NewtonSDFCollisionAPI` is authored, Newton detects it
  (`HasAPI(...)` is `True`) and reads the voxel size back, but simulated
  contact is identical with and without it. It is not a correctness problem —
  with `approximation="none"` the solver collides against the authored
  triangles and the cavity survives — but the catalog's `collider="sdf"` is
  presently decorative on this backend. Do not assume T3's collider-sanity
  check has teeth until this is understood.
- **The three vials are modelled as straight cylinders.** They need
  `necked_vessel`, whose builder exists and is tested, but converting them
  requires shoulder and neck dimensions nobody has sourced. The volume overshoot
  is the visible symptom; the real cost is that a gripper closes on a neck that
  is not in the mesh.
- **No catalog entry is sourced.** See above.

## What the real load established

`bench_5.usda` loads and simulates in Isaac Lab 3.0.0 / Newton 1.6.0rc1
(kit-less, `SolverMuJoCo`, CUDA). Getting there found four emitter bugs that
every structural test in this repo had passed over:

1. **`quatd` was written as a nested `(w, (x, y, z))` tuple.** USD text wants a
   flat 4-tuple. The file did not parse at all.
2. **Collision schemas were applied to the `Xform`, not the `Mesh`.**
   `UsdPhysicsCollisionAPI` applies to a `UsdGeomGprim`; on an `Xform` it is
   silently not a collider. Newton built shapes from the child meshes using its
   own defaults and ignored everything authored.
3. **`MaterialBindingAPI` was never applied**, so USD dropped
   `material:binding:physics`. Every object simulated at Newton's default
   friction of 1.0 instead of the catalog's values.
4. **`physics:approximation = "sdf"` is not a token.** Newton's map has no such
   key, so the lookup returned `None` and the vessels kept raw triangles with
   nothing reported. SDF is requested by applying `NewtonSDFCollisionAPI`
   alongside `approximation="none"`.

Each produced a stage that looked correct in every check written against a
reader in this repo. That is the argument for never letting the structural
tests stand alone.

**Hard rule 1 holds, and it was measured rather than assumed.** A 2 mL vial
dropped from 50 mm inside a 250 mL beaker settles at -0.3 mm — it falls through
the cavity to the bottom. Nothing seals the vessel. `examples/cavity_probe.json`
is that experiment, and it is worth re-running whenever the collider strategy
changes: it is the only check that distinguishes a real cavity from a
convincing one.

**SDF voxel sizing is derived, not defaulted.** `newton:sdfMaxResolution`
defaults to 64 over the longest axis, which for `test_tube_16x100` is a 1.56 mm
voxel against a 1.0 mm wall — the wall is thinner than one voxel, and an SDF
coarser than the wall rasterises the vessel solid. `usda.py` authors
`newton:sdfTargetVoxelSize` at wall/3 instead, so the setting is a property of
the object rather than of the default.

## Conventions

Metres, kilograms, seconds. Z-up, right-handed, `metersPerUnit = 1`.
Quaternions are `(w, x, y, z)`. Every object pose is in the robot base frame.
No centimetres or millimetres anywhere outside a docstring.

## Install

Core is pure Python and does not require Isaac Lab, Isaac Sim, or `pxr`. It is
importable and testable on any platform, which is deliberate — the validators
should run in CI without a GPU.

```sh
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest
```

### Validating emitted USD against real OpenUSD

`tests/test_usda.py` parses the emitted stage with a reader written in this
repo. That is a smoke test, not validation — a reader and an emitter written by
the same hand against the same wrong assumption agree with each other perfectly.
That is not hypothetical: the emitter wrote `quatd` as a nested
`(w, (x, y, z))` tuple, the repo's reader accepted it, 396 tests passed, and no
USD implementation would open the file.

So validate against actual OpenUSD, in a **separate venv**:

```sh
uv venv --python 3.12 .venv-usd
VIRTUAL_ENV=.venv-usd uv pip install usd-core -e ".[dev]"
.venv-usd/Scripts/python -m pytest        # Windows
.venv-usd/bin/python -m pytest            # Linux
```

Separate because Isaac Lab's environment uses `usd-exchange` as its pxr
provider, and its own dependency notes warn that two USD providers in one
environment overwrite each other's files — removing either then breaks `pxr`.
CI must run these; a skip here is the gap that let a non-parsing file through.

Do not install this into the Isaac Lab environment casually. That is a pinned
beta branch (Isaac Lab 3.0 Beta 2, `develop`) and adding a package can break
the install in a way that is slow to recover from. Purge stale
`mujoco` / `mujoco-warp` / `newton` wheels from the pip cache first if you do.

## Build order

`TASKS.md`, in order, each demoable on its own. Milestone one is T0–T2: a
hand-written `SceneSpec` JSON producing a `.usda` that loads in Isaac Lab with
five objects on a bench — **no video involved at all**.

Building the VLM stage first is the obvious temptation and it is the failure
mode: if identification is the first thing that exists, there is nothing to
check its output against.

---

¹ Darvish et al., *MATTERIX: toward a digital twin for robotics-assisted
chemistry laboratory automation*, [arXiv:2601.13232](https://arxiv.org/abs/2601.13232).
Collision-proxy framing follows Moonlake's
[evaluating 3D agents](https://moonlakeai.com/blog/evaluating-3d-agent) —
the proxy must be tight enough that a bad policy fails for the right reason.
