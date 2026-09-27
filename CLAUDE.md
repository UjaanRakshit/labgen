# CLAUDE.md

Read this fully before writing code. It contains decisions that are already
made and constraints that are not obvious from the codebase.

## If you are a session on a lab machine

The owner has pulled this repo onto a lab machine and expects you to act
without being briefed. Do this:

1. **Work out which machine you are on.** A machine with the YAM arms' CAN
   adapters and the lab teleop checkout (`yam_vr_teleop`) is the *arm laptop*.
   A machine with Isaac Lab and an NVIDIA GPU is the *sim machine*. It can be
   both.
2. **Follow `README.md` → "Lab runbook"** for that machine, and its
   **"Lab session checklist"**. It has the exact commands, the expected output
   of each, and how to hand results back.
3. **Real arms: never move one without the owner confirming in this session**
   that someone is at the e-stop and the workspace is clear. The recorder's
   `--i-have-cleared-the-workspace` flag is that confirmation; do not pass it on
   your own. Stop the lab teleop first. Never edit the teleop's configs to make
   a check pass.
4. **Hand results back as files in the repo** (`data/lab/<date>/`), as the
   checklist says, and report what you measured, not what you intended.

Scope has grown beyond the phase-1 text below, at the owner's direction: the
bimanual YAM sim, phone teleop, demo recording and MimicGen data generation
(`labgen_tasks/`, `scripts/teleop_task.py`, `scripts/mimic.py`) are in scope.
The rules below still hold everywhere, especially: no invented dimensions,
fail loudly, and anything unverified is reported as unverified.

## What this is

`labgen` turns video of a real chemistry lab bench into a scene that loads in
Isaac Lab with the right objects, at the right size, in the right places, in
the robot's coordinate frame.

This is **phase 1 of a larger project**. The larger project evaluates whether a
robot policy can complete a task. Do not build any of that. No policy loading,
no rollouts, no success predicates, no scoring. If you find yourself writing
something that mentions a policy, you have gone out of scope.

The deliverable for phase 1: point the tool at a video, get a `.usda` scene and
an Isaac Lab scene config, load it, and have it look like the real bench.

## The one design decision that matters

**Do not reconstruct objects from video.** Identify them and fit a parametric
template.

A chemistry lab is almost entirely catalog parts with published dimensions. A
250 mL Griffin beaker is 70 mm across and 95 mm tall whether or not your mesh
generator agrees. Photogrammetry and mesh-generation models both fail badly on
transparent and thin-walled objects, which is most of a chem lab, and they give
you no metric scale and no collision geometry.

So the pipeline is: recognise the object, look it up, instantiate the template,
estimate its pose. Mesh generation is a **fallback** for genuinely non-catalog
objects, and it should be clearly marked as low-confidence in the output.

`seed/catalog.py` is provided. It has real dimensions, masses, materials and
collision strategies for the common items. **Do not invent dimensions for new
catalog entries.** If you need an item that is not there, add it with
`verified=False` and a `TODO` naming what spec sheet needs checking. A
confidently wrong dimension is worse than a missing entry, because it produces a
scene that looks correct and grasps that silently miss.

## Hard rules

1. **Never wrap an open vessel in a convex hull.** A convex hull over a beaker
   seals the opening. Nothing can be placed inside it, and a policy will appear
   to succeed by resting an object on a lid that does not exist. Open vessels
   use SDF colliders. This is in the catalog already; do not "optimise" it away.
2. **Metres, kilograms, Z-up, right-handed.** `metersPerUnit = 1`. Every
   dimension in the codebase is SI. No centimetres, no millimetres outside of
   docstrings.
3. **Every object pose is expressed in the robot base frame**, not the
   reconstruction frame and not some arbitrary world origin. The registration
   step that gets you there is the highest-risk part of this project. See below.
4. **No silent defaults for physics properties.** If mass or friction is not
   known, the asset is marked unverified and the validator flags it. Do not let
   an object through with a placeholder density.
5. **Do not `pip install` into the Isaac Lab environment casually.** It is a
   beta branch with pinned versions. Adding a package can break the whole
   install and it is a slow recovery. Anything new goes in a separate venv or
   gets justified in the PR.

## Environment constraints

These are verified as of the time of writing and will drift:

- Newton lives on the **`develop` branch** of Isaac Lab (Isaac Lab 3.0 Beta 2),
  not `main`. `main` is frozen.
- Ubuntu 22.04 or 24.04. Python 3.12. Torch is pinned by the install script.
- Isaac Sim 6.0 is optional if you only use the kit-less Newton path.
- `uv` is the recommended package manager, and there are known dependency
  conflicts with stale `mujoco` / `mujoco-warp` / `newton` wheels in the pip
  cache. Purge before installing.
- NVIDIA states explicitly that the branch has breaking changes and that they
  will not provide debugging support until 3.0 ships. Pin a commit. Do not
  track the branch tip.
- Renderer: `NewtonWarpRendererCfg` is kit-less and supports **rgb and depth
  only**. Segmentation, normals and motion vectors require
  `IsaacRtxRendererCfg`, which needs Isaac Sim. Phase 1 does not need
  segmentation, but note it before designing around annotators.

## Physics backend

Write against Isaac Lab's backend-agnostic interfaces. Isaac Lab 3.0 refactored
`Articulation`, `ContactSensor` etc. into abstract interfaces so the same scene
runs on PhysX or Newton. Use that. Do not import PhysX-specific or
Newton-specific APIs into `labgen` core.

Practical reason: Newton's supported robot and feature list is still narrow, and
we may need to run on PhysX for a while. The scene generator should not care.

## Module layout and contracts

Each stage takes a typed artifact and emits a typed artifact. Each stage has a
validator that runs without an LLM in the loop. A stage does not advance until
its validator passes.

```
labgen/
  catalog.py        # provided in seed/ — parametric lab object templates
  identify.py       # video -> [ObjectObservation]   (VLM call, stub the model)
  fit.py            # ObjectObservation -> SceneObject (catalog resolve + pose)
  register.py       # reconstruction frame -> robot base frame
  meshes.py         # CatalogItem -> triangle mesh (OBJ), for templates
  usda.py           # SceneSpec -> .usda with UsdPhysics schemas
  isaaclab_cfg.py   # SceneSpec -> Isaac Lab InteractiveSceneCfg
  validate.py       # the gates
  cli.py
```

Core types, roughly:

```python
@dataclass
class ObjectObservation:      # what the VLM saw
    label: str                # free text, e.g. "250ml beaker"
    confidence: float
    bbox_2d: tuple             # in source frame
    frames: list[int]          # which video frames it appeared in

@dataclass
class SceneObject:            # what we decided it is
    instance_id: str
    catalog_key: str | None   # None => fell back to mesh generation
    position: tuple[float, float, float]   # robot base frame, metres
    orientation_wxyz: tuple[float, float, float, float]
    fixed: bool               # static fixture vs free rigid body
    confidence: float
    provenance: str           # "catalog" | "generated" | "manual"

@dataclass
class SceneSpec:
    name: str
    robot_base_frame: str
    objects: list[SceneObject]
    source: dict              # video paths, marker config, git sha
```

`SceneSpec` is JSON-serialisable and is the thing a human edits by hand when the
pipeline gets something wrong. Treat that round-trip as a first-class feature,
not a debugging escape hatch.

## Scale and registration

This is where the project actually fails if it fails.

**Scale:** monocular video has no metric scale. Do not try to infer it. Require
one of: a printed ArUco/AprilTag target of known size in frame, a stereo camera,
or a depth sensor. The marker config is part of `SceneSpec.source` and the
validator rejects a scene built without one.

**Registration:** every object pose must land in the robot base frame. Plan on
an ArUco target at a known offset from the robot base, or a hand-eye calibration
if a wrist camera is used. Write this as its own module with its own tests,
because a constant rotation error here makes every downstream number wrong in a
way that looks like a policy failure.

Note from the MATTERIX paper's real-world results, worth taking seriously: one
of their sim failures was caused purely by the simulated table being a different
size from the real one. Their reported failure causes were pose estimation
error, camera calibration, object frame definition, and sim/real asset size
mismatch. All four are registration and scale problems. None are modelling
problems.

## Validators (the gates)

`validate.py` must implement at minimum:

- **scale sanity** — every object's bounding box is within tolerance of its
  catalog dimensions; reject anything more than 5% off
- **support** — no object is floating; every free body rests on a surface within
  1 mm, or is flagged
- **interpenetration** — no two collision meshes overlap at rest
- **reachability** — every manipulable object is inside the robot's workspace
  envelope; warn if not
- **physics completeness** — every dynamic body has explicit mass and a bound
  physics material; every collider has an explicit approximation set
- **collider sanity** — no open vessel has a convex-hull collider (hard fail)
- **settle test** — step the scene 2 seconds with no actuation; nothing moves
  more than 2 mm and nothing explodes

The settle test is the cheapest high-value check in the whole project. Run it in
CI.

## Build order

Do these in order. Each one should be demoable on its own.

1. `catalog.py` + `meshes.py` + `usda.py` — hand-write a `SceneSpec` JSON by
   hand, produce a `.usda`, load it in Isaac Lab, see five objects on a bench.
   No video involved at all. **This is milestone one and it should land first.**
2. `validate.py` including the settle test.
3. `isaaclab_cfg.py` — emit a working `InteractiveSceneCfg`, arm present, arm
   does not clip the bench.
4. `register.py` — marker-based, with tests against synthetic poses.
5. `identify.py` + `fit.py` — the VLM stage, last, on top of a pipeline that
   already works end to end with hand-written input.

Building stage 5 first is the obvious temptation and it is the failure mode. If
the VLM stage is the first thing that exists, there is nothing to check its
output against.

## Definition of done for phase 1

One real bench. Five objects. Loads in Isaac Lab. A render from a known camera
pose put side by side with a real photo from the same pose looks like the same
room. Settle test passes. Validator report is clean or its warnings are
explained.
