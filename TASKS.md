# TASKS.md

Ordered. Do not start a task until the one above it passes its acceptance
check. Each task is sized to be a single session.

---

## T0 — Repo skeleton

Create the package layout from `CLAUDE.md`, a `pyproject.toml`, and a test
harness. Copy `seed/catalog.py` into `labgen/catalog.py` unchanged.

**Accept:** `pytest` runs and passes on an empty suite. `python -c "from labgen
import catalog; print(len(catalog.CATALOG))"` prints a non-zero count.

---

## T1 — Mesh generation for catalog templates

`meshes.py` generates watertight triangle meshes for each `ShapeKind`:
`open_vessel`, `conical_vessel`, `solid_cylinder`, `box`, `tube_rack`.

Open vessels are hollow: outer wall, inner wall, connected rim, closed base. The
interior cavity must be real, not implied. A beaker whose mesh is a solid
cylinder will pass every visual check and fail every pour task.

`tube_rack` needs actual holes, subtracted at the positions implied by
`rows`/`cols`/`hole_d`/`hole_depth`.

Write OBJ. Keep a separate lower-resolution collision mesh where it helps.

**Accept:**
- every catalog item generates a mesh with no degenerate or zero-area triangles
- every mesh is manifold (or the exceptions are enumerated and justified)
- computed volume of an open vessel's cavity is within 10% of its
  `capacity_ml`
- unit test asserts a beaker's cavity is non-empty by ray-casting down the axis
  and finding two surface crossings, not one

---

## T2 — USD emission with physics

`usda.py` takes a `SceneSpec` and writes a `.usda`. Text USD is fine, do not
require `pxr` unless it is already available.

Each object gets:
- `UsdGeomXform` at its pose, `UsdGeomMesh` child
- `PhysicsRigidBodyAPI` (dynamic) or nothing (static fixture)
- `PhysicsMassAPI` with explicit `mass`, never density-inferred
- `PhysicsCollisionAPI` + `PhysicsMeshCollisionAPI` with `approximation` set
  explicitly from `CatalogItem.collider`
- a bound `PhysicsMaterialAPI` from the catalog material
- keypoints written as custom attributes so task specs can reference them by
  name

Stage metadata: `metersPerUnit = 1`, `upAxis = "Z"`, `defaultPrim = "World"`.

**Accept:** hand-write `examples/bench_5.json` with a bench and five objects.
Generate the `.usda`. It opens in Isaac Lab without error and every object is
visible at plausible size.

---

## T3 — Validators

Implement every check listed in `CLAUDE.md`, including the settle test. Output a
readable report: per-object pass/warn/fail with the specific measured value that
triggered it.

The convex-hull-on-open-vessel check is a hard fail, not a warning.

**Accept:** deliberately corrupt `bench_5.json` five ways (float an object,
overlap two, strip a mass, hull a beaker, scale a beaker 20% up) and confirm
each produces the right specific failure and no false positives on the clean
scene.

---

## T4 — Isaac Lab scene config

`isaaclab_cfg.py` emits a working `InteractiveSceneCfg` from a `SceneSpec`,
using backend-agnostic interfaces only.

Include the arm. Getting the arm in is a sub-task with real risk: YAM ships
`yam.urdf`, `yam.xml` (MJCF) and STL meshes in the `i2rt` repo under
`i2rt/robot_models/arm/yam/`, plus per-revision docs with masses, COMs,
inertias, joint frames and DH parameters. Check whether the Isaac Lab path here
wants USD or can take MJCF directly before writing a converter.

Do not tune actuator gains in this task. Getting the arm to exist and not fall
over is enough.

**Accept:** scene loads, arm spawns at the origin of the robot base frame, arm
at its home configuration does not intersect the bench or any object. Settle
test passes with the arm present.

---

## T5 — Frame registration

`register.py` maps reconstruction-frame poses into the robot base frame via a
marker of known size at a known offset.

Test against synthetic data first: generate known poses, apply a known
transform, recover it, assert the residual is small. Only then touch real data.

**Accept:** synthetic recovery within 1 mm and 0.5 degrees. On real data,
reprojection error of marker corners under 2 px, and a physical spot-check where
a commanded arm pose reaches a measured real-world point within 5 mm.

---

## T6 — Identification and fitting

`identify.py` runs a VLM over video frames and emits `ObjectObservation`s.
`fit.py` resolves those onto catalog entries and estimates poses.

Stub the VLM behind an interface with a recorded-response fixture so the rest of
the pipeline is testable offline and in CI.

Two things it must do that are easy to skip:
- when the label does not resolve, **say so**. Emit the observation with
  `catalog_key=None` and low confidence. Do not silently pick the nearest match.
- read labels and graduations off glassware where visible. "250" printed on the
  side is better evidence than shape.

**Accept:** on a held-out video of a bench with known contents, correct catalog
key for at least 80% of objects, and zero cases where a wrong key is returned
with high confidence. A confident wrong answer counts as worse than an
abstention in the scoring.

---

## T7 — End-to-end CLI and the side-by-side

`labgen build <video> --markers <cfg> -o out/` runs the whole thing and writes
the scene, the config, and the validation report.

Then the actual deliverable: render from a camera pose matching a real photo,
put them side by side.

**Accept:** phase 1 definition of done in `CLAUDE.md`.

---

## Notes on working style

- After T2, every subsequent task should leave the example scene still loading.
  If a change breaks `bench_5.json`, that is a regression, fix it before moving
  on.
- Prefer failing loudly over degrading gracefully. This pipeline feeds a system
  that produces trust numbers. A scene that is quietly 8% wrong is worse than a
  scene that refuses to build.
- When you are unsure of a physical dimension, stop and ask rather than
  estimating. Estimated dimensions are the single most likely source of a
  plausible-looking wrong result in this codebase.
