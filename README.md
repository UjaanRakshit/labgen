# labgen

Turns video of a real chemistry bench into a scene that loads in Isaac Lab with
the right objects, at the right size, in the right places, **in the robot's
coordinate frame**.

This is phase 1 of a larger project. Phase 1 ends at "the scene is correct".
It contains no policy loading, no rollouts, no success predicates and no
scoring, and it should not grow any. See `CLAUDE.md`.

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
| `validate.py`     | the gates, including the settle test              | T3 |
| `isaaclab_cfg.py` | `SceneSpec` → `InteractiveSceneCfg`               | T4 |
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

## Known gaps

Things that are **not** covered, recorded so a later stage does not assume they
are:

- **`bench_5` does not exercise `convex_decomposition`.** It covers `box`
  (static and dynamic) and `sdf` (normal, small-feature, and extreme aspect
  ratio). Both racks build, but both have unsourced footprints, so neither
  belongs in the reference scene yet. T3's collider checks are untested against
  a decomposed collider.
- **"Opens in Isaac Lab without error" is unverified.** Isaac Lab is Linux-only
  and this suite runs without a GPU or `pxr`. `tests/test_usda.py` checks the
  emitted stage structurally — balanced scopes, stage metadata, and every
  physics invariant — but a green run is not a load test.
- **The three vials are modelled as straight cylinders.** They need
  `necked_vessel`, whose builder exists and is tested, but converting them
  requires shoulder and neck dimensions nobody has sourced. The volume overshoot
  is the visible symptom; the real cost is that a gripper closes on a neck that
  is not in the mesh.
- **No catalog entry is sourced.** See above.

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
