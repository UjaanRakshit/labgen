"""labgen -- video of a real chemistry bench to a loadable Isaac Lab scene.

Phase 1 scope (CLAUDE.md): point the tool at a video, get a `.usda` scene and
an Isaac Lab scene config, load it, and have it look like the real bench.
There is no policy evaluation in this package and there should not be.

Stage order, each gated by a validator that runs without an LLM in the loop:

    identify -> fit -> register -> usda / isaaclab_cfg

Importing `labgen` must not require Isaac Lab, Isaac Sim, or `pxr`.
"""

from __future__ import annotations

from .catalog import CATALOG, CatalogItem, PhysicsMaterial, resolve, search
from .types import (
    SCHEMA_VERSION,
    GeneratedAsset,
    ObjectObservation,
    SceneObject,
    SceneSpec,
    SpecError,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CATALOG",
    "CatalogItem",
    "PhysicsMaterial",
    "resolve",
    "search",
    "SCHEMA_VERSION",
    "GeneratedAsset",
    "ObjectObservation",
    "SceneObject",
    "SceneSpec",
    "SpecError",
]
