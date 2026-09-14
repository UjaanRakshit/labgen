"""Core artifacts passed between pipeline stages.

Every stage in `labgen` takes a typed artifact and emits a typed artifact
(CLAUDE.md, "Module layout and contracts"). These are those artifacts.

Conventions, enforced here rather than assumed:

* **Metres, kilograms, seconds.** No centimetres, no millimetres, no grams.
* **Z-up, right-handed.** `metersPerUnit = 1` downstream.
* **Every pose is in the robot base frame.** Not the reconstruction frame, not
  a world origin. `register.py` is what gets you here; nothing downstream of it
  should ever see a reconstruction-frame pose.
* **Quaternions are (w, x, y, z)**, unit norm.

`SceneSpec` is JSON-serialisable and is the thing a human edits by hand when
the pipeline gets something wrong. That round-trip is a first-class feature, so
loading is *strict*: an unknown key is an error, not a shrug. The failure this
prevents is real and quiet -- someone hand-edits `positon`, the loader ignores
it, keeps the old pose, and the scene is confidently wrong in a way that looks
like a policy failure two stages later.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

from .catalog import CATALOG, ColliderKind

__all__ = [
    "SCHEMA_VERSION",
    "QUAT_NORM_TOL",
    "SpecError",
    "Vec3",
    "QuatWXYZ",
    "Provenance",
    "GeneratedAsset",
    "ObjectObservation",
    "SceneObject",
    "SceneSpec",
]

# Bumped when the on-disk shape of SceneSpec changes incompatibly. A
# hand-edited scene from an older version should fail loudly on load rather
# than be silently reinterpreted.
SCHEMA_VERSION = 1

# A quaternion this far off unit norm is a typo or a bug, not float drift.
# We do not renormalise silently -- see module docstring.
QUAT_NORM_TOL = 1e-6

Vec3 = tuple[float, float, float]
QuatWXYZ = tuple[float, float, float, float]

Provenance = Literal["catalog", "generated", "manual"]


class SpecError(ValueError):
    """A scene artifact is malformed or internally inconsistent.

    Raised eagerly. CLAUDE.md: prefer failing loudly over degrading gracefully;
    this pipeline feeds a system that produces trust numbers.
    """


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _as_vec3(value: Any, what: str) -> Vec3:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            pass
    raise SpecError(f"{what}: expected 3 numbers in metres, got {value!r}")


def _as_quat(value: Any, what: str) -> QuatWXYZ:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            q = (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        except (TypeError, ValueError):
            q = None
        if q is not None:
            norm = math.sqrt(sum(c * c for c in q))
            if abs(norm - 1.0) > QUAT_NORM_TOL:
                raise SpecError(
                    f"{what}: quaternion (w,x,y,z)={q} has norm {norm:.9f}, "
                    f"expected 1.0 +/- {QUAT_NORM_TOL:g}. Not renormalising: a "
                    f"non-unit quaternion here is a typo or an upstream bug, and "
                    f"silently fixing it hides a rotation error that makes every "
                    f"downstream number wrong."
                )
            return q
    raise SpecError(f"{what}: expected 4 numbers (w,x,y,z), got {value!r}")


def _reject_unknown(data: dict, cls: type, what: str) -> None:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise SpecError(
            f"{what}: unknown field(s) {sorted(unknown)}. Valid fields are "
            f"{sorted(known)}. (Typo in a hand-edited scene? This is rejected "
            f"rather than ignored on purpose.)"
        )


def _require(data: dict, key: str, what: str) -> Any:
    if key not in data:
        raise SpecError(f"{what}: missing required field {key!r}")
    return data[key]


def _literal_values(tp) -> tuple:
    """Pull the allowed strings out of a Literal alias, incl. `X | None`."""
    if get_origin(tp) is Literal:
        return get_args(tp)
    out: list = []
    for arg in get_args(tp):
        out.extend(_literal_values(arg))
    return tuple(out)


_COLLIDER_KINDS = _literal_values(ColliderKind)
_PROVENANCES = _literal_values(Provenance)


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------

@dataclass
class ObjectObservation:
    """What the VLM saw. Emitted by `identify.py`, consumed by `fit.py`.

    Deliberately dumb: free text and pixels. No catalog resolution happens
    here, so that a bad identification is visible as a bad *label* rather than
    as a confident wrong catalog key.
    """

    label: str                 # free text, e.g. "250ml beaker"
    confidence: float          # 0..1, the model's own
    bbox_2d: tuple[float, float, float, float]   # (x0, y0, x1, y1) px, source frame
    frames: list[int] = field(default_factory=list)   # frame indices it appeared in

    # Text read off the object, when present. CLAUDE.md/T6: "250" printed on
    # the side is better evidence than shape, so it is carried separately
    # rather than being folded into `label`.
    read_text: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.label or not self.label.strip():
            raise SpecError("ObjectObservation.label must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise SpecError(
                f"ObjectObservation({self.label!r}).confidence must be in [0,1], "
                f"got {self.confidence}"
            )
        if len(self.bbox_2d) != 4:
            raise SpecError(
                f"ObjectObservation({self.label!r}).bbox_2d must be "
                f"(x0,y0,x1,y1), got {self.bbox_2d!r}"
            )
        x0, y0, x1, y1 = self.bbox_2d
        if x1 <= x0 or y1 <= y0:
            raise SpecError(
                f"ObjectObservation({self.label!r}).bbox_2d is empty or inverted: "
                f"{self.bbox_2d!r}"
            )

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox_2d": list(self.bbox_2d),
            "frames": list(self.frames),
            "read_text": list(self.read_text),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ObjectObservation:
        _reject_unknown(data, cls, "ObjectObservation")
        bbox = _require(data, "bbox_2d", "ObjectObservation")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise SpecError(f"ObjectObservation.bbox_2d: expected 4 numbers, got {bbox!r}")
        return cls(
            label=_require(data, "label", "ObjectObservation"),
            confidence=float(_require(data, "confidence", "ObjectObservation")),
            bbox_2d=tuple(float(v) for v in bbox),  # type: ignore[arg-type]
            frames=list(data.get("frames", [])),
            read_text=list(data.get("read_text", [])),
        )


@dataclass
class GeneratedAsset:
    """Physics facts for an object with no catalog entry.

    Only present when `SceneObject.catalog_key is None`, i.e. mesh generation
    was used as a fallback. A catalog object gets all of this from
    `catalog.py`; a generated one has nowhere else to get it from, and
    CLAUDE.md rule 4 forbids a silent default. So these are required, and
    `verified` is forced False -- a generated asset is low-confidence by
    construction and the validator says so in the report.
    """

    mesh_path: str                  # .obj, relative to the scene file
    mass_kg: float                  # explicit. no density inference.
    collider: ColliderKind          # explicit approximation
    material: str                   # key into catalog's PhysicsMaterial set
    collision_mesh_path: str | None = None   # optional lower-res proxy
    verified: bool = False          # always False; kept for report symmetry
    note: str = ""                  # what this is, and what to check

    def __post_init__(self) -> None:
        if self.mass_kg <= 0.0:
            raise SpecError(
                f"GeneratedAsset({self.mesh_path!r}).mass_kg must be > 0, got "
                f"{self.mass_kg}. A generated asset has no catalog mass to fall "
                f"back to and a placeholder density is not allowed."
            )
        if self.collider not in _COLLIDER_KINDS:
            raise SpecError(
                f"GeneratedAsset({self.mesh_path!r}).collider={self.collider!r} "
                f"is not one of {list(_COLLIDER_KINDS)}"
            )
        if self.verified:
            raise SpecError(
                "GeneratedAsset.verified cannot be True: a mesh-generated asset "
                "is low-confidence by construction (CLAUDE.md, 'The one design "
                "decision that matters'). Add a catalog entry instead."
            )

    def to_dict(self) -> dict:
        return {
            "mesh_path": self.mesh_path,
            "mass_kg": self.mass_kg,
            "collider": self.collider,
            "material": self.material,
            "collision_mesh_path": self.collision_mesh_path,
            "verified": self.verified,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GeneratedAsset:
        _reject_unknown(data, cls, "GeneratedAsset")
        return cls(
            mesh_path=_require(data, "mesh_path", "GeneratedAsset"),
            mass_kg=float(_require(data, "mass_kg", "GeneratedAsset")),
            collider=_require(data, "collider", "GeneratedAsset"),
            material=_require(data, "material", "GeneratedAsset"),
            collision_mesh_path=data.get("collision_mesh_path"),
            verified=bool(data.get("verified", False)),
            note=data.get("note", ""),
        )


@dataclass
class SceneObject:
    """What we decided an object is, and where it is.

    `position` / `orientation_wxyz` are **in the robot base frame**, metres.
    """

    instance_id: str
    catalog_key: str | None        # None => fell back to mesh generation
    position: Vec3                 # robot base frame, metres
    orientation_wxyz: QuatWXYZ
    fixed: bool                    # static fixture vs free rigid body
    confidence: float
    provenance: Provenance         # "catalog" | "generated" | "manual"

    # Present iff catalog_key is None. See GeneratedAsset.
    generated: GeneratedAsset | None = None

    # Free-text breadcrumb for hand edits: why a human moved this, what spec
    # sheet a dimension came from. Survives the round-trip.
    note: str = ""

    def __post_init__(self) -> None:
        if not self.instance_id or not self.instance_id.strip():
            raise SpecError("SceneObject.instance_id must be non-empty")
        self.position = _as_vec3(self.position, f"SceneObject({self.instance_id}).position")
        self.orientation_wxyz = _as_quat(
            self.orientation_wxyz, f"SceneObject({self.instance_id}).orientation_wxyz"
        )
        if not 0.0 <= self.confidence <= 1.0:
            raise SpecError(
                f"SceneObject({self.instance_id}).confidence must be in [0,1], "
                f"got {self.confidence}"
            )
        if self.provenance not in _PROVENANCES:
            raise SpecError(
                f"SceneObject({self.instance_id}).provenance={self.provenance!r} "
                f"is not one of {list(_PROVENANCES)}"
            )

        if self.catalog_key is None:
            if self.generated is None:
                raise SpecError(
                    f"SceneObject({self.instance_id}): catalog_key is None so this "
                    f"is a mesh-generation fallback, but `generated` is missing. "
                    f"It carries the mass, collider and material that would "
                    f"otherwise come from the catalog, and there is no legal "
                    f"default for them."
                )
            if self.provenance == "catalog":
                raise SpecError(
                    f"SceneObject({self.instance_id}): provenance='catalog' but "
                    f"catalog_key is None."
                )
        else:
            if self.catalog_key not in CATALOG:
                from .catalog import search

                near = search(self.catalog_key)
                hint = f" Did you mean one of {near}?" if near else ""
                raise SpecError(
                    f"SceneObject({self.instance_id}): catalog_key "
                    f"{self.catalog_key!r} is not in the catalog.{hint}"
                )
            if self.generated is not None:
                raise SpecError(
                    f"SceneObject({self.instance_id}): has both a catalog_key and "
                    f"a `generated` block. Exactly one source of physics facts."
                )

    # -- convenience ------------------------------------------------------

    @property
    def is_catalog(self) -> bool:
        return self.catalog_key is not None

    def catalog_item(self):
        """The CatalogItem backing this object, or None if generated."""
        return CATALOG[self.catalog_key] if self.catalog_key else None

    def mass_kg(self) -> float:
        """Explicit mass in kg. Never inferred from a density."""
        item = self.catalog_item()
        if item is not None:
            return item.mass_kg
        assert self.generated is not None  # guaranteed by __post_init__
        return self.generated.mass_kg

    def collider(self) -> ColliderKind:
        item = self.catalog_item()
        if item is not None:
            return item.collider
        assert self.generated is not None
        return self.generated.collider

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "catalog_key": self.catalog_key,
            "position": list(self.position),
            "orientation_wxyz": list(self.orientation_wxyz),
            "fixed": self.fixed,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "generated": self.generated.to_dict() if self.generated else None,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SceneObject:
        _reject_unknown(data, cls, "SceneObject")
        gen = data.get("generated")
        return cls(
            instance_id=_require(data, "instance_id", "SceneObject"),
            catalog_key=_require(data, "catalog_key", "SceneObject"),
            position=_require(data, "position", "SceneObject"),
            orientation_wxyz=_require(data, "orientation_wxyz", "SceneObject"),
            fixed=bool(_require(data, "fixed", "SceneObject")),
            confidence=float(_require(data, "confidence", "SceneObject")),
            provenance=_require(data, "provenance", "SceneObject"),
            generated=GeneratedAsset.from_dict(gen) if gen else None,
            note=data.get("note", ""),
        )


@dataclass
class SceneSpec:
    """A complete bench, in the robot base frame. The hand-editable artifact."""

    name: str
    robot_base_frame: str
    objects: list[SceneObject] = field(default_factory=list)

    # Provenance of the whole scene: video paths, marker config, git sha.
    # `validate.py` rejects a scene whose source carries no marker config,
    # because monocular video has no metric scale and a scene built without a
    # scale reference is wrong in a way nothing downstream can detect.
    source: dict[str, Any] = field(default_factory=dict)

    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise SpecError("SceneSpec.name must be non-empty")
        if not self.robot_base_frame or not self.robot_base_frame.strip():
            raise SpecError(
                "SceneSpec.robot_base_frame must be non-empty: every object pose "
                "is expressed in it, so an unnamed frame means the poses have no "
                "stated meaning."
            )
        if self.schema_version != SCHEMA_VERSION:
            raise SpecError(
                f"SceneSpec.schema_version={self.schema_version} but this build "
                f"of labgen writes {SCHEMA_VERSION}. Migrate the file rather than "
                f"letting it be reinterpreted."
            )
        seen: set[str] = set()
        for obj in self.objects:
            if obj.instance_id in seen:
                raise SpecError(
                    f"SceneSpec({self.name}): duplicate instance_id "
                    f"{obj.instance_id!r}. Prim paths collide in USD and one "
                    f"object silently replaces the other."
                )
            seen.add(obj.instance_id)

    # -- lookup -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.objects)

    def by_id(self, instance_id: str) -> SceneObject:
        for obj in self.objects:
            if obj.instance_id == instance_id:
                return obj
        raise KeyError(
            f"no object {instance_id!r} in scene {self.name!r}; have "
            f"{[o.instance_id for o in self.objects]}"
        )

    @property
    def free_bodies(self) -> list[SceneObject]:
        """Dynamic rigid bodies -- the things a settle test can move."""
        return [o for o in self.objects if not o.fixed]

    @property
    def fixtures(self) -> list[SceneObject]:
        return [o for o in self.objects if o.fixed]

    # -- round trip -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "robot_base_frame": self.robot_base_frame,
            "source": self.source,
            "objects": [o.to_dict() for o in self.objects],
        }

    @classmethod
    def from_dict(cls, data: dict) -> SceneSpec:
        _reject_unknown(data, cls, "SceneSpec")
        objects = _require(data, "objects", "SceneSpec")
        if not isinstance(objects, list):
            raise SpecError(f"SceneSpec.objects must be a list, got {type(objects).__name__}")
        return cls(
            name=_require(data, "name", "SceneSpec"),
            robot_base_frame=_require(data, "robot_base_frame", "SceneSpec"),
            objects=[SceneObject.from_dict(o) for o in objects],
            source=dict(data.get("source", {})),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent) + "\n"

    @classmethod
    def from_json(cls, text: str) -> SceneSpec:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SpecError(f"scene is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise SpecError(f"scene must be a JSON object, got {type(data).__name__}")
        return cls.from_dict(data)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: str | Path) -> SceneSpec:
        path = Path(path)
        try:
            return cls.from_json(path.read_text(encoding="utf-8"))
        except SpecError as exc:
            raise SpecError(f"{path}: {exc}") from exc


def _assert_dataclasses() -> None:
    for cls in (ObjectObservation, GeneratedAsset, SceneObject, SceneSpec):
        assert is_dataclass(cls), cls


_assert_dataclasses()
