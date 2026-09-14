"""The SceneSpec round-trip, which CLAUDE.md calls a first-class feature.

A human edits this file by hand when the pipeline gets something wrong. So the
tests here care less about the happy path than about what happens when the hand
edit is subtly wrong.
"""

from __future__ import annotations

import json
import math

import pytest

from labgen.types import (
    SCHEMA_VERSION,
    GeneratedAsset,
    ObjectObservation,
    SceneObject,
    SceneSpec,
    SpecError,
)

IDENT = (1.0, 0.0, 0.0, 0.0)


def _obj(instance_id="beaker_a", **kw) -> SceneObject:
    kw.setdefault("catalog_key", "beaker_250")
    kw.setdefault("position", (0.30, 0.10, 0.90))
    kw.setdefault("orientation_wxyz", IDENT)
    kw.setdefault("fixed", False)
    kw.setdefault("confidence", 0.92)
    kw.setdefault("provenance", "catalog")
    return SceneObject(instance_id=instance_id, **kw)


def _scene(*objects) -> SceneSpec:
    return SceneSpec(
        name="bench_test",
        robot_base_frame="yam_base_link",
        objects=list(objects) or [_obj()],
        source={"marker": {"kind": "aruco", "dict": "DICT_5X5_100", "size_m": 0.100}},
    )


# --- round trip ----------------------------------------------------------

def test_round_trip_is_lossless():
    before = _scene(
        _obj("beaker_a"),
        _obj("plate", catalog_key="hotplate_stirrer", fixed=True,
             position=(0.0, 0.25, 0.86), provenance="manual", note="measured with tape"),
    )
    after = SceneSpec.from_json(before.to_json())
    assert after.to_dict() == before.to_dict()


def test_round_trip_through_a_file(tmp_path):
    path = _scene().write(tmp_path / "nested" / "scene.json")
    assert path.exists()
    assert SceneSpec.read(path).to_dict() == _scene().to_dict()


def test_written_json_is_human_readable():
    text = _scene().to_json()
    assert text.endswith("\n")
    assert "\n  " in text, "must be indented; a human edits this by hand"
    json.loads(text)


def test_schema_version_is_recorded():
    assert _scene().to_dict()["schema_version"] == SCHEMA_VERSION


def test_future_schema_version_is_rejected():
    data = _scene().to_dict()
    data["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(SpecError, match="schema_version"):
        SceneSpec.from_dict(data)


# --- the hand-edit failure modes ----------------------------------------

def test_misspelled_field_is_rejected_not_ignored():
    """The quiet one. `positon` must not be silently dropped."""
    data = _scene().to_dict()
    obj = data["objects"][0]
    obj["positon"] = obj.pop("position")
    with pytest.raises(SpecError, match="positon"):
        SceneSpec.from_dict(data)


def test_missing_required_field_is_rejected():
    data = _scene().to_dict()
    del data["objects"][0]["fixed"]
    with pytest.raises(SpecError, match="fixed"):
        SceneSpec.from_dict(data)


def test_unknown_catalog_key_is_rejected_with_a_suggestion():
    with pytest.raises(SpecError, match="beaker_250"):
        _obj(catalog_key="beaker_205")


def test_duplicate_instance_ids_are_rejected():
    with pytest.raises(SpecError, match="duplicate"):
        _scene(_obj("beaker_a"), _obj("beaker_a", position=(0.1, 0.1, 0.9)))


def test_non_unit_quaternion_is_rejected_not_renormalised():
    """A silent renormalise hides the rotation error that CLAUDE.md calls the
    highest-risk failure in the project."""
    with pytest.raises(SpecError, match="norm"):
        _obj(orientation_wxyz=(0.9, 0.0, 0.0, 0.0))


def test_a_genuinely_unit_quaternion_passes():
    h = math.sqrt(0.5)
    o = _obj(orientation_wxyz=(h, 0.0, 0.0, h))  # 90 deg about Z
    assert o.orientation_wxyz[0] == pytest.approx(h)


def test_wrong_length_pose_is_rejected():
    with pytest.raises(SpecError, match="3 numbers"):
        _obj(position=(0.3, 0.1))
    with pytest.raises(SpecError, match="4 numbers"):
        _obj(orientation_wxyz=(1.0, 0.0, 0.0))


def test_confidence_out_of_range_is_rejected():
    with pytest.raises(SpecError, match="confidence"):
        _obj(confidence=1.4)


def test_empty_robot_base_frame_is_rejected():
    with pytest.raises(SpecError, match="robot_base_frame"):
        SceneSpec(name="x", robot_base_frame="", objects=[])


# --- generated-asset fallback -------------------------------------------

def _gen(**kw) -> GeneratedAsset:
    kw.setdefault("mesh_path", "assets/weird_clamp.obj")
    kw.setdefault("mass_kg", 0.085)
    kw.setdefault("collider", "convex_decomposition")
    kw.setdefault("material", "steel")
    return GeneratedAsset(**kw)


def test_generated_object_round_trips():
    obj = _obj("clamp", catalog_key=None, provenance="generated", generated=_gen())
    back = SceneObject.from_dict(obj.to_dict())
    assert back.generated is not None
    assert back.generated.mesh_path == "assets/weird_clamp.obj"
    assert back.mass_kg() == pytest.approx(0.085)


def test_generated_object_without_physics_facts_is_rejected():
    """CLAUDE.md rule 4. There is no legal default for the mass of a thing we
    reconstructed from video."""
    with pytest.raises(SpecError, match="generated"):
        _obj("clamp", catalog_key=None, provenance="generated", generated=None)


def test_generated_asset_cannot_claim_to_be_verified():
    with pytest.raises(SpecError, match="verified"):
        _gen(verified=True)


def test_generated_asset_rejects_zero_mass():
    with pytest.raises(SpecError, match="mass_kg"):
        _gen(mass_kg=0.0)


def test_catalog_object_cannot_also_carry_a_generated_block():
    with pytest.raises(SpecError, match="both"):
        _obj("beaker_a", catalog_key="beaker_250", generated=_gen())


def test_catalog_key_none_cannot_claim_catalog_provenance():
    with pytest.raises(SpecError, match="provenance"):
        _obj("clamp", catalog_key=None, provenance="catalog", generated=_gen())


# --- derived physics facts ----------------------------------------------

def test_mass_comes_from_the_catalog_never_a_density():
    from labgen.catalog import CATALOG

    assert _obj().mass_kg() == CATALOG["beaker_250"].mass_kg


def test_open_vessel_instance_reports_an_sdf_collider():
    assert _obj().collider() == "sdf"


def test_free_bodies_and_fixtures_partition_the_scene():
    scene = _scene(
        _obj("beaker_a"),
        _obj("bench", catalog_key="bench_top", fixed=True, position=(0.0, 0.0, 0.0)),
    )
    assert [o.instance_id for o in scene.free_bodies] == ["beaker_a"]
    assert [o.instance_id for o in scene.fixtures] == ["bench"]
    assert len(scene) == 2


def test_by_id_raises_with_the_available_ids():
    with pytest.raises(KeyError, match="beaker_a"):
        _scene().by_id("nope")


# --- observations --------------------------------------------------------

def test_observation_round_trips():
    obs = ObjectObservation(
        label="250ml beaker", confidence=0.81, bbox_2d=(10, 20, 110, 150),
        frames=[3, 4, 5], read_text=["250", "mL"],
    )
    assert ObjectObservation.from_dict(obs.to_dict()).to_dict() == obs.to_dict()


def test_observation_rejects_an_inverted_bbox():
    with pytest.raises(SpecError, match="inverted"):
        ObjectObservation(label="beaker", confidence=0.5, bbox_2d=(110, 20, 10, 150))


def test_observation_rejects_an_empty_label():
    with pytest.raises(SpecError, match="label"):
        ObjectObservation(label="   ", confidence=0.5, bbox_2d=(0, 0, 1, 1))
