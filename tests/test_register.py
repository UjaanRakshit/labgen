"""T5, synthetic half: recover a known transform within 1 mm and 0.5 degrees.

TASKS.md: generate known poses, apply a known transform, recover it, assert the
residual is small -- and only then touch real data. That is what every test here
does, across many random transforms and scales, with and without noise.

A registration step that returns an arbitrary rotation without complaint is the
worst thing this module could do, so the degenerate cases -- too few points,
points on a line -- are tested to RAISE, not to return something.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from labgen.register import (MarkerConfig, Similarity, UnsourcedMarker,
                             marker_corners, project, register,
                             reprojection_error_px, rotation_error_deg, umeyama)

MM = 1e-3
TOL_M = 1.0 * MM           # T5: 1 mm
TOL_DEG = 0.5              # T5: 0.5 deg
EDGE = 0.080               # an 80 mm printed marker (for the algebra tests)
SPEC_EDGE = 0.200          # the size that meets T5 under corner noise -- see below


def random_rotation(rng) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def random_similarity(rng, scale_range=(0.2, 5.0)) -> Similarity:
    return Similarity(random_rotation(rng), rng.uniform(-2, 2, 3),
                      float(rng.uniform(*scale_range)))


def measured_marker(base_T_marker=None) -> MarkerConfig:
    return MarkerConfig(edge_m=EDGE,
                        base_T_marker=base_T_marker or Similarity(np.eye(3), [0.30, 0.10, 0.0]),
                        marker_id=0, edge_source="calipers on the print, 80.0 mm",
                        offset_source="tape from base centre, 2026-09-22")


def observe(marker: MarkerConfig, recon_T_marker: Similarity, frames: int,
            noise_m: float, rng) -> np.ndarray:
    """Where a reconstruction with unknown scale would put the marker corners."""
    ref = marker_corners(marker.edge_m)
    obs = [recon_T_marker.apply(ref) for _ in range(frames)]
    obs = np.array(obs)
    # Noise is added in the RECONSTRUCTION's units, which differ from metres by
    # the unknown scale -- so a fixed metric noise is scaled accordingly.
    return obs + rng.normal(scale=noise_m * recon_T_marker.scale, size=obs.shape)


# --- the similarity algebra -----------------------------------------------

def test_compose_and_invert_are_consistent():
    rng = np.random.default_rng(0)
    a, b = random_similarity(rng), random_similarity(rng)
    p = rng.normal(size=(5, 3))
    assert (a @ b).apply(p) == pytest.approx(a.apply(b.apply(p)))
    assert (a @ a.inverse()).apply(p) == pytest.approx(p)


def test_a_non_rotation_is_refused():
    with pytest.raises(ValueError, match="proper rotation"):
        Similarity(np.diag([1.0, 1.0, -1.0]), np.zeros(3))      # a reflection


def test_scale_does_not_rotate_orientations():
    s = Similarity(np.eye(3), np.zeros(3), 3.0)
    assert s.rotate(np.eye(3)) == pytest.approx(np.eye(3))


# --- Umeyama --------------------------------------------------------------

def test_umeyama_recovers_an_exact_similarity():
    rng = np.random.default_rng(1)
    gt = random_similarity(rng)
    src = rng.normal(size=(20, 3))
    fit = umeyama(src, gt.apply(src))
    assert rotation_error_deg(fit.R, gt.R) < 1e-6
    assert fit.scale == pytest.approx(gt.scale, rel=1e-9)
    assert fit.t == pytest.approx(gt.t, abs=1e-9)


def test_umeyama_never_returns_a_reflection():
    """A reflection fits some point sets better than any rotation. Returning one
    would mirror the scene -- a box on the worktop reads as a hole in it."""
    rng = np.random.default_rng(2)
    src = rng.normal(size=(10, 3))
    dst = src * np.array([1.0, 1.0, -1.0])     # a mirror image
    fit = umeyama(src, dst)
    assert np.linalg.det(fit.R) == pytest.approx(1.0)


def test_too_few_points_raise():
    with pytest.raises(ValueError, match="at least 3"):
        umeyama(np.zeros((2, 3)), np.zeros((2, 3)))


def test_collinear_points_raise_instead_of_guessing():
    """Rotation about the line is undetermined; an arbitrary answer is worse
    than none."""
    line = np.outer(np.linspace(0, 1, 6), [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="collinear"):
        umeyama(line, line + 1.0)


# --- the T5 acceptance: 1 mm, 0.5 deg --------------------------------------

@pytest.mark.parametrize("seed", range(25))
def test_recovers_the_base_frame_to_t5_tolerance_noise_free(seed):
    rng = np.random.default_rng(100 + seed)
    marker = measured_marker(Similarity(random_rotation(rng), rng.uniform(-0.5, 0.5, 3)))
    recon_T_marker = random_similarity(rng)
    reg = register(observe(marker, recon_T_marker, frames=1, noise_m=0.0, rng=rng), marker)

    gt_base_T_recon = marker.base_T_marker @ recon_T_marker.inverse()
    probe = rng.uniform(-0.6, 0.6, size=(50, 3))                 # points in the reconstruction
    probe_recon = recon_T_marker.apply(marker.base_T_marker.inverse().apply(probe))
    assert np.abs(reg.to_base(probe_recon) - probe).max() < 1e-9
    assert rotation_error_deg(reg.base_T_recon.R, gt_base_T_recon.R) < 1e-6


def spec_marker() -> MarkerConfig:
    return MarkerConfig(edge_m=SPEC_EDGE, base_T_marker=Similarity(np.eye(3), [0.30, 0.10, 0.0]),
                        marker_id=0, edge_source="calipers on the print, 200.0 mm",
                        offset_source="tape from base centre, 2026-09-22")


@pytest.mark.parametrize("seed", range(10))
def test_recovers_to_t5_tolerance_with_realistic_corner_noise(seed):
    """0.5 mm corner noise per detection, 30 frames, a 200 mm marker: inside
    1 mm / 0.5 deg over a working volume of +/-0.4 m around the marker.

    The marker SIZE is what makes this pass. Under noise, a small angular error
    is multiplied by the distance to the object, and a bigger marker pins the
    angle down better. Measured worst case over the volume: 60 mm 2.82 mm,
    80 mm 2.10, 120 mm 1.40, 160 mm 1.05, 200 mm 0.85 -- roughly 1/edge.
    """
    rng = np.random.default_rng(200 + seed)
    marker = spec_marker()
    recon_T_marker = random_similarity(rng)
    reg = register(observe(marker, recon_T_marker, frames=30, noise_m=0.5 * MM, rng=rng), marker)

    gt = marker.base_T_marker @ recon_T_marker.inverse()
    assert rotation_error_deg(reg.base_T_recon.R, gt.R) < TOL_DEG
    probe_base = rng.uniform(-0.4, 0.4, size=(200, 3)) + marker.base_T_marker.t
    probe_recon = gt.inverse().apply(probe_base)
    err = np.linalg.norm(reg.to_base(probe_recon) - probe_base, axis=1)
    assert err.max() < TOL_M, f"worst {err.max()*1000:.3f} mm"


def test_an_80_mm_marker_does_not_meet_1_mm_over_the_bench():
    """The sizing finding, pinned so nobody prints a small marker assuming it is
    fine. Same noise and volume as the passing test; only the edge changes. The
    worst point across 40 random setups exceeds 1 mm with an 80 mm marker.

    This rests on an ASSUMPTION: 0.5 mm of corner noise per detection. Real
    noise depends on the camera, its resolution and the viewing distance, and
    must be measured before the 200 mm recommendation is trusted.
    """
    worst = 0.0
    for seed in range(40):
        rng = np.random.default_rng(1000 + seed)
        m = measured_marker()
        rtm = random_similarity(rng)
        reg = register(observe(m, rtm, 30, 0.5 * MM, rng), m)
        gt = m.base_T_marker @ rtm.inverse()
        pb = rng.uniform(-0.4, 0.4, size=(300, 3)) + m.base_T_marker.t
        worst = max(worst, np.linalg.norm(reg.to_base(gt.inverse().apply(pb)) - pb, axis=1).max())
    assert worst > TOL_M, f"an 80 mm marker reached {worst*1000:.2f} mm; the sizing finding no longer holds"


def test_scale_is_recovered_from_the_marker_edge_alone():
    """The whole reason a marker is required: monocular video has no scale."""
    rng = np.random.default_rng(3)
    marker = measured_marker()
    recon_T_marker = Similarity(np.eye(3), [0.0, 0.0, 0.0], 0.37)   # reconstruction is 0.37x
    reg = register(observe(marker, recon_T_marker, 1, 0.0, rng), marker)
    assert reg.marker_T_recon.scale == pytest.approx(1 / 0.37, rel=1e-9)


def test_a_wrong_marker_size_scales_every_distance_by_the_same_ratio():
    """Why the print must be measured: an 80 mm marker believed to be 100 mm
    makes every object 25% further from the robot than it is."""
    rng = np.random.default_rng(4)
    truth = measured_marker()
    recon_T_marker = random_similarity(rng)
    obs = observe(truth, recon_T_marker, 1, 0.0, rng)
    wrong = MarkerConfig(edge_m=0.100, base_T_marker=truth.base_T_marker, marker_id=0,
                         edge_source="assumed from the PDF, not measured",
                         offset_source=truth.offset_source)
    good, bad = register(obs, truth), register(obs, wrong)
    p = recon_T_marker.apply(np.array([[0.3, 0.2, 0.1]]))
    d_good = np.linalg.norm(good.marker_T_recon.apply(p))
    d_bad = np.linalg.norm(bad.marker_T_recon.apply(p))
    assert d_bad / d_good == pytest.approx(1.25, rel=1e-9)


def test_orientations_are_carried_through():
    rng = np.random.default_rng(5)
    marker = measured_marker(Similarity(random_rotation(rng), [0.2, 0.0, 0.0]))
    recon_T_marker = random_similarity(rng)
    reg = register(observe(marker, recon_T_marker, 1, 0.0, rng), marker)
    R_obj_base = random_rotation(rng)
    gt = marker.base_T_marker @ recon_T_marker.inverse()
    R_obj_recon = gt.inverse().rotate(R_obj_base)
    assert rotation_error_deg(reg.orientation_to_base(R_obj_recon), R_obj_base) < 1e-6


def test_fit_residual_is_reported_in_metres():
    rng = np.random.default_rng(6)
    marker = measured_marker()
    reg = register(observe(marker, random_similarity(rng), 30, 0.5 * MM, rng), marker)
    assert 0.1 * MM < reg.corner_rms_m < 1.5 * MM
    assert reg.n_corners == 120


# --- provenance: the offset and the edge are measurements ------------------

def test_an_unmeasured_offset_is_refused():
    with pytest.raises(UnsourcedMarker, match="offset has no source"):
        MarkerConfig(edge_m=EDGE, base_T_marker=Similarity.identity(), marker_id=0,
                     edge_source="calipers", offset_source="")


def test_a_todo_edge_is_refused():
    with pytest.raises(UnsourcedMarker, match="edge has no source"):
        MarkerConfig(edge_m=EDGE, base_T_marker=Similarity.identity(), marker_id=0,
                     edge_source="TODO measure the print", offset_source="tape")


def test_a_placeholder_builds_but_is_not_verified():
    m = MarkerConfig(edge_m=EDGE, base_T_marker=Similarity.identity(), marker_id=0,
                     edge_source="calipers", offset_source="PLACEHOLDER until taped")
    rng = np.random.default_rng(7)
    reg = register(observe(m, random_similarity(rng), 1, 0.0, rng), m)
    assert not m.verified and not reg.verified


def test_a_measured_marker_is_verified():
    assert measured_marker().verified


def test_the_marker_offset_must_be_rigid():
    with pytest.raises(ValueError, match="rigid"):
        MarkerConfig(edge_m=EDGE, base_T_marker=Similarity(np.eye(3), np.zeros(3), 2.0),
                     marker_id=0, edge_source="c", offset_source="t")


# --- the real-data half's arithmetic ---------------------------------------

def test_reprojection_is_zero_for_a_perfect_pose():
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    cam_T_marker = Similarity(np.eye(3), [0.02, -0.01, 0.5])
    px = project(cam_T_marker.apply(marker_corners(EDGE)), K)
    assert reprojection_error_px(px, cam_T_marker, K, EDGE) < 1e-9


def test_reprojection_error_sees_a_one_pixel_shift():
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    cam_T_marker = Similarity(np.eye(3), [0.0, 0.0, 0.5])
    px = project(cam_T_marker.apply(marker_corners(EDGE)), K) + np.array([1.0, 0.0])
    assert reprojection_error_px(px, cam_T_marker, K, EDGE) == pytest.approx(1.0)


def test_a_point_behind_the_camera_is_refused():
    with pytest.raises(ValueError, match="behind the camera"):
        project(np.array([[0.0, 0.0, -1.0]]), np.eye(3))


def test_marker_corners_follow_the_aruco_order():
    c = marker_corners(0.1)
    assert c[0] == pytest.approx([-0.05, 0.05, 0.0])     # top-left
    assert c[2] == pytest.approx([0.05, -0.05, 0.0])     # bottom-right


def test_register_needs_no_opencv_or_physics():
    import subprocess
    import sys
    code = ("import sys; import labgen.register;"
            "banned=('cv2','newton','warp','torch','isaaclab');"
            "print(','.join(m for m in banned if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".")
    assert out.returncode == 0, out.stderr
    assert not [m for m in out.stdout.strip().split(",") if m]
