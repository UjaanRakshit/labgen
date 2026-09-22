"""Marker detection, end to end, against synthetic views whose pose is known.

Skipped where OpenCV is absent (the core CI environment), run in .venv-vision.
The render is exact -- a planar target maps to the image by a homography -- so
a recovered pose can be compared against the truth, and the corner noise that
labgen.register's sizing recommendation ASSUMED can be measured instead.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
if not hasattr(cv2, "aruco"):
    pytest.skip("OpenCV without the aruco module", allow_module_level=True)

from labgen.detect import DetectionFailed, detect_marker, render_marker_view  # noqa: E402
from labgen.register import Similarity, rotation_error_deg                    # noqa: E402

# A 1080p camera with a ~60 deg horizontal field of view.
W, H = 1920, 1080
F = 1660.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
EDGE = 0.200


# A marker FACING the camera: its +z (out of the printed face) points back at the
# camera, i.e. along camera -z. The first version of this test used identity,
# which points the marker's face AWAY -- the camera then sees the back of the
# print, mirrored, and a mirrored ArUco code does not decode. Every detection
# failed with "no markers detected" on a render that looked perfectly fine.
FACING = np.diag([1.0, -1.0, -1.0])


def pose(distance_m: float, tilt_deg: float = 20.0, yaw_deg: float = 15.0) -> Similarity:
    tx, ty = math.radians(tilt_deg), math.radians(yaw_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(tx), -math.sin(tx)], [0, math.sin(tx), math.cos(tx)]])
    Ry = np.array([[math.cos(ty), 0, math.sin(ty)], [0, 1, 0], [-math.sin(ty), 0, math.cos(ty)]])
    return Similarity(Ry @ Rx @ FACING, [0.03, -0.02, distance_m])


def test_recovers_a_known_pose_from_a_clean_view():
    truth = pose(1.0)
    img = render_marker_view(truth, K, (W, H), EDGE, marker_id=7)
    det = detect_marker(img, K, None, EDGE, marker_id=7)
    assert np.linalg.norm(det.cam_T_marker.t - truth.t) < 1e-3
    assert rotation_error_deg(det.cam_T_marker.R, truth.R) < 0.5
    assert det.reprojection_px < 2.0, "T5 real-data bar"


def test_survives_sensor_noise_and_blur():
    truth = pose(1.2)
    img = render_marker_view(truth, K, (W, H), EDGE, marker_id=3, noise_std=6.0,
                             blur_px=1.0, rng=np.random.default_rng(1))
    det = detect_marker(img, K, None, EDGE, marker_id=3)
    assert np.linalg.norm(det.cam_T_marker.t - truth.t) < 5e-3
    assert det.reprojection_px < 2.0


def test_a_missing_marker_raises_rather_than_returning_a_pose():
    img = np.full((H, W), 200, np.uint8)
    with pytest.raises(DetectionFailed, match="no markers"):
        detect_marker(img, K, None, EDGE, marker_id=0)


def test_rendering_the_back_of_the_marker_is_refused():
    away = Similarity(np.eye(3), [0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="faces away"):
        render_marker_view(away, K, (W, H), EDGE, marker_id=0)


def test_the_wrong_id_is_not_accepted_as_the_right_one():
    img = render_marker_view(pose(1.0), K, (W, H), EDGE, marker_id=5)
    with pytest.raises(DetectionFailed, match="not found"):
        detect_marker(img, K, None, EDGE, marker_id=9)


def test_a_wrong_edge_length_puts_the_marker_at_the_wrong_distance():
    """Why the print must be measured, not taken from the PDF: believe a 200 mm
    marker is 250 mm and it reads as 25% further away."""
    truth = pose(1.0)
    img = render_marker_view(truth, K, (W, H), EDGE, marker_id=2)
    right = detect_marker(img, K, None, EDGE, marker_id=2)
    wrong = detect_marker(img, K, None, 0.250, marker_id=2)
    assert wrong.distance_m / right.distance_m == pytest.approx(1.25, rel=1e-3)


def test_measured_corner_noise_grounds_the_sizing_recommendation():
    """labgen.register recommends a >= 200 mm marker ASSUMING 0.5 mm of corner
    noise per detection. Measure it for this camera at bench distances instead.

    Noisy, blurred renders at 0.8-1.5 m; the spread of recovered positions over
    independent noise draws is the per-detection noise, in metres.
    """
    rng = np.random.default_rng(42)
    spreads = []
    for d in (0.8, 1.0, 1.2, 1.5):
        truth = pose(d)
        ts = []
        for _ in range(12):
            img = render_marker_view(truth, K, (W, H), EDGE, marker_id=1,
                                     noise_std=5.0, blur_px=0.8, rng=rng)
            ts.append(detect_marker(img, K, None, EDGE, marker_id=1).cam_T_marker.t)
        spreads.append(float(np.std(np.array(ts), axis=0).max()))
    worst = max(spreads)
    # Recorded, not merely bounded: this is the number the sizing finding needs.
    print(f"\nper-detection position spread, 200 mm marker, 1080p, 0.8-1.5 m: "
          f"{[round(s*1000, 3) for s in spreads]} mm (worst {worst*1000:.3f} mm)")
    assert worst < 0.5e-3, (
        f"corner noise {worst*1000:.3f} mm exceeds the 0.5 mm the 200 mm "
        f"recommendation assumed; revisit labgen.register's sizing")


@pytest.mark.parametrize("blur", [0.0, 0.8, 1.5])
def test_corner_refinement_does_not_bias_depth_under_blur(blur):
    """A SYSTEMATIC error, so frame averaging cannot remove it. cornerSubPix
    (CORNER_REFINE_SUBPIX) pulled the corners of a solid square inward: 2.1 mm
    at 1.5 m sharp, 4.9 mm blurred. AprilTag edge-line refinement measured
    0.6 mm at 1.5 m regardless of blur. Guard against a regression to it."""
    truth = pose(1.5)
    img = render_marker_view(truth, K, (W, H), EDGE, marker_id=4, blur_px=blur)
    det = detect_marker(img, K, None, EDGE, marker_id=4)
    assert np.linalg.norm(det.cam_T_marker.t - truth.t) < 1e-3


def test_reprojection_error_cannot_see_a_miscalibrated_camera():
    """The T5 real-data gate (< 2 px) passes a camera whose focal length is 1%
    wrong, while the marker lands ~1% of its distance off. A calibrated camera
    is therefore a precondition that has to be established separately; this
    test pins that the gate does not establish it."""
    truth = pose(1.0)
    img = render_marker_view(truth, K, (W, H), EDGE, marker_id=6)
    K_off = K.copy()
    K_off[0, 0] *= 1.01
    K_off[1, 1] *= 1.01
    det = detect_marker(img, K_off, None, EDGE, marker_id=6)
    assert det.reprojection_px < 0.5, "the gate passes it comfortably"
    assert np.linalg.norm(det.cam_T_marker.t - truth.t) > 8e-3, "yet it is ~1 cm off"
