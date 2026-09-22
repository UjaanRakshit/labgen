"""Find the printed marker in a camera image and recover its pose.

The detection half of T5. labgen.register takes marker corner positions in and
fits the frame; this finds those corners in real images. Kept apart because it
needs OpenCV, which labgen core must not require -- cv2 is imported inside the
functions, never at module import, so `import labgen.detect` works on a machine
without it and fails only when detection is actually attempted.

Run in `.venv-vision` (opencv-contrib-python), never in the Isaac Lab
environment (CLAUDE.md rule 5).

Pose is solved with SOLVEPNP_IPPE_SQUARE, OpenCV's solver made for exactly this:
four coplanar corners of a square of known edge. It returns the correct pose for
a planar target where the generic iterative solver can flip between two
mirror-ambiguous solutions.

T5's real-data acceptance is reprojection error under 2 px; every detection
reports it, so a bad frame is visible rather than silently averaged in.

What that gate cannot see: a wrong camera intrinsic. Detect with a focal length
1% off and the marker lands ~1% of its distance away from the truth (10 mm at
1 m) while reprojection stays at ~0.2 px, because the pose absorbs the error. A
clean reprojection number says the corners are consistent, not that the camera
is calibrated. Measured in tests/test_detect.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .register import Similarity, marker_corners, reprojection_error_px

__all__ = ["Detection", "detect_marker", "DEFAULT_DICTIONARY", "render_marker_view"]

# A 4x4 dictionary with 50 ids: coarse cells detect robustly at a distance and
# under blur, which matters more on a bench video than having many ids.
DEFAULT_DICTIONARY = "DICT_4X4_50"


class DetectionFailed(RuntimeError):
    """The marker was not found, or not found cleanly enough to trust."""


@dataclass(frozen=True)
class Detection:
    marker_id: int
    corners_px: np.ndarray          # (4, 2), ArUco order: TL, TR, BR, BL
    cam_T_marker: Similarity        # marker pose in the camera frame, metres
    reprojection_px: float

    @property
    def distance_m(self) -> float:
        return float(np.linalg.norm(self.cam_T_marker.t))


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "labgen.detect needs OpenCV with the aruco module "
            "(opencv-contrib-python). Use the .venv-vision environment, not the "
            "Isaac Lab one.") from exc
    return cv2


def _rodrigues_to_R(rvec) -> np.ndarray:
    cv2 = _cv2()
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    return R


def detect_marker(image, K, dist, edge_m: float, marker_id: int,
                  dictionary: str = DEFAULT_DICTIONARY,
                  max_reprojection_px: float = 2.0) -> Detection:
    """Detect one specific marker id and solve its pose.

    Refuses rather than guesses: a missing marker, a duplicate id, or a solve
    whose reprojection error exceeds `max_reprojection_px` all raise. T5 sets
    2 px as the real-data bar, so a frame above it is not evidence.
    """
    cv2 = _cv2()
    img = np.asarray(image)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    params = cv2.aruco.DetectorParameters()
    # APRILTAG refinement: fits a line to each edge of the square and intersects
    # them. Measured against SUBPIX on exact renders of a 200 mm marker
    # (1080p, f=1660 px), depth bias at 0.8 / 1.2 / 1.5 m:
    #     SUBPIX    0.3-1.1 / 1.2-3.0 / 2.1-4.9 mm, growing with blur
    #     CONTOUR   2.2     / 4.7     / 7.2-7.9 mm
    #     APRILTAG  0.3     / 0.5     / 0.6 mm, independent of blur
    # SUBPIX is cornerSubPix, built for checkerboard saddle points; on the convex
    # outer corner of a solid square it pulls every corner inward, the marker
    # reads small, and so reads FAR. A systematic bias: averaging frames does not
    # remove it. Pinned by tests/test_detect.py.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    detector = cv2.aruco.ArucoDetector(d, params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        raise DetectionFailed("no markers detected")
    ids = ids.ravel().tolist()
    hits = [i for i, v in enumerate(ids) if v == marker_id]
    if not hits:
        raise DetectionFailed(f"marker {marker_id} not found (saw {sorted(ids)})")
    if len(hits) > 1:
        raise DetectionFailed(f"marker {marker_id} detected {len(hits)} times; ambiguous")
    c = np.asarray(corners[hits[0]], float).reshape(4, 2)

    obj = marker_corners(edge_m).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, c, np.asarray(K, float),
                                  None if dist is None else np.asarray(dist, float),
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise DetectionFailed("pose solve failed")
    pose = Similarity(_rodrigues_to_R(rvec), np.asarray(tvec, float).ravel(), 1.0)
    rep = reprojection_error_px(c, pose, K, edge_m)
    if rep > max_reprojection_px:
        raise DetectionFailed(
            f"reprojection error {rep:.2f} px exceeds {max_reprojection_px} px; "
            f"this frame is not good enough to register from")
    return Detection(marker_id=marker_id, corners_px=c, cam_T_marker=pose,
                     reprojection_px=rep)


def render_marker_view(cam_T_marker: Similarity, K, size_px, edge_m: float,
                       marker_id: int, dictionary: str = DEFAULT_DICTIONARY,
                       noise_std: float = 0.0, blur_px: float = 0.0,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Synthesise what a pinhole camera sees of the marker at a known pose.

    Exact, not approximate: a planar target maps to the image by the homography
    H = K [r1 r2 t], so warping the flat marker by H is the true projection. Lets
    detection be tested end to end against a pose whose answer is known, and
    lets corner noise be measured for a given camera rather than assumed.
    """
    cv2 = _cv2()
    R, t = cam_T_marker.R, cam_T_marker.t
    # The printed face is the marker's +z. If it points away from the camera the
    # warp still draws a convincing marker -- the back of the print, mirrored --
    # which no detector decodes. Refuse it rather than hand back that picture.
    if float(R[:, 2] @ t) >= 0:
        raise ValueError(
            "marker faces away from the camera (its +z must point back toward "
            "the camera); the render would be a mirrored, undecodable print")
    w, h = size_px
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    side = 400
    tag = cv2.aruco.generateImageMarker(d, marker_id, side)
    # White quiet zone around the tag, as a real print has; detection needs it.
    pad = side // 4
    canvas = np.full((side + 2 * pad, side + 2 * pad), 255, np.uint8)
    canvas[pad:pad + side, pad:pad + side] = tag
    # canvas pixel -> marker-frame metres (the TAG spans edge_m, not the canvas).
    # OpenCV pixel coordinates are pixel CENTRES, so the tag's outer edge -- the
    # boundary of pixel `pad` -- is at pad - 0.5 and its centre at
    # pad + side/2 - 0.5. Dropping the 0.5 displaced the whole marker by half a
    # canvas pixel, read as a uniform -0.5 px corner bias at 0.8 m.
    m_per_px = edge_m / side
    c0 = pad + side / 2 - 0.5
    A = np.array([[m_per_px, 0, -c0 * m_per_px],
                  [0, -m_per_px, c0 * m_per_px],
                  [0, 0, 1]])
    H = np.asarray(K, float) @ np.column_stack([R[:, 0], R[:, 1], t]) @ A
    img = cv2.warpPerspective(canvas, H, (w, h), flags=cv2.INTER_LINEAR,
                              borderValue=200)
    if blur_px > 0:
        k = int(2 * round(3 * blur_px) + 1)
        img = cv2.GaussianBlur(img, (k, k), blur_px)
    if noise_std > 0:
        rng = rng or np.random.default_rng(0)
        img = np.clip(img.astype(float) + rng.normal(0, noise_std, img.shape), 0, 255).astype(np.uint8)
    return img
