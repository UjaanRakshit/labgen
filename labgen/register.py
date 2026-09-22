"""Reconstruction frame -> robot base frame, through a marker of known size.

CLAUDE.md is blunt about why this module matters: every one of MATTERIX's
real-world sim failures was a registration or scale problem, and a constant
rotation error here makes every downstream number wrong in a way that looks
like a policy failure.

Two things have to be recovered, not one.

* SCALE. Monocular reconstruction has none; it is only defined up to a
  similarity. The marker's printed edge length is the only metric reference in
  the scene, so the transform from the reconstruction frame is a SIMILARITY
  (rotation, translation AND scale), fitted to the marker's corners.
* PLACEMENT. The marker frame is not the robot base frame. The offset between
  them is a physical measurement on the real bench, and it gets the same rule as
  a catalog dimension and a second arm's mounting offset: no source, no number.
  A placeholder builds and runs but reports the registration unverified.

The chain, for a point p seen in the reconstruction:

    p_base = T_base_marker * S_marker_recon * p_recon

where S_marker_recon is the similarity fitted from corner correspondences
(Umeyama, closed form, least squares over every corner in every frame), and
T_base_marker is the measured rigid offset.

Pure numpy. No OpenCV: corner DETECTION is a separate script with its own
dependencies; this module takes corner positions in, which is what makes it
testable against synthetic poses to the stated tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Similarity",
    "MarkerConfig",
    "Registration",
    "UnsourcedMarker",
    "umeyama",
    "marker_corners",
    "register",
    "rotation_error_deg",
    "project",
    "reprojection_error_px",
]


class UnsourcedMarker(ValueError):
    """A marker offset from the robot base that nobody measured."""


@dataclass(frozen=True)
class Similarity:
    """x -> scale * R @ x + t. Rigid when scale == 1."""

    R: np.ndarray
    t: np.ndarray
    scale: float = 1.0

    def __post_init__(self) -> None:
        R = np.asarray(self.R, float)
        if R.shape != (3, 3):
            raise ValueError("R must be 3x3")
        if abs(np.linalg.det(R) - 1.0) > 1e-6 or not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
            raise ValueError("R is not a proper rotation (orthonormal, det +1)")
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", np.asarray(self.t, float).reshape(3))

    @classmethod
    def identity(cls) -> "Similarity":
        return cls(np.eye(3), np.zeros(3), 1.0)

    def apply(self, p) -> np.ndarray:
        p = np.asarray(p, float)
        return self.scale * p @ self.R.T + self.t

    def rotate(self, R_obj) -> np.ndarray:
        """An orientation carried through the transform. Scale does not rotate."""
        return self.R @ np.asarray(R_obj, float)

    def __matmul__(self, other: "Similarity") -> "Similarity":
        """(self @ other).apply(p) == self.apply(other.apply(p))."""
        return Similarity(self.R @ other.R,
                          self.scale * self.R @ other.t + self.t,
                          self.scale * other.scale)

    def inverse(self) -> "Similarity":
        Rt = self.R.T
        return Similarity(Rt, -(Rt @ self.t) / self.scale, 1.0 / self.scale)


def umeyama(src, dst, *, with_scale: bool = True) -> Similarity:
    """Least-squares similarity mapping `src` points onto `dst` (Umeyama 1991).

    Closed form via SVD of the cross-covariance. Refuses degenerate input --
    fewer than three points, or points on a line -- because the rotation about
    that line is then undetermined, and a silently arbitrary rotation is the
    worst possible output from a registration step.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src and dst must both be (N, 3)")
    n = len(src)
    if n < 3:
        raise ValueError(f"need at least 3 correspondences, got {n}")
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    sv = np.linalg.svd(xs, compute_uv=False)
    if sv[1] < 1e-9 * max(sv[0], 1e-12):
        raise ValueError("points are collinear; the rotation about their line is undetermined")
    cov = xd.T @ xs / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / n
    scale = float(np.trace(np.diag(D) @ S) / var_s) if with_scale else 1.0
    t = mu_d - scale * R @ mu_s
    return Similarity(R, t, scale)


def marker_corners(edge_m: float) -> np.ndarray:
    """The four corners of a square marker in its own frame (z out of the page).

    Order follows OpenCV's ArUco convention: top-left, top-right, bottom-right,
    bottom-left, so detections can be matched without reordering.
    """
    if edge_m <= 0:
        raise ValueError("marker edge must be positive")
    h = edge_m / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


def _is_sourced(source: str) -> bool:
    s = (source or "").strip()
    return bool(s) and not s.upper().startswith("TODO")


def _is_placeholder(source: str) -> bool:
    return (source or "").strip().upper().startswith("PLACEHOLDER")


@dataclass(frozen=True)
class MarkerConfig:
    """The printed marker: its size, and where it sits relative to the robot base.

    Both numbers are physical measurements. `edge_m` is the printed square's
    edge -- measure the print, not the PDF, because printers scale. And
    `base_T_marker` is the marker's pose in the robot base frame, measured on
    the bench. A TODO blocks; a PLACEHOLDER builds and poisons `verified`.
    """

    edge_m: float
    base_T_marker: Similarity
    marker_id: int
    edge_source: str
    offset_source: str

    def __post_init__(self) -> None:
        if self.edge_m <= 0:
            raise ValueError("marker edge must be positive")
        if abs(self.base_T_marker.scale - 1.0) > 1e-9:
            raise ValueError("the marker offset is a rigid transform; scale must be 1")
        for name, src in (("edge", self.edge_source), ("offset", self.offset_source)):
            if _is_placeholder(src):
                continue
            if not _is_sourced(src):
                raise UnsourcedMarker(
                    f"marker {name} has no source ({src!r}). It is a physical "
                    f"measurement of the real bench and cannot be chosen. Supply "
                    f"it, or mark it PLACEHOLDER to build without trusting it.")

    @property
    def verified(self) -> bool:
        return not (_is_placeholder(self.edge_source) or _is_placeholder(self.offset_source))


@dataclass(frozen=True)
class Registration:
    """The fitted chain reconstruction -> base, and how well it fitted."""

    base_T_recon: Similarity
    marker_T_recon: Similarity
    corner_rms_m: float          # fit residual, in METRES (after scale recovery)
    n_corners: int
    marker: MarkerConfig

    @property
    def verified(self) -> bool:
        return self.marker.verified

    def to_base(self, p_recon) -> np.ndarray:
        return self.base_T_recon.apply(p_recon)

    def orientation_to_base(self, R_recon) -> np.ndarray:
        return self.base_T_recon.rotate(R_recon)


def register(corner_obs_recon, marker: MarkerConfig) -> Registration:
    """Fit reconstruction -> base from observed marker corners.

    `corner_obs_recon` is (F, 4, 3): the marker's four corners as the
    reconstruction placed them, in each of F frames. Every corner in every frame
    enters one least-squares fit, so independent noise averages down.
    """
    obs = np.asarray(corner_obs_recon, float)
    if obs.ndim == 2:
        obs = obs[None]
    if obs.shape[1:] != (4, 3):
        raise ValueError(f"corner observations must be (frames, 4, 3), got {obs.shape}")
    ref = marker_corners(marker.edge_m)
    src = obs.reshape(-1, 3)
    dst = np.tile(ref, (len(obs), 1))
    marker_T_recon = umeyama(src, dst, with_scale=True)
    resid = marker_T_recon.apply(src) - dst
    rms = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
    return Registration(base_T_recon=marker.base_T_marker @ marker_T_recon,
                        marker_T_recon=marker_T_recon, corner_rms_m=rms,
                        n_corners=len(src), marker=marker)


def rotation_error_deg(R_a, R_b) -> float:
    """Angle of the rotation taking R_b to R_a, in degrees.

    atan2(sin, cos), not acos(cos). acos is ill-conditioned near identity --
    acos(1 - e) ~ sqrt(2e) -- so float rounding alone in the trace reads as
    ~1e-6 deg of error between two matrices equal to eight digits. The sine term
    comes from the skew part and keeps small angles exact.
    """
    r = np.asarray(R_a, float) @ np.asarray(R_b, float).T
    cos = (np.trace(r) - 1.0) / 2.0
    skew = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / 2.0
    return float(math.degrees(math.atan2(float(np.linalg.norm(skew)), float(cos))))


def project(points_cam, K) -> np.ndarray:
    """Pinhole projection of camera-frame points to pixels. Points must be in front."""
    p = np.asarray(points_cam, float)
    if (p[:, 2] <= 0).any():
        raise ValueError("a point is behind the camera")
    uv = p @ np.asarray(K, float).T
    return uv[:, :2] / uv[:, 2:3]


def reprojection_error_px(corners_px, cam_T_marker: Similarity, K, edge_m: float) -> float:
    """RMS pixel error of the marker corners reprojected through a pose.

    The real-data half of T5's acceptance: under 2 px. Kept here, pure, so the
    arithmetic is tested even though the detections come from elsewhere.
    """
    pred = project(cam_T_marker.apply(marker_corners(edge_m)), K)
    d = np.asarray(corners_px, float) - pred
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))
