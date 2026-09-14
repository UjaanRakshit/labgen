"""CatalogItem -> watertight triangle mesh (OBJ).

The rule this module exists to enforce: **an open vessel's cavity must be
real, not implied.** A beaker whose mesh is a solid cylinder passes every
visual check and fails every pour task, and it fails it silently -- the render
looks right, the grasp looks right, and the liquid ends up nowhere.

So every hollow shape here is built as a genuine surface of revolution over a
*closed* profile that goes up the outside, across the rim, down the inside, and
across the floor. There is no place in this module where a cavity is faked.

Geometry conventions:

* metres, Z-up, right-handed (CLAUDE.md rule 2)
* every object sits on z = 0, so a pose places its base, not its centre
* faces are counter-clockwise seen from outside, i.e. outward normals, which
  makes the divergence-theorem volume positive and lets the manifold check
  double as an orientation check

No dependency beyond numpy. In particular no `trimesh`, no `pxr` -- this has to
import cleanly in CI without a GPU or an Isaac install.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .catalog import CATALOG, CatalogItem

__all__ = [
    "Mesh",
    "MeshError",
    "RENDER_SEGMENTS",
    "COLLISION_SEGMENTS",
    "revolve",
    "build",
    "build_collision",
    "build_all",
    "cavity_volume_m3",
]

# Enough that a 16 mm test tube does not look faceted, few enough that an SDF
# bake over a whole bench stays cheap.
RENDER_SEGMENTS = 64

# Only used for shapes where a reduced proxy is safe. Never for an open vessel
# -- see build_collision().
COLLISION_SEGMENTS = 24

# Triangles with area below this are degenerate in single precision and make
# both SDF baking and convex decomposition unstable.
MIN_TRIANGLE_AREA = 1e-12   # m^2


class MeshError(ValueError):
    """The geometry implied by a catalog entry cannot be built.

    Raised rather than worked around. A rack whose holes do not fit in its
    footprint is a catalog error, and quietly shrinking the holes to make the
    mesh build would produce a rack that no real vial fits into.
    """


# --------------------------------------------------------------------------
# mesh container
# --------------------------------------------------------------------------

@dataclass
class Mesh:
    vertices: np.ndarray        # (N, 3) float64, metres
    faces: np.ndarray           # (M, 3) int32, CCW from outside
    name: str = "mesh"

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int32).reshape(-1, 3)
        if len(self.faces) and self.faces.max() >= len(self.vertices):
            raise MeshError(
                f"{self.name}: face references vertex {self.faces.max()} but there "
                f"are only {len(self.vertices)} vertices"
            )

    # -- measurements -----------------------------------------------------

    def triangle_areas(self) -> np.ndarray:
        a, b, c = (self.vertices[self.faces[:, i]] for i in range(3))
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    def degenerate_faces(self) -> np.ndarray:
        """Indices of faces with a repeated vertex or near-zero area."""
        f = self.faces
        repeated = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
        return np.flatnonzero(repeated | (self.triangle_areas() < MIN_TRIANGLE_AREA))

    def volume_m3(self) -> float:
        """Signed volume by the divergence theorem.

        Positive iff the surface is closed and outward-oriented, so this is
        also a cheap orientation check.
        """
        a, b, c = (self.vertices[self.faces[:, i]] for i in range(3))
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def extents(self) -> np.ndarray:
        lo, hi = self.bounds()
        return hi - lo

    # -- topology ---------------------------------------------------------

    def manifold_errors(self) -> dict[str, list]:
        """Empty dict iff the mesh is closed, manifold and consistently wound.

        Test: in a closed, consistently oriented triangle mesh every directed
        edge (a, b) occurs exactly once, and (b, a) occurs exactly once. A
        directed edge seen twice means two faces disagree about which way is
        out; an edge whose reverse is missing means a hole.
        """
        seen: dict[tuple[int, int], int] = {}
        for f in self.faces:
            for i in range(3):
                e = (int(f[i]), int(f[(i + 1) % 3]))
                seen[e] = seen.get(e, 0) + 1

        duplicated = [e for e, n in seen.items() if n > 1]
        boundary = [e for e in seen if (e[1], e[0]) not in seen]

        errors: dict[str, list] = {}
        if duplicated:
            errors["duplicated_directed_edges"] = sorted(duplicated)[:20]
        if boundary:
            errors["boundary_edges"] = sorted(boundary)[:20]
        return errors

    @property
    def is_manifold(self) -> bool:
        return not self.manifold_errors()

    # -- ray casting ------------------------------------------------------

    def ray_crossings_z(self, x: float, y: float) -> list[float]:
        """Z values where a +Z ray from (x, y, -inf) crosses the surface.

        Used by the cavity test. Note that the *count* alone does not prove a
        cavity: a solid cylinder crosses twice too (base and top). What
        distinguishes a real open vessel is *where* the second crossing is --
        at the cavity floor, with nothing at all at rim height, because the
        vessel is open. See tests/test_meshes.py.

        Cast the ray slightly off-axis. A ray through a triangle-fan apex hits
        every triangle sharing that vertex and reports the crossing once per
        triangle, which is correct-but-useless.

        Moller-Trumbore, specialised to a vertical ray so it stays readable.
        """
        v = self.vertices
        a, b, c = (v[self.faces[:, i]] for i in range(3))

        # Barycentric test in the XY plane only, since the ray is vertical.
        p = np.array([x, y])
        d0 = b[:, :2] - a[:, :2]
        d1 = c[:, :2] - a[:, :2]
        d2 = p[None, :] - a[:, :2]

        denom = d0[:, 0] * d1[:, 1] - d1[:, 0] * d0[:, 1]
        ok = np.abs(denom) > 1e-18
        if not ok.any():
            return []

        u = np.zeros(len(a))
        w = np.zeros(len(a))
        u[ok] = (d2[ok, 0] * d1[ok, 1] - d1[ok, 0] * d2[ok, 1]) / denom[ok]
        w[ok] = (d0[ok, 0] * d2[ok, 1] - d2[ok, 0] * d0[ok, 1]) / denom[ok]

        inside = ok & (u >= 0) & (w >= 0) & (u + w <= 1.0)
        if not inside.any():
            return []

        z = (a[inside, 2]
             + u[inside] * (b[inside, 2] - a[inside, 2])
             + w[inside] * (c[inside, 2] - a[inside, 2]))
        return sorted(float(t) for t in z)

    # -- io ---------------------------------------------------------------

    def to_obj(self) -> str:
        lines = [
            f"# labgen generated mesh: {self.name}",
            "# units: metres, Z-up, right-handed. Do not rescale.",
            f"o {self.name}",
        ]
        lines += [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in self.vertices]
        lines += [f"f {i + 1} {j + 1} {k + 1}" for i, j, k in self.faces]
        return "\n".join(lines) + "\n"

    def write_obj(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_obj(), encoding="utf-8")
        return path


# --------------------------------------------------------------------------
# primitive: surface of revolution
# --------------------------------------------------------------------------

def revolve(profile: list[tuple[float, float]], segments: int = RENDER_SEGMENTS,
            name: str = "revolved") -> Mesh:
    """Revolve a closed (r, z) profile about the Z axis.

    `profile` is a closed loop -- the last point joins back to the first -- and
    must be ordered so that the outer wall is traversed **upward**. That
    ordering is what makes the resulting normals point out of the solid; see
    the module docstring.

    Points with r == 0 lie on the axis and become triangle-fan apexes rather
    than rings, which is how the degenerate sliver triangles that would
    otherwise appear at the axis are avoided. A profile edge with both
    endpoints on the axis sweeps zero area and is dropped.
    """
    if segments < 3:
        raise MeshError(f"{name}: need at least 3 segments, got {segments}")
    if len(profile) < 3:
        raise MeshError(f"{name}: profile needs at least 3 points, got {len(profile)}")
    for r, z in profile:
        if r < 0:
            raise MeshError(f"{name}: profile radius {r} is negative")

    theta = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    vertices: list[np.ndarray] = []
    rings: list[int | np.ndarray] = []   # per profile point: apex index, or ring indices

    for r, z in profile:
        if r == 0.0:
            rings.append(len(vertices))
            vertices.append(np.array([0.0, 0.0, z]))
        else:
            idx = np.arange(len(vertices), len(vertices) + segments)
            rings.append(idx)
            vertices.extend(np.stack([r * cos_t, r * sin_t, np.full(segments, z)], axis=1))

    faces: list[tuple[int, int, int]] = []
    n = len(profile)
    for k in range(n):
        lo, hi = rings[k], rings[(k + 1) % n]
        lo_axis, hi_axis = isinstance(lo, int), isinstance(hi, int)

        if lo_axis and hi_axis:
            continue                        # edge lies on the axis: zero area

        if lo_axis:                         # fan from apex `lo` up to ring `hi`
            for s in range(segments):
                t = (s + 1) % segments
                faces.append((lo, hi[t], hi[s]))
        elif hi_axis:                       # fan from ring `lo` up to apex `hi`
            for s in range(segments):
                t = (s + 1) % segments
                faces.append((lo[s], lo[t], hi))
        else:                               # quad strip between two rings
            for s in range(segments):
                t = (s + 1) % segments
                faces.append((lo[s], lo[t], hi[t]))
                faces.append((lo[s], hi[t], hi[s]))

    return Mesh(np.asarray(vertices), np.asarray(faces, dtype=np.int32), name=name)


# --------------------------------------------------------------------------
# shape builders
# --------------------------------------------------------------------------

def _open_vessel_profile(outer_d: float, height: float, wall: float,
                         base_d: float | None = None) -> list[tuple[float, float]]:
    """Closed profile for a straight-sided open vessel.

    Order matters: up the outside, in across the rim, down the inside, in
    across the cavity floor, down the axis, out across the base.

        (Ro,0) -> (Ro,H) -> (Ri,H) -> (Ri,w) -> (0,w) -> (0,0) -> back

    `base_d` wider than `outer_d` is the graduated-cylinder case: a narrow
    column on a wide foot. The foot is a separate step in the profile, not a
    taper, because that is what the real part looks like and a taper would put
    the centre of mass in the wrong place.
    """
    outer_r = outer_d / 2.0
    inner_r = outer_r - wall
    if inner_r <= 0:
        raise MeshError(
            f"wall {wall} m leaves no cavity in a {outer_d} m vessel. "
            f"A vessel with no cavity is a solid cylinder."
        )
    if height <= wall:
        raise MeshError(f"height {height} m is not greater than floor thickness {wall} m")

    profile: list[tuple[float, float]] = []
    base_r = (base_d / 2.0) if base_d else outer_r

    if base_r > outer_r + 1e-9:
        # Wide foot: out along the base, up the foot rim, in to the column.
        foot_h = min(wall * 2.0, height * 0.02)
        profile += [(base_r, 0.0), (base_r, foot_h), (outer_r, foot_h)]
    else:
        base_r = outer_r
        profile += [(outer_r, 0.0)]

    profile += [
        (outer_r, height),      # up the outside
        (inner_r, height),      # across the rim
        (inner_r, wall),        # down the inside
        (0.0, wall),            # across the cavity floor
        (0.0, 0.0),             # down the axis (zero-area, dropped)
    ]
    return profile


def build_open_vessel(dims: dict, segments: int, name: str) -> Mesh:
    return revolve(
        _open_vessel_profile(
            outer_d=dims["outer_d"], height=dims["height"],
            wall=dims["wall"], base_d=dims.get("base_d"),
        ),
        segments=segments, name=name,
    )


def build_conical_vessel(dims: dict, segments: int, name: str) -> Mesh:
    """Erlenmeyer: flat base, conical shoulder, cylindrical neck, open top."""
    base_r = dims["base_d"] / 2.0
    neck_r = dims["neck_d"] / 2.0
    height = dims["height"]
    neck_h = dims["neck_height"]
    wall = dims["wall"]

    shoulder_z = height - neck_h
    if shoulder_z <= wall:
        raise MeshError(f"{name}: neck_height {neck_h} leaves no body below the neck")
    if neck_r - wall <= 0:
        raise MeshError(f"{name}: wall {wall} closes off a {dims['neck_d']} m neck")

    # The inner surface is the outer surface offset horizontally by `wall`.
    # On the cone this gives a perpendicular thickness of wall*cos(taper), i.e.
    # very slightly thin -- 0.4% on a 250 mL Erlenmeyer. Recorded rather than
    # corrected: the error is far below the 5% scale tolerance and a true
    # normal offset would complicate the profile for no measurable gain.
    profile = [
        (base_r, 0.0),              # up the outside: base rim
        (neck_r, shoulder_z),       #   cone to the shoulder
        (neck_r, height),           #   up the neck
        (neck_r - wall, height),    # across the rim
        (neck_r - wall, shoulder_z),   # down the inside: neck
        (base_r - wall, wall),         #   cone back down to the floor
        (0.0, wall),                # across the cavity floor
        (0.0, 0.0),                 # down the axis (dropped)
    ]
    return revolve(profile, segments=segments, name=name)


def build_solid_cylinder(dims: dict, segments: int, name: str) -> Mesh:
    r, h = dims["outer_d"] / 2.0, dims["height"]
    return revolve([(r, 0.0), (r, h), (0.0, h), (0.0, 0.0)], segments=segments, name=name)


def build_box(dims: dict, name: str) -> Mesh:
    """Axis-aligned box, centred in X and Y, sitting on z = 0."""
    hx, hy, z = dims["x"] / 2.0, dims["y"] / 2.0, dims["z"]
    v = np.array([
        [-hx, -hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0],
        [-hx, -hy, z], [hx, -hy, z], [hx, hy, z], [-hx, hy, z],
    ])
    f = np.array([
        [0, 3, 2], [0, 2, 1],      # bottom  (-Z)
        [4, 5, 6], [4, 6, 7],      # top     (+Z)
        [0, 1, 5], [0, 5, 4],      # -Y
        [1, 2, 6], [1, 6, 5],      # +X
        [2, 3, 7], [2, 7, 6],      # +Y
        [3, 0, 4], [3, 4, 7],      # -X
    ], dtype=np.int32)
    return Mesh(v, f, name=name)


class _VertexPool:
    """Vertex list that welds coincident points.

    The rack's top face is tiled cell by cell, so neighbouring cells emit the
    same point on their shared edge. Without welding, every internal cell
    boundary becomes a pair of boundary edges and the mesh is a shell of
    disconnected patches -- it renders identically and is not watertight, which
    is exactly the kind of defect that survives a visual check and then makes
    an SDF bake or a convex decomposition produce nonsense.

    Welding is by coordinate at 1 nm, far below any dimension in the catalog
    and far above float64 noise on metre-scale values.
    """

    QUANTUM = 1e-9   # metres

    def __init__(self) -> None:
        self._index: dict[tuple[int, int, int], int] = {}
        self.vertices: list[tuple[float, float, float]] = []

    def add(self, x: float, y: float, z: float) -> int:
        kx = round(x / self.QUANTUM)
        ky = round(y / self.QUANTUM)
        kz = round(z / self.QUANTUM)
        # Probe the neighbouring quantum cells too. Two cells computing the
        # same shared-edge point by different-but-equivalent arithmetic can
        # differ by an ulp, and if that ulp happens to straddle a rounding
        # boundary a plain key lookup misses and the seam silently opens.
        for dx in (0, -1, 1):
            for dy in (0, -1, 1):
                for dz in (0, -1, 1):
                    hit = self._index.get((kx + dx, ky + dy, kz + dz))
                    if hit is not None:
                        return hit
        idx = len(self.vertices)
        self._index[(kx, ky, kz)] = idx
        self.vertices.append((x, y, z))
        return idx

    def add_ring(self, pts: np.ndarray) -> list[int]:
        return [self.add(float(p[0]), float(p[1]), float(p[2])) for p in pts]

    def as_array(self) -> np.ndarray:
        return np.asarray(self.vertices, dtype=np.float64)


def _cell_boundary(x0: float, x1: float, y0: float, y1: float,
                   per_edge: int, z: float) -> np.ndarray:
    """The cell rectangle [x0,x1] x [y0,y1], walked CCW from the (x1, y0) corner.

    Sampled by subdividing each edge between its two corners, **not** by
    casting rays from the cell centre. Ray casting was the first attempt and it
    is subtly wrong: the sample angles rarely land on the rectangle's corners,
    so consecutive samples chord across them and the cell polygons stop tiling
    the plane, leaving a diamond-shaped gap at every grid vertex.

    Subdividing between corners makes neighbouring cells agree because both
    derive the shared edge from the same grid-line coordinates. Corners are
    always included.
    """
    def lerp(a: float, b: float, n: int) -> np.ndarray:
        return a + (b - a) * (np.arange(n, dtype=np.float64) / n)

    xs = np.concatenate([
        np.full(per_edge, x1),          # right edge, going +y
        lerp(x1, x0, per_edge),         # top edge, going -x
        np.full(per_edge, x0),          # left edge, going -y
        lerp(x0, x1, per_edge),         # bottom edge, going +x
    ])
    ys = np.concatenate([
        lerp(y0, y1, per_edge),
        np.full(per_edge, y1),
        lerp(y1, y0, per_edge),
        np.full(per_edge, y0),
    ])
    return np.stack([xs, ys, np.full(len(xs), z)], axis=1)


def _boundary_loops(faces: list[tuple[int, int, int]]) -> list[list[int]]:
    """Ordered boundary loops of an open surface, following its winding."""
    directed = {(f[i], f[(i + 1) % 3]) for f in faces for i in range(3)}
    boundary = {a: b for (a, b) in directed if (b, a) not in directed}

    loops: list[list[int]] = []
    unused = dict(boundary)
    while unused:
        start = next(iter(unused))
        loop = [start]
        node = unused.pop(start)
        while node != start:
            loop.append(node)
            nxt = unused.pop(node, None)
            if nxt is None:                      # open chain: not a clean loop
                break
            node = nxt
        loops.append(loop)
    return loops


def build_tube_rack(dims: dict, segments: int, name: str) -> Mesh:
    """Rack: a box whose top face is punched with real blind holes.

    The holes are subtracted, not painted on. A rack with a flat top face lets
    a policy 'insert' a tube by resting it on the lid -- the same class of
    silent false success as hulling a beaker.

    Construction: tile the top face with one rectangle-minus-circle patch per
    hole, weld the shared cell edges, then extrude the outer boundary of that
    tiled face down to z = 0 and cap it. Building the sides from the actual top
    boundary rather than from four assumed corners is what keeps the result
    watertight -- the top face's perimeter has a vertex everywhere a cell edge
    meets it, and a four-corner box side would not.
    """
    x, y, z = dims["x"], dims["y"], dims["z"]
    rows, cols = int(dims["rows"]), int(dims["cols"])
    hole_r = dims["hole_d"] / 2.0
    depth = dims["hole_depth"]

    if depth >= z:
        raise MeshError(
            f"{name}: hole_depth {depth} m is not less than rack height {z} m. "
            f"A through-hole is a different shape; blind holes need a floor."
        )

    pitch_x, pitch_y = x / cols, y / rows
    if dims["hole_d"] >= min(pitch_x, pitch_y):
        raise MeshError(
            f"{name}: hole_d {dims['hole_d']} m does not fit a "
            f"{pitch_x:.4f} x {pitch_y:.4f} m grid cell "
            f"({cols} cols over {x} m, {rows} rows over {y} m). The holes would "
            f"overlap each other. This is a catalog dimension error, not a mesh "
            f"error -- do not shrink the holes to make it build, because then no "
            f"real tube fits. Check the manufacturer footprint."
        )

    # Capped and a multiple of 4 (one quarter per cell edge): 72 bores at 64
    # segments is 30k triangles for a part whose collider is a convex
    # decomposition anyway. 16 keeps an 18 mm bore round to 0.09 mm, well
    # inside the 1 mm support tolerance in validate.py.
    seg = min(segments, 16)
    seg -= seg % 4
    per_edge = seg // 4

    # Circle sampled from the same corner direction the cell ring starts at, so
    # index i on the rectangle pairs with a roughly radial index i on the hole.
    theta = np.linspace(0.0, 2.0 * math.pi, seg, endpoint=False) + math.atan2(-pitch_y, pitch_x)
    unit = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    hx, hy = x / 2.0, y / 2.0
    pool = _VertexPool()
    top_faces: list[tuple[int, int, int]] = []
    faces: list[tuple[int, int, int]] = []
    floor_z = z - depth

    for r in range(rows):
        for c in range(cols):
            # Grid lines, so the shared edge between two cells is derived from
            # identical arithmetic on both sides.
            x0, x1 = -hx + c * pitch_x, -hx + (c + 1) * pitch_x
            y0, y1 = -hy + r * pitch_y, -hy + (r + 1) * pitch_y
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

            outer = pool.add_ring(_cell_boundary(x0, x1, y0, y1, per_edge, z))
            rim = pool.add_ring(np.column_stack([
                cx + hole_r * unit[:, 0], cy + hole_r * unit[:, 1], np.full(seg, z)]))
            bore = pool.add_ring(np.column_stack([
                cx + hole_r * unit[:, 0], cy + hole_r * unit[:, 1], np.full(seg, floor_z)]))
            centre = pool.add(cx, cy, floor_z)

            for s in range(seg):
                t = (s + 1) % seg
                # Top face, +Z, CCW from above.
                top_faces.append((outer[s], outer[t], rim[t]))
                top_faces.append((outer[s], rim[t], rim[s]))
                # Bore wall: normals point into the bore, i.e. out of the solid,
                # exactly as a vessel's inner wall does.
                faces.append((rim[s], rim[t], bore[t]))
                faces.append((rim[s], bore[t], bore[s]))
                # Bore floor, +Z (you can see it from above, through the hole).
                faces.append((bore[s], bore[t], centre))

    # The top face's boundary is its outer perimeter plus one loop per bore
    # rim; the rims get closed by the bore walls already emitted, so only the
    # perimeter needs extruding. Anything beyond that count is an unwelded seam.
    loops = _boundary_loops(top_faces)
    expected = 1 + rows * cols
    if len(loops) != expected:
        raise MeshError(
            f"{name}: top face has {len(loops)} boundary loops, expected "
            f"{expected} (1 perimeter + {rows * cols} bore rims). Cell seams "
            f"failed to weld, so the rack would not be watertight."
        )
    perimeter = max(loops, key=len)

    faces.extend(top_faces)

    # Extrude the perimeter down to z = 0 and cap the bottom.
    lower = [pool.add(pool.vertices[i][0], pool.vertices[i][1], 0.0) for i in perimeter]
    n = len(perimeter)
    for s in range(n):
        t = (s + 1) % n
        faces.append((perimeter[s], lower[s], lower[t]))
        faces.append((perimeter[s], lower[t], perimeter[t]))

    base_centre = pool.add(0.0, 0.0, 0.0)
    for s in range(n):
        t = (s + 1) % n
        faces.append((base_centre, lower[t], lower[s]))   # -Z

    return Mesh(pool.as_array(), np.asarray(faces, dtype=np.int32), name=name)


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------

def build(item: CatalogItem | str, segments: int = RENDER_SEGMENTS) -> Mesh:
    """Render mesh for a catalog item, in metres, sitting on z = 0."""
    if isinstance(item, str):
        if item not in CATALOG:
            raise MeshError(f"{item!r} is not a catalog key")
        item = CATALOG[item]

    name = item.key
    if item.shape == "open_vessel":
        return build_open_vessel(item.dims, segments, name)
    if item.shape == "conical_vessel":
        return build_conical_vessel(item.dims, segments, name)
    if item.shape == "solid_cylinder":
        return build_solid_cylinder(item.dims, segments, name)
    if item.shape == "box":
        return build_box(item.dims, name)
    if item.shape == "tube_rack":
        return build_tube_rack(item.dims, segments, name)
    raise MeshError(f"{name}: no builder for shape {item.shape!r}")


def build_collision(item: CatalogItem | str,
                    segments: int = COLLISION_SEGMENTS) -> Mesh | None:
    """Lower-resolution collision proxy, or None if the render mesh must be used.

    None is returned for every SDF collider, and that is the important case.
    An SDF is baked *from* the mesh it is given, so handing it a decimated
    beaker moves the cavity wall inward by the decimation error. On a 33.5 mm
    inner radius at 24 segments the chord sag is about 0.29 mm, which is a
    quarter of the 1 mm support tolerance in `validate.py` and is spent for
    nothing. Open vessels are cheap meshes already.

    Decimation is only offered where the collider is a primitive or a convex
    decomposition, neither of which is sensitive to the extra triangles in the
    way an SDF cavity is.
    """
    if isinstance(item, str):
        item = CATALOG[item]

    if item.collider in ("sdf", "box", "cylinder"):
        # sdf: see above. box/cylinder: the collider is an analytic primitive,
        # so a proxy mesh is never consulted at all.
        return None
    if segments >= RENDER_SEGMENTS:
        return None

    proxy = build(item, segments=segments)
    if len(proxy.faces) >= len(build(item).faces):
        # Some builders already cap their own resolution (the rack does, since
        # 72 bores at full segments is absurd), so the "reduced" mesh comes out
        # the same size. Shipping a second identical file helps nobody.
        return None
    return proxy


def build_all(segments: int = RENDER_SEGMENTS,
              skip_unbuildable: bool = False) -> dict[str, Mesh]:
    """Every catalog item, so a regression can diff the whole set at once.

    Raises on the first item whose dimensions cannot produce a mesh. Pass
    `skip_unbuildable=True` only for reporting over the whole catalog -- never
    on a path that feeds a scene, because the item that got skipped is exactly
    the one someone needs to be told about.
    """
    out: dict[str, Mesh] = {}
    for key, item in CATALOG.items():
        try:
            out[key] = build(item, segments=segments)
        except MeshError:
            if not skip_unbuildable:
                raise
    return out


# --------------------------------------------------------------------------
# cavity measurement
# --------------------------------------------------------------------------

def cavity_volume_m3(item: CatalogItem | str, segments: int = 256) -> float:
    """Brim-full interior volume of a hollow vessel, from the generated cavity.

    Computed as (solid envelope) - (mesh volume): the mesh encloses only the
    glass, so subtracting it from the volume of the same profile with the
    cavity filled in leaves exactly the cavity. Uses a high segment count
    because this is a measurement, not a render -- at 64 segments a circle is
    under-measured by 0.08%, which is small but pointless to carry.

    This is **brim** volume. A Griffin beaker's marked capacity sits well below
    its rim, so brim volume exceeding `capacity_ml` is expected and correct;
    see `tests/test_meshes.py` for what is actually asserted.
    """
    if isinstance(item, str):
        item = CATALOG[item]
    if item.shape not in ("open_vessel", "conical_vessel"):
        raise MeshError(f"{item.key}: {item.shape} has no cavity")

    hollow = build(item, segments=segments).volume_m3()

    d = item.dims
    if item.shape == "open_vessel":
        outer_r = d["outer_d"] / 2.0
        base_r = max(d.get("base_d", d["outer_d"]) / 2.0, outer_r)
        profile = [(base_r, 0.0)]
        if base_r > outer_r + 1e-9:
            foot_h = min(d["wall"] * 2.0, d["height"] * 0.02)
            profile = [(base_r, 0.0), (base_r, foot_h), (outer_r, foot_h)]
        profile += [(outer_r, d["height"]), (0.0, d["height"]), (0.0, 0.0)]
    else:
        shoulder_z = d["height"] - d["neck_height"]
        profile = [
            (d["base_d"] / 2.0, 0.0),
            (d["neck_d"] / 2.0, shoulder_z),
            (d["neck_d"] / 2.0, d["height"]),
            (0.0, d["height"]),
            (0.0, 0.0),
        ]
    solid = revolve(profile, segments=segments, name=f"{item.key}_solid").volume_m3()
    return solid - hollow
