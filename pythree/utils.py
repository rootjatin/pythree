# pythree/utils.py
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

from .sphere import Mesh  # type: ignore

Vec3 = Tuple[float, float, float]
Tri = Tuple[int, int, int]
Bounds = Tuple[Vec3, Vec3]


# -------------------------
# Basic vector math
# -------------------------

def vec_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_mul(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def vec_div(a: Vec3, s: float) -> Vec3:
    if s == 0:
        raise ZeroDivisionError("Cannot divide vector by zero")
    return (a[0] / s, a[1] / s, a[2] / s)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def length(v: Vec3) -> float:
    return math.sqrt(dot(v, v))


def distance(a: Vec3, b: Vec3) -> float:
    return length(vec_sub(a, b))


def normalize(v: Vec3, *, eps: float = 1e-12) -> Vec3:
    n = length(v)
    if n <= eps:
        return (0.0, 0.0, 0.0)
    return vec_div(v, n)


def lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def is_finite_vec3(v: Vec3) -> bool:
    return math.isfinite(v[0]) and math.isfinite(v[1]) and math.isfinite(v[2])


# -------------------------
# Triangle helpers
# -------------------------

def triangle_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    """
    Return the unit normal of triangle (a, b, c).
    Degenerate triangles return (0,0,0).
    """
    ab = vec_sub(b, a)
    ac = vec_sub(c, a)
    return normalize(cross(ab, ac))


def triangle_area(a: Vec3, b: Vec3, c: Vec3) -> float:
    """
    Return the area of triangle (a, b, c).
    """
    ab = vec_sub(b, a)
    ac = vec_sub(c, a)
    return 0.5 * length(cross(ab, ac))


def triangle_centroid(a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    return (
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    )


# -------------------------
# Bounds / bbox helpers
# -------------------------

def bounds_from_points(points: Sequence[Vec3]) -> Bounds:
    """
    Compute axis-aligned bounds from a sequence of 3D points.
    """
    if not points:
        raise ValueError("points must not be empty")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]

    return (
        (min(xs), min(ys), min(zs)),
        (max(xs), max(ys), max(zs)),
    )


def bounds_union(a: Bounds, b: Bounds) -> Bounds:
    """
    Return the union of two axis-aligned bounding boxes.
    """
    (ax0, ay0, az0), (ax1, ay1, az1) = a
    (bx0, by0, bz0), (bx1, by1, bz1) = b

    return (
        (min(ax0, bx0), min(ay0, by0), min(az0, bz0)),
        (max(ax1, bx1), max(ay1, by1), max(az1, bz1)),
    )


def bounds_size(bounds: Bounds) -> Vec3:
    (mn, mx) = bounds
    return (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])


def bounds_center(bounds: Bounds) -> Vec3:
    (mn, mx) = bounds
    return (
        0.5 * (mn[0] + mx[0]),
        0.5 * (mn[1] + mx[1]),
        0.5 * (mn[2] + mx[2]),
    )


# -------------------------
# Mesh inspection helpers
# -------------------------

def mesh_bounds(mesh: Mesh) -> Bounds:
    if not mesh.verts:
        raise ValueError("mesh has no vertices")
    return bounds_from_points(mesh.verts)


def mesh_centroid(mesh: Mesh) -> Vec3:
    """
    Vertex-average centroid.
    """
    if not mesh.verts:
        raise ValueError("mesh has no vertices")

    sx = sy = sz = 0.0
    for x, y, z in mesh.verts:
        sx += x
        sy += y
        sz += z

    n = float(len(mesh.verts))
    return (sx / n, sy / n, sz / n)


def mesh_surface_area(mesh: Mesh) -> float:
    """
    Sum of triangle areas.
    """
    area = 0.0
    for i, j, k in mesh.faces:
        a = mesh.verts[i]
        b = mesh.verts[j]
        c = mesh.verts[k]
        area += triangle_area(a, b, c)
    return area


def mesh_stats(mesh: Mesh) -> dict:
    """
    Return basic mesh stats as a dictionary.
    """
    stats = {
        "name": getattr(mesh, "name", "mesh"),
        "vertex_count": len(mesh.verts),
        "face_count": len(mesh.faces),
        "is_empty": len(mesh.verts) == 0 or len(mesh.faces) == 0,
    }

    if mesh.verts:
        b = mesh_bounds(mesh)
        stats["bounds"] = b
        stats["size"] = bounds_size(b)
        stats["center"] = bounds_center(b)
        stats["centroid"] = mesh_centroid(mesh)
    else:
        stats["bounds"] = None
        stats["size"] = None
        stats["center"] = None
        stats["centroid"] = None

    if mesh.faces:
        stats["surface_area"] = mesh_surface_area(mesh)
    else:
        stats["surface_area"] = 0.0

    return stats


# -------------------------
# Mesh validation
# -------------------------

def validate_faces(mesh: Mesh) -> None:
    """
    Raise ValueError if any face has invalid indices.
    """
    n = len(mesh.verts)
    for fi, face in enumerate(mesh.faces):
        if len(face) != 3:
            raise ValueError(f"Face {fi} is not a triangle: {face}")

        i, j, k = face
        if not (0 <= i < n and 0 <= j < n and 0 <= k < n):
            raise ValueError(
                f"Face {fi} contains out-of-range vertex index: {face} (vertex_count={n})"
            )


def validate_vertices(mesh: Mesh) -> None:
    """
    Raise ValueError if any vertex is not finite.
    """
    for vi, v in enumerate(mesh.verts):
        if not is_finite_vec3(v):
            raise ValueError(f"Vertex {vi} is not finite: {v}")


def validate_mesh(mesh: Mesh) -> None:
    validate_vertices(mesh)
    validate_faces(mesh)


# -------------------------
# Mesh transforms
# -------------------------

def translated(mesh: Mesh, offset: Vec3, *, name: str | None = None) -> Mesh:
    dx, dy, dz = offset
    verts = [(x + dx, y + dy, z + dz) for x, y, z in mesh.verts]
    faces = list(mesh.faces)
    return Mesh(verts, faces, name=name or getattr(mesh, "name", "mesh"))


def scaled(
    mesh: Mesh,
    scale: Vec3 | float,
    *,
    center: Vec3 = (0.0, 0.0, 0.0),
    name: str | None = None,
) -> Mesh:
    if isinstance(scale, (int, float)):
        sx = sy = sz = float(scale)
    else:
        sx, sy, sz = scale

    cx, cy, cz = center

    verts = [
        (
            cx + (x - cx) * sx,
            cy + (y - cy) * sy,
            cz + (z - cz) * sz,
        )
        for x, y, z in mesh.verts
    ]
    faces = list(mesh.faces)
    return Mesh(verts, faces, name=name or getattr(mesh, "name", "mesh"))


def recentered(mesh: Mesh, *, name: str | None = None) -> Mesh:
    """
    Return a copy of the mesh translated so its centroid is at the origin.
    """
    c = mesh_centroid(mesh)
    return translated(mesh, (-c[0], -c[1], -c[2]), name=name)


# -------------------------
# Mesh combination
# -------------------------

def merge_meshes(meshes: Iterable[Mesh], *, name: str = "merged") -> Mesh:
    """
    Merge multiple meshes into one mesh.

    Vertex indices in faces are re-based automatically.
    """
    verts: List[Vec3] = []
    faces: List[Tri] = []

    offset = 0
    count = 0

    for mesh in meshes:
        count += 1
        verts.extend(mesh.verts)
        faces.extend((a + offset, b + offset, c + offset) for a, b, c in mesh.faces)
        offset += len(mesh.verts)

    if count == 0:
        return Mesh([], [], name=name)

    return Mesh(verts, faces, name=name)


# -------------------------
# Mesh cleanup helpers
# -------------------------

def remove_degenerate_faces(mesh: Mesh, *, eps: float = 1e-12, name: str | None = None) -> Mesh:
    """
    Remove triangles with repeated indices or near-zero area.
    """
    faces: List[Tri] = []

    for i, j, k in mesh.faces:
        if i == j or j == k or i == k:
            continue

        a = mesh.verts[i]
        b = mesh.verts[j]
        c = mesh.verts[k]

        if triangle_area(a, b, c) <= eps:
            continue

        faces.append((i, j, k))

    return Mesh(list(mesh.verts), faces, name=name or getattr(mesh, "name", "mesh"))


def unique_edges(mesh: Mesh) -> List[Tuple[int, int]]:
    """
    Return sorted unique undirected edges from triangle faces.
    """
    edges = set()

    for i, j, k in mesh.faces:
        e0 = (i, j) if i < j else (j, i)
        e1 = (j, k) if j < k else (k, j)
        e2 = (k, i) if k < i else (i, k)
        edges.add(e0)
        edges.add(e1)
        edges.add(e2)

    return sorted(edges)


# -------------------------
# Convenience
# -------------------------

def print_mesh_stats(mesh: Mesh) -> None:
    s = mesh_stats(mesh)
    print(f"Mesh: {s['name']}")
    print(f"  vertices: {s['vertex_count']}")
    print(f"  faces: {s['face_count']}")
    print(f"  empty: {s['is_empty']}")
    print(f"  bounds: {s['bounds']}")
    print(f"  size: {s['size']}")
    print(f"  center: {s['center']}")
    print(f"  centroid: {s['centroid']}")
    print(f"  surface_area: {s['surface_area']}")
