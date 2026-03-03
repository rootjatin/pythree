# pythree/mesh.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

Vec3 = Tuple[float, float, float]
Tri = Tuple[int, int, int]


def _vec_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_mul(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(v: Vec3) -> float:
    return math.sqrt(_dot(v, v))


def _normalize(v: Vec3, *, eps: float = 1e-12) -> Vec3:
    n = _length(v)
    if n <= eps:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _triangle_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    ab = _vec_sub(b, a)
    ac = _vec_sub(c, a)
    return _normalize(_cross(ab, ac))


def _triangle_area_weighted_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    """
    Returns the unnormalized face normal.
    Its magnitude is 2 * triangle area, which is useful for
    area-weighted vertex normal accumulation.
    """
    ab = _vec_sub(b, a)
    ac = _vec_sub(c, a)
    return _cross(ab, ac)


@dataclass
class Mesh:
    """
    Basic triangle mesh container.

    Attributes
    ----------
    verts:
        List of vertex positions as (x, y, z).
    faces:
        List of triangle indices as (i, j, k).
    name:
        Human-readable mesh name.
    normals:
        Optional per-vertex normals. Usually computed via compute_normals(True).
    face_normals:
        Optional per-face normals. Usually computed via compute_normals(False).
    """

    verts: List[Vec3]
    faces: List[Tri]
    name: str = "mesh"
    normals: Optional[List[Vec3]] = field(default=None)
    face_normals: Optional[List[Vec3]] = field(default=None)

    def __post_init__(self) -> None:
        self.verts = list(self.verts)
        self.faces = [tuple(face) for face in self.faces]

    def copy(self, *, name: Optional[str] = None) -> "Mesh":
        return Mesh(
            verts=list(self.verts),
            faces=list(self.faces),
            name=self.name if name is None else name,
            normals=None if self.normals is None else list(self.normals),
            face_normals=None if self.face_normals is None else list(self.face_normals),
        )

    def __len__(self) -> int:
        return len(self.verts)

    @property
    def vertex_count(self) -> int:
        return len(self.verts)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    def is_empty(self) -> bool:
        return not self.verts or not self.faces

    def validate(self) -> "Mesh":
        n = len(self.verts)

        for vi, v in enumerate(self.verts):
            if len(v) != 3:
                raise ValueError(f"Vertex {vi} is not a 3D tuple: {v}")
            if not all(math.isfinite(x) for x in v):
                raise ValueError(f"Vertex {vi} is not finite: {v}")

        for fi, f in enumerate(self.faces):
            if len(f) != 3:
                raise ValueError(f"Face {fi} is not a triangle: {f}")

            i, j, k = f
            if not (0 <= i < n and 0 <= j < n and 0 <= k < n):
                raise ValueError(
                    f"Face {fi} contains out-of-range indices: {f} (vertex_count={n})"
                )

        return self

    def bounds(self) -> Tuple[Vec3, Vec3]:
        if not self.verts:
            raise ValueError("Cannot compute bounds of an empty mesh")

        xs = [v[0] for v in self.verts]
        ys = [v[1] for v in self.verts]
        zs = [v[2] for v in self.verts]

        return (
            (min(xs), min(ys), min(zs)),
            (max(xs), max(ys), max(zs)),
        )

    def centroid(self) -> Vec3:
        if not self.verts:
            raise ValueError("Cannot compute centroid of an empty mesh")

        sx = sy = sz = 0.0
        for x, y, z in self.verts:
            sx += x
            sy += y
            sz += z

        n = float(len(self.verts))
        return (sx / n, sy / n, sz / n)

    def translated(self, dx: float, dy: float, dz: float, *, name: Optional[str] = None) -> "Mesh":
        verts = [(x + dx, y + dy, z + dz) for x, y, z in self.verts]
        return Mesh(verts, list(self.faces), name=name or self.name)

    def scaled(
        self,
        sx: float,
        sy: Optional[float] = None,
        sz: Optional[float] = None,
        *,
        center: Vec3 = (0.0, 0.0, 0.0),
        name: Optional[str] = None,
    ) -> "Mesh":
        if sy is None:
            sy = sx
        if sz is None:
            sz = sx

        cx, cy, cz = center
        verts = [
            (
                cx + (x - cx) * sx,
                cy + (y - cy) * sy,
                cz + (z - cz) * sz,
            )
            for x, y, z in self.verts
        ]
        return Mesh(verts, list(self.faces), name=name or self.name)

    def compute_normals(self, smooth: bool = True) -> "Mesh":
        """
        Compute normals.

        Parameters
        ----------
        smooth:
            If True:
                compute per-vertex normals by averaging adjacent face normals.
                Result is stored in self.normals.
            If False:
                compute per-face normals only.
                Result is stored in self.face_normals.

        Returns
        -------
        Mesh
            Returns self for chaining.
        """
        if not self.verts:
            self.normals = []
            self.face_normals = []
            return self

        if not self.faces:
            self.normals = [(0.0, 0.0, 0.0) for _ in self.verts]
            self.face_normals = []
            return self

        self.face_normals = []
        for i, j, k in self.faces:
            a = self.verts[i]
            b = self.verts[j]
            c = self.verts[k]
            self.face_normals.append(_triangle_normal(a, b, c))

        if not smooth:
            self.normals = None
            return self

        accum: List[Vec3] = [(0.0, 0.0, 0.0) for _ in self.verts]

        for (i, j, k) in self.faces:
            a = self.verts[i]
            b = self.verts[j]
            c = self.verts[k]

            # area-weighted accumulation gives better results than
            # averaging already-normalized face normals
            wn = _triangle_area_weighted_normal(a, b, c)

            accum[i] = _vec_add(accum[i], wn)
            accum[j] = _vec_add(accum[j], wn)
            accum[k] = _vec_add(accum[k], wn)

        self.normals = [_normalize(n) for n in accum]
        return self

    def surface_area(self) -> float:
        area = 0.0
        for i, j, k in self.faces:
            a = self.verts[i]
            b = self.verts[j]
            c = self.verts[k]
            area += 0.5 * _length(_cross(_vec_sub(b, a), _vec_sub(c, a)))
        return area

    def append(self, other: "Mesh") -> "Mesh":
        """
        Append another mesh into this one in-place.
        """
        offset = len(self.verts)
        self.verts.extend(other.verts)
        self.faces.extend((a + offset, b + offset, c + offset) for a, b, c in other.faces)

        # normals become stale after topology changes
        self.normals = None
        self.face_normals = None
        return self

    @classmethod
    def merged(cls, meshes: Iterable["Mesh"], *, name: str = "merged") -> "Mesh":
        verts: List[Vec3] = []
        faces: List[Tri] = []

        offset = 0
        for mesh in meshes:
            verts.extend(mesh.verts)
            faces.extend((a + offset, b + offset, c + offset) for a, b, c in mesh.faces)
            offset += len(mesh.verts)

        return cls(verts, faces, name=name)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "verts": list(self.verts),
            "faces": list(self.faces),
            "normals": None if self.normals is None else list(self.normals),
            "face_normals": None if self.face_normals is None else list(self.face_normals),
        }

    def __repr__(self) -> str:
        return (
            f"Mesh(name={self.name!r}, "
            f"verts={len(self.verts)}, "
            f"faces={len(self.faces)})"
        )
