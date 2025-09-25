"""
geomesh: a tiny, dependency‑free Python toolkit for building, transforming,
and exporting triangle meshes (OBJ / binary STL / ASCII PLY).

Highlights
---------
• Mesh class with vertices, faces, normals, uvs, and name
• Clean transform API (translate / rotate / scale / arbitrary 4×4 matrix)
• Mesh composition (merge / + operator), bounding box, vertex welding
• Procedural primitives: uv_sphere, ico_sphere, cube, plane, cylinder,
  cone, torus, grid, capsule, parametric surface, polygon extrusion
• Normal generation (smooth or flat) and basic spherical/cylindrical UVs
• Exporters: Wavefront OBJ (with vt/vn), binary STL, ASCII PLY
• Tiny CLI to quickly generate shapes

Pure Python, no numpy required. Python 3.8+ recommended.

This module is intentionally compact and commented so you can extend it
into a full package (e.g., add glTF export, boolean ops via external libs,
bezier surfaces, etc.).
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]
Tri = Tuple[int, int, int]
Mat4 = List[List[float]]

# -----------------------------
# Small vector/matrix utilities
# -----------------------------

def v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def v_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def v_len(a: Vec3) -> float:
    return math.sqrt(v_dot(a, a))


def v_norm(a: Vec3) -> Vec3:
    l = v_len(a)
    if l == 0:
        return (0.0, 0.0, 0.0)
    return (a[0] / l, a[1] / l, a[2] / l)


def mat_identity() -> Mat4:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat_mul(a: Mat4, b: Mat4) -> Mat4:
    out = [[0.0] * 4 for _ in range(4)]
    for r in range(4):
        for c in range(4):
            out[r][c] = (
                a[r][0] * b[0][c]
                + a[r][1] * b[1][c]
                + a[r][2] * b[2][c]
                + a[r][3] * b[3][c]
            )
    return out


def mat_translate(dx: float, dy: float, dz: float) -> Mat4:
    m = mat_identity()
    m[0][3], m[1][3], m[2][3] = dx, dy, dz
    return m


def mat_scale(sx: float, sy: Optional[float] = None, sz: Optional[float] = None) -> Mat4:
    if sy is None:
        sy = sx
    if sz is None:
        sz = sx
    m = mat_identity()
    m[0][0], m[1][1], m[2][2] = sx, sy, sz
    return m


def mat_rotate_x(a: float) -> Mat4:
    c, s = math.cos(a), math.sin(a)
    return [
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1],
    ]


def mat_rotate_y(a: float) -> Mat4:
    c, s = math.cos(a), math.sin(a)
    return [
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1],
    ]


def mat_rotate_z(a: float) -> Mat4:
    c, s = math.cos(a), math.sin(a)
    return [
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


def apply_mat(v: Vec3, m: Mat4) -> Vec3:
    x, y, z = v
    # v' = M * [x, y, z, 1]
    xp = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3]
    yp = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3]
    zp = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3]
    wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3]
    if wp != 0 and wp != 1:
        return (xp / wp, yp / wp, zp / wp)
    return (xp, yp, zp)


# --------------
# Mesh container
# --------------

@dataclass
class Mesh:
    vertices: List[Vec3] = field(default_factory=list)
    faces: List[Tri] = field(default_factory=list)
    normals: Optional[List[Vec3]] = None  # aligned 1:1 with vertices when present
    uvs: Optional[List[Vec2]] = None      # aligned 1:1 with vertices when present
    name: str = "mesh"

    # ---- transforms ----
    def transform(self, m: Mat4) -> "Mesh":
        self.vertices = [apply_mat(v, m) for v in self.vertices]
        if self.normals:
            # ignore translation; apply rotation+scale via inverse-transpose of 3x3
            # For simplicity, assuming uniform scale — normalize afterward.
            self.normals = [v_norm(apply_mat(n, m)) for n in self.normals]
        return self

    def translated(self, dx: float, dy: float, dz: float) -> "Mesh":
        return self.copy().transform(mat_translate(dx, dy, dz))

    def translate(self, dx: float, dy: float, dz: float) -> "Mesh":
        return self.transform(mat_translate(dx, dy, dz))

    def scaled(self, sx: float, sy: Optional[float] = None, sz: Optional[float] = None) -> "Mesh":
        return self.copy().transform(mat_scale(sx, sy, sz))

    def scale(self, sx: float, sy: Optional[float] = None, sz: Optional[float] = None) -> "Mesh":
        return self.transform(mat_scale(sx, sy, sz))

    def rotate_x(self, a: float) -> "Mesh":
        return self.transform(mat_rotate_x(a))

    def rotate_y(self, a: float) -> "Mesh":
        return self.transform(mat_rotate_y(a))

    def rotate_z(self, a: float) -> "Mesh":
        return self.transform(mat_rotate_z(a))

    def copy(self) -> "Mesh":
        return Mesh(self.vertices.copy(), self.faces.copy(),
                    None if self.normals is None else self.normals.copy(),
                    None if self.uvs is None else self.uvs.copy(),
                    self.name)

    # ---- composition ----
    def merge(self, other: "Mesh", rename: bool = False) -> "Mesh":
        offset = len(self.vertices)
        self.vertices.extend(other.vertices)
        self.faces.extend([(a + offset, b + offset, c + offset) for (a, b, c) in other.faces])
        if self.normals is not None and other.normals is not None:
            self.normals.extend(other.normals)
        else:
            self.normals = None  # mixed state; recompute if needed
        if self.uvs is not None and other.uvs is not None:
            self.uvs.extend(other.uvs)
        else:
            self.uvs = None
        if rename:
            self.name += "+" + other.name
        return self

    def __add__(self, other: "Mesh") -> "Mesh":
        return self.copy().merge(other)

    # ---- analysis ----
    def bounds(self) -> Tuple[Vec3, Vec3]:
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    def weld(self, eps: float = 1e-7) -> "Mesh":
        """Deduplicate vertices; updates faces accordingly. Keeps first occurrence.
        eps: positional tolerance.
        """
        key = lambda v: (round(v[0] / eps), round(v[1] / eps), round(v[2] / eps))
        index_map = {}
        new_verts: List[Vec3] = []
        new_norms: Optional[List[Vec3]] = [] if self.normals is not None else None
        new_uvs: Optional[List[Vec2]] = [] if self.uvs is not None else None
        for i, v in enumerate(self.vertices):
            k = key(v)
            if k in index_map:
                index_map[i] = index_map[k]  # map to existing
            else:
                index_map[k] = len(new_verts)
                index_map[i] = index_map[k]
                new_verts.append(v)
                if new_norms is not None and self.normals is not None:
                    new_norms.append(self.normals[i])
                if new_uvs is not None and self.uvs is not None:
                    new_uvs.append(self.uvs[i])
        self.vertices = new_verts
        if new_norms is not None:
            self.normals = new_norms
        if new_uvs is not None:
            self.uvs = new_uvs
        self.faces = [(index_map[a], index_map[b], index_map[c]) for (a, b, c) in self.faces]
        return self

    # ---- shading ----
    def compute_normals(self, smooth: bool = True) -> "Mesh":
        if not self.vertices or not self.faces:
            self.normals = []
            return self
        if not smooth:
            # Flat: duplicate vertices per face so vn aligns; simpler export
            v_out: List[Vec3] = []
            n_out: List[Vec3] = []
            f_out: List[Tri] = []
            for (a, b, c) in self.faces:
                va, vb, vc = self.vertices[a], self.vertices[b], self.vertices[c]
                n = v_norm(v_cross(v_sub(vb, va), v_sub(vc, va)))
                base = len(v_out)
                v_out.extend([va, vb, vc])
                n_out.extend([n, n, n])
                f_out.append((base, base + 1, base + 2))
            self.vertices, self.faces, self.normals = v_out, f_out, n_out
            return self
        # Smooth: average adjacent face normals per vertex
        normals = [(0.0, 0.0, 0.0) for _ in self.vertices]
        for (a, b, c) in self.faces:
            va, vb, vc = self.vertices[a], self.vertices[b], self.vertices[c]
            n = v_norm(v_cross(v_sub(vb, va), v_sub(vc, va)))
            normals[a] = v_add(normals[a], n)
            normals[b] = v_add(normals[b], n)
            normals[c] = v_add(normals[c], n)
        self.normals = [v_norm(n) for n in normals]
        return self

    # ---- UV mapping helpers ----
    def spherical_uv(self) -> "Mesh":
        """Assigns spherical (longitude/latitude) UVs aligned with vertices."""
        if not self.vertices:
            self.uvs = []
            return self
        uvs: List[Vec2] = []
        for x, y, z in self.vertices:
            r = math.sqrt(x * x + y * y + z * z) or 1.0
            theta = math.acos(max(-1.0, min(1.0, z / r)))  # 0..pi
            phi = (math.atan2(y, x) + 2 * math.pi) % (2 * math.pi)  # 0..2pi
            u = phi / (2 * math.pi)
            v = 1.0 - theta / math.pi
            uvs.append((u, v))
        self.uvs = uvs
        return self

    def cylindrical_uv(self, axis: str = "y") -> "Mesh":
        if not self.vertices:
            self.uvs = []
            return self
        uvs: List[Vec2] = []
        for x, y, z in self.vertices:
            if axis == "y":
                ang = (math.atan2(z, x) + 2 * math.pi) % (2 * math.pi)
                u = ang / (2 * math.pi)
                v = (y - self.bounds()[0][1]) / max(1e-9, self.bounds()[1][1] - self.bounds()[0][1])
            elif axis == "x":
                ang = (math.atan2(y, z) + 2 * math.pi) % (2 * math.pi)
                u = ang / (2 * math.pi)
                v = (x - self.bounds()[0][0]) / max(1e-9, self.bounds()[1][0] - self.bounds()[0][0])
            else:  # z
                ang = (math.atan2(y, x) + 2 * math.pi) % (2 * math.pi)
                u = ang / (2 * math.pi)
                v = (z - self.bounds()[0][2]) / max(1e-9, self.bounds()[1][2] - self.bounds()[0][2])
            uvs.append((u, v))
        self.uvs = uvs
        return self


# -----------------------
# Primitive constructors
# -----------------------

def uv_sphere(radius: float = 1.0, segments: int = 32, rings: Optional[int] = None, name: str = "uv_sphere") -> Mesh:
    if rings is None:
        rings = segments // 2
    verts: List[Vec3] = []
    faces: List[Tri] = []
    for i in range(rings + 1):
        v = i / rings
        theta = v * math.pi  # 0..pi
        st, ct = math.sin(theta), math.cos(theta)
        for j in range(segments + 1):
            u = j / segments
            phi = u * 2.0 * math.pi
            sp, cp = math.sin(phi), math.cos(phi)
            x = radius * st * cp
            y = radius * st * sp
            z = radius * ct
            verts.append((x, y, z))
            if i < rings and j < segments:
                a = i * (segments + 1) + j
                b = a + 1
                c = (i + 1) * (segments + 1) + j
                d = c + 1
                faces.append((a, c, b))
                faces.append((b, c, d))
    m = Mesh(verts, faces, name=name)
    return m.compute_normals(smooth=True).spherical_uv()


def _midpoint_cache_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


def ico_sphere(radius: float = 1.0, subdivisions: int = 2, name: str = "ico_sphere") -> Mesh:
    # Based on subdividing an icosahedron
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts: List[Vec3] = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    verts = [v_norm(v) for v in verts]
    faces: List[Tri] = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]

    midpoint_cache = {}

    def midpoint(i: int, j: int) -> int:
        key = _midpoint_cache_key(i, j)
        if key in midpoint_cache:
            return midpoint_cache[key]
        vi, vj = verts[i], verts[j]
        vm = v_norm(((vi[0] + vj[0]) * 0.5, (vi[1] + vj[1]) * 0.5, (vi[2] + vj[2]) * 0.5))
        verts.append(vm)
        k = len(verts) - 1
        midpoint_cache[key] = k
        return k

    for _ in range(subdivisions):
        new_faces: List[Tri] = []
        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces += [
                (a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)
            ]
        faces = new_faces

    # project to sphere
    verts = [v_scale(v_norm(v), radius) for v in verts]
    m = Mesh(verts, faces, name=name).compute_normals(smooth=True).spherical_uv()
    return m


def plane(size_x: float = 1.0, size_y: float = 1.0, segments_x: int = 1, segments_y: int = 1, name: str = "plane") -> Mesh:
    verts: List[Vec3] = []
    faces: List[Tri] = []
    for iy in range(segments_y + 1):
        v = iy / segments_y
        y = (v - 0.5) * size_y
        for ix in range(segments_x + 1):
            u = ix / segments_x
            x = (u - 0.5) * size_x
            verts.append((x, 0.0, y))
            if ix < segments_x and iy < segments_y:
                a = iy * (segments_x + 1) + ix
                b = a + 1
                c = (iy + 1) * (segments_x + 1) + ix
                d = c + 1
                faces.append((a, c, b))
                faces.append((b, c, d))
    m = Mesh(verts, faces, name=name)
    return m.compute_normals(smooth=True)


def cube(size: float = 1.0, name: str = "cube") -> Mesh:
    # Build from 6 planes to get decent normals/uvs
    s = size
    px = plane(s, s, 1, 1, "px").rotate_z(-math.pi / 2).rotate_y(math.pi / 2).translate(s / 2, 0, 0)
    nx = plane(s, s, 1, 1, "nx").rotate_z(math.pi / 2).rotate_y(-math.pi / 2).translate(-s / 2, 0, 0)
    py = plane(s, s, 1, 1, "py").rotate_x(-math.pi / 2).translate(0, s / 2, 0)
    ny = plane(s, s, 1, 1, "ny").rotate_x(math.pi / 2).translate(0, -s / 2, 0)
    pz = plane(s, s, 1, 1, "pz").translate(0, 0, s / 2)
    nz = plane(s, s, 1, 1, "nz").rotate_y(math.pi).translate(0, 0, -s / 2)
    return Mesh(name=name).merge(px).merge(nx).merge(py).merge(ny).merge(pz).merge(nz)


def cylinder(radius: float = 0.5, height: float = 1.0, radial_segments: int = 32, height_segments: int = 1,
             cap_top: bool = True, cap_bottom: bool = True, name: str = "cylinder") -> Mesh:
    verts: List[Vec3] = []
    faces: List[Tri] = []
    # side
    for iy in range(height_segments + 1):
        v = iy / height_segments
        y = (v - 0.5) * height
        for ix in range(radial_segments + 1):
            u = ix / radial_segments
            ang = u * 2 * math.pi
            x = radius * math.cos(ang)
            z = radius * math.sin(ang)
            verts.append((x, y, z))
            if ix < radial_segments and iy < height_segments:
                a = iy * (radial_segments + 1) + ix
                b = a + 1
                c = (iy + 1) * (radial_segments + 1) + ix
                d = c + 1
                faces.append((a, c, b))
                faces.append((b, c, d))
    # caps
    base_index = len(verts)
    if cap_top:
        y = height / 2
        center_top = len(verts)
        verts.append((0.0, y, 0.0))
        for i in range(radial_segments):
            ang0 = i * 2 * math.pi / radial_segments
            ang1 = (i + 1) * 2 * math.pi / radial_segments
            v0 = (radius * math.cos(ang0), y, radius * math.sin(ang0))
            v1 = (radius * math.cos(ang1), y, radius * math.sin(ang1))
            verts.extend([v0, v1])
            a = center_top
            b = center_top + 1 + (i * 2)
            c = center_top + 2 + (i * 2)
            faces.append((a, b, c))
    if cap_bottom:
        y = -height / 2
        center_bottom = len(verts)
        verts.append((0.0, y, 0.0))
        for i in range(radial_segments):
            ang0 = i * 2 * math.pi / radial_segments
            ang1 = (i + 1) * 2 * math.pi / radial_segments
            v0 = (radius * math.cos(ang0), y, radius * math.sin(ang0))
            v1 = (radius * math.cos(ang1), y, radius * math.sin(ang1))
            verts.extend([v1, v0])  # reverse winding for bottom
            a = center_bottom
            b = center_bottom + 1 + (i * 2)
            c = center_bottom + 2 + (i * 2)
            faces.append((a, b, c))
    m = Mesh(verts, faces, name=name)
    m.compute_normals(smooth=True)
    return m


def cone(radius: float = 0.5, height: float = 1.0, radial_segments: int = 32, cap: bool = True, name: str = "cone") -> Mesh:
    verts: List[Vec3] = []
    faces: List[Tri] = []
    # side
    for ix in range(radial_segments + 1):
        u = ix / radial_segments
        ang = u * 2 * math.pi
        x = radius * math.cos(ang)
        z = radius * math.sin(ang)
        verts.append((x, -height / 2, z))
        if ix < radial_segments:
            a = ix
            b = ix + 1
            c = radial_segments + 1  # apex to be appended later
            faces.append((a, c, b))
    # apex
    verts.append((0.0, height / 2, 0.0))

    # base cap
    if cap:
        center = len(verts)
        verts.append((0.0, -height / 2, 0.0))
        for i in range(radial_segments):
            ang0 = i * 2 * math.pi / radial_segments
            ang1 = (i + 1) * 2 * math.pi / radial_segments
            v0 = (radius * math.cos(ang0), -height / 2, radius * math.sin(ang0))
            v1 = (radius * math.cos(ang1), -height / 2, radius * math.sin(ang1))
            verts.extend([v1, v0])
            a = center
            b = center + 1 + (i * 2)
            c = center + 2 + (i * 2)
            faces.append((a, b, c))
    m = Mesh(verts, faces, name=name)
    return m.compute_normals(smooth=True)


def torus(R: float = 1.0, r: float = 0.3, radial_segments: int = 32, tubular_segments: int = 24, name: str = "torus") -> Mesh:
    verts: List[Vec3] = []
    faces: List[Tri] = []
    for i in range(radial_segments + 1):
        u = i / radial_segments * 2 * math.pi
        cu, su = math.cos(u), math.sin(u)
        for j in range(tubular_segments + 1):
            v = j / tubular_segments * 2 * math.pi
            cv, sv = math.cos(v), math.sin(v)
            x = (R + r * cv) * cu
            y = (R + r * cv) * su
            z = r * sv
            verts.append((x, z, y))  # swap to make torus "stand" on Y
            if i < radial_segments and j < tubular_segments:
                a = i * (tubular_segments + 1) + j
                b = a + 1
                c = (i + 1) * (tubular_segments + 1) + j
                d = c + 1
                faces.append((a, c, b))
                faces.append((b, c, d))
    m = Mesh(verts, faces, name=name)
    return m.compute_normals(smooth=True)


def grid(width: int, depth: int, cell: float = 1.0, name: str = "grid") -> Mesh:
    m = plane(width * cell, depth * cell, width, depth, name)
    return m


def capsule(radius: float = 0.5, height: float = 1.5, segments: int = 24, name: str = "capsule") -> Mesh:
    # Cylinder with two hemispheres
    cy_h = max(0.0, height - 2 * radius)
    c = cylinder(radius, cy_h, segments, 1, cap_top=False, cap_bottom=False, name="capsule_body")
    top = uv_sphere(radius, segments).rotate_x(0).translate(0, cy_h / 2 + radius, 0)
    bottom = uv_sphere(radius, segments).rotate_z(math.pi).translate(0, -(cy_h / 2 + radius), 0)
    return Mesh(name=name).merge(c).merge(top).merge(bottom).compute_normals(True)


def parametric_surface(func: Callable[[float, float], Vec3], u_segments: int, v_segments: int,
                       u_range: Tuple[float, float] = (0.0, 1.0), v_range: Tuple[float, float] = (0.0, 1.0),
                       wrap_u: bool = False, wrap_v: bool = False, name: str = "parametric") -> Mesh:
    verts: List[Vec3] = []
    faces: List[Tri] = []
    umin, umax = u_range
    vmin, vmax = v_range
    ucount = u_segments + (0 if wrap_u else 1)
    vcount = v_segments + (0 if wrap_v else 1)
    for i in range(ucount):
        u = i / u_segments
        uu = umin + u * (umax - umin)
        for j in range(vcount):
            v = j / v_segments
            vv = vmin + v * (vmax - vmin)
            verts.append(func(uu, vv))
    def idx(i: int, j: int) -> int:
        return (i % (u_segments) if wrap_u else i) * vcount + (j % (v_segments) if wrap_v else j)
    for i in range(u_segments):
        for j in range(v_segments):
            a = idx(i, j)
            b = idx(i + 1, j)
            c = idx(i, j + 1)
            d = idx(i + 1, j + 1)
            faces.append((a, b, c))
            faces.append((b, d, c))
    m = Mesh(verts, faces, name=name)
    return m.compute_normals(True)


def extrude_polygon(polygon_xy: Sequence[Tuple[float, float]], height: float, name: str = "prism") -> Mesh:
    """Extrude a simple (non-self-intersecting) 2D polygon in XY by height along +Z.
    No triangulation of concave polygons here; expects convex or already triangulated rings.
    """
    verts: List[Vec3] = []
    faces: List[Tri] = []
    n = len(polygon_xy)
    # bottom ring (z=0) and top ring (z=height)
    for x, y in polygon_xy:
        verts.append((x, y, 0.0))
    for x, y in polygon_xy:
        verts.append((x, y, height))
    # sides
    for i in range(n):
        a = i
        b = (i + 1) % n
        c = n + i
        d = n + ((i + 1) % n)
        faces.append((a, b, c))
        faces.append((b, d, c))
    # caps as fan (works for convex polygons)
    center_bottom = len(verts)
    verts.append((sum(x for x, _ in polygon_xy) / n, sum(y for _, y in polygon_xy) / n, 0.0))
    for i in range(n - 1):
        faces.append((center_bottom, i + 1, i))
    center_top = len(verts)
    verts.append((sum(x for x, _ in polygon_xy) / n, sum(y for _, y in polygon_xy) / n, height))
    for i in range(n - 1):
        faces.append((center_top, n + i, n + i + 1))
    m = Mesh(verts, faces, name=name)
    return m.compute_normals(True)


# ---------------
# File exporters
# ---------------

def save_obj(path: str, mesh: Mesh) -> None:
    """Save OBJ with optional vt/vn (assumes they are aligned with vertices)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"o {mesh.name}\n")
        for x, y, z in mesh.vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        if mesh.uvs:
            for u, v in mesh.uvs:
                f.write(f"vt {u:.6f} {v:.6f}\n")
        if mesh.normals:
            for nx, ny, nz in mesh.normals:
                f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        use_vt = mesh.uvs is not None
        use_vn = mesh.normals is not None
        for a, b, c in mesh.faces:
            def idx(i: int) -> str:
                vi = i + 1
                ti = vi if use_vt else None
                ni = vi if use_vn else None
                if use_vt and use_vn:
                    return f"{vi}/{ti}/{ni}"
                elif use_vt:
                    return f"{vi}/{ti}"
                elif use_vn:
                    return f"{vi}//{ni}"
                else:
                    return f"{vi}"
            f.write(f"f {idx(a)} {idx(b)} {idx(c)}\n")


def save_stl_binary(path: str, mesh: Mesh) -> None:
    """Write a binary STL. Normals are per face (flat)."""
    with open(path, "wb") as f:
        f.write(b"geomesh STL export" + bytes(80 - len("geomesh STL export")))
        f.write(struct.pack("<I", len(mesh.faces)))
        # If vertex normals are present, compute a face normal per triangle anyway
        for a, b, c in mesh.faces:
            va, vb, vc = mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]
            n = v_norm(v_cross(v_sub(vb, va), v_sub(vc, va)))
            f.write(struct.pack("<3f", *n))
            f.write(struct.pack("<3f", *va))
            f.write(struct.pack("<3f", *vb))
            f.write(struct.pack("<3f", *vc))
            f.write(struct.pack("<H", 0))  # attribute byte count


def save_ply_ascii(path: str, mesh: Mesh) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(mesh.vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(mesh.faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for x, y, z in mesh.vertices:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in mesh.faces:
            f.write(f"3 {a} {b} {c}\n")


# --------
#   CLI
# --------

_DEF_HELP = """
Examples:
  python -m geomesh --shape uv_sphere --radius 1 --segments 48 --out sphere.obj
  python -m geomesh --shape ico_sphere --subdiv 2 --out unit_ico.obj
  python -m geomesh --shape cylinder --radius 0.5 --height 2 --radial 64 --out cyl.obj
  python -m geomesh --shape torus --R 1.2 --r 0.4 --out torus.obj
  python -m geomesh --shape cube --size 1 --out cube.stl --stl
  python -m geomesh --shape param --u 64 --v 32 --out mobius.obj
"""


def _cli():
    import argparse
    p = argparse.ArgumentParser(description="geomesh: tiny 3D mesh generator", epilog=_DEF_HELP,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--shape", required=True, choices=[
        "uv_sphere", "ico_sphere", "cylinder", "cone", "torus", "cube", "plane", "grid", "capsule", "param"
    ])
    p.add_argument("--out", required=True, help="Output path (.obj/.stl/.ply). Alternatively use --stl/--ply flags.")
    p.add_argument("--stl", action="store_true", help="Force binary STL output")
    p.add_argument("--ply", action="store_true", help="Force ASCII PLY output")
    # Common params
    p.add_argument("--segments", type=int, default=32)
    p.add_argument("--rings", type=int)
    p.add_argument("--subdiv", type=int, default=2)
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--R", type=float, default=1.0)
    p.add_argument("--r", type=float, default=0.3)
    p.add_argument("--height", type=float, default=1.0)
    p.add_argument("--radial", type=int, default=32)
    p.add_argument("--hseg", type=int, default=1)
    p.add_argument("--size", type=float, default=1.0)
    p.add_argument("--sizex", type=float, default=1.0)
    p.add_argument("--sizey", type=float, default=1.0)
    p.add_argument("--gridx", type=int, default=8)
    p.add_argument("--gridy", type=int, default=8)
    p.add_argument("--capsule_h", type=float, default=1.5)
    p.add_argument("--u", type=int, default=64)
    p.add_argument("--v", type=int, default=32)
    p.add_argument("--param", choices=["mobius", "saddle"], default="mobius")

    args = p.parse_args()

    if args.shape == "uv_sphere":
        mesh = uv_sphere(radius=args.radius, segments=args.segments, rings=args.rings)
    elif args.shape == "ico_sphere":
        mesh = ico_sphere(radius=args.radius, subdivisions=args.subdiv)
    elif args.shape == "cylinder":
        mesh = cylinder(radius=args.radius, height=args.height, radial_segments=args.radial, height_segments=args.hseg)
    elif args.shape == "cone":
        mesh = cone(radius=args.radius, height=args.height, radial_segments=args.radial)
    elif args.shape == "torus":
        mesh = torus(R=args.R, r=args.r, radial_segments=args.segments, tubular_segments=args.radial)
    elif args.shape == "cube":
        mesh = cube(size=args.size)
    elif args.shape == "plane":
        mesh = plane(size_x=args.sizex, size_y=args.sizey, segments_x=args.gridx, segments_y=args.gridy)
    elif args.shape == "grid":
        mesh = grid(args.gridx, args.gridy, cell=1.0)
    elif args.shape == "capsule":
        mesh = capsule(radius=args.radius, height=args.capsule_h, segments=args.segments)
    elif args.shape == "param":
        if args.param == "mobius":
            def mf(u: float, v: float) -> Vec3:
                # u in [0, 2pi], v in [-1, 1]
                x = (1 + (v / 2) * math.cos(u / 2)) * math.cos(u)
                y = (1 + (v / 2) * math.cos(u / 2)) * math.sin(u)
                z = (v / 2) * math.sin(u / 2)
                return (x, y, z)
            mesh = parametric_surface(mf, args.u, args.v, (0, 2 * math.pi), (-1, 1), wrap_u=True)
        else:  # saddle
            def sf(u: float, v: float) -> Vec3:
                # simple saddle: z = u^2 - v^2
                return (u, v, (u * u - v * v))
            mesh = parametric_surface(sf, args.u, args.v, (-1, 1), (-1, 1))
    else:
        raise SystemExit("Unknown shape")

    # Decide format
    out_lower = args.out.lower()
    if args.stl or out_lower.endswith(".stl"):
        save_stl_binary(args.out, mesh)
    elif args.ply or out_lower.endswith(".ply"):
        save_ply_ascii(args.out, mesh)
    else:
        save_obj(args.out, mesh)


if __name__ == "__main__":
    _cli()
