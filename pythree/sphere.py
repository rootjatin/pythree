from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

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
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat_rotate_y(a: float) -> Mat4:
    c, s = math.cos(a), math.sin(a)
    return [
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat_rotate_z(a: float) -> Mat4:
    c, s = math.cos(a), math.sin(a)
    return [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def apply_mat(v: Vec3, m: Mat4) -> Vec3:
    x, y, z = v
    xp = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3]
    yp = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3]
    zp = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3]
    wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3]
    if wp != 0.0 and wp != 1.0:
        return (xp / wp, yp / wp, zp / wp)
    return (xp, yp, zp)


def apply_mat_dir(v: Vec3, m3: List[List[float]]) -> Vec3:
    """Apply a 3x3 matrix to a direction vector."""
    x, y, z = v
    return (
        m3[0][0] * x + m3[0][1] * y + m3[0][2] * z,
        m3[1][0] * x + m3[1][1] * y + m3[1][2] * z,
        m3[2][0] * x + m3[2][1] * y + m3[2][2] * z,
    )


def normal_matrix(m: Mat4) -> List[List[float]]:
    """Inverse-transpose of the upper-left 3x3."""
    a00, a01, a02 = m[0][0], m[0][1], m[0][2]
    a10, a11, a12 = m[1][0], m[1][1], m[1][2]
    a20, a21, a22 = m[2][0], m[2][1], m[2][2]

    det = (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )

    if abs(det) < 1e-12:
        # Fallback for singular transforms
        return [
            [a00, a10, a20],
            [a01, a11, a21],
            [a02, a12, a22],
        ]

    inv_det = 1.0 / det
    inv = [
        [
            (a11 * a22 - a12 * a21) * inv_det,
            (a02 * a21 - a01 * a22) * inv_det,
            (a01 * a12 - a02 * a11) * inv_det,
        ],
        [
            (a12 * a20 - a10 * a22) * inv_det,
            (a00 * a22 - a02 * a20) * inv_det,
            (a02 * a10 - a00 * a12) * inv_det,
        ],
        [
            (a10 * a21 - a11 * a20) * inv_det,
            (a01 * a20 - a00 * a21) * inv_det,
            (a00 * a11 - a01 * a10) * inv_det,
        ],
    ]

    # transpose(inv)
    return [
        [inv[0][0], inv[1][0], inv[2][0]],
        [inv[0][1], inv[1][1], inv[2][1]],
        [inv[0][2], inv[1][2], inv[2][2]],
    ]


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

        if self.normals is not None:
            nm = normal_matrix(m)
            self.normals = [v_norm(apply_mat_dir(n, nm)) for n in self.normals]

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
        return Mesh(
            self.vertices.copy(),
            self.faces.copy(),
            None if self.normals is None else self.normals.copy(),
            None if self.uvs is None else self.uvs.copy(),
            self.name,
        )

    # ---- composition ----
    def merge(self, other: "Mesh", rename: bool = False) -> "Mesh":
        if not other.vertices:
            return self

        if not self.vertices:
            self.vertices = other.vertices.copy()
            self.faces = other.faces.copy()
            self.normals = None if other.normals is None else other.normals.copy()
            self.uvs = None if other.uvs is None else other.uvs.copy()
        else:
            offset = len(self.vertices)
            self.vertices.extend(other.vertices)
            self.faces.extend((a + offset, b + offset, c + offset) for (a, b, c) in other.faces)

            if self.normals is not None and other.normals is not None:
                self.normals.extend(other.normals)
            else:
                self.normals = None

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
        if not self.vertices:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    def weld(self, eps: float = 1e-7) -> "Mesh":
        """Deduplicate vertices; updates faces accordingly. Keeps first occurrence."""
        if eps <= 0:
            raise ValueError("eps must be > 0")

        def key(v: Vec3) -> Tuple[int, int, int]:
            return (round(v[0] / eps), round(v[1] / eps), round(v[2] / eps))

        lookup = {}
        remap: List[int] = [-1] * len(self.vertices)
        new_verts: List[Vec3] = []
        new_norms: Optional[List[Vec3]] = [] if self.normals is not None else None
        new_uvs: Optional[List[Vec2]] = [] if self.uvs is not None else None

        for i, v in enumerate(self.vertices):
            k = key(v)
            if k in lookup:
                remap[i] = lookup[k]
            else:
                new_index = len(new_verts)
                lookup[k] = new_index
                remap[i] = new_index
                new_verts.append(v)

                if new_norms is not None and self.normals is not None:
                    new_norms.append(self.normals[i])

                if new_uvs is not None and self.uvs is not None:
                    new_uvs.append(self.uvs[i])

        self.vertices = new_verts
        self.faces = [(remap[a], remap[b], remap[c]) for (a, b, c) in self.faces]
        self.normals = new_norms
        self.uvs = new_uvs
        return self

    # ---- shading ----
    def compute_normals(self, smooth: bool = True) -> "Mesh":
        if not self.vertices or not self.faces:
            self.normals = []
            return self

        if not smooth:
            v_out: List[Vec3] = []
            n_out: List[Vec3] = []
            f_out: List[Tri] = []
            uv_out: Optional[List[Vec2]] = [] if self.uvs is not None else None

            for (a, b, c) in self.faces:
                va, vb, vc = self.vertices[a], self.vertices[b], self.vertices[c]
                n = v_norm(v_cross(v_sub(vb, va), v_sub(vc, va)))

                base = len(v_out)
                v_out.extend([va, vb, vc])
                n_out.extend([n, n, n])
                f_out.append((base, base + 1, base + 2))

                if uv_out is not None and self.uvs is not None:
                    uv_out.extend([self.uvs[a], self.uvs[b], self.uvs[c]])

            self.vertices = v_out
            self.faces = f_out
            self.normals = n_out
            self.uvs = uv_out
            return self

        normals = [(0.0, 0.0, 0.0) for _ in self.vertices]

        for (a, b, c) in self.faces:
            va, vb, vc = self.vertices[a], self.vertices[b], self.vertices[c]
            face_n = v_cross(v_sub(vb, va), v_sub(vc, va))  # area-weighted
            normals[a] = v_add(normals[a], face_n)
            normals[b] = v_add(normals[b], face_n)
            normals[c] = v_add(normals[c], face_n)

        self.normals = [v_norm(n) for n in normals]
        return self

    # ---- UV mapping helpers ----
    def spherical_uv(self) -> "Mesh":
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

        bmin, bmax = self.bounds()
        uvs: List[Vec2] = []

        for x, y, z in self.vertices:
            if axis == "y":
                ang = (math.atan2(z, x) + 2 * math.pi) % (2 * math.pi)
                u = ang / (2 * math.pi)
                v = (y - bmin[1]) / max(1e-9, bmax[1] - bmin[1])
            elif axis == "x":
                ang = (math.atan2(y, z) + 2 * math.pi) % (2 * math.pi)
                u = ang / (2 * math.pi)
                v = (x - bmin[0]) / max(1e-9, bmax[0] - bmin[0])
            elif axis == "z":
                ang = (math.atan2(y, x) + 2 * math.pi) % (2 * math.pi)
                u = ang / (2 * math.pi)
                v = (z - bmin[2]) / max(1e-9, bmax[2] - bmin[2])
            else:
                raise ValueError("axis must be 'x', 'y', or 'z'")

            uvs.append((u, v))

        self.uvs = uvs
        return self


# -----------------------
# Primitive constructors
# -----------------------


def uv_sphere(
    radius: float = 1.0,
    segments: int = 32,
    rings: Optional[int] = None,
    name: str = "uv_sphere",
) -> Mesh:
    """UV sphere with seam-friendly duplicated vertices and no degenerate pole triangles."""
    if segments < 3:
        raise ValueError("segments must be >= 3")

    if rings is None:
        rings = max(2, segments // 2)

    if rings < 2:
        raise ValueError("rings must be >= 2")

    verts: List[Vec3] = []
    normals: List[Vec3] = []
    uvs: List[Vec2] = []
    faces: List[Tri] = []

    row_stride = segments + 1

    for i in range(rings + 1):
        v = 1.0 - (i / rings)  # top=1, bottom=0
        theta = (1.0 - v) * math.pi
        st, ct = math.sin(theta), math.cos(theta)

        for j in range(segments + 1):
            u = j / segments
            phi = u * 2.0 * math.pi
            cp, sp = math.cos(phi), math.sin(phi)

            if i == 0:
                pos = (0.0, 0.0, radius)
                nrm = (0.0, 0.0, 1.0)
            elif i == rings:
                pos = (0.0, 0.0, -radius)
                nrm = (0.0, 0.0, -1.0)
            else:
                x = radius * st * cp
                y = radius * st * sp
                z = radius * ct
                pos = (x, y, z)
                nrm = v_norm(pos)

            verts.append(pos)
            normals.append(nrm)
            uvs.append((u, v))

    def idx(i: int, j: int) -> int:
        return i * row_stride + j

    for i in range(rings):
        for j in range(segments):
            a = idx(i, j)
            b = idx(i, j + 1)
            c = idx(i + 1, j)
            d = idx(i + 1, j + 1)

            if i == 0:
                # top cap
                faces.append((a, c, d))
            elif i == rings - 1:
                # bottom cap
                faces.append((a, c, b))
            else:
                faces.append((a, c, b))
                faces.append((b, c, d))

    return Mesh(
        vertices=verts,
        faces=faces,
        normals=normals,
        uvs=uvs,
        name=name,
    )


def _midpoint_cache_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


def ico_sphere(radius: float = 1.0, subdivisions: int = 2, name: str = "ico_sphere") -> Mesh:
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
            new_faces.extend([
                (a, ab, ca),
                (b, bc, ab),
                (c, ca, bc),
                (ab, bc, ca),
            ])
        faces = new_faces

    verts = [v_scale(v_norm(v), radius) for v in verts]
    return Mesh(verts, faces, name=name).compute_normals(smooth=True).spherical_uv()


def plane(
    size_x: float = 1.0,
    size_y: float = 1.0,
    segments_x: int = 1,
    segments_y: int = 1,
    name: str = "plane",
) -> Mesh:
    if segments_x < 1 or segments_y < 1:
        raise ValueError("segments_x and segments_y must be >= 1")

    verts: List[Vec3] = []
    faces: List[Tri] = []
    uvs: List[Vec2] = []

    for iy in range(segments_y + 1):
        v = iy / segments_y
        y = (v - 0.5) * size_y
        for ix in range(segments_x + 1):
            u = ix / segments_x
            x = (u - 0.5) * size_x
            verts.append((x, 0.0, y))
            uvs.append((u, v))

            if ix < segments_x and iy < segments_y:
                a = iy * (segments_x + 1) + ix
                b = a + 1
                c = (iy + 1) * (segments_x + 1) + ix
                d = c + 1
                faces.append((a, c, b))
                faces.append((b, c, d))

    return Mesh(verts, faces, uvs=uvs, name=name).compute_normals(smooth=True)


def cube(size: float = 1.0, name: str = "cube") -> Mesh:
    s = size
    px = plane(s, s, 1, 1, "px").rotate_z(-math.pi / 2).rotate_y(math.pi / 2).translate(s / 2, 0, 0)
    nx = plane(s, s, 1, 1, "nx").rotate_z(math.pi / 2).rotate_y(-math.pi / 2).translate(-s / 2, 0, 0)
    py = plane(s, s, 1, 1, "py").rotate_x(-math.pi / 2).translate(0, s / 2, 0)
    ny = plane(s, s, 1, 1, "ny").rotate_x(math.pi / 2).translate(0, -s / 2, 0)
    pz = plane(s, s, 1, 1, "pz").translate(0, 0, s / 2)
    nz = plane(s, s, 1, 1, "nz").rotate_y(math.pi).translate(0, 0, -s / 2)
    return Mesh(name=name).merge(px).merge(nx).merge(py).merge(ny).merge(pz).merge(nz)


def cylinder(
    radius: float = 0.5,
    height: float = 1.0,
    radial_segments: int = 32,
    height_segments: int = 1,
    cap_top: bool = True,
    cap_bottom: bool = True,
    name: str = "cylinder",
) -> Mesh:
    if radial_segments < 3:
        raise ValueError("radial_segments must be >= 3")
    if height_segments < 1:
        raise ValueError("height_segments must be >= 1")

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

    return Mesh(verts, faces, name=name).compute_normals(smooth=True)


def cone(
    radius: float = 0.5,
    height: float = 1.0,
    radial_segments: int = 32,
    cap: bool = True,
    name: str = "cone",
) -> Mesh:
    if radial_segments < 3:
        raise ValueError("radial_segments must be >= 3")

    verts: List[Vec3] = []
    faces: List[Tri] = []

    # base ring
    for ix in range(radial_segments + 1):
        u = ix / radial_segments
        ang = u * 2 * math.pi
        x = radius * math.cos(ang)
        z = radius * math.sin(ang)
        verts.append((x, -height / 2, z))

    apex = len(verts)
    verts.append((0.0, height / 2, 0.0))

    for ix in range(radial_segments):
        a = ix
        b = ix + 1
        faces.append((a, apex, b))

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

    return Mesh(verts, faces, name=name).compute_normals(smooth=True)


def torus(
    R: float = 1.0,
    r: float = 0.3,
    radial_segments: int = 32,
    tubular_segments: int = 24,
    name: str = "torus",
) -> Mesh:
    if radial_segments < 3 or tubular_segments < 3:
        raise ValueError("radial_segments and tubular_segments must be >= 3")

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

            verts.append((x, z, y))  # stand on Y axis

            if i < radial_segments and j < tubular_segments:
                a = i * (tubular_segments + 1) + j
                b = a + 1
                c = (i + 1) * (tubular_segments + 1) + j
                d = c + 1
                faces.append((a, c, b))
                faces.append((b, c, d))

    return Mesh(verts, faces, name=name).compute_normals(smooth=True)


def grid(width: int, depth: int, cell: float = 1.0, name: str = "grid") -> Mesh:
    if width < 1 or depth < 1:
        raise ValueError("width and depth must be >= 1")
    return plane(width * cell, depth * cell, width, depth, name)


def capsule(radius: float = 0.5, height: float = 1.5, segments: int = 24, name: str = "capsule") -> Mesh:
    cy_h = max(0.0, height - 2 * radius)
    body = cylinder(radius, cy_h, segments, 1, cap_top=False, cap_bottom=False, name="capsule_body")
    top = uv_sphere(radius, segments).translate(0.0, cy_h / 2 + radius, 0.0)
    bottom = uv_sphere(radius, segments).rotate_z(math.pi).translate(0.0, -(cy_h / 2 + radius), 0.0)
    return Mesh(name=name).merge(body).merge(top).merge(bottom).compute_normals(True)


def parametric_surface(
    func: Callable[[float, float], Vec3],
    u_segments: int,
    v_segments: int,
    u_range: Tuple[float, float] = (0.0, 1.0),
    v_range: Tuple[float, float] = (0.0, 1.0),
    wrap_u: bool = False,
    wrap_v: bool = False,
    name: str = "parametric",
) -> Mesh:
    if u_segments < 1 or v_segments < 1:
        raise ValueError("u_segments and v_segments must be >= 1")

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
        ii = i % u_segments if wrap_u else i
        jj = j % v_segments if wrap_v else j
        return ii * vcount + jj

    for i in range(u_segments):
        for j in range(v_segments):
            a = idx(i, j)
            b = idx(i + 1, j)
            c = idx(i, j + 1)
            d = idx(i + 1, j + 1)
            faces.append((a, b, c))
            faces.append((b, d, c))

    return Mesh(verts, faces, name=name).compute_normals(True)


def extrude_polygon(polygon_xy: Sequence[Tuple[float, float]], height: float, name: str = "prism") -> Mesh:
    """Extrude a convex 2D polygon in XY by height along +Z."""
    n = len(polygon_xy)
    if n < 3:
        raise ValueError("polygon must have at least 3 points")

    verts: List[Vec3] = []
    faces: List[Tri] = []

    # bottom ring
    for x, y in polygon_xy:
        verts.append((x, y, 0.0))

    # top ring
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

    cx = sum(x for x, _ in polygon_xy) / n
    cy = sum(y for _, y in polygon_xy) / n

    center_bottom = len(verts)
    verts.append((cx, cy, 0.0))
    for i in range(n - 1):
        faces.append((center_bottom, i + 1, i))

    center_top = len(verts)
    verts.append((cx, cy, height))
    for i in range(n - 1):
        faces.append((center_top, n + i, n + i + 1))

    return Mesh(verts, faces, name=name).compute_normals(True)


# ---------------
# File exporters
# ---------------


def save_obj(path: str, mesh: Mesh) -> None:
    """Save OBJ with optional vt/vn (assumes they are aligned with vertices)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"o {mesh.name}\n")

        for x, y, z in mesh.vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

        if mesh.uvs is not None:
            for u, v in mesh.uvs:
                f.write(f"vt {u:.6f} {v:.6f}\n")

        if mesh.normals is not None:
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
                if use_vt:
                    return f"{vi}/{ti}"
                if use_vn:
                    return f"{vi}//{ni}"
                return f"{vi}"

            f.write(f"f {idx(a)} {idx(b)} {idx(c)}\n")


def save_stl_binary(path: str, mesh: Mesh) -> None:
    """Write a binary STL. Normals are written per-face."""
    header = b"geomesh STL export"
    header = header + bytes(80 - len(header))

    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(mesh.faces)))

        for a, b, c in mesh.faces:
            va, vb, vc = mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]
            n = v_norm(v_cross(v_sub(vb, va), v_sub(vc, va)))

            f.write(struct.pack("<3f", *n))
            f.write(struct.pack("<3f", *va))
            f.write(struct.pack("<3f", *vb))
            f.write(struct.pack("<3f", *vc))
            f.write(struct.pack("<H", 0))


def save_ply_ascii(path: str, mesh: Mesh) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(mesh.vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(mesh.faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

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

    p = argparse.ArgumentParser(
        description="geomesh: tiny 3D mesh generator",
        epilog=_DEF_HELP,
        formatter_class=argparse.RawTextHelpFormatter,
    )

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
        mesh = cylinder(
            radius=args.radius,
            height=args.height,
            radial_segments=args.radial,
            height_segments=args.hseg,
        )

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
                x = (1 + (v / 2) * math.cos(u / 2)) * math.cos(u)
                y = (1 + (v / 2) * math.cos(u / 2)) * math.sin(u)
                z = (v / 2) * math.sin(u / 2)
                return (x, y, z)

            mesh = parametric_surface(mf, args.u, args.v, (0, 2 * math.pi), (-1, 1), wrap_u=True)

        else:
            def sf(u: float, v: float) -> Vec3:
                return (u, v, (u * u - v * v))

            mesh = parametric_surface(sf, args.u, args.v, (-1, 1), (-1, 1))

    else:
        raise SystemExit("Unknown shape")

    out_lower = args.out.lower()
    if args.stl or out_lower.endswith(".stl"):
        save_stl_binary(args.out, mesh)
    elif args.ply or out_lower.endswith(".ply"):
        save_ply_ascii(args.out, mesh)
    else:
        save_obj(args.out, mesh)


if __name__ == "__main__":
    _cli()
