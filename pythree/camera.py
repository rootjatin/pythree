# pythree/camera.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

Vec3 = Tuple[float, float, float]
Vec4 = Tuple[float, float, float, float]
Mat4 = Tuple[
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
]


# -------------------------
# Small math helpers
# -------------------------

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


def _mat4_identity() -> Mat4:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _mat4_mul(a: Mat4, b: Mat4) -> Mat4:
    out = []
    for i in range(4):
        row = []
        for j in range(4):
            v = (
                a[i][0] * b[0][j]
                + a[i][1] * b[1][j]
                + a[i][2] * b[2][j]
                + a[i][3] * b[3][j]
            )
            row.append(v)
        out.append(tuple(row))
    return tuple(out)  # type: ignore[return-value]


def _mat4_mul_vec4(m: Mat4, v: Vec4) -> Vec4:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3] * v[3],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3] * v[3],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3] * v[3],
        m[3][0] * v[0] + m[3][1] * v[1] + m[3][2] * v[2] + m[3][3] * v[3],
    )


# -------------------------
# Base camera
# -------------------------

@dataclass
class Camera:
    """
    Base camera storing pose and orientation.

    Notes
    -----
    - Matrices are returned in row-major form.
    - The camera uses a standard right-handed look-at convention.
    - By default, the camera looks from `position` toward `target`.
    """

    position: Vec3 = (0.0, 0.0, 5.0)
    target: Vec3 = (0.0, 0.0, 0.0)
    up: Vec3 = (0.0, 1.0, 0.0)
    near: float = 0.1
    far: float = 1000.0
    name: str = "camera"

    def __post_init__(self) -> None:
        if self.near <= 0:
            raise ValueError("near must be > 0")
        if self.far <= self.near:
            raise ValueError("far must be > near")

    def copy(self, *, name: str | None = None) -> "Camera":
        return Camera(
            position=self.position,
            target=self.target,
            up=self.up,
            near=self.near,
            far=self.far,
            name=self.name if name is None else name,
        )

    def look_at(self, target: Vec3) -> "Camera":
        self.target = target
        return self

    def translated(self, dx: float, dy: float, dz: float) -> "Camera":
        delta = (dx, dy, dz)
        self.position = _vec_add(self.position, delta)
        self.target = _vec_add(self.target, delta)
        return self

    def direction(self) -> Vec3:
        return _normalize(_vec_sub(self.target, self.position))

    def right(self) -> Vec3:
        f = self.direction()
        r = _cross(f, self.up)
        return _normalize(r)

    def true_up(self) -> Vec3:
        r = self.right()
        f = self.direction()
        return _normalize(_cross(r, f))

    def view_matrix(self) -> Mat4:
        """
        Return a standard right-handed look-at view matrix.
        """
        eye = self.position
        center = self.target
        up = self.up

        f = _normalize(_vec_sub(center, eye))
        if _length(f) <= 1e-12:
            return _mat4_identity()

        s = _normalize(_cross(f, up))
        if _length(s) <= 1e-12:
            # fall back if up is parallel to forward
            alt_up = (0.0, 0.0, 1.0) if abs(f[1]) > 0.99 else (0.0, 1.0, 0.0)
            s = _normalize(_cross(f, alt_up))

        u = _cross(s, f)

        ex = -_dot(s, eye)
        ey = -_dot(u, eye)
        ez = _dot(f, eye)

        return (
            (s[0], s[1], s[2], ex),
            (u[0], u[1], u[2], ey),
            (-f[0], -f[1], -f[2], ez),
            (0.0, 0.0, 0.0, 1.0),
        )

    def projection_matrix(self) -> Mat4:
        raise NotImplementedError("Subclasses must implement projection_matrix()")

    def view_projection_matrix(self) -> Mat4:
        return _mat4_mul(self.projection_matrix(), self.view_matrix())

    def project_point(self, p: Vec3) -> Vec3:
        """
        Project a world-space point into normalized device coordinates (NDC).

        Returns
        -------
        (x, y, z) where x/y/z are typically in [-1, 1] if the point is inside clip volume.
        """
        clip = _mat4_mul_vec4(self.view_projection_matrix(), (p[0], p[1], p[2], 1.0))
        w = clip[3]
        if abs(w) <= 1e-12:
            raise ZeroDivisionError("Point projects with w ~= 0")
        return (clip[0] / w, clip[1] / w, clip[2] / w)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, position={self.position}, target={self.target})"
        )


# -------------------------
# Perspective camera
# -------------------------

@dataclass
class PerspectiveCamera(Camera):
    """
    Perspective camera.

    Parameters
    ----------
    fov_y:
        Vertical field of view in degrees.
    aspect:
        Viewport aspect ratio = width / height.
    """

    fov_y: float = 60.0
    aspect: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.fov_y <= 0.0 or self.fov_y >= 180.0:
            raise ValueError("fov_y must be in the range (0, 180)")
        if self.aspect <= 0.0:
            raise ValueError("aspect must be > 0")

    def copy(self, *, name: str | None = None) -> "PerspectiveCamera":
        return PerspectiveCamera(
            position=self.position,
            target=self.target,
            up=self.up,
            near=self.near,
            far=self.far,
            name=self.name if name is None else name,
            fov_y=self.fov_y,
            aspect=self.aspect,
        )

    def projection_matrix(self) -> Mat4:
        f = 1.0 / math.tan(math.radians(self.fov_y) * 0.5)
        n = self.near
        fa = self.far

        return (
            (f / self.aspect, 0.0, 0.0, 0.0),
            (0.0, f, 0.0, 0.0),
            (0.0, 0.0, (fa + n) / (n - fa), (2.0 * fa * n) / (n - fa)),
            (0.0, 0.0, -1.0, 0.0),
        )


# -------------------------
# Orthographic camera
# -------------------------

@dataclass
class OrthographicCamera(Camera):
    """
    Orthographic camera.

    Parameters
    ----------
    left, right, bottom, top:
        Bounds of the orthographic view volume.
    """

    left: float = -1.0
    right: float = 1.0
    bottom: float = -1.0
    top: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.left == self.right:
            raise ValueError("left and right must differ")
        if self.bottom == self.top:
            raise ValueError("bottom and top must differ")

    def copy(self, *, name: str | None = None) -> "OrthographicCamera":
        return OrthographicCamera(
            position=self.position,
            target=self.target,
            up=self.up,
            near=self.near,
            far=self.far,
            name=self.name if name is None else name,
            left=self.left,
            right=self.right,
            bottom=self.bottom,
            top=self.top,
        )

    def projection_matrix(self) -> Mat4:
        l = self.left
        r = self.right
        b = self.bottom
        t = self.top
        n = self.near
        fa = self.far

        return (
            (2.0 / (r - l), 0.0, 0.0, -(r + l) / (r - l)),
            (0.0, 2.0 / (t - b), 0.0, -(t + b) / (t - b)),
            (0.0, 0.0, -2.0 / (fa - n), -(fa + n) / (fa - n)),
            (0.0, 0.0, 0.0, 1.0),
        )
