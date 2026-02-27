# pythree/equations.py
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .sphere import Mesh  # type: ignore

Vec3 = Tuple[float, float, float]
Tri = Tuple[int, int, int]


"""
This module provides utilities to build triangle meshes from math expressions:

1) Heightfield surface:
      z = f(x, y)

2) Parametric surface:
      x = fx(u, v)
      y = fy(u, v)
      z = fz(u, v)

3) Implicit surface (iso-surface):
      f(x, y, z) = iso

Expressions are compiled using a strict AST validator so you can accept user input
without allowing arbitrary Python execution. Only a small set of math functions and
constants are allowed (see _ALLOWED_FUNCS/_ALLOWED_CONSTS).

------------------------------------------------------------
Choosing resolution / segments
------------------------------------------------------------

Mesh quality is driven primarily by sampling density:

- Heightfield: x_segments, y_segments
- Parametric: u_segments, v_segments
- Implicit: resolution = (nx, ny, nz) sample points of the scalar field

For implicit surfaces, increasing nx/ny/nz improves detail but cost grows ~O(nx*ny*nz).
Doubling each axis increases work about 8x.

To make it easier to choose, use:
    resolution_from_cell_size(bounds, cell_size)

Example:
    bounds = ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5))
    res = resolution_from_cell_size(bounds, cell_size=0.04, max_resolution=160)
    mesh = mesh_from_implicit(expr, bounds=bounds, resolution=res)

Or simply:
    mesh = mesh_from_implicit_cell_size(expr, bounds=bounds, cell_size=0.04)

For heightfields/parametric meshes, use:
    segments_from_step((a,b), step)
to choose segments such that adjacent samples are about 'step' apart in parameter units.
"""


# -----------------------------
# Safe math expression compiler
# -----------------------------

_ALLOWED_FUNCS: Dict[str, Callable[..., float]] = {
    # trig
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    # exp/log/pow
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "pow": pow,
    # misc
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "min": min,
    "max": max,
}

_ALLOWED_CONSTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": getattr(math, "tau", 2 * math.pi),
}

_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,  # py3.8+
)


class _MathExprValidator(ast.NodeVisitor):
    def __init__(self, allowed_vars: Sequence[str]) -> None:
        self.allowed_vars = set(allowed_vars)

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise ValueError(f"Disallowed syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Only allow f(x) where f is a simple name in _ALLOWED_FUNCS
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed (e.g., sin(x))")
        if node.func.id not in _ALLOWED_FUNCS:
            raise ValueError(f"Function not allowed: {node.func.id}")
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed in function calls")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        ident = node.id
        if (
            ident not in self.allowed_vars
            and ident not in _ALLOWED_FUNCS
            and ident not in _ALLOWED_CONSTS
        ):
            raise ValueError(f"Name not allowed: {ident}")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        # Only allow numeric constants
        if not isinstance(node.value, (int, float)):
            raise ValueError(
                f"Only numeric constants are allowed (got {type(node.value).__name__})"
            )
        self.generic_visit(node)


@dataclass(frozen=True)
class CompiledMathExpr:
    """
    A validated expression compiled into a lambda for fast repeated evaluation.

    - Supports positional calls in the declared var order
    - Supports keyword calls for clarity

    Example:
        f = compile_math_expr("sin(x) + y*y", vars=("x", "y"))
        z1 = f(0.5, 2.0)
        z2 = f(x=0.5, y=2.0)
    """

    expr: str
    vars: Tuple[str, ...]
    _fn: Callable[..., float]

    def __call__(self, *args: float, **kwargs: float) -> float:
        if args and kwargs:
            raise TypeError("Use either positional args OR keyword args, not both")

        if kwargs:
            try:
                args = tuple(float(kwargs[name]) for name in self.vars)
            except KeyError as e:
                raise TypeError(f"Missing variable: {e.args[0]}") from None
        else:
            if len(args) != len(self.vars):
                raise TypeError(
                    f"Expected {len(self.vars)} args ({', '.join(self.vars)}), got {len(args)}"
                )
            args = tuple(float(a) for a in args)

        return float(self._fn(*args))


def compile_math_expr(expr: str, *, vars: Sequence[str]) -> CompiledMathExpr:
    """
    Compile a restricted math expression into a callable.

    Allowed:
      - numeric literals, + - * / % **, parentheses
      - variables in `vars`
      - functions in _ALLOWED_FUNCS
      - constants pi, e, tau

    Disallowed:
      - attribute access, indexing, comprehensions, lambdas, imports, etc.
      - keyword arguments in calls
      - non-numeric constants (strings, None, etc.)
    """
    tree = ast.parse(expr, mode="eval")
    _MathExprValidator(vars).visit(tree)

    # Compile to a lambda for speed (avoids per-sample eval + dict creation).
    arglist = ", ".join(vars)
    src = f"lambda {arglist}: ({expr})"

    base_scope: Dict[str, object] = {"__builtins__": {}}
    base_scope.update(_ALLOWED_FUNCS)
    base_scope.update(_ALLOWED_CONSTS)

    fn = eval(src, base_scope, {})  # safe: AST validated + builtins removed
    return CompiledMathExpr(expr=expr, vars=tuple(vars), _fn=fn)


# -------------------------
# Resolution helper methods
# -------------------------

def segments_from_step(
    r: Tuple[float, float],
    step: float,
    *,
    min_segments: int = 4,
    max_segments: int = 4096,
) -> int:
    """
    Pick a segment count so adjacent samples are about 'step' apart in parameter units.

    Example:
        x_segments = segments_from_step((-2.0, 2.0), step=0.02)
    """
    a, b = r
    length = abs(b - a)
    if step <= 0:
        raise ValueError("step must be > 0")
    n = int(math.ceil(length / step))
    n = max(min_segments, n)
    n = min(max_segments, n)
    return n


def resolution_from_cell_size(
    bounds: Tuple[Vec3, Vec3],
    cell_size: float,
    *,
    min_resolution: int = 12,
    max_resolution: int = 256,
) -> Tuple[int, int, int]:
    """
    Pick (nx, ny, nz) for implicit meshing so grid cell edges are roughly 'cell_size'
    in world-space units.

    - nx/ny/nz are clamped to [min_resolution, max_resolution]
    - Always returns at least (2,2,2)

    Example:
        res = resolution_from_cell_size(((-1,-1,-1),(1,1,1)), cell_size=0.03, max_resolution=200)
    """
    (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0")

    lx = abs(xmax - xmin)
    ly = abs(ymax - ymin)
    lz = abs(zmax - zmin)

    nx = int(math.ceil(lx / cell_size)) + 1
    ny = int(math.ceil(ly / cell_size)) + 1
    nz = int(math.ceil(lz / cell_size)) + 1

    nx = max(2, min(max_resolution, max(min_resolution, nx)))
    ny = max(2, min(max_resolution, max(min_resolution, ny)))
    nz = max(2, min(max_resolution, max(min_resolution, nz)))

    return (nx, ny, nz)


# -------------------------
# Small helpers
# -------------------------

def _require_segments(name: str, n: int) -> None:
    if n < 1:
        raise ValueError(f"{name} must be >= 1 (got {n})")


def _linspace(a: float, b: float, segments: int) -> List[float]:
    # segments = number of intervals; points = segments + 1
    step = (b - a) / segments
    return [a + i * step for i in range(segments + 1)]


def _is_finite(x: float) -> bool:
    return math.isfinite(x) and not math.isnan(x)


# -------------------------
# 1) Heightfield: z = f(x,y)
# -------------------------

def mesh_from_heightfield(
    expr: str,
    *,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (-1.0, 1.0),
    x_segments: int = 100,
    y_segments: int = 100,
    name: str = "heightfield",
    skip_nonfinite: bool = True,
) -> Mesh:
    """
    Build a triangle mesh for a heightfield z = f(x,y).

    Parameters
    ----------
    expr:
        Math expression using variables x and y.
        Example: "sin(x) * cos(y)" or "x*x - y*y".
    x_range, y_range:
        Sampling ranges for x and y.
    x_segments, y_segments:
        Number of intervals along each axis (points = segments + 1).
        Higher values -> denser mesh.
    skip_nonfinite:
        If True, triangles touching NaN/Inf samples are omitted, producing holes
        instead of exploding geometry.

    Returns
    -------
    Mesh:
        Mesh with computed vertex normals.
    """
    _require_segments("x_segments", x_segments)
    _require_segments("y_segments", y_segments)

    f = compile_math_expr(expr, vars=("x", "y"))

    xs0, xs1 = x_range
    ys0, ys1 = y_range
    xs = _linspace(xs0, xs1, x_segments)
    ys = _linspace(ys0, ys1, y_segments)

    verts: List[Vec3] = []
    faces: List[Tri] = []

    # Vertex id grid; -1 means "invalid"
    vid: List[List[int]] = [[-1] * (x_segments + 1) for _ in range(y_segments + 1)]

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            z = f(x, y)
            if (not skip_nonfinite) or _is_finite(z):
                vid[iy][ix] = len(verts)
                verts.append((x, y, z))

    for iy in range(y_segments):
        for ix in range(x_segments):
            a = vid[iy][ix]
            b = vid[iy][ix + 1]
            c = vid[iy + 1][ix]
            d = vid[iy + 1][ix + 1]
            if skip_nonfinite and (a < 0 or b < 0 or c < 0 or d < 0):
                continue
            faces.append((a, c, b))
            faces.append((b, c, d))

    return Mesh(verts, faces, name=name).compute_normals(True)


# ---------------------------------------------------
# 2) Parametric surface: x(u,v), y(u,v), z(u,v)
# ---------------------------------------------------

def mesh_from_parametric(
    x_expr: str,
    y_expr: str,
    z_expr: str,
    *,
    u_range: Tuple[float, float] = (0.0, 1.0),
    v_range: Tuple[float, float] = (0.0, 1.0),
    u_segments: int = 200,
    v_segments: int = 80,
    wrap_u: bool = False,
    wrap_v: bool = False,
    name: str = "parametric",
    skip_nonfinite: bool = True,
) -> Mesh:
    """
    Build a triangle mesh for a parametric surface.

    Parameters
    ----------
    x_expr, y_expr, z_expr:
        Math expressions using variables u and v.
    u_range, v_range:
        Parameter sampling ranges.
    u_segments, v_segments:
        Number of intervals in u and v.
    wrap_u, wrap_v:
        If True, treats the surface as periodic in that parameter direction and
        stitches the seam by omitting the last row/column of points.
        Typical for tori, spheres in uv, etc.
    skip_nonfinite:
        If True, samples that produce NaN/Inf are omitted and triangles touching them
        are skipped (holes).

    Returns
    -------
    Mesh:
        Mesh with computed vertex normals.
    """
    _require_segments("u_segments", u_segments)
    _require_segments("v_segments", v_segments)

    fx = compile_math_expr(x_expr, vars=("u", "v"))
    fy = compile_math_expr(y_expr, vars=("u", "v"))
    fz = compile_math_expr(z_expr, vars=("u", "v"))

    u0, u1 = u_range
    v0, v1 = v_range

    ucount = u_segments + (0 if wrap_u else 1)
    vcount = v_segments + (0 if wrap_v else 1)

    us = [u0 + (u1 - u0) * (i / u_segments) for i in range(ucount)]
    vs = [v0 + (v1 - v0) * (j / v_segments) for j in range(vcount)]

    verts: List[Vec3] = []
    faces: List[Tri] = []

    vid: List[List[int]] = [[-1] * vcount for _ in range(ucount)]

    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            x = fx(u, v)
            y = fy(u, v)
            z = fz(u, v)
            if (not skip_nonfinite) or (_is_finite(x) and _is_finite(y) and _is_finite(z)):
                vid[i][j] = len(verts)
                verts.append((x, y, z))

    def idx(i: int, j: int) -> int:
        ii = (i % u_segments) if wrap_u else i
        jj = (j % v_segments) if wrap_v else j
        return vid[ii][jj]

    for i in range(u_segments):
        for j in range(v_segments):
            a = idx(i, j)
            b = idx(i + 1, j)
            c = idx(i, j + 1)
            d = idx(i + 1, j + 1)
            if skip_nonfinite and (a < 0 or b < 0 or c < 0 or d < 0):
                continue
            faces.append((a, b, c))
            faces.append((b, d, c))

    return Mesh(verts, faces, name=name).compute_normals(True)


# ---------------------------------------------------------
# 3) Implicit surface: f(x,y,z) = iso (marching tetrahedra)
#    Improvement: edge-cache keyed by grid corner ids
# ---------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _interp(p0: Vec3, p1: Vec3, v0: float, v1: float, iso: float) -> Vec3:
    dv = v1 - v0
    if abs(dv) < 1e-12:
        t = 0.5
    else:
        t = (iso - v0) / dv

    # clamp improves stability around flat regions / numeric noise
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    return (
        _lerp(p0[0], p1[0], t),
        _lerp(p0[1], p1[1], t),
        _lerp(p0[2], p1[2], t),
    )


def mesh_from_implicit(
    expr: str,
    *,
    bounds: Tuple[Vec3, Vec3] = ((-1.2, -1.2, -1.2), (1.2, 1.2, 1.2)),
    resolution: Tuple[int, int, int] = (48, 48, 48),
    iso: float = 0.0,
    name: str = "implicit",
    skip_nonfinite: bool = True,
) -> Mesh:
    """
    Build an iso-surface mesh for f(x,y,z) = iso using marching tetrahedra.

    Parameters
    ----------
    expr:
        Math expression using variables x, y, z.
        Example: "x*x + y*y + z*z - 1"  (sphere)
    bounds:
        ((xmin,ymin,zmin),(xmax,ymax,zmax)) sampling volume.
        Your surface must be inside these bounds.
    resolution:
        (nx, ny, nz) number of sample points along each axis.
        Higher values -> more detail (cost grows ~nx*ny*nz).
    iso:
        Extract surface where f(x,y,z) == iso.
    skip_nonfinite:
        If True, cubes that touch NaN/Inf samples are ignored.

    Mesh quality notes
    ------------------
    This implementation uses an edge-vertex cache keyed by *grid corner IDs*,
    which significantly reduces cracks and duplicated vertices compared to
    naive "welding by rounding".

    Returns
    -------
    Mesh:
        Mesh with computed vertex normals.
    """
    f = compile_math_expr(expr, vars=("x", "y", "z"))

    (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
    nx, ny, nz = resolution
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError("resolution must be at least (2,2,2)")

    xs = [xmin + (xmax - xmin) * (i / (nx - 1)) for i in range(nx)]
    ys = [ymin + (ymax - ymin) * (j / (ny - 1)) for j in range(ny)]
    zs = [zmin + (zmax - zmin) * (k / (nz - 1)) for k in range(nz)]

    def gidx(i: int, j: int, k: int) -> int:
        # Flattened scalar field: index = (i*ny + j)*nz + k
        return (i * ny + j) * nz + k

    vals = [0.0] * (nx * ny * nz)
    finite = [True] * (nx * ny * nz)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            base = (i * ny + j) * nz
            for k, z in enumerate(zs):
                v = f(x, y, z)
                vals[base + k] = v
                finite[base + k] = _is_finite(v)

    corner_off = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ]

    # Split cube into 6 tetrahedra along diagonal (0 -> 6)
    tets = [
        (0, 5, 1, 6),
        (0, 1, 2, 6),
        (0, 2, 3, 6),
        (0, 3, 7, 6),
        (0, 7, 4, 6),
        (0, 4, 5, 6),
    ]

    verts: List[Vec3] = []
    faces: List[Tri] = []

    # Edge cache: (gridCornerIdA, gridCornerIdB) -> vertexIndex
    edge_cache: Dict[Tuple[int, int], int] = {}

    def add_vert(p: Vec3) -> int:
        verts.append(p)
        return len(verts) - 1

    def edge_vertex(
        id0: int,
        id1: int,
        p0: Vec3,
        p1: Vec3,
        v0: float,
        v1: float,
    ) -> int:
        key = (id0, id1) if id0 < id1 else (id1, id0)
        hit = edge_cache.get(key)
        if hit is not None:
            return hit
        p = _interp(p0, p1, v0, v1, iso)
        vi = add_vert(p)
        edge_cache[key] = vi
        return vi

    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                # Gather cube corners (positions, scalar values, and global ids)
                P: List[Vec3] = []
                V: List[float] = []
                ID: List[int] = []

                ok = True
                for dx, dy, dz in corner_off:
                    ii, jj, kk = i + dx, j + dy, k + dz
                    gid = gidx(ii, jj, kk)
                    if skip_nonfinite and not finite[gid]:
                        ok = False
                        break
                    ID.append(gid)
                    P.append((xs[ii], ys[jj], zs[kk]))
                    V.append(vals[gid])

                if not ok:
                    continue

                # Polygonize each tetrahedron
                for a, b, c, d in tets:
                    pv = (P[a], P[b], P[c], P[d])
                    sv = (V[a], V[b], V[c], V[d])
                    iv = (ID[a], ID[b], ID[c], ID[d])

                    inside = (sv[0] < iso, sv[1] < iso, sv[2] < iso, sv[3] < iso)
                    n_in = int(inside[0]) + int(inside[1]) + int(inside[2]) + int(inside[3])

                    if n_in == 0 or n_in == 4:
                        continue

                    in_idx = [q for q in range(4) if inside[q]]
                    out_idx = [q for q in range(4) if not inside[q]]

                    if n_in == 1:
                        vi_in = in_idx[0]
                        o0, o1, o2 = out_idx
                        i0 = edge_vertex(iv[vi_in], iv[o0], pv[vi_in], pv[o0], sv[vi_in], sv[o0])
                        i1 = edge_vertex(iv[vi_in], iv[o1], pv[vi_in], pv[o1], sv[vi_in], sv[o1])
                        i2 = edge_vertex(iv[vi_in], iv[o2], pv[vi_in], pv[o2], sv[vi_in], sv[o2])
                        faces.append((i0, i1, i2))

                    elif n_in == 3:
                        vi_out = out_idx[0]
                        i0, i1, i2 = in_idx
                        a0 = edge_vertex(iv[vi_out], iv[i0], pv[vi_out], pv[i0], sv[vi_out], sv[i0])
                        a1 = edge_vertex(iv[vi_out], iv[i1], pv[vi_out], pv[i1], sv[vi_out], sv[i1])
                        a2 = edge_vertex(iv[vi_out], iv[i2], pv[vi_out], pv[i2], sv[vi_out], sv[i2])
                        faces.append((a0, a2, a1))  # flipped orientation

                    else:
                        # n_in == 2 => quad => 2 triangles
                        i0, i1 = in_idx
                        o0, o1 = out_idx
                        a0 = edge_vertex(iv[i0], iv[o0], pv[i0], pv[o0], sv[i0], sv[o0])
                        a1 = edge_vertex(iv[i0], iv[o1], pv[i0], pv[o1], sv[i0], sv[o1])
                        a2 = edge_vertex(iv[i1], iv[o0], pv[i1], pv[o0], sv[i1], sv[o0])
                        a3 = edge_vertex(iv[i1], iv[o1], pv[i1], pv[o1], sv[i1], sv[o1])
                        faces.append((a0, a1, a2))
                        faces.append((a2, a1, a3))

    return Mesh(verts, faces, name=name).compute_normals(True)


def mesh_from_implicit_cell_size(
    expr: str,
    *,
    bounds: Tuple[Vec3, Vec3],
    cell_size: float,
    iso: float = 0.0,
    name: str = "implicit",
    min_resolution: int = 12,
    max_resolution: int = 256,
    skip_nonfinite: bool = True,
) -> Mesh:
    """
    Convenience wrapper around mesh_from_implicit() that chooses resolution from a
    desired world-space grid cell size.

    Example:
        mesh = mesh_from_implicit_cell_size(
            "x*x + y*y + z*z - 1",
            bounds=((-1.3,-1.3,-1.3),(1.3,1.3,1.3)),
            cell_size=0.03,
            max_resolution=200
        )
    """
    res = resolution_from_cell_size(
        bounds,
        cell_size,
        min_resolution=min_resolution,
        max_resolution=max_resolution,
    )
    return mesh_from_implicit(
        expr,
        bounds=bounds,
        resolution=res,
        iso=iso,
        name=name,
        skip_nonfinite=skip_nonfinite,
    )
