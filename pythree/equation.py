# pythree/equations.py
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .sphere import Mesh  # type: ignore

Vec3 = Tuple[float, float, float]
Tri = Tuple[int, int, int]


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

# NOTE: ast.Load is required for modern Python AST traversal
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
            raise ValueError(f"Only numeric constants are allowed (got {type(node.value).__name__})")
        self.generic_visit(node)


@dataclass(frozen=True)
class CompiledMathExpr:
    """
    A validated expression compiled into a lambda for fast repeated evaluation.
    Supports positional calls in the declared var order, and keyword calls.
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
                raise TypeError(f"Expected {len(self.vars)} args ({', '.join(self.vars)}), got {len(args)}")
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

    fn = eval(src, base_scope, {})  # safe because AST is validated + builtins removed
    return CompiledMathExpr(expr=expr, vars=tuple(vars), _fn=fn)


# -------------------------
# Helpers
# -------------------------

def _require_segments(name: str, n: int) -> None:
    if n < 1:
        raise ValueError(f"{name} must be >= 1 (got {n})")


def _linspace(a: float, b: float, segments: int) -> List[float]:
    # segments = number of intervals; points = segments + 1
    step = (b - a) / segments
    return [a + i * step for i in range(segments + 1)]


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
) -> Mesh:
    _require_segments("x_segments", x_segments)
    _require_segments("y_segments", y_segments)

    f = compile_math_expr(expr, vars=("x", "y"))

    xs0, xs1 = x_range
    ys0, ys1 = y_range
    xs = _linspace(xs0, xs1, x_segments)
    ys = _linspace(ys0, ys1, y_segments)

    verts: List[Vec3] = []
    faces: List[Tri] = []

    row_stride = x_segments + 1

    for iy, y in enumerate(ys):
        row_base = iy * row_stride
        for ix, x in enumerate(xs):
            verts.append((x, y, f(x, y)))

            if ix < x_segments and iy < y_segments:
                a = row_base + ix
                b = a + 1
                c = a + row_stride
                d = c + 1
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
) -> Mesh:
    _require_segments("u_segments", u_segments)
    _require_segments("v_segments", v_segments)

    fx = compile_math_expr(x_expr, vars=("u", "v"))
    fy = compile_math_expr(y_expr, vars=("u", "v"))
    fz = compile_math_expr(z_expr, vars=("u", "v"))

    u0, u1 = u_range
    v0, v1 = v_range

    ucount = u_segments + (0 if wrap_u else 1)
    vcount = v_segments + (0 if wrap_v else 1)

    # Build parameter samples (note: if wrapping, last seam point is omitted)
    us = [u0 + (u1 - u0) * (i / u_segments) for i in range(ucount)]
    vs = [v0 + (v1 - v0) * (j / v_segments) for j in range(vcount)]

    verts: List[Vec3] = []
    faces: List[Tri] = []

    for u in us:
        for v in vs:
            verts.append((fx(u, v), fy(u, v), fz(u, v)))

    def idx(i: int, j: int) -> int:
        ii = (i % u_segments) if wrap_u else i
        jj = (j % v_segments) if wrap_v else j
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


# ---------------------------------------------------------
# 3) Implicit surface: f(x,y,z) = iso (marching tetrahedra)
# ---------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _interp(p0: Vec3, p1: Vec3, v0: float, v1: float, iso: float) -> Vec3:
    dv = v1 - v0
    t = 0.5 if abs(dv) < 1e-12 else (iso - v0) / dv
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
    weld_eps: Optional[float] = 1e-6,
) -> Mesh:
    """
    Build an iso-surface mesh for f(x,y,z) = iso using marching tetrahedra.

    Tips:
      - Higher resolution => smoother but slower.
      - bounds should fully contain your surface.
      - weld_eps controls vertex welding during polygonization (None disables).
    """
    f = compile_math_expr(expr, vars=("x", "y", "z"))

    (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
    nx, ny, nz = resolution
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError("resolution must be at least (2,2,2)")

    xs = [xmin + (xmax - xmin) * (i / (nx - 1)) for i in range(nx)]
    ys = [ymin + (ymax - ymin) * (j / (ny - 1)) for j in range(ny)]
    zs = [zmin + (zmax - zmin) * (k / (nz - 1)) for k in range(nz)]

    # Flattened scalar field: index = (i*ny + j)*nz + k
    def vidx(i: int, j: int, k: int) -> int:
        return (i * ny + j) * nz + k

    vals = [0.0] * (nx * ny * nz)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            base = (i * ny + j) * nz
            for k, z in enumerate(zs):
                vals[base + k] = f(x, y, z)

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

    def add_v(p: Vec3) -> int:
        verts.append(p)
        return len(verts) - 1

    cache: Dict[Tuple[int, int, int], int] = {}

    def add_v_cached(p: Vec3) -> int:
        if weld_eps is None:
            return add_v(p)
        eps = weld_eps
        key = (round(p[0] / eps), round(p[1] / eps), round(p[2] / eps))
        hit = cache.get(key)
        if hit is not None:
            return hit
        i = add_v(p)
        cache[key] = i
        return i

    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                # Gather cube corners
                P: List[Vec3] = []
                V: List[float] = []
                for dx, dy, dz in corner_off:
                    ii, jj, kk = i + dx, j + dy, k + dz
                    P.append((xs[ii], ys[jj], zs[kk]))
                    V.append(vals[vidx(ii, jj, kk)])

                # Polygonize each tetrahedron
                for a, b, c, d in tets:
                    ids = (a, b, c, d)
                    pv = (P[ids[0]], P[ids[1]], P[ids[2]], P[ids[3]])
                    sv = (V[ids[0]], V[ids[1]], V[ids[2]], V[ids[3]])

                    inside = (sv[0] < iso, sv[1] < iso, sv[2] < iso, sv[3] < iso)
                    n_in = int(inside[0]) + int(inside[1]) + int(inside[2]) + int(inside[3])

                    if n_in == 0 or n_in == 4:
                        continue

                    in_idx = [q for q in range(4) if inside[q]]
                    out_idx = [q for q in range(4) if not inside[q]]

                    if n_in == 1:
                        vi = in_idx[0]
                        o0, o1, o2 = out_idx
                        p0 = _interp(pv[vi], pv[o0], sv[vi], sv[o0], iso)
                        p1 = _interp(pv[vi], pv[o1], sv[vi], sv[o1], iso)
                        p2 = _interp(pv[vi], pv[o2], sv[vi], sv[o2], iso)
                        i0 = add_v_cached(p0)
                        i1 = add_v_cached(p1)
                        i2 = add_v_cached(p2)
                        faces.append((i0, i1, i2))

                    elif n_in == 3:
                        vo = out_idx[0]
                        i0, i1, i2 = in_idx
                        p0 = _interp(pv[vo], pv[i0], sv[vo], sv[i0], iso)
                        p1 = _interp(pv[vo], pv[i1], sv[vo], sv[i1], iso)
                        p2 = _interp(pv[vo], pv[i2], sv[vo], sv[i2], iso)
                        a0 = add_v_cached(p0)
                        a1 = add_v_cached(p1)
                        a2 = add_v_cached(p2)
                        faces.append((a0, a2, a1))  # flipped orientation

                    else:
                        # n_in == 2 => quad => 2 triangles
                        i0, i1 = in_idx
                        o0, o1 = out_idx
                        p0 = _interp(pv[i0], pv[o0], sv[i0], sv[o0], iso)
                        p1 = _interp(pv[i0], pv[o1], sv[i0], sv[o1], iso)
                        p2 = _interp(pv[i1], pv[o0], sv[i1], sv[o0], iso)
                        p3 = _interp(pv[i1], pv[o1], sv[i1], sv[o1], iso)
                        a0 = add_v_cached(p0)
                        a1 = add_v_cached(p1)
                        a2 = add_v_cached(p2)
                        a3 = add_v_cached(p3)
                        faces.append((a0, a1, a2))
                        faces.append((a2, a1, a3))

    return Mesh(verts, faces, name=name).compute_normals(True)
