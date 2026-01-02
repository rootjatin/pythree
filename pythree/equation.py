# pythree/equations.py
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Change this import if you rename sphere.py -> something else (e.g., core.py)
from .sphere import Mesh  # type: ignore

Vec3 = Tuple[float, float, float]
Tri = Tuple[int, int, int]


# -----------------------------
# Safe math expression compiler
# -----------------------------

_ALLOWED_FUNCS: Dict[str, Callable[..., float]] = {
    # trig
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    # exp/log/pow
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10,
    "pow": pow,
    # misc
    "abs": abs,
    "floor": math.floor, "ceil": math.ceil,
    "min": min, "max": max,
}

_ALLOWED_CONSTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau if hasattr(math, "tau") else 2 * math.pi,
}

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp, ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Call,
    ast.Name,
    ast.Constant,  # py3.8+
)


def compile_math_expr(expr: str, *, vars: Sequence[str]) -> Callable[..., float]:
    """
    Compile a restricted math expression like "sin(x) + y*y" into a function.
    Allowed:
      - numbers, + - * / % **, parentheses
      - variables in `vars`
      - functions in _ALLOWED_FUNCS
      - constants pi, e, tau
    Disallowed:
      - attribute access, indexing, comprehensions, lambdas, imports, etc.
    """
    tree = ast.parse(expr, mode="eval")

    def _check(node: ast.AST) -> None:
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Disallowed syntax: {type(node).__name__}")

        # Disallow things not in the node whitelist even if they appear as subnodes
        for child in ast.iter_child_nodes(node):
            _check(child)

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are allowed (e.g., sin(x))")
            if node.func.id not in _ALLOWED_FUNCS:
                raise ValueError(f"Function not allowed: {node.func.id}")

        if isinstance(node, ast.Name):
            if node.id not in vars and node.id not in _ALLOWED_FUNCS and node.id not in _ALLOWED_CONSTS:
                raise ValueError(f"Name not allowed: {node.id}")

    _check(tree)

    code = compile(tree, "<math_expr>", "eval")

    def _fn(**kwargs: float) -> float:
        scope = {"__builtins__": {}}
        scope.update(_ALLOWED_FUNCS)
        scope.update(_ALLOWED_CONSTS)
        scope.update(kwargs)
        return float(eval(code, scope, {}))  # safe because AST is restricted

    return _fn


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
    f = compile_math_expr(expr, vars=("x", "y"))

    xs0, xs1 = x_range
    ys0, ys1 = y_range

    verts: List[Vec3] = []
    faces: List[Tri] = []

    for iy in range(y_segments + 1):
        ty = iy / y_segments
        y = ys0 + ty * (ys1 - ys0)
        for ix in range(x_segments + 1):
            tx = ix / x_segments
            x = xs0 + tx * (xs1 - xs0)
            z = f(x=x, y=y)
            verts.append((x, y, z))

            if ix < x_segments and iy < y_segments:
                a = iy * (x_segments + 1) + ix
                b = a + 1
                c = (iy + 1) * (x_segments + 1) + ix
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
    fx = compile_math_expr(x_expr, vars=("u", "v"))
    fy = compile_math_expr(y_expr, vars=("u", "v"))
    fz = compile_math_expr(z_expr, vars=("u", "v"))

    u0, u1 = u_range
    v0, v1 = v_range

    ucount = u_segments + (0 if wrap_u else 1)
    vcount = v_segments + (0 if wrap_v else 1)

    verts: List[Vec3] = []
    faces: List[Tri] = []

    for i in range(ucount):
        tu = i / u_segments
        u = u0 + tu * (u1 - u0)
        for j in range(vcount):
            tv = j / v_segments
            v = v0 + tv * (v1 - v0)
            verts.append((fx(u=u, v=v), fy(u=u, v=v), fz(u=u, v=v)))

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
    dv = (v1 - v0)
    if abs(dv) < 1e-12:
        t = 0.5
    else:
        t = (iso - v0) / dv
    return (_lerp(p0[0], p1[0], t), _lerp(p0[1], p1[1], t), _lerp(p0[2], p1[2], t))

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
    - Higher resolution => smoother but slower.
    - bounds should fully contain your surface.
    """
    f = compile_math_expr(expr, vars=("x", "y", "z"))

    (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
    nx, ny, nz = resolution
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError("resolution must be at least (2,2,2)")

    # Sample grid values
    xs = [xmin + (xmax - xmin) * (i / (nx - 1)) for i in range(nx)]
    ys = [ymin + (ymax - ymin) * (j / (ny - 1)) for j in range(ny)]
    zs = [zmin + (zmax - zmin) * (k / (nz - 1)) for k in range(nz)]

    vals = [[[0.0 for _ in range(nz)] for _ in range(ny)] for _ in range(nx)]
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                vals[i][j][k] = f(x=x, y=y, z=z)

    # cube corner offsets (0..7)
    corner_off = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
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

    # For optional welding without calling mesh.weld later:
    cache: Dict[Tuple[int, int, int], int] = {}

    def add_v_cached(p: Vec3) -> int:
        if weld_eps is None:
            return add_v(p)
        k = (round(p[0] / weld_eps), round(p[1] / weld_eps), round(p[2] / weld_eps))
        hit = cache.get(k)
        if hit is not None:
            return hit
        i = add_v(p)
        cache[k] = i
        return i

    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                # positions + scalar values at 8 cube corners
                P: List[Vec3] = []
                V: List[float] = []
                for (dx, dy, dz) in corner_off:
                    x = xs[i + dx]
                    y = ys[j + dy]
                    z = zs[k + dz]
                    P.append((x, y, z))
                    V.append(vals[i + dx][j + dy][k + dz])

                # polygonize each tetra
                for (a, b, c, d) in tets:
                    ids = [a, b, c, d]
                    pv = [P[t] for t in ids]
                    sv = [V[t] for t in ids]
                    inside = [sv[q] < iso for q in range(4)]
                    n_in = sum(1 for x in inside if x)

                    if n_in == 0 or n_in == 4:
                        continue

                    # helper lists
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
                        # 1 outside => triangle too, flip orientation
                        vo = out_idx[0]
                        i0, i1, i2 = in_idx
                        p0 = _interp(pv[vo], pv[i0], sv[vo], sv[i0], iso)
                        p1 = _interp(pv[vo], pv[i1], sv[vo], sv[i1], iso)
                        p2 = _interp(pv[vo], pv[i2], sv[vo], sv[i2], iso)
                        a0 = add_v_cached(p0)
                        a1 = add_v_cached(p1)
                        a2 = add_v_cached(p2)
                        faces.append((a0, a2, a1))

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
