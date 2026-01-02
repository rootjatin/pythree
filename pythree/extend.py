# pythree/ext.py
from __future__ import annotations

import base64
import json
import math
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---- import Mesh from your core module ----
# If your file is pythree/geomesh.py, keep this.
# If it's pythree/mesh.py, change to: from .mesh import Mesh
try:
    from .geomesh import Mesh  # type: ignore
except Exception:
    # fallback if you re-export Mesh in pythree/__init__.py
    from . import Mesh  # type: ignore


Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]
Tri = Tuple[int, int, int]
Vec2f = Tuple[float, float]


# ----------------------------
# Small local vector utilities
# ----------------------------

def v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def v_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )

def v_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def v_len(a: Vec3) -> float:
    return math.sqrt(v_dot(a, a))

def v_norm(a: Vec3) -> Vec3:
    l = v_len(a)
    if l == 0.0:
        return (0.0, 0.0, 0.0)
    return (a[0] / l, a[1] / l, a[2] / l)


# ----------------------------
# Mesh analysis: area & volume
# ----------------------------

def triangle_area(a: Vec3, b: Vec3, c: Vec3) -> float:
    ab = v_sub(b, a)
    ac = v_sub(c, a)
    cr = v_cross(ab, ac)
    return 0.5 * v_len(cr)

def mesh_surface_area(mesh: "Mesh") -> float:
    area = 0.0
    for ia, ib, ic in mesh.faces:
        a, b, c = mesh.vertices[ia], mesh.vertices[ib], mesh.vertices[ic]
        area += triangle_area(a, b, c)
    return area

def mesh_volume(mesh: "Mesh") -> float:
    """
    Signed volume for a closed, consistently oriented triangle mesh.
    Uses origin-based tetrahedron summation: V = sum(dot(a, cross(b,c))) / 6
    """
    vol6 = 0.0
    for ia, ib, ic in mesh.faces:
        a, b, c = mesh.vertices[ia], mesh.vertices[ib], mesh.vertices[ic]
        vol6 += v_dot(a, v_cross(b, c))
    return vol6 / 6.0


# -----------------------------------
# 2D polygon helpers + triangulation
# -----------------------------------

def _poly_area2(poly: Sequence[Vec2f]) -> float:
    """Signed 2D polygon area * 2. CCW => positive."""
    s = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return s

def _is_ccw(poly: Sequence[Vec2f]) -> bool:
    return _poly_area2(poly) > 0.0

def _cross2(a: Vec2f, b: Vec2f, c: Vec2f) -> float:
    """2D cross (b-a)x(c-a) z-component."""
    abx, aby = (b[0] - a[0], b[1] - a[1])
    acx, acy = (c[0] - a[0], c[1] - a[1])
    return abx * acy - aby * acx

def _pt_in_tri(p: Vec2f, a: Vec2f, b: Vec2f, c: Vec2f) -> bool:
    """Barycentric point-in-triangle (inclusive)."""
    px, py = p
    ax, ay = a
    bx, by = b
    cx, cy = c

    v0x, v0y = (cx - ax, cy - ay)
    v1x, v1y = (bx - ax, by - ay)
    v2x, v2y = (px - ax, py - ay)

    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot02 = v0x * v2x + v0y * v2y
    dot11 = v1x * v1x + v1y * v1y
    dot12 = v1x * v2x + v1y * v2y

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-18:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return (u >= -1e-12) and (v >= -1e-12) and (u + v <= 1.0 + 1e-12)

def triangulate_polygon_earclip(poly: Sequence[Vec2f]) -> List[Tri]:
    """
    Ear-clipping triangulation for a simple polygon (single ring).
    Supports concave polygons. No holes/self-intersections.
    Returns triangle indices into the original poly list.
    """
    if len(poly) < 3:
        return []

    idx = list(range(len(poly)))
    if not _is_ccw(poly):
        idx.reverse()

    result: List[Tri] = []
    guard = 0

    while len(idx) > 3 and guard < 100000:
        guard += 1
        ear_found = False

        for i in range(len(idx)):
            i0 = idx[(i - 1) % len(idx)]
            i1 = idx[i]
            i2 = idx[(i + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]

            # Must be convex in CCW polygon
            if _cross2(a, b, c) <= 1e-15:
                continue

            # No other point inside
            ok = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                if _pt_in_tri(poly[j], a, b, c):
                    ok = False
                    break
            if not ok:
                continue

            result.append((i0, i1, i2))
            del idx[i]
            ear_found = True
            break

        if not ear_found:
            # fallback fan
            base = idx[0]
            for k in range(1, len(idx) - 1):
                result.append((base, idx[k], idx[k + 1]))
            return result

    if len(idx) == 3:
        result.append((idx[0], idx[1], idx[2]))

    return result


def extrude_polygon_any(polygon_xy: Sequence[Vec2f], height: float, name: str = "prism") -> "Mesh":
    """
    Extrude a simple polygon (convex or concave) in XY by height along +Z.
    Single ring only (no holes).
    """
    poly = list(polygon_xy)
    n = len(poly)
    if n < 3:
        return Mesh(name=name)

    verts: List[Vec3] = [(x, y, 0.0) for (x, y) in poly] + [(x, y, height) for (x, y) in poly]
    faces: List[Tri] = []

    # sides
    for i in range(n):
        a = i
        b = (i + 1) % n
        c = n + i
        d = n + ((i + 1) % n)
        faces.append((a, b, c))
        faces.append((b, d, c))

    tris = triangulate_polygon_earclip(poly)
    poly_ccw = _is_ccw(poly)

    # top (+Z)
    for i0, i1, i2 in tris:
        if poly_ccw:
            faces.append((n + i0, n + i1, n + i2))
        else:
            faces.append((n + i2, n + i1, n + i0))

    # bottom (-Z)
    for i0, i1, i2 in tris:
        if poly_ccw:
            faces.append((i2, i1, i0))
        else:
            faces.append((i0, i1, i2))

    return Mesh(verts, faces, name=name).compute_normals(True)


# -------------------------
# Minimal glTF 2.0 exporter
# -------------------------

@dataclass
class _GltfBuffers:
    bin: bytearray
    views: List[Dict[str, Any]]
    accessors: List[Dict[str, Any]]

def _pad4(n: int) -> int:
    return (n + 3) & ~3

def _pack_f32(xs: List[float]) -> bytes:
    return struct.pack("<" + "f" * len(xs), *xs)

def _pack_u16(xs: List[int]) -> bytes:
    return struct.pack("<" + "H" * len(xs), *xs)

def _pack_u32(xs: List[int]) -> bytes:
    return struct.pack("<" + "I" * len(xs), *xs)

def _flatten_vec3(vs: List[Vec3]) -> List[float]:
    out: List[float] = []
    for x, y, z in vs:
        out += [x, y, z]
    return out

def _flatten_vec2(vs: List[Vec2]) -> List[float]:
    out: List[float] = []
    for u, v in vs:
        out += [u, v]
    return out

def _minmax_vec3(vs: List[Vec3]) -> Tuple[List[float], List[float]]:
    xs = [v[0] for v in vs]
    ys = [v[1] for v in vs]
    zs = [v[2] for v in vs]
    return [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]

def _add_buffer_view(buffers: _GltfBuffers, blob: bytes, target: int) -> int:
    offset = len(buffers.bin)
    buffers.bin.extend(blob)
    pad = _pad4(len(buffers.bin)) - len(buffers.bin)
    if pad:
        buffers.bin.extend(b"\x00" * pad)

    view_i = len(buffers.views)
    buffers.views.append({
        "buffer": 0,
        "byteOffset": offset,
        "byteLength": len(blob),
        "target": target,  # 34962 ARRAY_BUFFER / 34963 ELEMENT_ARRAY_BUFFER
    })
    return view_i

def _add_accessor(buffers: _GltfBuffers, view_index: int, component_type: int, count: int,
                  type_str: str, *, minv: Optional[List[float]] = None, maxv: Optional[List[float]] = None) -> int:
    acc: Dict[str, Any] = {
        "bufferView": view_index,
        "componentType": component_type,
        "count": count,
        "type": type_str,
    }
    if minv is not None:
        acc["min"] = minv
    if maxv is not None:
        acc["max"] = maxv
    i = len(buffers.accessors)
    buffers.accessors.append(acc)
    return i

def _mesh_indices(mesh: "Mesh") -> List[int]:
    out: List[int] = []
    for a, b, c in mesh.faces:
        out += [a, b, c]
    return out

def save_gltf(path: str, mesh: "Mesh", *, embed_buffer: bool = True, ensure_normals: bool = True) -> None:
    """
    Save glTF 2.0 .gltf.
    - embed_buffer=True => one .gltf file with base64 buffer
    - embed_buffer=False => writes sibling .bin
    """
    if ensure_normals and (mesh.normals is None or len(mesh.normals) != len(mesh.vertices)):
        mesh = mesh.copy().compute_normals(True)

    buffers = _GltfBuffers(bin=bytearray(), views=[], accessors=[])

    # positions
    pos_blob = _pack_f32(_flatten_vec3(mesh.vertices))
    pos_view = _add_buffer_view(buffers, pos_blob, 34962)
    pos_min, pos_max = _minmax_vec3(mesh.vertices)
    pos_acc = _add_accessor(buffers, pos_view, 5126, len(mesh.vertices), "VEC3", minv=pos_min, maxv=pos_max)

    attrs: Dict[str, int] = {"POSITION": pos_acc}

    # normals
    if mesh.normals and len(mesh.normals) == len(mesh.vertices):
        nrm_blob = _pack_f32(_flatten_vec3(mesh.normals))
        nrm_view = _add_buffer_view(buffers, nrm_blob, 34962)
        nrm_acc = _add_accessor(buffers, nrm_view, 5126, len(mesh.vertices), "VEC3")
        attrs["NORMAL"] = nrm_acc

    # uvs
    if mesh.uvs and len(mesh.uvs) == len(mesh.vertices):
        uv_blob = _pack_f32(_flatten_vec2(mesh.uvs))
        uv_view = _add_buffer_view(buffers, uv_blob, 34962)
        uv_acc = _add_accessor(buffers, uv_view, 5126, len(mesh.vertices), "VEC2")
        attrs["TEXCOORD_0"] = uv_acc

    # indices
    idx = _mesh_indices(mesh)
    max_index = max(idx) if idx else 0
    if max_index < 65536:
        idx_blob = _pack_u16(idx)
        idx_type = 5123  # UNSIGNED_SHORT
    else:
        idx_blob = _pack_u32(idx)
        idx_type = 5125  # UNSIGNED_INT

    idx_view = _add_buffer_view(buffers, idx_blob, 34963)
    idx_acc = _add_accessor(buffers, idx_view, idx_type, len(idx), "SCALAR")

    gltf: Dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "pythree.ext"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": mesh.name}],
        "meshes": [{
            "name": mesh.name,
            "primitives": [{
                "attributes": attrs,
                "indices": idx_acc,
                "mode": 4,  # TRIANGLES
            }]
        }],
        "buffers": [{"byteLength": len(buffers.bin)}],
        "bufferViews": buffers.views,
        "accessors": buffers.accessors,
    }

    if embed_buffer:
        uri = "data:application/octet-stream;base64," + base64.b64encode(bytes(buffers.bin)).decode("ascii")
        gltf["buffers"][0]["uri"] = uri
        with open(path, "w", encoding="utf-8") as f:
            json.dump(gltf, f, ensure_ascii=False, separators=(",", ":"))
    else:
        bin_path = path.rsplit(".", 1)[0] + ".bin"
        gltf["buffers"][0]["uri"] = bin_path.split("/")[-1].split("\\")[-1]
        with open(bin_path, "wb") as bf:
            bf.write(bytes(buffers.bin))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(gltf, f, ensure_ascii=False, indent=2)

def save_glb(path: str, mesh: "Mesh", *, ensure_normals: bool = True) -> None:
    """Save GLB (binary glTF 2.0) in one file."""
    if ensure_normals and (mesh.normals is None or len(mesh.normals) != len(mesh.vertices)):
        mesh = mesh.copy().compute_normals(True)

    buffers = _GltfBuffers(bin=bytearray(), views=[], accessors=[])

    pos_blob = _pack_f32(_flatten_vec3(mesh.vertices))
    pos_view = _add_buffer_view(buffers, pos_blob, 34962)
    pos_min, pos_max = _minmax_vec3(mesh.vertices)
    pos_acc = _add_accessor(buffers, pos_view, 5126, len(mesh.vertices), "VEC3", minv=pos_min, maxv=pos_max)

    attrs: Dict[str, int] = {"POSITION": pos_acc}

    if mesh.normals and len(mesh.normals) == len(mesh.vertices):
        nrm_blob = _pack_f32(_flatten_vec3(mesh.normals))
        nrm_view = _add_buffer_view(buffers, nrm_blob, 34962)
        nrm_acc = _add_accessor(buffers, nrm_view, 5126, len(mesh.vertices), "VEC3")
        attrs["NORMAL"] = nrm_acc

    if mesh.uvs and len(mesh.uvs) == len(mesh.vertices):
        uv_blob = _pack_f32(_flatten_vec2(mesh.uvs))
        uv_view = _add_buffer_view(buffers, uv_blob, 34962)
        uv_acc = _add_accessor(buffers, uv_view, 5126, len(mesh.vertices), "VEC2")
        attrs["TEXCOORD_0"] = uv_acc

    idx = _mesh_indices(mesh)
    max_index = max(idx) if idx else 0
    if max_index < 65536:
        idx_blob = _pack_u16(idx)
        idx_type = 5123
    else:
        idx_blob = _pack_u32(idx)
        idx_type = 5125

    idx_view = _add_buffer_view(buffers, idx_blob, 34963)
    idx_acc = _add_accessor(buffers, idx_view, idx_type, len(idx), "SCALAR")

    gltf: Dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "pythree.ext"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": mesh.name}],
        "meshes": [{
            "name": mesh.name,
            "primitives": [{
                "attributes": attrs,
                "indices": idx_acc,
                "mode": 4,
            }]
        }],
        "buffers": [{"byteLength": len(buffers.bin)}],
        "bufferViews": buffers.views,
        "accessors": buffers.accessors,
    }

    json_bytes = json.dumps(gltf, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    json_pad = _pad4(len(json_bytes)) - len(json_bytes)
    json_chunk = json_bytes + (b" " * json_pad)

    bin_bytes = bytes(buffers.bin)
    bin_pad = _pad4(len(bin_bytes)) - len(bin_bytes)
    bin_chunk = bin_bytes + (b"\x00" * bin_pad)

    total_len = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)

    with open(path, "wb") as f:
        f.write(b"glTF")
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<I", total_len))

        f.write(struct.pack("<I", len(json_chunk)))
        f.write(b"JSON")
        f.write(json_chunk)

        f.write(struct.pack("<I", len(bin_chunk)))
        f.write(b"BIN\x00")
        f.write(bin_chunk)
