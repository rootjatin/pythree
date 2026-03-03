# pythree/__init__.py
"""
pythree

A lightweight 3D geometry / mesh package with helpers for:
- safe math expression compilation
- heightfield meshes
- parametric surfaces
- implicit surfaces
- vector and mesh utilities
"""

from .mesh import Mesh
from .equations import (
    CompiledMathExpr,
    compile_math_expr,
    segments_from_step,
    resolution_from_cell_size,
    mesh_from_heightfield,
    mesh_from_parametric,
    mesh_from_implicit,
    mesh_from_implicit_cell_size,
)
from .utils import (
    vec_add,
    vec_sub,
    vec_mul,
    vec_div,
    dot,
    cross,
    length,
    distance,
    normalize,
    lerp,
    triangle_normal,
    triangle_area,
    triangle_centroid,
    bounds_from_points,
    bounds_union,
    bounds_size,
    bounds_center,
    mesh_bounds,
    mesh_centroid,
    mesh_surface_area,
    mesh_stats,
    validate_faces,
    validate_vertices,
    validate_mesh,
    translated,
    scaled,
    recentered,
    merge_meshes,
    remove_degenerate_faces,
    unique_edges,
    print_mesh_stats,
)

__version__ = "0.1.0"

__all__ = [
    "Mesh",
    "CompiledMathExpr",
    "compile_math_expr",
    "segments_from_step",
    "resolution_from_cell_size",
    "mesh_from_heightfield",
    "mesh_from_parametric",
    "mesh_from_implicit",
    "mesh_from_implicit_cell_size",
    "vec_add",
    "vec_sub",
    "vec_mul",
    "vec_div",
    "dot",
    "cross",
    "length",
    "distance",
    "normalize",
    "lerp",
    "triangle_normal",
    "triangle_area",
    "triangle_centroid",
    "bounds_from_points",
    "bounds_union",
    "bounds_size",
    "bounds_center",
    "mesh_bounds",
    "mesh_centroid",
    "mesh_surface_area",
    "mesh_stats",
    "validate_faces",
    "validate_vertices",
    "validate_mesh",
    "translated",
    "scaled",
    "recentered",
    "merge_meshes",
    "remove_degenerate_faces",
    "unique_edges",
    "print_mesh_stats",
]
