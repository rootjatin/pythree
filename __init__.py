# pythree/__init__.py
"""
pythree

A lightweight 3D geometry / mesh package with helpers for:
- safe math expression compilation
- heightfield meshes
- parametric surfaces
- implicit surfaces
"""

from .sphere import Mesh
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
]
