# Practical Guide to `pythree`

This README is a plain-English usage guide for the [`rootjatin/pythree`](https://github.com/rootjatin/pythree) package, based on reading the repository source instead of relying on the current official docs.

## What `pythree` is

`pythree` is best understood as a **small 3D triangle-mesh generation toolkit**.

It is useful for:
- generating meshes from mathematical equations
- creating sampled surfaces
- building implicit 3D shapes
- inspecting and transforming meshes
- exporting meshes to formats like `.obj`

It is **not** a full 3D renderer or scene engine.

---

## What the package contains

The repository includes a few main modules:

- `pythree.mesh` — core `Mesh` container and mesh operations
- `pythree.equation` — generate meshes from heightfields, parametric equations, and implicit surfaces
- `pythree.utils` — validation, statistics, merging, scaling, translating, cleanup helpers
- `pythree.material` — simple material/color metadata
- `pythree.sphere` — a separate primitive/export-oriented module with overlapping functionality

---

## Important warning before you use it

The repo is currently **inconsistent in a few places**, which is why it feels hard to understand.

Examples:
- top-level imports appear to reference `.equations`, while the actual file is `equation.py`
- version numbers are not fully consistent across files
- there appear to be **two partially overlapping mesh APIs** in the repository

Because of that, the safest way to use it is:

```python
from pythree.equation import ...
from pythree.mesh import Mesh
from pythree.utils import ...
```

Instead of relying on:

```python
import pythree
```

---

## Best mental model

Think of `pythree` like this:

> **Generate triangle meshes from equations, inspect or transform them, then export them elsewhere.**

That is the cleanest and most reliable way to use it.

---

## Recommended workflow

The most reliable workflow is:

1. Generate a mesh from an equation
2. Validate the mesh
3. Inspect stats like bounds, centroid, area, face count
4. Transform the mesh if needed
5. Export it to `.obj` or another format

---

## The cleanest usable API

## 1. Heightfield meshes

Use this when your surface is of the form:

```text
z = f(x, y)
```

Example:

```python
from pythree.equation import segments_from_step, mesh_from_heightfield
from pythree.utils import print_mesh_stats, validate_mesh

x_segments = segments_from_step((-3, 3), step=0.05)
y_segments = segments_from_step((-3, 3), step=0.05)

mesh = mesh_from_heightfield(
    "sin(x) * cos(y)",
    x_range=(-3, 3),
    y_range=(-3, 3),
    x_segments=x_segments,
    y_segments=y_segments,
    name="wave_surface",
)

validate_mesh(mesh)
print_mesh_stats(mesh)

print(mesh.vertex_count)
print(mesh.face_count)
print(mesh.surface_area())
```

### When to use this
- terrain-like surfaces
- wave surfaces
- scalar fields over a 2D domain
- math visualizations where `z` depends on `x` and `y`

---

## 2. Parametric surfaces

Use this when your shape is naturally written as:

```text
x = X(u, v)
y = Y(u, v)
z = Z(u, v)
```

Example:

```python
from pythree.equation import mesh_from_parametric
from math import pi

mesh = mesh_from_parametric(
    x_expr="(1 + 0.3*cos(v)) * cos(u)",
    y_expr="(1 + 0.3*cos(v)) * sin(u)",
    z_expr="0.3*sin(v)",
    u_range=(0, 2*pi),
    v_range=(0, 2*pi),
    u_segments=120,
    v_segments=40,
    wrap_u=True,
    wrap_v=True,
    name="torus_like",
)
```

### When to use this
- torus-like surfaces
- Möbius-strip-style objects
- custom surfaces where you directly control `x`, `y`, and `z`
- periodic/wrapped surfaces

---

## 3. Implicit surfaces

Use this when your shape is defined by an equation like:

```text
f(x, y, z) = 0
```

Example:

```python
from pythree.equation import mesh_from_implicit_cell_size
from pythree.utils import print_mesh_stats

bounds = ((-1.3, -1.3, -1.3), (1.3, 1.3, 1.3))

mesh = mesh_from_implicit_cell_size(
    "x*x + y*y + z*z - 1",
    bounds=bounds,
    cell_size=0.05,
    name="unit_sphere",
)

print_mesh_stats(mesh)
```

### When to use this
- spheres
- blobs
- signed-distance-like surfaces
- closed equation-defined 3D shapes

### Important note
Smaller `cell_size` gives more detail, but also makes generation slower and heavier.

---

## Expression syntax supported by `pythree`

The equation system uses a restricted expression parser.

Typical supported items include:
- `sin`, `cos`, `tan`
- `sqrt`, `exp`, `log`
- `pow`, `abs`
- `floor`, `ceil`
- `min`, `max`
- constants like `pi`, `e`, `tau`

So expressions usually look like:

```python
"sin(x) * cos(y)"
"x*x + y*y + z*z - 1"
"sqrt(x*x + y*y)"
```

---

## The `Mesh` object you get back

The clean mesh API in `pythree.mesh` works mainly with:

- `mesh.verts` — vertex positions
- `mesh.faces` — triangle indices
- `mesh.normals` — vertex normals after computation
- `mesh.face_normals` — face normals after computation

Typical operations include:

```python
mesh = mesh.compute_normals(True)
moved = mesh.translated(1.0, 0.0, 0.0)
bigger = mesh.scaled(2.0)
area = mesh.surface_area()
```

Other useful methods/helpers include:
- `validate()`
- `bounds()`
- `centroid()`
- `append()`
- `merged()`

---

## Useful utilities

The `pythree.utils` module appears to be one of the more practical parts of the project.

Common helpers include:
- `validate_mesh(mesh)`
- `mesh_stats(mesh)`
- `print_mesh_stats(mesh)`
- `translated(mesh, dx, dy, dz)`
- `scaled(mesh, scale)`
- `recentered(mesh)`
- `merge_meshes([...])`
- `remove_degenerate_faces(mesh)`
- `unique_edges(mesh)`

A small example:

```python
from pythree.utils import validate_mesh, print_mesh_stats, recentered

validate_mesh(mesh)
print_mesh_stats(mesh)
mesh = recentered(mesh)
```

---

## Why the package feels confusing

There seems to be a second, overlapping system in `pythree.sphere`.

That module appears to define:
- another `Mesh` class
- primitive generators such as sphere/cylinder/cone/torus/cube/plane-like helpers
- some exporter functions

The confusing part is that this `Mesh` does **not** appear to match the cleaner `pythree.mesh.Mesh` exactly.

For example, one side of the repo uses names like:
- `verts`
- `faces`

while the other side appears to use:
- `vertices`
- `faces`

So the repo behaves more like **two partially merged APIs** than one polished library.

---

## Best practice: use only the stable-looking parts

If you want the least painful experience, stick to:

- `pythree.equation`
- `pythree.mesh`
- `pythree.utils`
- `pythree.material`

Avoid assuming that all modules interoperate perfectly.

---

## Exporting to OBJ

The README mentions OBJ export, but given the internal inconsistencies, the safest option is to write a tiny exporter yourself.

Example:

```python
def save_obj_basic(mesh, path):
    with open(path, "w", encoding="utf-8") as f:
        for x, y, z in mesh.verts:
            f.write(f"v {x} {y} {z}\n")

        # OBJ uses 1-based indexing
        for i, j, k in mesh.faces:
            f.write(f"f {i+1} {j+1} {k+1}\n")
```

Usage:

```python
save_obj_basic(mesh, "output.obj")
```

This is the simplest way to get a generated mesh into Blender, MeshLab, or another 3D tool.

---

## A complete starter example

This is the most practical end-to-end example for getting started.

```python
from pythree.equation import segments_from_step, mesh_from_heightfield
from pythree.utils import validate_mesh, print_mesh_stats


def save_obj_basic(mesh, path):
    with open(path, "w", encoding="utf-8") as f:
        for x, y, z in mesh.verts:
            f.write(f"v {x} {y} {z}\n")
        for i, j, k in mesh.faces:
            f.write(f"f {i+1} {j+1} {k+1}\n")


x_segments = segments_from_step((-3, 3), step=0.05)
y_segments = segments_from_step((-3, 3), step=0.05)

mesh = mesh_from_heightfield(
    "sin(x) * cos(y)",
    x_range=(-3, 3),
    y_range=(-3, 3),
    x_segments=x_segments,
    y_segments=y_segments,
    name="wave_surface",
)

validate_mesh(mesh)
print_mesh_stats(mesh)

mesh = mesh.compute_normals(True)
save_obj_basic(mesh, "output.obj")

print("Saved output.obj")
```

---

## What I would personally use `pythree` for

I would use it for:
- procedural geometry experiments
- math-surface visualization
- generating simple meshes for export
- educational geometry scripting
- quick custom geometry prototypes

I would **not** currently use it as:
- a production rendering engine
- a polished CAD-style toolkit
- a stable all-in-one 3D framework

---

## Final recommendation

If you want to work with this package successfully:

1. install from source
2. import directly from submodules
3. use the equation-based mesh generators first
4. inspect the returned mesh through `pythree.mesh` and `pythree.utils`
5. export with a tiny manual OBJ writer when needed

In one sentence:

> `pythree` is a small equation-driven mesh generator with useful ideas, but the repository structure is inconsistent, so you should use only the clean submodules directly.

---

## Source used

This guide was compiled by inspecting the repository source here:
- https://github.com/rootjatin/pythree
