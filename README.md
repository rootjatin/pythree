# pythree
Python package for making geometric shapes obj file and for 3d plotting and 3d obj file 

Pypi : https://pypi.org/project/pythree/
# pythree (pythree)

**pythree** (aka **pythreemaker**) is a Python package for generating simple 3D geometric meshes and exporting them as **Wavefront `.obj`** files.

**Links**
- PyPI: https://pypi.org/project/pythree/
- GitHub: https://github.com/rootjatin/pythree

---

## Features

- Create basic 3D geometric shapes (triangle-mesh style)
- Export models as `.obj` files for Blender / MeshLab / other viewers
- Lightweight and script-friendly

---

## Installation

### From PyPI

```bash
pip install pythree
```
### From source
```
git clone https://github.com/rootjatin/pythree.git
cd pythree
pip install -e .
```
```
import pythree
print(dir(pythree))
help(pythree)
```
### From Equations
```
bounds = ((-1.3, -1.3, -1.3), (1.3, 1.3, 1.3))
mesh = mesh_from_implicit_cell_size(
    "x*x + y*y + z*z - 1",
    bounds=bounds,
    cell_size=0.03,      # smaller = higher resolution
    max_resolution=220,  # safety clamp (prevents accidental huge grids)
)
```
#### Heightfied parameter (equations)
```
x_segments = segments_from_step((-3, 3), step=0.02)
y_segments = segments_from_step((-3, 3), step=0.02)

mesh = mesh_from_heightfield(
    "sin(x)*cos(y)",
    x_range=(-3,3),
    y_range=(-3,3),
    x_segments=x_segments,
    y_segments=y_segments,
)
```
