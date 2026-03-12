# pythree/material.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

ColorRGB = Tuple[float, float, float]
ColorRGBA = Tuple[float, float, float, float]

__all__ = [
    "ColorRGB",
    "ColorRGBA",
    "rgb",
    "rgba",
    "hex_to_rgb",
    "hex_to_rgba",
    "Material",
]


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _validate_rgb(c: ColorRGB) -> ColorRGB:
    if len(c) != 3:
        raise ValueError("RGB color must have exactly 3 components")
    return (_clamp01(c[0]), _clamp01(c[1]), _clamp01(c[2]))


def _validate_rgba(c: ColorRGBA) -> ColorRGBA:
    if len(c) != 4:
        raise ValueError("RGBA color must have exactly 4 components")
    return (_clamp01(c[0]), _clamp01(c[1]), _clamp01(c[2]), _clamp01(c[3]))


def rgb(r: float, g: float, b: float) -> ColorRGB:
    """
    Create a clamped RGB color in [0, 1].
    """
    return _validate_rgb((r, g, b))


def rgba(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    """
    Create a clamped RGBA color in [0, 1].
    """
    return _validate_rgba((r, g, b, a))


def hex_to_rgb(value: str) -> ColorRGB:
    """
    Convert '#RRGGBB' or 'RRGGBB' to an RGB tuple in [0, 1].
    """
    s = value.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError("Expected a 6-digit hex color like '#ff8800'")

    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return (r, g, b)


def hex_to_rgba(value: str, alpha: float = 1.0) -> ColorRGBA:
    """
    Convert '#RRGGBB' or 'RRGGBB' to an RGBA tuple in [0, 1].
    """
    r, g, b = hex_to_rgb(value)
    return rgba(r, g, b, alpha)


@dataclass
class Material:
    """
    Simple material description for meshes.

    Notes
    -----
    This is intentionally renderer-agnostic, but the fields map well to
    modern PBR-style materials and are suitable for future glTF export.

    Parameters
    ----------
    name:
        Material name.
    base_color:
        RGBA tuple in [0, 1].
    emissive:
        RGB tuple in [0, 1].
    metallic:
        Metallic factor in [0, 1].
    roughness:
        Roughness factor in [0, 1].
    double_sided:
        Whether both sides of faces should be rendered.
    wireframe:
        Hint for debug / viewer rendering.
    """

    name: str = "material"
    base_color: ColorRGBA = (0.8, 0.8, 0.8, 1.0)
    emissive: ColorRGB = (0.0, 0.0, 0.0)
    metallic: float = 0.0
    roughness: float = 1.0
    double_sided: bool = False
    wireframe: bool = False

    def __post_init__(self) -> None:
        self.base_color = _validate_rgba(self.base_color)
        self.emissive = _validate_rgb(self.emissive)
        self.metallic = _clamp01(self.metallic)
        self.roughness = _clamp01(self.roughness)

    @property
    def opacity(self) -> float:
        return self.base_color[3]

    @property
    def is_transparent(self) -> bool:
        return self.opacity < 1.0

    def copy(self, **changes: object) -> "Material":
        return Material(
            name=str(changes.get("name", self.name)),
            base_color=changes.get("base_color", self.base_color),  # type: ignore[arg-type]
            emissive=changes.get("emissive", self.emissive),  # type: ignore[arg-type]
            metallic=float(changes.get("metallic", self.metallic)),
            roughness=float(changes.get("roughness", self.roughness)),
            double_sided=bool(changes.get("double_sided", self.double_sided)),
            wireframe=bool(changes.get("wireframe", self.wireframe)),
        )

    def with_color(self, color: ColorRGB | ColorRGBA) -> "Material":
        if len(color) == 3:
            c3 = _validate_rgb(color)  # type: ignore[arg-type]
            c4 = (c3[0], c3[1], c3[2], self.opacity)
        else:
            c4 = _validate_rgba(color)  # type: ignore[arg-type]
        return self.copy(base_color=c4)

    def with_opacity(self, alpha: float) -> "Material":
        r, g, b, _ = self.base_color
        return self.copy(base_color=(r, g, b, _clamp01(alpha)))

    def to_dict(self) -> Dict[str, object]:
        """
        Convert the material to a plain dictionary.
        Useful for debugging, serialization, or exporters.
        """
        return {
            "name": self.name,
            "base_color": self.base_color,
            "emissive": self.emissive,
            "metallic": self.metallic,
            "roughness": self.roughness,
            "double_sided": self.double_sided,
            "wireframe": self.wireframe,
            "transparent": self.is_transparent,
        }

    def __repr__(self) -> str:
        return (
            f"Material(name={self.name!r}, base_color={self.base_color}, "
            f"metallic={self.metallic}, roughness={self.roughness})"
        )
