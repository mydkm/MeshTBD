"""Neutral mesh representation and lazily loaded geometry-library adapters."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from mesh_interlibrary_formatter.calibration import calibrate_mesh_scale, compute_uniform_scale
from mesh_interlibrary_formatter.core import MeshData


_LAZY_IMPORTS = {
    "load_with_pyvista": ("mesh_interlibrary_formatter.adapters.pyvista_adapter", "load_with_pyvista"),
    "from_pyvista": ("mesh_interlibrary_formatter.adapters.pyvista_adapter", "from_pyvista"),
    "to_pyvista": ("mesh_interlibrary_formatter.adapters.pyvista_adapter", "to_pyvista"),
    "load_with_open3d": ("mesh_interlibrary_formatter.adapters.open3d_adapter", "load_with_open3d"),
    "from_open3d_triangle_mesh": ("mesh_interlibrary_formatter.adapters.open3d_adapter", "from_open3d_triangle_mesh"),
    "from_open3d_point_cloud": ("mesh_interlibrary_formatter.adapters.open3d_adapter", "from_open3d_point_cloud"),
    "to_open3d_triangle_mesh": ("mesh_interlibrary_formatter.adapters.open3d_adapter", "to_open3d_triangle_mesh"),
    "to_open3d_point_cloud": ("mesh_interlibrary_formatter.adapters.open3d_adapter", "to_open3d_point_cloud"),
    "to_open3d": ("mesh_interlibrary_formatter.adapters.open3d_adapter", "to_open3d"),
    "load_with_trimesh": ("mesh_interlibrary_formatter.adapters.trimesh_adapter", "load_with_trimesh"),
    "from_trimesh": ("mesh_interlibrary_formatter.adapters.trimesh_adapter", "from_trimesh"),
    "to_trimesh": ("mesh_interlibrary_formatter.adapters.trimesh_adapter", "to_trimesh"),
    "load_with_pymeshlab": ("mesh_interlibrary_formatter.adapters.pymeshlab_adapter", "load_with_pymeshlab"),
    "from_pymeshlab": ("mesh_interlibrary_formatter.adapters.pymeshlab_adapter", "from_pymeshlab"),
    "to_pymeshlab": ("mesh_interlibrary_formatter.adapters.pymeshlab_adapter", "to_pymeshlab"),
    "to_pymeshlab_mesh": ("mesh_interlibrary_formatter.adapters.pymeshlab_adapter", "to_pymeshlab_mesh"),
    "to_pymeshlab_meshset": ("mesh_interlibrary_formatter.adapters.pymeshlab_adapter", "to_pymeshlab_meshset"),
    "from_bpy_object": ("mesh_interlibrary_formatter.adapters.bpy_adapter", "from_bpy_object"),
    "from_bpy_mesh_data": ("mesh_interlibrary_formatter.adapters.bpy_adapter", "from_bpy_mesh_data"),
    "to_bpy_object": ("mesh_interlibrary_formatter.adapters.bpy_adapter", "to_bpy_object"),
    "to_bpy_object_linked": ("mesh_interlibrary_formatter.adapters.bpy_adapter", "to_bpy_object_linked"),
}

__all__ = [
    "MeshData",
    "compute_uniform_scale",
    "calibrate_mesh_scale",
    *_LAZY_IMPORTS,
]


def __getattr__(name: str) -> Any:
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
