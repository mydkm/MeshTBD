from meshtbd.core import MeshData

from meshtbd.adapters.pyvista_adapter import (
    load_with_pyvista,
    from_pyvista,
    to_pyvista,
)

from meshtbd.adapters.trimesh_adapter import (
    load_with_trimesh,
    from_trimesh,
    to_trimesh,
)

from meshtbd.adapters.pymeshlab_adapter import (
    load_with_pymeshlab,
    from_pymeshlab,
    to_pymeshlab,
    to_pymeshlab_mesh,
    to_pymeshlab_meshset,
)

from meshtbd.adapters.bpy_adapter import (
    from_bpy_object,
    from_bpy_mesh_data,
    to_bpy_object,
    to_bpy_object_linked,
)

__all__ = [
    "MeshData",

    # PyVista
    "load_with_pyvista",
    "from_pyvista",
    "to_pyvista",

    # trimesh
    "load_with_trimesh",
    "from_trimesh",
    "to_trimesh",

    # PyMeshLab
    "load_with_pymeshlab",
    "from_pymeshlab",
    "to_pymeshlab",
    "to_pymeshlab_mesh",
    "to_pymeshlab_meshset",

    # bpy / Blender
    "from_bpy_object",
    "from_bpy_mesh_data",
    "to_bpy_object",
    "to_bpy_object_linked",
]