from __future__ import annotations

import numpy as np
import pymeshlab

from meshtbd.core import MeshData


def from_pymeshlab(mesh_or_meshset) -> MeshData:
    """
    Convert a pymeshlab.Mesh or pymeshlab.MeshSet current mesh into MeshData.
    """
    if isinstance(mesh_or_meshset, pymeshlab.MeshSet):
        mesh = mesh_or_meshset.current_mesh()
    elif isinstance(mesh_or_meshset, pymeshlab.Mesh):
        mesh = mesh_or_meshset
    else:
        raise TypeError(
            "from_pymeshlab expects a pymeshlab.Mesh or pymeshlab.MeshSet"
        )

    V = np.asarray(mesh.vertex_matrix(), dtype=np.float32)

    F = None
    try:
        face_matrix = mesh.face_matrix()
        if face_matrix is not None and len(face_matrix) > 0:
            F = np.asarray(face_matrix, dtype=np.int32)
    except Exception:
        F = None

    VN = None
    try:
        vertex_normals = mesh.vertex_normal_matrix()
        if vertex_normals is not None and len(vertex_normals) == len(V):
            VN = np.asarray(vertex_normals, dtype=np.float32)
    except Exception:
        VN = None

    FN = None
    try:
        face_normals = mesh.face_normal_matrix()
        if F is not None and face_normals is not None and len(face_normals) == len(F):
            FN = np.asarray(face_normals, dtype=np.float32)
    except Exception:
        FN = None

    C = None
    try:
        vertex_colors = mesh.vertex_color_matrix()
        if vertex_colors is not None and len(vertex_colors) == len(V):
            C = np.asarray(vertex_colors)
    except Exception:
        C = None

    return MeshData(V=V, F=F, VN=VN, FN=FN, C=C)


def to_pymeshlab_mesh(meshdata: MeshData) -> pymeshlab.Mesh:
    """
    Convert MeshData into a pymeshlab.Mesh.
    """
    kwargs = {
        "vertex_matrix": np.asarray(meshdata.V, dtype=np.float32),
    }

    if meshdata.F is not None:
        kwargs["face_matrix"] = np.asarray(meshdata.F, dtype=np.int32)

    if meshdata.VN is not None:
        kwargs["v_normals_matrix"] = np.asarray(meshdata.VN, dtype=np.float32)

    if meshdata.C is not None:
        colors = np.asarray(meshdata.C)

        # PyMeshLab generally works well with float colors.
        # Keep RGB or RGBA only.
        if colors.ndim == 2 and colors.shape[1] in (3, 4):
            kwargs["v_color_matrix"] = colors.astype(np.float32)

    return pymeshlab.Mesh(**kwargs)


def to_pymeshlab_meshset(meshdata: MeshData, name: str = "MILF_mesh") -> pymeshlab.MeshSet:
    """
    Convert MeshData into a pymeshlab.MeshSet containing one mesh.
    """
    mesh = to_pymeshlab_mesh(meshdata)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh, name)
    return ms


def to_pymeshlab(meshdata: MeshData) -> pymeshlab.Mesh:
    """
    Default MeshData -> PyMeshLab conversion.
    Returns a pymeshlab.Mesh.
    """
    return to_pymeshlab_mesh(meshdata)


def load_with_pymeshlab(path: str) -> MeshData:
    """
    Load a mesh using PyMeshLab and convert it into MeshData.
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    return from_pymeshlab(ms)