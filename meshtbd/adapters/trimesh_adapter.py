from __future__ import annotations

import numpy as np
import trimesh

from meshtbd.core import MeshData


def from_trimesh(mesh: trimesh.Trimesh) -> MeshData:
    """
    Convert a trimesh.Trimesh object into MeshData.
    """
    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.faces, dtype=np.int32) if mesh.faces is not None else None

    VN = None
    if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(V):
        VN = np.asarray(mesh.vertex_normals, dtype=np.float32)

    FN = None
    if hasattr(mesh, "face_normals") and mesh.face_normals is not None and F is not None and len(mesh.face_normals) == len(F):
        FN = np.asarray(mesh.face_normals, dtype=np.float32)

    C = None
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        vc = np.asarray(mesh.visual.vertex_colors)
        if vc.ndim == 2 and vc.shape[0] == len(V) and vc.shape[1] in (3, 4):
            C = vc

    return MeshData(V=V, F=F, VN=VN, FN=FN, C=C)


def to_trimesh(meshdata: MeshData) -> trimesh.Trimesh:
    """
    Convert MeshData into a trimesh.Trimesh object.
    """
    if meshdata.F is None:
        raise ValueError("Cannot convert point cloud MeshData to trimesh.Trimesh because F is None")

    mesh = trimesh.Trimesh(
        vertices=np.asarray(meshdata.V, dtype=np.float32),
        faces=np.asarray(meshdata.F, dtype=np.int32),
        process=False,
    )

    if meshdata.VN is not None:
        try:
            mesh.vertex_normals = np.asarray(meshdata.VN, dtype=np.float32)
        except Exception:
            pass

    if meshdata.C is not None:
        try:
            mesh.visual.vertex_colors = np.asarray(meshdata.C)
        except Exception:
            pass

    return mesh


def load_with_trimesh(path: str) -> MeshData:
    """
    Load a mesh using trimesh and convert it into MeshData.
    """
    mesh = trimesh.load(path, process=False)

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"trimesh could not load a valid Trimesh from: {path}")

    return from_trimesh(mesh)