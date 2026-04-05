from __future__ import annotations

import numpy as np
import pyvista as pv

from meshtbd.core import MeshData


def _extract_triangle_faces_from_pyvista(mesh: pv.PolyData) -> np.ndarray | None:
    """
    Convert PyVista's padded face array format into an (M, 3) int32 array.
    Assumes the mesh has already been triangulated.
    """
    if mesh.faces.size == 0:
        return None

    raw = np.asarray(mesh.faces, dtype=np.int64)
    faces = []

    i = 0
    while i < len(raw):
        n = raw[i]
        face = raw[i + 1 : i + 1 + n]
        if len(face) == 3:
            faces.append(face)
        i += n + 1

    if not faces:
        return None

    return np.asarray(faces, dtype=np.int32)


def _find_vertex_colors(mesh: pv.PolyData, n_vertices: int) -> np.ndarray | None:
    """
    Heuristic: return the first point_data array shaped like vertex colors,
    i.e. (N, 3) or (N, 4).
    """
    for _, arr in mesh.point_data.items():
        a = np.asarray(arr)
        if a.ndim == 2 and a.shape[0] == n_vertices and a.shape[1] in (3, 4):
            return a
    return None


def from_pyvista(mesh) -> MeshData:
    """
    Convert a PyVista mesh-like object into MeshData.
    """
    # Normalize to PolyData
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()

    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()

    mesh = mesh.triangulate()

    V = np.asarray(mesh.points, dtype=np.float32)
    F = _extract_triangle_faces_from_pyvista(mesh)

    VN = None
    try:
        point_normals = mesh.point_normals
        if point_normals is not None and len(point_normals) == len(V):
            VN = np.asarray(point_normals, dtype=np.float32)
    except Exception:
        VN = None

    FN = None
    try:
        face_normals = mesh.face_normals
        if face_normals is not None and F is not None and len(face_normals) == len(F):
            FN = np.asarray(face_normals, dtype=np.float32)
    except Exception:
        FN = None

    C = _find_vertex_colors(mesh, len(V))
    if C is not None:
        C = np.asarray(C)

    return MeshData(V=V, F=F, VN=VN, FN=FN, C=C)


def to_pyvista(meshdata: MeshData) -> pv.PolyData:
    """
    Convert MeshData into a PyVista PolyData object.
    """
    V = np.asarray(meshdata.V, dtype=np.float32)

    if meshdata.F is None:
        mesh = pv.PolyData(V)
    else:
        F = np.asarray(meshdata.F, dtype=np.int32)

        # Convert (M, 3) faces into PyVista padded format:
        # [3, i0, i1, i2, 3, j0, j1, j2, ...]
        padded_faces = np.hstack(
            [np.column_stack([np.full(len(F), 3, dtype=np.int32), F]).reshape(-1)]
        )

        mesh = pv.PolyData(V, padded_faces)

    if meshdata.VN is not None:
        mesh.point_data["Normals"] = np.asarray(meshdata.VN, dtype=np.float32)

    if meshdata.FN is not None and meshdata.F is not None:
        mesh.cell_data["Normals"] = np.asarray(meshdata.FN, dtype=np.float32)

    if meshdata.C is not None:
        mesh.point_data["Colors"] = np.asarray(meshdata.C)

    return mesh


def load_with_pyvista(path: str) -> MeshData:
    """
    Load a mesh using PyVista and convert it into canonical MeshData.
    """
    mesh = pv.read(path)
    return from_pyvista(mesh)