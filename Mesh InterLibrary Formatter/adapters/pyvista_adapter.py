from __future__ import annotations

import numpy as np
import pyvista as pv

from ..core import MeshData


def load_with_pyvista(path: str) -> MeshData:
    """
    Load a mesh using PyVista and convert it into canonical MeshData.
    """

    mesh = pv.read(path)

    # Ensure PolyData
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()

    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()

    mesh = mesh.triangulate()

    # Vertices
    V = mesh.points.astype(np.float32)

    # Faces (convert padded format)
    F = None
    if mesh.faces.size > 0:
        raw = mesh.faces.astype(np.int64)
        faces = []
        i = 0
        while i < len(raw):
            n = raw[i]
            faces.append(raw[i + 1 : i + 1 + n])
            i += n + 1

        F = np.array([f for f in faces if len(f) == 3], dtype=np.int32)

    # Normals
    VN = None
    if mesh.point_normals is not None and len(mesh.point_normals) == len(V):
        VN = mesh.point_normals.astype(np.float32)

    FN = None
    if mesh.face_normals is not None and F is not None:
        FN = mesh.face_normals.astype(np.float32)

    # Colors (heuristic: first 3/4 channel point_data array)
    C = None
    for _, arr in mesh.point_data.items():
        a = np.asarray(arr)
        if a.ndim == 2 and a.shape[0] == len(V) and a.shape[1] in (3, 4):
            C = a
            break

    return MeshData(V=V, F=F, VN=VN, FN=FN, C=C)
