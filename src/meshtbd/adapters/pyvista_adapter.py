from __future__ import annotations

from ..core import MeshData


def load_with_pyvista(path: str) -> MeshData:
    import numpy as np
    import pyvista as pv

    mesh = pv.read(path)

    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()

    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()

    mesh = mesh.triangulate()

    vertices = mesh.points.astype(np.float32)

    faces = None
    if mesh.faces.size > 0:
        raw = mesh.faces.astype(np.int64)
        items = []
        i = 0
        while i < len(raw):
            n = raw[i]
            items.append(raw[i + 1 : i + 1 + n])
            i += n + 1
        faces = np.array([face for face in items if len(face) == 3], dtype=np.int32)

    vertex_normals = None
    if mesh.point_normals is not None and len(mesh.point_normals) == len(vertices):
        vertex_normals = mesh.point_normals.astype(np.float32)

    face_normals = None
    if mesh.face_normals is not None and faces is not None:
        face_normals = mesh.face_normals.astype(np.float32)

    colors = None
    for _, arr in mesh.point_data.items():
        array = np.asarray(arr)
        if array.ndim == 2 and array.shape[0] == len(vertices) and array.shape[1] in (3, 4):
            colors = array
            break

    return MeshData(V=vertices, F=faces, VN=vertex_normals, FN=face_normals, C=colors)
