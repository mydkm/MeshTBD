from __future__ import annotations

from ..core import MeshData


def load_with_open3d(path: str) -> MeshData:
    import numpy as np
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(path)

    if len(mesh.triangles) > 0:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.int32)

        vertex_normals = None
        if mesh.has_vertex_normals():
            vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

        face_normals = None
        if mesh.has_triangle_normals():
            face_normals = np.asarray(mesh.triangle_normals, dtype=np.float32)

        colors = None
        if mesh.has_vertex_colors():
            colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
    else:
        pcd = o3d.io.read_point_cloud(path)

        vertices = np.asarray(pcd.points, dtype=np.float32)
        faces = None
        vertex_normals = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else None
        face_normals = None
        colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None

    return MeshData(V=vertices, F=faces, VN=vertex_normals, FN=face_normals, C=colors)
