from __future__ import annotations

import numpy as np
import open3d as o3d

from ..core import MeshData


def load_with_open3d(path: str) -> MeshData:
    """
    Load mesh or point cloud using Open3D and convert to MeshData.
    """

    mesh = o3d.io.read_triangle_mesh(path)

    if len(mesh.triangles) > 0:
        # Triangle mesh
        V = np.asarray(mesh.vertices, dtype=np.float32)
        F = np.asarray(mesh.triangles, dtype=np.int32)

        VN = None
        if mesh.has_vertex_normals():
            VN = np.asarray(mesh.vertex_normals, dtype=np.float32)

        FN = None
        if mesh.has_triangle_normals():
            FN = np.asarray(mesh.triangle_normals, dtype=np.float32)

        C = None
        if mesh.has_vertex_colors():
            C = np.asarray(mesh.vertex_colors, dtype=np.float32)

    else:
        # Point cloud fallback
        pcd = o3d.io.read_point_cloud(path)

        V = np.asarray(pcd.points, dtype=np.float32)
        F = None
        VN = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else None
        FN = None
        C = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None

    return MeshData(V=V, F=F, VN=VN, FN=FN, C=C)
