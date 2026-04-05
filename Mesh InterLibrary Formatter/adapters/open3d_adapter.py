from __future__ import annotations

import numpy as np
import open3d as o3d

from meshtbd.core import MeshData


def from_open3d_triangle_mesh(mesh: o3d.geometry.TriangleMesh) -> MeshData:
    """
    Convert an Open3D TriangleMesh to MeshData.
    """
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

    return MeshData(V=V, F=F, VN=VN, FN=FN, C=C)


def from_open3d_point_cloud(pcd: o3d.geometry.PointCloud) -> MeshData:
    """
    Convert an Open3D PointCloud to MeshData.
    """
    V = np.asarray(pcd.points, dtype=np.float32)
    F = None

    VN = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else None
    FN = None
    C = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None

    return MeshData(V=V, F=F, VN=VN, FN=FN, C=C)


def to_open3d_triangle_mesh(meshdata: MeshData) -> o3d.geometry.TriangleMesh:
    """
    Convert MeshData to an Open3D TriangleMesh.
    """
    if meshdata.F is None:
        raise ValueError("Cannot convert point cloud MeshData to TriangleMesh because F is None")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(meshdata.V, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(meshdata.F, dtype=np.int32))

    if meshdata.VN is not None:
        mesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(meshdata.VN, dtype=np.float64))

    if meshdata.FN is not None:
        mesh.triangle_normals = o3d.utility.Vector3dVector(np.asarray(meshdata.FN, dtype=np.float64))

    if meshdata.C is not None:
        colors = np.asarray(meshdata.C, dtype=np.float64)

        # Open3D expects RGB, not RGBA
        if colors.shape[1] == 4:
            colors = colors[:, :3]

        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return mesh


def to_open3d_point_cloud(meshdata: MeshData) -> o3d.geometry.PointCloud:
    """
    Convert MeshData to an Open3D PointCloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(meshdata.V, dtype=np.float64))

    if meshdata.VN is not None:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(meshdata.VN, dtype=np.float64))

    if meshdata.C is not None:
        colors = np.asarray(meshdata.C, dtype=np.float64)

        # Open3D point clouds also expect RGB, not RGBA
        if colors.shape[1] == 4:
            colors = colors[:, :3]

        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def to_open3d(meshdata: MeshData):
    """
    Convert MeshData to the appropriate Open3D geometry type.
    Returns TriangleMesh if faces exist, otherwise PointCloud.
    """
    if meshdata.F is None:
        return to_open3d_point_cloud(meshdata)
    return to_open3d_triangle_mesh(meshdata)


def load_with_open3d(path: str) -> MeshData:
    """
    Load mesh or point cloud from file using Open3D and convert to MeshData.
    """
    mesh = o3d.io.read_triangle_mesh(path)

    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        return from_open3d_triangle_mesh(mesh)

    pcd = o3d.io.read_point_cloud(path)

    if len(pcd.points) > 0:
        return from_open3d_point_cloud(pcd)

    raise ValueError(f"Open3D could not load a triangle mesh or point cloud from: {path}")