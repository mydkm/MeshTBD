#!/usr/bin/env python3
"""
Turn every edge of a triangle mesh into a tube, then decimate to ~200 k verts
"""

import numpy as np
import open3d as o3d


# ──────────────────────────────────────────────────────────────────────────────
# cylinder helper
# ──────────────────────────────────────────────────────────────────────────────
def cylinder_between(p0, p1, radius, resolution=8):
    """
    Return an Open3D TriangleMesh cylinder whose end‑caps lie at p0 and p1.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    vec = p1 - p0
    h   = np.linalg.norm(vec)
    if h < 1e-8:                       # degenerate segment
        return None

    # 1. raw cylinder on +Z
    cyl = o3d.geometry.TriangleMesh.create_cylinder(
        radius=float(radius), height=float(h),
        resolution=int(resolution), split=1)
    # 2. rotate so +Z → vec
    z_axis = np.array([0.0, 0.0, 1.0])
    axis   = vec / h
    cross  = np.cross(z_axis, axis)
    s      = np.linalg.norm(cross)
    c      = np.dot(z_axis, axis)
    if s < 1e-8:                       # parallel / anti‑parallel
        R = np.eye(3) if c > 0 else \
            o3d.geometry.get_rotation_matrix_from_axis_angle(
                np.array([1, 0, 0]) * np.pi)
    else:
        angle = np.arctan2(s, c)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
            cross / s * angle)
    cyl.rotate(R, center=np.zeros(3))
    # 3. move to midpoint
    cyl.translate((p0 + p1) * 0.5)
    return cyl


# ──────────────────────────────────────────────────────────────────────────────
# parameters
# ──────────────────────────────────────────────────────────────────────────────
TARGET_VERTS     = 200_000          # after decimation
CIRCLE_RES       = 8                # 8‑sided tube
RADIUS_FRACTION  = 0.01             # 1 % of model diagonal


# ──────────────────────────────────────────────────────────────────────────────
# 1. load and clean the triangle mesh
# ──────────────────────────────────────────────────────────────────────────────
m = o3d.io.read_triangle_mesh("a3.ply")
if m.is_empty():
    raise RuntimeError("Failed to load mesh!")

m.remove_duplicated_triangles()
m.remove_degenerate_triangles()
m.remove_duplicated_vertices()
m.remove_unreferenced_vertices()
m.remove_non_manifold_edges()
m.orient_triangles()
bbox = m.get_axis_aligned_bounding_box()
diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())

print(f"Original mesh: {len(m.vertices)} verts – diag ≈ {diag:.3f}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. edges → LineSet
# ──────────────────────────────────────────────────────────────────────────────
ls = o3d.geometry.LineSet.create_from_triangle_mesh(m)
V   = np.asarray(ls.points)
E   = np.asarray(ls.lines)
print(f"Unique edges: {len(E)} (each will become one tube)")

# ──────────────────────────────────────────────────────────────────────────────
# 3. build the tubes
# ──────────────────────────────────────────────────────────────────────────────
tube_radius = RADIUS_FRACTION * diag
big_mesh    = o3d.geometry.TriangleMesh()

for i0, i1 in E:
    cyl = cylinder_between(V[i0], V[i1], radius=tube_radius,
                           resolution=CIRCLE_RES)
    if cyl is not None:
        big_mesh += cyl

big_mesh.remove_duplicated_vertices()
big_mesh.remove_degenerate_triangles()
big_mesh.compute_vertex_normals()
print(f"Before decimation: {len(big_mesh.vertices):,} verts, "
      f"{len(big_mesh.triangles):,} tris")

# ──────────────────────────────────────────────────────────────────────────────
# 4. simplify to ~TARGET_VERTS
# ──────────────────────────────────────────────────────────────────────────────
# each tube face is a triangle, so verts ≈ tris/2 → aim for ~2*TARGET_VERTS tris
target_tris  = 2 * TARGET_VERTS
big_mesh.remove_degenerate_triangles()
big_mesh.remove_duplicated_vertices()
big_mesh.compute_vertex_normals()

voxel_size = max(big_mesh.get_max_bound() - big_mesh.get_min_bound()) / 128
print(f'voxel_size = {voxel_size:e}')
big_mesh = big_mesh.simplify_vertex_clustering(
    voxel_size=voxel_size,
    contraction=o3d.geometry.SimplificationContraction.Average)
print(f"After decimation:  {len(big_mesh.vertices):,} verts, "
      f"{len(big_mesh.triangles):,} tris")

o3d.io.write_triangle_mesh("a6.ply", big_mesh)
big_mesh_out = big_mesh.filter_smooth_taubin(number_of_iterations=25)
big_mesh_out.remove_degenerate_triangles()
big_mesh_out.remove_duplicated_triangles()
big_mesh_out.remove_duplicated_vertices()
big_mesh_out.remove_unreferenced_vertices()
big_mesh_out.remove_non_manifold_edges()
big_mesh_out.compute_vertex_normals()
o3d.io.write_triangle_mesh("a7.ply", big_mesh_out)
# ──────────────────────────────────────────────────────────────────────────────
# 5. visualisecd
# ──────────────────────────────────────────────────────────────────────────────
o3d.visualization.draw_geometries(
    [big_mesh_out], mesh_show_back_face=True
)
