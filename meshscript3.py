import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib as mpl

## PREPROCESSING
# 1. Load STL -----------------------------------------------------------------
mesh = o3d.io.read_triangle_mesh("Just forearm.stl")
if mesh.is_empty():
    raise RuntimeError("Failed to load mesh!")
print("Loaded mesh!")

# 2. Point-cloud sampling -----------------------------------------------------
pcd = mesh.sample_points_poisson_disk(100000)  # denser cloud
print('Generated reconstruction PCD!')

# 3. Normal estimation & orientation ------------------------------------------
bbox = mesh.get_axis_aligned_bounding_box()
bounding_diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())

pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=bounding_diag * 0.02,  # 2 % of bbox diagonal
        max_nn=30))
pcd.orient_normals_consistent_tangent_plane(100)
print('Normal estimation and orientation completed!')

# 4. Select data-driven ball radii --------------------------------------------
dists = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(dists)
radii = [avg_dist * k for k in (1.5, 2.5, 3.5)]
print('Computed ball radii!')

# 5. Ball-pivoting reconstruction ---------------------------------------------
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))
print('BPA completed!')

# 6. Mesh cleanup -------------------------------------------------------------
rec_mesh.remove_degenerate_triangles()
rec_mesh.remove_duplicated_triangles()
rec_mesh.remove_duplicated_vertices()
rec_mesh.remove_unreferenced_vertices()
rec_mesh.remove_non_manifold_edges()
rec_mesh.compute_vertex_normals()
print('Mesh cleaned up!')
print(
    f'Simplified mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
)

## MESHIFICATION
# 8. Mesh point-cloud sampling ------------------------------------------------
regpcd = mesh.sample_points_poisson_disk(50, use_triangle_normal=True)
vertpcd = o3d.geometry.PointCloud()
vertpcd.points = rec_mesh.vertices
print('Poisson disk sampling completed!')

# # 9. Distance per mesh vertex -------------------------------------------------
# dists = np.asarray(vertpcd.compute_point_cloud_distance(regpcd))
# print('Vertex-PCD distance computed!')

# ------------------------------------------------------------------
# 8.  Build an undirected edge‑length graph of the reconstructed mesh
# ------------------------------------------------------------------
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra          # multi‑source Dijkstra:contentReference[oaicite:0]{index=0}
from scipy.spatial import cKDTree

V = np.asarray(rec_mesh.vertices)
F = np.asarray(rec_mesh.triangles)

# all unique (undirected) edges
E = np.vstack((F[:, [0, 1]],
               F[:, [1, 2]],
               F[:, [2, 0]]))
E = np.sort(E, axis=1)
E = np.unique(E, axis=0)

# edge lengths become graph weights
edge_lens = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)

# sparse adjacency matrix  (symmetric → undirected graph)
n = len(V)
rows = np.hstack((E[:, 0], E[:, 1]))
cols = np.hstack((E[:, 1], E[:, 0]))
data = np.hstack((edge_lens, edge_lens))
G = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

# ------------------------------------------------------------------
# 9.  Identify source vertices and run multi‑source Dijkstra
# ------------------------------------------------------------------
# map each point in regpcd to its nearest *mesh* vertex index
kdt = cKDTree(V)
src_indices = kdt.query(np.asarray(regpcd.points))[1]

# fastest option: return only the minimum distance from *any* source
dists = dijkstra(G,
                 directed=False,
                 indices=src_indices,
                 min_only=True)          # 1‑D array, length = n_vertices

print('Geodesic distances computed!')

# 10. Painting the mesh -------------------------------------------------------
vmin, vmax = np.percentile(dists, [2, 98])
norm = mpl.colors.PowerNorm(gamma=0.6, vmin=vmin, vmax=vmax, clip=True)
colors = plt.cm.plasma_r(norm(dists))[:, :3]   # drop the alpha channel
rec_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
print('Mesh painted!')


# 11. Visualise ----------------------------------------------------------------
if rec_mesh.is_empty():
    print("No triangles produced – increase sample density or radii.")
else:
    # o3d.visualization.draw_geometries([regpcd])
    o3d.visualization.draw_geometries([rec_mesh])
