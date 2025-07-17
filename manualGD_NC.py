import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
import open3d as o3d  # type: ignore
import pymeshlab as pyml  # type: ignore
import matplotlib.pyplot as plt
import matplotlib as mpl

## PREPROCESSING
# 1. load STL
ms = pyml.MeshSet()
ms.load_new_mesh("Just forearm.stl")
surface_id = ms.current_mesh_id()
pymeshlab_mesh = ms.current_mesh()
print("Loaded mesh!")

# 2. Pyml surface reconstruction
ms.generate_surface_reconstruction_vcg(voxsize=pyml.PercentageValue(0.500000))
ms.set_current_mesh(1)
ms.save_current_mesh("plymcout.ply")
ms.load_new_mesh("plymcout.ply")
ms.meshing_surface_subdivision_loop(threshold=pyml.PercentageValue(0.500000))
print("Reconstruction complete!")

# 3. Converting Pyml mesh to Open3D mesh
processed_ml_mesh = ms.current_mesh()
vertices = processed_ml_mesh.vertex_matrix()
faces = processed_ml_mesh.face_matrix()
open3d_mesh = o3d.geometry.TriangleMesh()
open3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
open3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
print("Converted mesh successfully!")

# 4. Mesh cleanup
open3d_mesh.remove_degenerate_triangles()
open3d_mesh.remove_duplicated_triangles()
open3d_mesh.remove_duplicated_vertices()
open3d_mesh.remove_unreferenced_vertices()
open3d_mesh.remove_non_manifold_edges()
open3d_mesh.compute_vertex_normals()
print("Mesh cleaned up!")
print(
    f"Simplified mesh has {len(open3d_mesh.vertices)} vertices and {len(open3d_mesh.triangles)} triangles"
)

## MESHIFICATION
# 5. Mesh point‑cloud sampling (using PyMeshLab only)
ms.generate_sampling_poisson_disk(samplenum=50, exactnumflag=True)
sample_layer = ms.current_mesh()
sample_pts = sample_layer.vertex_matrix()

regpcd = o3d.geometry.PointCloud()
regpcd.points = o3d.utility.Vector3dVector(sample_pts)

print("Poisson‑disk sampling completed!")


# 6.  Build an undirected edge‑length graph of the reconstructed mesh
V = np.asarray(open3d_mesh.vertices)
F = np.asarray(open3d_mesh.triangles)

# all unique (undirected) edges
E = np.vstack((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]))
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

# 7.  Identify source vertices and run multi‑source Dijkstra
# map each point in regpcd to its nearest *mesh* vertex index
kdt = cKDTree(V)
src_indices = kdt.query(np.asarray(regpcd.points))[1]

# fastest option: return only the minimum distance from *any* source
dists = dijkstra(
    G, directed=False, indices=src_indices, min_only=True
)  # 1‑D array, length = n_vertices

print("Geodesic distances computed!")

# 8. Painting the mesh
vmin, vmax = np.percentile(dists, [0, 100])
norm = mpl.colors.PowerNorm(gamma=0.70, vmin=vmin, vmax=vmax, clip=True)
colors = plt.cm.plasma_r(norm(dists))[          #type: ignore
    :, :3
]  # drop the alpha channel            
open3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
print("Mesh painted!")

## POSTPROCESSING
# . Visualize
if open3d_mesh.is_empty():
    print("No triangles produced – increase sample density or radii.")
else:
    # o3d.visualization.draw_geometries([regpcd])
    o3d.visualization.draw_geometries([open3d_mesh])
