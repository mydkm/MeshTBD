import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
import open3d as o3d              # type: ignore
import pymeshlab as pyml          # type: ignore
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.max_open_warning"] = 0    # keep the console tidy

# ════════════════════════════════════════════════════════════════════════════
# 1.  Load STL with PyMeshLab
# ════════════════════════════════════════════════════════════════════════════
ms = pyml.MeshSet()
ms.load_new_mesh("Just forearm.stl")
print("Loaded mesh!")

# 2.  Surface reconstruction (unchanged)
ms.generate_surface_reconstruction_vcg(voxsize=pyml.PercentageValue(0.50))
ms.set_current_mesh(1)
ms.save_current_mesh("plymcout.ply")
ms.load_new_mesh("plymcout.ply")
ms.meshing_surface_subdivision_loop(threshold=pyml.PercentageValue(0.50))
print("Reconstruction complete!")

# 3.  Convert PyMeshLab mesh → Open3D mesh (unchanged)
processed_ml_mesh = ms.current_mesh()
vertices = processed_ml_mesh.vertex_matrix()
faces    = processed_ml_mesh.face_matrix()

open3d_mesh = o3d.geometry.TriangleMesh()
open3d_mesh.vertices  = o3d.utility.Vector3dVector(vertices)
open3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
print("Converted mesh successfully!")

# 4.  Mesh cleanup (unchanged)
open3d_mesh.remove_degenerate_triangles()
open3d_mesh.remove_duplicated_triangles()
open3d_mesh.remove_duplicated_vertices()
open3d_mesh.remove_unreferenced_vertices()
open3d_mesh.remove_non_manifold_edges()
open3d_mesh.compute_vertex_normals()
print(
    f"Mesh cleaned!  {len(open3d_mesh.vertices)} verts, {len(open3d_mesh.triangles)} tris"
)

# ════════════════════════════════════════════════════════════════════════════
# 5.  Poisson‑disk sampling  (uses PyMeshLab only)           ← SAME AS BEFORE
# ════════════════════════════════════════════════════════════════════════════
ms.generate_sampling_poisson_disk(samplenum=50, exactnumflag=True)
sample_layer = ms.current_mesh()
sample_pts   = sample_layer.vertex_matrix()

regpcd = o3d.geometry.PointCloud()
regpcd.points = o3d.utility.Vector3dVector(sample_pts)
print("Poisson‑disk sampling completed!")

# ── FINAL ▶  robust seed colouring  ─────────────────────────────────────────
ml_col_rgb = sample_layer.vertex_color_matrix()           # shape (N,3 or 4)

use_ml = ml_col_rgb.size != 0 and np.any(ml_col_rgb)

if use_ml:
    # keep only first 3 channels (RGB)
    if ml_col_rgb.shape[1] > 3:
        ml_col_rgb = ml_col_rgb[:, :3]

    ml_col_rgb = ml_col_rgb.astype(np.float64)

    # ↳ normalise only if values look like 0‑to‑255 integers
    if ml_col_rgb.max() > 1.0:
        ml_col_rgb /= 255.0

    # ↳ if every row is (almost) identical, we consider it “un‑coloured”
    unique_rows = np.unique(np.round(ml_col_rgb, 6), axis=0)
    if len(unique_rows) < 2:           # all‑white or all‑something
        use_ml = False                 # force fallback palette

if not use_ml:
    # fabricate a qualitative palette with distinct hues
    ml_col_rgb = plt.cm.tab20(np.linspace(0, 1, len(sample_pts)))[:, :3] # type: ignore

# make sure the array is contiguous float64 → Open3D friendly
seed_rgb = np.ascontiguousarray(ml_col_rgb, dtype=np.float64)

# hand colours to Open3D
regpcd.colors = o3d.utility.Vector3dVector(seed_rgb)
# ────────────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════════════
# 6.  Build an undirected edge‑length graph of the mesh       ← SAME AS BEFORE
# ════════════════════════════════════════════════════════════════════════════
V = np.asarray(open3d_mesh.vertices)
F = np.asarray(open3d_mesh.triangles)

# all unique (undirected) edges
E = np.vstack((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]))
E = np.sort(E, axis=1)
E = np.unique(E, axis=0)

edge_lens = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)

n = len(V)
rows = np.hstack((E[:, 0], E[:, 1]))
cols = np.hstack((E[:, 1], E[:, 0]))
data = np.hstack((edge_lens, edge_lens))
G = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

# ════════════════════════════════════════════════════════════════════════════
# 7.  Map seeds to mesh vertices and run *multi‑source* Dijkstra
# ════════════════════════════════════════════════════════════════════════════
kdt         = cKDTree(V)
src_indices = kdt.query(np.asarray(regpcd.points))[1]      # seed → nearest vertex

# ── NEW ▶ 7a.  Full distance matrix (k × n) – we need *which* seed wins
dist_matrix = dijkstra(G, directed=False, indices=src_indices)
winner_seed = np.argmin(dist_matrix, axis=0)               # shape (n_vertices,)

print("Voronoi regions computed!")

# ════════════════════════════════════════════════════════════════════════════
# 8.  Colour the mesh with the RGB of the *winning* seed       ← CHANGED
# ════════════════════════════════════════════════════════════════════════════
vertex_rgb = seed_rgb[winner_seed]                         # shape (n,3)
open3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_rgb)
print("Mesh painted with Voronoi colours!")

# ════════════════════════════════════════════════════════════════════════════
# 9.  Visualise (unchanged)
# ════════════════════════════════════════════════════════════════════════════
if open3d_mesh.is_empty():
    print("No triangles produced – increase sample density.")
else:
    o3d.visualization.draw_geometries([open3d_mesh])
