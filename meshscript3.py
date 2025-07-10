import copy
import numpy as np
import open3d as o3d

print("Testing mesh...")
mesh = o3d.io.read_triangle_mesh("Just forearm.stl")
print(np.asarray(mesh.vertices))
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])