import open3d as o3d

mesh = o3d.io.read_triangle_mesh("SabrinaOutput.ply")

o3d.io.write_triangle_mesh("SabrinaOutput.obj", mesh, write_triangle_uvs=True)
