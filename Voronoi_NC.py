import pymeshlab as pyml  # type: ignore
import pyvista as pv  # type: ignore
import numpy as np  # type: ignore
from matplotlib.colors import rgb_to_hsv  # type: ignore
from pathlib import Path
import bpy # type: ignore
import bmesh # type: ignore
import os  # to implement: support of input args

# PyMeshlab preprocessing
ms = pyml.MeshSet()
ms.load_new_mesh("Just forearm.stl")

ms.generate_surface_reconstruction_vcg(voxsize=pyml.PercentageValue(0.50))
print("Reconstruction complete!")
ms.load_new_mesh("plymcout.ply")
surface_id = ms.current_mesh_id()

ms.meshing_surface_subdivision_loop(threshold=pyml.PercentageValue(0.50))
print("Subdivision complete!")
ms.generate_sampling_poisson_disk(samplenum=50, exactnumflag=True)
print("Point cloud generated!")
pointcloud_id = ms.current_mesh_id()

ms.set_current_mesh(surface_id)
ms.compute_color_by_point_cloud_voronoi_projection(
    coloredmesh=surface_id,
    vertexmesh=pointcloud_id,
    backward=True,
)
print("Color computed!")

# Meshification in PyVista
csurface_id = ms.current_mesh_id()
csurface = ms.current_mesh()
cvertices = csurface.vertex_matrix() # (N, 3) float64
cfaces = csurface.face_matrix() # (F, 3) int32
colors = csurface.vertex_color_matrix() # (N, 4)
cfaces_pv = np.hstack(
    [np.full((cfaces.shape[0], 1), 3, dtype=np.int64), cfaces]
).ravel()
cmesh = pv.PolyData(cvertices, cfaces_pv)
cmesh.point_data["RGBA"] = colors
n_pts = cmesh.n_points
print("PyVista conversion completed!")
# cmesh.plot(rgb=True)

if "RGBA" in cmesh.point_data:
    rgba = np.asarray(cmesh.point_data["RGBA"])
    if rgba.ndim == 1:
        rgba = rgba.reshape(n_pts, 4)
    rgb = rgba[:, :3].astype(np.uint8).copy()
    cmesh.point_data.pop("RGBA")
    cmesh.point_data["RGB"] = rgb

# Identify red/orange vertices (HSV filter)
rgb_norm = cmesh["RGB"].astype(float) / 255.0
hsv      = rgb_to_hsv(rgb_norm)

hue        = hsv[:, 0]                       # 0 → red, 0.14 → 50°
saturation = hsv[:, 1]

red_like = (hue <= 50/360.0) & (saturation >= 0.25)
cmesh["keep"] = red_like.astype(np.uint8)
print("Selected vertices to delete!")

# Extract cells whose *all* vertices are red/orange
red_vol = cmesh.threshold(
    (0.5, 1.5),          # keep == 1
    scalars="keep",
    all_scalars=False
)                         # returns an UnstructuredGrid
print("Deleted selected vertices!")

# print(f"[threshold] kept {red_vol.n_points} verts, {red_vol.n_cells} cells")

red_mesh = red_vol.extract_surface()       # PolyData
red_mesh = red_mesh.triangulate()          # ensure triangle faces only
red_faces = red_mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32)   # drop the leading 3’s
red_verts = red_mesh.points.astype(np.float64)
# print(f"[surface] final surface has {red_mesh.n_points} pts, "
#       f"{red_mesh.n_cells} tris")
mesh_kwargs = dict(vertex_matrix=red_verts, face_matrix=red_faces)
ml_mesh = pyml.Mesh(**mesh_kwargs)
ms.add_mesh(ml_mesh, "red_mesh")
print("New mesh uploaded to MeshLab!")

ms.apply_coord_laplacian_smoothing(stepsmoothnum = 50, cotangentweight = False)
print("Laplacian smooth complete!")
red_verts = ml_mesh.vertex_matrix()          
red_faces = ml_mesh.face_matrix()

me = bpy.data.meshes.new('pymlMesh')
bm = bmesh.new()

for v in red_verts:
    bm.verts.new(v)
bm.verts.ensure_lookup_table()

for tri in red_faces:
    try:
        bm.faces.new([bm.verts[i] for i in tri])
    except ValueError:
        pass

bm.to_mesh(me)
bm.free()

obj = bpy.data.objects.new('pymlMesh', me)
bpy.context.collection.objects.link(obj)

solid = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
solid.thickness  = 1.5       # metres; tweak to taste
solid.offset     = 1.0         # -1 = inward, +1 = outward, 0 = both sides
solid.use_even_offset = False   # uniform thickness around sharp bends
bpy.ops.object.modifier_apply(modifier=solid.name)

filepath = "output_file.ply" 

bpy.ops.export_mesh.ply(
    filepath=filepath,
    use_selection=False,  # Export whole mesh
    use_normals=True,   
    use_uv_coords=True,  
    use_colors=True,     
    global_scale=1.0,    
    axis_forward='Y',    
    axis_up='Z'          
)
