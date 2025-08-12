import bpy  # type: ignore
import pymeshlab as ml  # type: ignore
import pyvista as pv  # type: ignore
import numpy as np  # type: ignore
from matplotlib.colors import rgb_to_hsv  # type: ignore
from pathlib import Path
import bmesh  # type: ignore
import os  # to implement: support of input args

# PyMeshlab preprocessing
ms = ml.MeshSet()
ms.load_new_mesh("Sabrina-scanSuccess-in.stl")

ms.generate_surface_reconstruction_vcg(voxsize=ml.PercentageValue(0.50))
print("Reconstruction complete!")
surface_id = ms.current_mesh_id()

ms.meshing_surface_subdivision_loop(threshold=ml.PercentageValue(0.50))
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
cvertices = csurface.vertex_matrix()  # (N, 3) float64
cfaces = csurface.face_matrix()  # (F, 3) int32
colors = csurface.vertex_color_matrix()  # (N, 4)
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
hsv = rgb_to_hsv(rgb_norm)

hue = hsv[:, 0]  # 0 → red, 0.14 → 50°
saturation = hsv[:, 1]

red_like = (hue <= 50 / 360.0) & (saturation >= 0.25)
cmesh["keep"] = red_like.astype(np.uint8)
print("Selected vertices to delete!")

# Extract cells whose *all* vertices are red/orange
red_vol = cmesh.threshold(
    (0.5, 1.5), scalars="keep", all_scalars=False  # keep == 1
)  # returns an UnstructuredGrid
print("Deleted selected vertices!")

# print(f"[threshold] kept {red_vol.n_points} verts, {red_vol.n_cells} cells")

red_mesh = red_vol.extract_surface()  # PolyData
red_mesh = red_mesh.triangulate()  # ensure triangle faces only
red_faces = red_mesh.faces.reshape(-1, 4)[:, 1:].astype(
    np.int32
)  # drop the leading 3’s
red_verts = red_mesh.points.astype(np.float64)
# print(f"[surface] final surface has {red_mesh.n_points} pts, "
#       f"{red_mesh.n_cells} tris")
mesh_kwargs = dict(vertex_matrix=red_verts, face_matrix=red_faces)
ml_mesh = ml.Mesh(**mesh_kwargs)
ms.add_mesh(ml_mesh, "red_mesh")
print("New mesh uploaded to MeshLab!")

ms.apply_coord_laplacian_smoothing(stepsmoothnum=50, cotangentweight=False)
print("Laplacian smooth complete!")
red_verts = ml_mesh.vertex_matrix()
red_faces = ml_mesh.face_matrix()

me = bpy.data.meshes.new("pymlMesh")
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

obj = bpy.data.objects.new("pymlMesh", me)
bpy.context.collection.objects.link(obj)

displace = obj.modifiers.new(name="Displace", type='DISPLACE')
displace.strength = 1.5  # Adjust displacement strength
displace.mid_level = 0.5 # Adjust mid-level (value that gives no displacement)
displace.direction = 'NORMAL' # Displacement direction (e.g., 'NORMAL', 'X', 'Y', 'Z', 'RGB_TO_XYZ')
displace.space = 'LOCAL' # Displacement space (e.g., 'LOCAL', 'GLOBAL')
solid = obj.modifiers.new(name="Solidify", type="SOLIDIFY")
solid.thickness = 1.5  # metres; tweak to taste
solid.offset = 1.0  # -1 = inward, +1 = outward, 0 = both sides
solid.use_even_offset = True  # uniform thickness around sharp bends
bpy.ops.object.modifier_apply(modifier=displace.name)
bpy.ops.object.modifier_apply(modifier=solid.name)
print("Mesh thickened!")

filepath = "output_file.ply"
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

bpy.ops.wm.ply_export(
    filepath=filepath,
    export_selected_objects=False,  # Export whole mesh
    export_normals=True,
    export_uv=True,
    global_scale=1.0,
    forward_axis="Y",
    up_axis="Z",
)
