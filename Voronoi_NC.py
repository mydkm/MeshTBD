import bpy  
import pymeshlab as ml  
import pyvista as pv  
import numpy as np  
from matplotlib.colors import rgb_to_hsv  
from pathlib import Path
import bmesh  
import os
import argparse

parser = argparse.ArgumentParser(
    description="Hello fellow meshers, step right up to see a body part (currently accepting .stl/.ply IO's) be turned into a cast. We're the only magicians that reveal our secrets though, come see how we do it at 'https://github.com/mydkm/MeshTBD'!"
)
parser.add_argument("-i", "--input", required=True)
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()
inputfile = os.path.relpath(args.input)
outputfile = os.path.relpath(args.output)

print("Input file is", inputfile)
print("Output file is", outputfile)

# PyMeshlab preprocessing
ms = ml.MeshSet()
ms.load_new_mesh(inputfile)

# ms.compute_matrix_from_scaling_or_normalization(axisx = 15.000000, axisy = 15.000000, axisz = 15.000000)
# note the scaling feature is still in progress, need to figure out how to choose an appropiate value
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

red_mesh = red_vol.extract_surface()  # PolyData
red_mesh = red_mesh.triangulate()  # ensure triangle faces only
red_faces = red_mesh.faces.reshape(-1, 4)[:, 1:].astype(
    np.int32
)  # drop the leading 3’s
red_verts = red_mesh.points.astype(np.float64)
mesh_kwargs = dict(vertex_matrix=red_verts, face_matrix=red_faces)
ml_mesh = ml.Mesh(**mesh_kwargs)
ms.add_mesh(ml_mesh, "red_mesh")
print("New mesh uploaded to MeshLab!")

# this filter is computationally expensive, maybe I can play around with this targetlen number
ms.meshing_isotropic_explicit_remeshing(
    iterations=10,
    adaptive=True,
    checksurfdist=True,
    targetlen=ml.PercentageValue(0.250),
)
print("Surface remeshed!")
smooth_mesh = ms.current_mesh()
red_verts = smooth_mesh.vertex_matrix()
red_faces = smooth_mesh.face_matrix()

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

displace = obj.modifiers.new(name="Displace", type="DISPLACE")
displace.strength = 1.5  # Adjust displacement strength
displace.mid_level = 0.5  # Adjust mid-level (value that gives no displacement)
displace.direction = (
    "NORMAL"  # Displacement direction (e.g., 'NORMAL', 'X', 'Y', 'Z', 'RGB_TO_XYZ')
)
displace.space = "LOCAL"  # Displacement space (e.g., 'LOCAL', 'GLOBAL')
solid = obj.modifiers.new(name="Solidify", type="SOLIDIFY")
solid.thickness = 1.5  # metres; tweak to taste
solid.offset = 1.0  # -1 = inward, +1 = outward, 0 = both sides
solid.use_even_offset = True  # uniform thickness around sharp bends
# include weld modifier characteristics here (smoothing of vertices closer to holes should follow)

bpy.ops.object.modifier_apply(modifier=displace.name)
bpy.ops.object.modifier_apply(modifier=solid.name)
print("Mesh thickened!")

filepath = outputfile
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

ext = Path(outputfile).suffix.lower()
if ext == ".ply":
    bpy.ops.wm.ply_export(
        filepath=filepath,
        export_selected_objects=False,  # Export whole mesh
        export_normals=True,
        export_uv=True,
        global_scale=1.0,
        forward_axis="Y",
        up_axis="Z",
    )
elif ext == ".stl":
    bpy.ops.wm.stl_export(
        filepath=filepath,
        export_selected_objects=False,  # Export whole mesh
        global_scale=1.0,
        forward_axis="Y",
        up_axis="Z",
    )
else:
    print("Export failed!")
