import bpy  
import pymeshlab as ml  
import pyvista as pv  
import numpy as np 
from matplotlib.colors import rgb_to_hsv  
from pathlib import Path
import bmesh  
import os
import argparse
import polyscope as ps

def mesh_clean(ms: ml.MeshSet):
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    print("Mesh hygiene accounted for!")

parser = argparse.ArgumentParser(
    description="Hello fellow meshers, step right up to see a body part (currently accepting .stl/.ply IO's) be turned into a cast. We're the only magicians that reveal our secrets though, come see how we do it at 'https://github.com/mydkm/MeshTBD'!"
)
parser.add_argument("-i", "--input", required=True)
# parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()
inputfile = os.path.relpath(args.input)
# outputfile = os.path.relpath(args.output)

print("Input file is", inputfile)
# print("Output file is", outputfile)

print("Input the length from the tip of your middle finger to your cubital fossa (in mm):")
armlength = float(input().strip())

# PyMeshlab preprocessing
ms = ml.MeshSet()
ms.load_new_mesh(inputfile)
inputfile_id = ms.current_mesh_id()

mesh_clean(ms)
ms.compute_selection_by_small_disconnected_components_per_face()
ms.meshing_remove_selected_faces()
ms.meshing_remove_selected_vertices()
bbox = ms.current_mesh().bounding_box()
bboxlength = bbox.dim_y()
scalefactor = armlength / bboxlength
ms.compute_matrix_from_scaling_or_normalization(axisx = scalefactor, axisy = scalefactor, axisz = scalefactor)
ms.generate_surface_reconstruction_vcg(voxsize=ml.PercentageValue(0.50))
print("Reconstruction complete!")
surface_id = ms.current_mesh_id()

ms.meshing_surface_subdivision_loop(threshold=ml.PercentageValue(0.50))
print("Subdivision complete!")
ms.generate_sampling_poisson_disk(samplenum=75, exactnumflag=True)
print("Point cloud generated!")
pointcloud_id = ms.current_mesh_id()

ms.save_current_mesh('result.obj')
ms.set_current_mesh(surface_id)
ms.save_current_mesh('result2.obj')
