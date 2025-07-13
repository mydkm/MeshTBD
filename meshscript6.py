import pymeshlab as pyml
import pyvista as pv
import numpy as np
from matplotlib.colors import rgb_to_hsv

# ── 1. colour‑transfer pipeline (unchanged) ──────────────────────────────────
ms = pyml.MeshSet()
ms.load_new_mesh("Just forearm.stl")
surface_id = ms.current_mesh_id()

ms.generate_surface_reconstruction_vcg(voxsize=pyml.PercentageValue(0.499991))
ms.set_current_mesh(1)
ms.save_current_mesh("plymcout.ply")
ms.load_new_mesh("plymcout.ply")

ms.meshing_surface_subdivision_loop(threshold=pyml.PercentageValue(0.500009))
ms.generate_sampling_poisson_disk(samplenum=50, exactnumflag=True)
pointcloud_id = ms.current_mesh_id()

ms.set_current_mesh(surface_id)
ms.compute_color_by_point_cloud_voronoi_projection(
    coloredmesh=surface_id,
    vertexmesh=pointcloud_id,
    backward=True,
)
ms.save_current_mesh("z.ply")          # coloured full arm

# ── 2. PyVista: keep only red/orange faces ───────────────────────────────────
mesh = pv.read("z.ply")

rgb = np.asarray(mesh.point_data["RGBA"])[:, :3].astype(np.uint8)
mesh.point_data.pop("RGBA")
mesh["RGB"] = rgb

hsv = rgb_to_hsv(rgb.astype(float) / 255.0)
red_like = (hsv[:, 0] <= 50/360.0) & (hsv[:, 1] >= 0.25)
mesh["keep"] = red_like.astype(np.uint8)

red_mesh = mesh.threshold((0.5, 1.5), scalars="keep", all_scalars=False)
red_mesh = red_mesh.extract_surface()      # ensure triangle faces
red_mesh.save("a.ply")                     # ← TRUE hand‑off

# ── 3. PyMeshLab: resample → Laplacian smooth ────────────────────────────────
ms2 = pyml.MeshSet()
ms2.load_new_mesh("a.ply")

ms2.generate_resampled_uniform_mesh(mergeclosevert=True,
                                    discretize=False,
                                    multisample=True,
                                    absdist=True)

# ms2.apply_coord_laplacian_smoothing(stepsmoothnum=50,
#                                     cotangentweight=False)

ms2.save_current_mesh("new.ply")           # ← save from ms2

# ── 4. Visualise result ──────────────────────────────────────────────────────
pv.read("new.ply").plot()
