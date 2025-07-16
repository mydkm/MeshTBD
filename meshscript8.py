import pymeshlab as pyml              # type: ignore
import pyvista as pv                  # type: ignore
import numpy as np
from matplotlib.colors import rgb_to_hsv
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 1.  PyMeshLab pipeline (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
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

ms.save_current_mesh("z.ply")   # coloured, *unfiltered* mesh

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Load with PyVista and strip the (transparent) alpha channel
# ──────────────────────────────────────────────────────────────────────────────
mesh = pv.read("z.ply")
n_pts = mesh.n_points

if "RGBA" in mesh.point_data:
    rgba = np.asarray(mesh.point_data["RGBA"])
    if rgba.ndim == 1:
        rgba = rgba.reshape(n_pts, 4)
    rgb = rgba[:, :3].astype(np.uint8).copy()
    mesh.point_data.pop("RGBA")
    mesh.point_data["RGB"] = rgb

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Identify red/orange vertices (HSV filter)
# ──────────────────────────────────────────────────────────────────────────────
rgb_norm = mesh["RGB"].astype(float) / 255.0
hsv      = rgb_to_hsv(rgb_norm)

hue        = hsv[:, 0]                       # 0 → red, 0.14 → 50°
saturation = hsv[:, 1]

red_like = (hue <= 50/360.0) & (saturation >= 0.25)
mesh["keep"] = red_like.astype(np.uint8)

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Extract cells whose *all* vertices are red/orange
# ──────────────────────────────────────────────────────────────────────────────
red_vol = mesh.threshold(
    (0.5, 1.5),          # keep == 1
    scalars="keep",
    all_scalars=True
)                         # ← returns an UnstructuredGrid
print(f"[threshold] kept {red_vol.n_points} pts, {red_vol.n_cells} cells")

# ── convert back to a surface so we can save as .ply ─────────────────────────
red_mesh = red_vol.extract_surface()       # PolyData
red_mesh = red_mesh.triangulate()          # ensure triangle faces only
print(f"[surface]  final surface has {red_mesh.n_points} pts, "
      f"{red_mesh.n_cells} tris")

# ──────────────────────────────────────────────────────────────────────────────
# 4b. Save the CLEANED mesh to disk
# ──────────────────────────────────────────────────────────────────────────────
out_path = "only_red.ply"
red_mesh.save(out_path)                    # works: PolyData → .ply
print(f"Wrote filtered mesh to {Path(out_path).resolve()}")

# (Optional) bring it back into PyMeshLab
ms_red = pyml.MeshSet()
ms_red.load_new_mesh(out_path)

# ──────────────────────────────────────────────────────────────────────────────
# 5.  Plot the cleaned mesh
# ──────────────────────────────────────────────────────────────────────────────
red_mesh.plot(
    rgb=True,
    show_scalar_bar=False,
    background="white",
)

