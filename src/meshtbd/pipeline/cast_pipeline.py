from __future__ import annotations

import bpy  # noqa: F401
import pymeshlab as ml  # noqa: F401
import pyvista as pv
import numpy as np
from matplotlib.colors import rgb_to_hsv  # noqa: F401
from pathlib import Path  # noqa: F401
import bmesh  # noqa: F401

from ..calibration import collect_pick_result, compute_scale_factor, resolve_real_world_distance
from ..models import PipelineConfig
from ..ops.blender_ops import (
    apply_cast_modifiers,
    create_object_from_triangle_mesh,
    export_object,
    smooth_hole_boundaries,
)
from ..ops.meshlab_ops import (
    add_triangle_mesh,
    build_scaled_voronoi_projection,
    current_mesh_arrays,
    remesh_current_selection,
)
from ..ops.pyvista_ops import (
    attach_rgb_point_data,
    build_colorized_polydata,
    compute_red_like_mask,
    extract_red_surface,
    preview_colorized_mesh,
    triangle_arrays_from_polydata,
)


def run_pipeline(config: PipelineConfig) -> Path:
    pick_result, _ = collect_pick_result(config.calibration)

    print("\n=== Pick result ===")
    print(f"P0 (vertex {pick_result.v0}): {pick_result.p0.tolist()}")
    print(f"P1 (vertex {pick_result.v1}): {pick_result.p1.tolist()}")
    print(f"Geodesic distance:   {pick_result.geodesic_distance:.10g}")
    print("(Units match the mesh coordinate units.)\n")

    real_world_distance = resolve_real_world_distance(config.calibration, pick_result)
    scale = compute_scale_factor(pick_result.geodesic_distance, real_world_distance)
    print(
        f"\nScale factor = real / mesh = "
        f"{real_world_distance:.3f} / {pick_result.geodesic_distance:.3f} = {scale:.6f}"
    )

    print("\nLoading mesh in PyMeshLab and applying geodesic-derived scale...")
    ms = build_scaled_voronoi_projection(
        config.input_path,
        scale,
        config.export.scaled_polydata_out,
    )

    vertices, faces, colors = current_mesh_arrays(ms)
    color_mesh = build_colorized_polydata(vertices, faces, colors)
    print("PyVista conversion completed!")
    if config.export.preview_color_mesh:
        preview_colorized_mesh(color_mesh)

    rgb = attach_rgb_point_data(color_mesh)
    red_like = compute_red_like_mask(rgb)
    color_mesh["keep"] = red_like.astype(np.uint8)
    print("Selected vertices to delete!")

    red_mesh = extract_red_surface(color_mesh)
    red_vertices, red_faces = triangle_arrays_from_polydata(red_mesh)
    add_triangle_mesh(ms, red_vertices, red_faces)

    smooth_mesh = remesh_current_selection(ms)
    smooth_vertices = smooth_mesh.vertex_matrix()
    smooth_faces = smooth_mesh.face_matrix()

    obj = create_object_from_triangle_mesh(smooth_vertices, smooth_faces)
    apply_cast_modifiers(obj)
    smooth_hole_boundaries(obj)
    export_object(config.export.output_path)
    print(f"Final mesh exported to: {config.export.output_path.resolve()}")
    return config.export.output_path
