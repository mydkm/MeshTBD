from __future__ import annotations

from pathlib import Path


def mesh_clean(ms) -> None:
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    print("Mesh hygiene accounted for!")


def build_scaled_voronoi_projection(
    input_path: Path,
    scale: float,
    scaled_polydata_out: Path,
):
    import pymeshlab as ml

    ms = ml.MeshSet()
    ms.load_new_mesh(str(input_path))
    ms.compute_selection_by_small_disconnected_components_per_face()
    mesh_clean(ms)
    ms.meshing_decimation_quadric_edge_collapse()
    mesh_clean(ms)
    ms.meshing_remove_selected_faces()
    ms.meshing_remove_selected_vertices()
    ms.compute_matrix_from_scaling_or_normalization(axisx=scale, axisy=scale, axisz=scale)
    scaled_polydata_out.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(scaled_polydata_out))
    print(f"Scaled PolyData exported to: {scaled_polydata_out.resolve()}")

    ms.generate_surface_reconstruction_vcg(voxsize=ml.PercentageValue(0.50))
    print("Reconstruction complete!")
    surface_id = ms.current_mesh_id()

    ms.meshing_surface_subdivision_loop(threshold=ml.PercentageValue(0.50))
    print("Subdivision complete!")
    ms.generate_sampling_poisson_disk(samplenum=75, exactnumflag=True)
    print("Point cloud generated!")
    pointcloud_id = ms.current_mesh_id()

    ms.set_current_mesh(surface_id)
    ms.compute_color_by_point_cloud_voronoi_projection(
        coloredmesh=surface_id,
        vertexmesh=pointcloud_id,
        backward=True,
    )
    print("Color computed!")
    return ms


def current_mesh_arrays(ms):
    mesh = ms.current_mesh()
    return (
        mesh.vertex_matrix(),
        mesh.face_matrix(),
        mesh.vertex_color_matrix(),
    )


def add_triangle_mesh(ms, vertices, faces, name: str = "red_mesh") -> None:
    import numpy as np
    import pymeshlab as ml

    ml_mesh = ml.Mesh(
        vertex_matrix=np.asarray(vertices, dtype=np.float64),
        face_matrix=np.asarray(faces, dtype=np.int32),
    )
    ms.add_mesh(ml_mesh, name)
    print("New mesh uploaded to MeshLab!")


def remesh_current_selection(ms):
    import pymeshlab as ml

    mesh_clean(ms)
    ms.meshing_close_holes(maxholesize=50)
    ms.meshing_isotropic_explicit_remeshing(
        iterations=10,
        adaptive=True,
        checksurfdist=True,
        targetlen=ml.PercentageValue(0.250),
    )
    print("Surface remeshed!")
    return ms.current_mesh()
