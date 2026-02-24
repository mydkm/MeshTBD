#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import bpy  # noqa: F401
import pymeshlab as ml  # noqa: F401
import pyvista as pv
import numpy as np
from matplotlib.colors import rgb_to_hsv  # noqa: F401
from pathlib import Path  # noqa: F401
import bmesh  # noqa: F401

# Prefer PyVista's VTK shim; fall back to vtk if needed.
try:
    import pyvista._vtk as vtk  # type: ignore
except Exception:  # pragma: no cover
    import vtk  # type: ignore


@dataclass(frozen=True)
class PickResult:
    p0: np.ndarray
    p1: np.ndarray
    v0: int
    v1: int
    geodesic_distance: float

def mesh_clean(ms: ml.MeshSet):
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    print("Mesh hygiene accounted for!")

def load_surface_mesh(path: str) -> pv.PolyData:
    """Read a mesh file and return a clean, triangulated PolyData surface."""
    mesh = pv.read(path)

    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()

    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()

    mesh = mesh.triangulate()
    mesh = mesh.clean(tolerance=0.0)
    return mesh


def _safe_remove_observer(iren, obs_id: Optional[int]) -> None:
    """Remove a VTK observer in a version-tolerant way."""
    if obs_id is None:
        return
    if hasattr(iren, "remove_observer"):
        try:
            iren.remove_observer(obs_id)
            return
        except Exception:
            pass
    if hasattr(iren, "RemoveObserver"):
        try:
            iren.RemoveObserver(obs_id)
        except Exception:
            pass


def _mark_dataset_modified(dataset) -> None:
    """
    Mark a VTK/PyVista dataset as modified in a version-tolerant way.

    - Some versions expose only VTK's `Modified()` (capital M).
    - Some wrappers may not expose it directly, so we fall back to points.Modified().
    """
    if hasattr(dataset, "Modified"):
        dataset.Modified()
        return

    if hasattr(dataset, "GetPoints"):
        pts = dataset.GetPoints()
        if pts is not None and hasattr(pts, "Modified"):
            pts.Modified()


def pick_two_points_and_geodesic(
    stl_path: str,
    *,
    left_click: bool = True,
    auto_close: bool = False,
    picker: str = "cell",
) -> tuple[PickResult, pv.PolyData]:
    """
    Open an interactive viewer, pick two points, and compute geodesic distance.

    Returns:
        (PickResult, surf_mesh)
    """
    surf = load_surface_mesh(stl_path)

    pl = pv.Plotter()
    mesh_actor = pl.add_mesh(surf, show_edges=False)

    pl.add_text(
        "Hover to preview pick (red).\n"
        "Pick TWO points on the surface.\n"
        "After the 2nd pick, the geodesic is computed.\n"
        "Close window to finish (or use --auto-close).",
        font_size=12,
        name="__msg__",
    )

    marker_radius = max(float(surf.length) * 0.0075, 1e-6)

    picked_points: list[np.ndarray] = []
    picked_vids: list[int] = []
    result: dict[str, Optional[PickResult]] = {"value": None}

    # ----------------------------
    # Hover indicator (red)
    # ----------------------------
    hover_cloud = pv.PolyData(np.array([[0.0, 0.0, 0.0]], dtype=float))
    hover_actor = pl.add_points(
        hover_cloud,
        name="__hover_point__",
        color="red",
        point_size=14,
        render_points_as_spheres=True,
        pickable=False,
    )
    hover_actor.SetVisibility(False)

    hover_picker = vtk.vtkCellPicker()
    hover_picker.SetTolerance(0.0005)
    hover_picker.AddPickList(mesh_actor)
    hover_picker.PickFromListOn()

    pl.track_mouse_position()

    _last_vid: Optional[int] = None
    mouse_move_obs_id: Optional[int] = None

    def update_hover_point(p_snap: np.ndarray) -> None:
        hover_cloud.points[0] = p_snap
        _mark_dataset_modified(hover_cloud)
        hover_actor.SetVisibility(True)
        pl.render()

    def on_mouse_move(*_args) -> None:
        nonlocal _last_vid

        if result["value"] is not None or len(picked_vids) >= 2:
            hover_actor.SetVisibility(False)
            return

        if pl.mouse_position is None:
            return

        x, y = pl.mouse_position
        ok = hover_picker.Pick(x, y, 0, pl.renderer)
        if not ok:
            hover_actor.SetVisibility(False)
            return

        p = np.array(hover_picker.GetPickPosition(), dtype=float)

        vid = int(surf.find_closest_point(p))
        if _last_vid == vid:
            return
        _last_vid = vid

        p_snap = np.asarray(surf.points[vid], dtype=float)
        update_hover_point(p_snap)

    mouse_move_obs_id = pl.iren.add_observer("MouseMoveEvent", on_mouse_move)

    # ----------------------------
    # Click picking callback
    # ----------------------------
    def on_pick(*_args) -> None:
        nonlocal mouse_move_obs_id

        if result["value"] is not None:
            return

        p = pl.picked_point
        if p is None:
            return

        vid = int(surf.find_closest_point(p))
        p_snap = np.asarray(surf.points[vid], dtype=float)

        picked_points.append(p_snap)
        picked_vids.append(vid)

        sphere = pv.Sphere(radius=marker_radius, center=p_snap)
        pl.add_mesh(sphere, color="red", name=f"pick_sphere_{len(picked_points)}")
        pl.add_point_labels(
            [p_snap],
            [f"P{len(picked_points)}"],
            font_size=12,
            name=f"pick_label_{len(picked_points)}",
        )
        pl.render()

        if len(picked_points) < 2:
            return

        hover_actor.SetVisibility(False)
        _safe_remove_observer(pl.iren, mouse_move_obs_id)
        mouse_move_obs_id = None

        v0, v1 = picked_vids[0], picked_vids[1]
        path = surf.geodesic(v0, v1, keep_order=True)
        dist = float(surf.geodesic_distance(v0, v1))

        pl.add_mesh(path, line_width=6, name="__geodesic_path__")
        pl.add_text(
            f"Done.\n"
            f"v0={v0}, v1={v1}\n"
            f"Geodesic distance: {dist:.6g}\n"
            f"{'(auto-closing...)' if auto_close else 'Close window to print results.'}",
            font_size=12,
            name="__msg__",
        )
        pl.render()

        result["value"] = PickResult(
            p0=picked_points[0],
            p1=picked_points[1],
            v0=v0,
            v1=v1,
            geodesic_distance=dist,
        )

        if auto_close:
            pl.close()

    pl.enable_surface_point_picking(
        callback=on_pick,
        left_clicking=left_click,
        picker=picker,
        show_message=False,
        show_point=True,
        color="red",
        point_size=12,
    )

    pl.show()

    if result["value"] is None:
        raise RuntimeError("Window closed before two points were picked.")

    return result["value"], surf




def main() -> int:
    ap = argparse.ArgumentParser()
    
    ap.add_argument(
            "stl",
            nargs="?",
            help="Path to input mesh file (.stl/.ply).",
        )
    ap.add_argument(
            "-i",
            "--input",
            dest="input",
            default=None,
            help="Path to input mesh file (.stl/.ply). Alias for positional stl.",
        )
    ap.add_argument(
            "-o",
            "--output",
            dest="output",
            default=None,
            help="Path to final output mesh (.stl/.ply).",
        )
    ap.add_argument(
            "--right-click",
            action="store_true",
            help="Use right-click instead of left-click",
        )
    ap.add_argument(
            "--auto-close",
            action="store_true",
            help="Automatically close the window after the 2nd pick",
        )
    ap.add_argument(
            "--picker",
            default="cell",
            choices=["hardware", "cell", "point", "volume"],
            help="VTK picker type used for clicks (default: cell).",
        )
    ap.add_argument(
            "--scaled-polydata-out",
            default=None,
            help=(
                "Optional output path for scaled PolyData (e.g. .ply/.stl/.obj). "
                "Defaults to <input_stem>_scaled_polydata.ply"
            ),
        )
    args = ap.parse_args()
    
    if args.stl and args.input and Path(args.stl) != Path(args.input):
        ap.error("Input path conflict: positional 'stl' and '--input' differ. Use only one.")
    
    input_arg = args.input or args.stl
    if input_arg is None:
        ap.error("Missing input mesh. Provide positional 'stl' or '-i/--input'.")
    
    inputfile = os.path.relpath(input_arg)
    
    if args.output:
        outputfile = os.path.relpath(args.output)
    else:
        input_path = Path(inputfile)
        outputfile = os.path.relpath(input_path.with_name(f"{input_path.stem}_output.stl"))
    
    # 1) Interactive picking + geodesic measurement
    res, surf = pick_two_points_and_geodesic(
            inputfile,
            left_click=not args.right_click,
            auto_close=args.auto_close,
            picker=args.picker,
        )
    
    print("\n=== Pick result ===")
    print(f"P0 (vertex {res.v0}): {res.p0.tolist()}")
    print(f"P1 (vertex {res.v1}): {res.p1.tolist()}")
    print(f"Geodesic distance:   {res.geodesic_distance:.10g}")
    print("(Units match the STL coordinate units.)\n")
    
    # 2) Scaling feature: map mesh geodesic to real-world distance
    geo = float(res.geodesic_distance)
    if geo <= 0:
        raise SystemExit("Geodesic distance is non-positive; cannot calibrate scale.")
    
    user_input = input(
        "Enter real-world distance between these two landmarks "
        "(e.g. middle finger tip to cubital fossa, in mm): "
    ).strip()
    
    try:
        real_mm = float(user_input)
    except ValueError:
        raise SystemExit(f"Invalid numeric value: {user_input!r}")
    if real_mm <= 0:
        raise SystemExit("Real-world distance must be positive.")
    
    scale = real_mm / geo
    print(f"\nScale factor = real / mesh = {real_mm:.3f} / {geo:.3f} = {scale:.6f}")
    
    # 4) Load into PyMeshLab and apply scale there to match VoronoiArm_NC behavior.
    print("\nLoading mesh in PyMeshLab and applying geodesic-derived scale...")
    ms = ml.MeshSet()
    ms.load_new_mesh(inputfile)
    mesh_clean(ms)
    ms.compute_selection_by_small_disconnected_components_per_face()
    ms.meshing_remove_selected_faces()
    ms.meshing_remove_selected_vertices()
    ms.compute_matrix_from_scaling_or_normalization(axisx=scale, axisy=scale, axisz=scale)
    
    if args.scaled_polydata_out:
        scaled_polydata_out = Path(args.scaled_polydata_out)
    else:
        in_path = Path(inputfile)
        scaled_polydata_out = in_path.with_name(f"{in_path.stem}_scaled_polydata.ply")
    
    scaled_polydata_out = scaled_polydata_out.resolve()
    # ms.save_current_mesh(str(scaled_polydata_out))
    print(f"Scaled PolyData exported to: {scaled_polydata_out}")
    
    ms.generate_surface_reconstruction_vcg(voxsize=ml.PercentageValue(0.50))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum = 50000)
    print("Reconstruction complete!")
    surface_id = ms.current_mesh_id()
    
    ms.meshing_surface_subdivision_loop(threshold=ml.PercentageValue(0.50))
    print("Subdivision complete!")
    ms.generate_sampling_poisson_disk(samplenum=75, exactnumflag=True)
    print("Point cloud generated!")
    pointcloud_id = ms.current_mesh_id()
    
    csurface = ms.set_current_mesh(surface_id)
    ms.compute_color_by_point_cloud_voronoi_projection(
        coloredmesh=surface_id,
        vertexmesh=pointcloud_id,
        backward=True,
    )
    print("Color computed!")
    
    # Meshification in PyVista
    cvertices = csurface.vertex_matrix()  # (N, 3) float64
    cfaces = csurface.face_matrix()  # (F, 3) int32
    colors = csurface.vertex_color_matrix()  # (N, 4)
    cfaces_pv = np.hstack(
        [np.full((cfaces.shape[0], 1), 3, dtype=np.int64), cfaces]
    ).ravel() # for polydata conversion
    cmesh = pv.PolyData(cvertices, cfaces_pv)
    cmesh.point_data["RGBA"] = colors
    n_pts = cmesh.n_points
    print("PyVista conversion completed!")
    cmesh.plot(rgb=True)
    
    
    if "RGBA" in cmesh.point_data:
        rgba = np.asarray(cmesh.point_data["RGBA"])
        if rgba.ndim == 1:
            rgba = rgba.reshape(n_pts, 4)
        # PyMeshLab colors are float in [0, 1]; scale before uint8 conversion.
        rgb = np.clip((rgba[:, :3] * 255.0).round(), 0, 255).astype(np.uint8)
        cmesh.point_data.pop("RGBA")
        cmesh.point_data["RGB"] = rgb
    
    # Identify red/orange vertices (HSV filter)
    rgb_norm = cmesh["RGB"].astype(float) / 255.0
    hsv = rgb_to_hsv(rgb_norm)
    
    hue = hsv[:, 0]  # 0 → red, 0.14 → 50°
    saturation = hsv[:, 1]
    
    red_like = (hue <= 50.0 / 360.0) & (saturation >= 0.25)
    cmesh["keep"] = red_like.astype(np.uint8)
    print("Selected vertices to delete!")
    
    # Extract cells whose *all* vertices are red/orange
    red_vol = cmesh.threshold(
        (0.5, 1.5), scalars="keep", all_scalars=True 
    )  # returns an UnstructuredGrid
    print("Deleted selected vertices!")
    
    red_mesh = red_vol.extract_surface()  # PolyData
    red_mesh = red_mesh.triangulate()  # ensure triangle faces only
    if red_mesh.n_points == 0 or red_mesh.n_cells == 0:
        raise SystemExit("No Voronoi cells selected by keep-mask; cannot continue.")
    red_faces = red_mesh.faces.reshape(-1, 4)[:, 1:].astype(
        np.int32
    )  # drop the leading 3’s
    red_verts = red_mesh.points.astype(np.float64)
    mesh_kwargs = dict(vertex_matrix=red_verts, face_matrix=red_faces)
    ml_mesh = ml.Mesh(**mesh_kwargs)
    ms.add_mesh(ml_mesh, "red_mesh")
    print("New mesh uploaded to MeshLab!")
    
    # this filter is computationally expensive, maybe I can play around with this targetlen number
    mesh_clean(ms)
    ms.meshing_close_holes(maxholesize = 50)
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
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    displace = obj.modifiers.new(name="Displace", type="DISPLACE")
    displace.strength = 1.5  # Adjust displacement strength
    displace.mid_level = 0.5  # Adjust mid-level (value that gives no displacement)
    displace.direction = (
        "NORMAL"  # Displacement direction (e.g., 'NORMAL', 'X', 'Y', 'Z', 'RGB_TO_XYZ')
    )
    displace.space = "LOCAL"  # Displacement space (e.g., 'LOCAL', 'GLOBAL')
    solid = obj.modifiers.new(name="Solidify", type="SOLIDIFY")
    solid.thickness = 2.5  # tweak to taste
    solid.offset = 1.0  # -1 = inward, +1 = outward, 0 = both sides
    solid.use_even_offset = True  # uniform thickness around sharp bends
    
    bpy.ops.object.modifier_apply(modifier=displace.name)
    bpy.ops.object.modifier_apply(modifier=solid.name)
    print("Mesh thickened!")
    
    # Smoothing hole edge faces
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    
    boundary_verts = set()
    for e in bm.edges:
        if len(e.link_faces) == 1:  # boundary edge
            boundary_verts.add(e.verts[0])
            boundary_verts.add(e.verts[1])
    
    def grow_region(verts, n=2):
        sel = set(verts)
        for _ in range(n):
            new_sel = set(sel)
            for v in sel:
                for e in v.link_edges:
                    new_sel.add(e.other_vert(v))
            sel = new_sel
        return sel
    
    smooth_region = grow_region(boundary_verts, n=3)
    
    bmesh.ops.smooth_vert(
        bm,
        verts=list(smooth_region),
        factor=0.8,   # smoothing strength (0–1)
        use_axis_x=True,
        use_axis_y=True,
        use_axis_z=True
    )
    
    bm.to_mesh(me)
    bm.free()
    print(f"Localized smoothing applied to {len(smooth_region)} vertices near holes!")
    
    ext = Path(outputfile).suffix.lower()
    if ext == ".ply":
        bpy.ops.wm.ply_export(
            filepath=outputfile,
            export_selected_objects=False,  # Export whole mesh
            export_normals=True,
            export_uv=True,
            global_scale=1.0,
            forward_axis="Y",
            up_axis="Z",
        )
    elif ext == ".stl":
        bpy.ops.wm.stl_export(
            filepath=outputfile,
            export_selected_objects=False,  # Export whole mesh
            global_scale=1.0,
            forward_axis="Y",
            up_axis="Z",
        )
    else:
        print("Export failed!")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())