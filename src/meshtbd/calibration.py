from __future__ import annotations

from pathlib import Path
from typing import Optional

from .models import CalibrationInput, PickResult


def load_surface_mesh(path: str | Path):
    import pyvista as pv

    mesh = pv.read(str(path))

    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()

    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()

    mesh = mesh.triangulate()
    mesh = mesh.clean(tolerance=0.0)
    return mesh


def _safe_remove_observer(iren, obs_id: Optional[int]) -> None:
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
    if hasattr(dataset, "Modified"):
        dataset.Modified()
        return

    if hasattr(dataset, "GetPoints"):
        pts = dataset.GetPoints()
        if pts is not None and hasattr(pts, "Modified"):
            pts.Modified()


def compute_scale_factor(geodesic_distance: float, real_world_distance_mm: float) -> float:
    if geodesic_distance <= 0:
        raise ValueError("Geodesic distance must be positive.")
    if real_world_distance_mm <= 0:
        raise ValueError("Real-world distance must be positive.")
    return real_world_distance_mm / geodesic_distance


def default_scaled_polydata_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_scaled_polydata.ply")


def measure_geodesic_by_vertex_ids(
    input_path: str | Path,
    v0: int,
    v1: int,
):
    import numpy as np

    surf = load_surface_mesh(input_path)
    n_points = surf.n_points
    if not (0 <= v0 < n_points and 0 <= v1 < n_points):
        raise ValueError(f"Vertex ids must be within [0, {n_points - 1}].")

    result = PickResult(
        p0=np.asarray(surf.points[v0], dtype=float),
        p1=np.asarray(surf.points[v1], dtype=float),
        v0=int(v0),
        v1=int(v1),
        geodesic_distance=float(surf.geodesic_distance(v0, v1)),
    )
    return result, surf


def pick_two_points_and_geodesic(
    stl_path: str | Path,
    *,
    left_click: bool = True,
    auto_close: bool = False,
    picker: str = "cell",
):
    import numpy as np
    import pyvista as pv

    try:
        import pyvista._vtk as vtk  # type: ignore
    except Exception:  # pragma: no cover
        import vtk  # type: ignore

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

    last_vid: Optional[int] = None
    mouse_move_obs_id: Optional[int] = None

    def update_hover_point(p_snap: np.ndarray) -> None:
        hover_cloud.points[0] = p_snap
        _mark_dataset_modified(hover_cloud)
        hover_actor.SetVisibility(True)
        pl.render()

    def on_mouse_move(*_args) -> None:
        nonlocal last_vid

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

        point = np.array(hover_picker.GetPickPosition(), dtype=float)
        vid = int(surf.find_closest_point(point))
        if last_vid == vid:
            return
        last_vid = vid

        p_snap = np.asarray(surf.points[vid], dtype=float)
        update_hover_point(p_snap)

    mouse_move_obs_id = pl.iren.add_observer("MouseMoveEvent", on_mouse_move)

    def on_pick(*_args) -> None:
        nonlocal mouse_move_obs_id

        if result["value"] is not None:
            return

        point = pl.picked_point
        if point is None:
            return

        vid = int(surf.find_closest_point(point))
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


def collect_pick_result(calibration: CalibrationInput):
    if calibration.landmark_vertices is not None:
        v0, v1 = calibration.landmark_vertices
        return measure_geodesic_by_vertex_ids(calibration.input_path, v0, v1)

    return pick_two_points_and_geodesic(
        calibration.input_path,
        left_click=calibration.left_click,
        auto_close=calibration.auto_close,
        picker=calibration.picker,
    )


def resolve_real_world_distance(
    calibration: CalibrationInput,
    pick_result: PickResult,
) -> float:
    if calibration.real_world_distance_mm is not None:
        return float(calibration.real_world_distance_mm)

    user_input = input(
        "Enter real-world distance between these two landmarks "
        "(e.g. middle finger tip to cubital fossa, in mm): "
    ).strip()

    try:
        real_mm = float(user_input)
    except ValueError as exc:  # pragma: no cover
        raise SystemExit(f"Invalid numeric value: {user_input!r}") from exc

    if real_mm <= 0:
        raise SystemExit("Real-world distance must be positive.")

    if pick_result.geodesic_distance <= 0:
        raise SystemExit("Geodesic distance is non-positive; cannot calibrate scale.")

    return real_mm
