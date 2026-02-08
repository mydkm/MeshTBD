#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyvista as pv

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

    # Fallback: mark the points array as modified
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
        # Update the single-point PolyData in-place.
        hover_cloud.points[0] = p_snap
        _mark_dataset_modified(hover_cloud)

        hover_actor.SetVisibility(True)
        pl.render()

    def on_mouse_move(*_args) -> None:
        nonlocal _last_vid

        # Stop hover updates after two picks or once result computed
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

        # Snap to nearest vertex (matches geodesic usage)
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

        # Disable hover now that we are done picking
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

    # Return both the pick result and the surface mesh we used
    return result["value"], surf


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Pick 2 points on an STL surface and compute geodesic distance "
            "(with hover preview), then write a uniformly scaled STL based "
            "on a real-world measurement."
        )
    )
    ap.add_argument("stl", help="Path to input .stl file")
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
        "--scaled-out",
        help=(
            "Optional path for saving a uniformly scaled STL. "
            "If omitted, will use <input_stem>_scaled.stl"
        ),
    )
    args = ap.parse_args()

    # 1) Interactive picking + geodesic measurement
    res, surf = pick_two_points_and_geodesic(
        args.stl,
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

    # 3) Decide output path for scaled mesh
    if args.scaled_out:
        scaled_path = args.scaled_out
    else:
        base, ext = os.path.splitext(args.stl)
        if not ext:
            ext = ".stl"
        scaled_path = f"{base}_scaled{ext}"

    # 4) Apply scaling to the already-loaded mesh
    print("\nReusing in-memory mesh to apply scaling (no second load)...")
    mesh = surf.copy()
    mesh.points *= scale

    # Optional sanity-check geodesic on scaled mesh
    try:
        path_scaled = mesh.geodesic(res.v0, res.v1, keep_order=True)
        print(
            f"Geodesic after scaling: {path_scaled.length:.3f} "
            f"(expected ≈ {real_mm:.3f} mm)"
        )
    except Exception:
        print("Warning: could not recompute geodesic on scaled mesh for sanity check.")

    # 5) Save scaled STL
    mesh.save(scaled_path)
    print(f"\nScaled STL written to: {scaled_path}")
    print(
        "You can now feed this scaled STL into VoronoiArm_NC.py "
        "and the rest of your pipeline."
    )


if __name__ == "__main__":
    main()
