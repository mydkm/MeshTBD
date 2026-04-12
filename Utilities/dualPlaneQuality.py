#!/usr/bin/env python3
"""
dual_plane_quality.py

Compute both longitudinal and transverse plane-based quality maps for a mesh,
attach them as point-data arrays, optionally export the enriched mesh, and
display an interactive viewer that lets the user switch between the two maps.

This script intentionally mirrors the CLI shape of Utilities/planeCalc.py,
except it does not expose --plane-mode because it always computes both modes.

Keyboard shortcuts in the viewer:
  L / 1 : show longitudinal quality
  T / 2 : show transverse quality
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv

from planeCalc import (
    closest_point_on_surface_vtk,
    compute_diffusion_distance_to_band,
    compute_euclidean_distance_to_band,
    compute_volume_and_centroid_signed_tets,
    diffusion_embedding,
    ensure_polydata,
    geodesic_distance_to_band,
    make_plane_patch,
    mesh_diag,
    normalize,
    normalize_01,
    pca_axes,
    pick_nonparallel_axis,
    warn_if_not_watertight,
)


@dataclass
class GeometryContext:
    mesh: pv.PolyData
    points: np.ndarray
    faces: np.ndarray
    diag: float
    ref: np.ndarray
    volume_signed: float
    centroid: np.ndarray
    closest_point: np.ndarray
    closest_cell_id: int
    closest_dist2: float
    arrow_dir: np.ndarray
    axes: np.ndarray
    ep: float


@dataclass
class PlaneQualityResult:
    mode: str
    plane_normal: np.ndarray
    plane_offset: float
    plane_patch: pv.PolyData
    band_points: np.ndarray
    band_mask: np.ndarray
    band_idx: np.ndarray
    plane_signed_dist: np.ndarray
    dist_to_band: np.ndarray
    dist_norm: np.ndarray
    dist_signed: np.ndarray
    dist_signed_norm: np.ndarray
    quality_name: str
    quality: np.ndarray


def compute_geometry_context(mesh: pv.PolyData, ref_mode: str, ep: float | None) -> GeometryContext:
    mesh = ensure_polydata(mesh).triangulate()
    warn_if_not_watertight(mesh)

    points = np.asarray(mesh.points, dtype=np.float64)
    faces_raw = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)
    if faces_raw.size == 0:
        raise ValueError("Mesh has no faces.")
    faces = faces_raw[:, 1:4]

    diag = mesh_diag(mesh)

    ref = np.zeros(3, dtype=np.float64)
    if ref_mode == "mesh_center":
        ref = np.array(mesh.center, dtype=np.float64)

    volume_signed, centroid = compute_volume_and_centroid_signed_tets(mesh, ref=ref)
    closest_point, closest_cell_id, closest_dist2 = closest_point_on_surface_vtk(mesh, centroid)
    if np.allclose(closest_point, centroid):
        raise ValueError("Closest surface point equals centroid; cannot define arrow direction.")

    arrow_dir = normalize(closest_point - centroid)
    axes = pca_axes(points, center=centroid)
    band_thickness = ep if ep is not None else 0.01 * (diag if diag > 0 else 1.0)

    return GeometryContext(
        mesh=mesh,
        points=points,
        faces=faces,
        diag=diag,
        ref=ref,
        volume_signed=volume_signed,
        centroid=centroid,
        closest_point=closest_point,
        closest_cell_id=closest_cell_id,
        closest_dist2=closest_dist2,
        arrow_dir=arrow_dir,
        axes=axes,
        ep=band_thickness,
    )


def compute_plane_normal(mode: str, arrow_dir: np.ndarray, axes: np.ndarray) -> np.ndarray:
    u0, u1, u2 = axes[:, 0], axes[:, 1], axes[:, 2]

    if mode == "normal":
        return normalize(arrow_dir)

    if mode == "longitudinal":
        u = u0
        u_perp = u - np.dot(u, arrow_dir) * arrow_dir
        if np.linalg.norm(u_perp) < 1e-6:
            u = u1
            u_perp = u - np.dot(u, arrow_dir) * arrow_dir
        if np.linalg.norm(u_perp) < 1e-6:
            u = pick_nonparallel_axis(arrow_dir)
            u_perp = u - np.dot(u, arrow_dir) * arrow_dir
        u_perp = normalize(u_perp)
        return normalize(np.cross(arrow_dir, u_perp))

    if mode == "transverse":
        u = u0
        m = u - np.dot(u, arrow_dir) * arrow_dir
        if np.linalg.norm(m) < 1e-6:
            u = u1
            m = u - np.dot(u, arrow_dir) * arrow_dir
        if np.linalg.norm(m) < 1e-6:
            u = u2
            m = u - np.dot(u, arrow_dir) * arrow_dir
        if np.linalg.norm(m) < 1e-6:
            u = pick_nonparallel_axis(arrow_dir)
            m = u - np.dot(u, arrow_dir) * arrow_dir
        return normalize(m)

    raise ValueError(f"Unsupported plane mode: {mode}")


def compute_distance_to_band(
    ctx: GeometryContext,
    band_idx: np.ndarray,
    distance_mode: str,
    diffusion_data: tuple[np.ndarray, np.ndarray] | None,
    geodesic_backend: str,
) -> np.ndarray:
    if distance_mode == "euclidean":
        return compute_euclidean_distance_to_band(ctx.points, ctx.points[band_idx])

    if distance_mode == "diffusion":
        if diffusion_data is None:
            raise RuntimeError("Diffusion embedding was not prepared.")
        embedding, _ = diffusion_data
        return compute_diffusion_distance_to_band(embedding, band_idx)

    d_to_band = geodesic_distance_to_band(ctx.points, ctx.faces, band_idx, backend=geodesic_backend)
    if np.all(np.isfinite(d_to_band)):
        return d_to_band

    finite = d_to_band[np.isfinite(d_to_band)]
    if finite.size == 0:
        print("[WARN] All geodesic distances are infinite. Setting all distances to 0.")
        return np.zeros_like(d_to_band)

    max_finite = float(np.max(finite))
    print(
        f"[WARN] Some vertices were unreachable from the {band_idx.size}-vertex band. "
        f"Clamping inf to max finite ({max_finite:.6g}) for normalization."
    )
    return np.where(np.isfinite(d_to_band), d_to_band, max_finite)


def compute_plane_quality(
    ctx: GeometryContext,
    args: argparse.Namespace,
    mode: str,
    diffusion_data: tuple[np.ndarray, np.ndarray] | None,
) -> PlaneQualityResult:
    plane_normal = compute_plane_normal(mode, ctx.arrow_dir, ctx.axes)
    plane_offset = -float(np.dot(plane_normal, ctx.centroid))
    plane_signed_dist = ctx.points @ plane_normal + plane_offset

    band_mask = np.abs(plane_signed_dist) <= ctx.ep
    band_idx = np.where(band_mask)[0]
    if band_idx.size == 0:
        band_idx = np.argsort(np.abs(plane_signed_dist))[:1]
        band_mask = np.zeros_like(plane_signed_dist, dtype=bool)
        band_mask[band_idx] = True
        print(
            f"[WARN] {mode}: no vertices satisfied abs(Ax+By+Cz+D) <= ep (ep={ctx.ep:.6g}). "
            "Falling back to the closest-to-plane vertex."
        )

    dist_to_band = compute_distance_to_band(
        ctx,
        band_idx=band_idx,
        distance_mode=args.distance_mode,
        diffusion_data=diffusion_data,
        geodesic_backend=args.geodesic_backend,
    )

    dist_norm = normalize_01(dist_to_band)
    side = np.where(plane_signed_dist < 0.0, -1.0, 1.0)
    dist_signed = dist_to_band * side
    dist_signed_norm = dist_norm * side

    quality_name = f"quality_{mode}"
    quality = dist_signed_norm if args.signed_under_plane else dist_norm

    return PlaneQualityResult(
        mode=mode,
        plane_normal=plane_normal,
        plane_offset=plane_offset,
        plane_patch=make_plane_patch(
            center=ctx.centroid,
            normal=plane_normal,
            mesh=ctx.mesh,
            scale=args.plane_scale,
        ),
        band_points=ctx.points[band_idx],
        band_mask=band_mask,
        band_idx=band_idx,
        plane_signed_dist=plane_signed_dist,
        dist_to_band=dist_to_band,
        dist_norm=dist_norm,
        dist_signed=dist_signed,
        dist_signed_norm=dist_signed_norm,
        quality_name=quality_name,
        quality=quality,
    )


def attach_plane_quality(mesh: pv.PolyData, result: PlaneQualityResult) -> None:
    prefix = result.mode
    mesh.point_data[f"{prefix}_plane_signed_dist"] = result.plane_signed_dist
    mesh.point_data[f"{prefix}_in_band"] = result.band_mask.astype(np.uint8)
    mesh.point_data[f"{prefix}_dist_to_band"] = result.dist_to_band
    mesh.point_data[f"{prefix}_dist_norm"] = result.dist_norm
    mesh.point_data[f"{prefix}_dist_signed"] = result.dist_signed
    mesh.point_data[f"{prefix}_dist_signed_norm"] = result.dist_signed_norm
    mesh.point_data[result.quality_name] = result.quality


def save_ply_with_dual_quality(
    path: str | Path,
    verts: np.ndarray,
    faces: np.ndarray,
    quality: np.ndarray,
    quality_longitudinal: np.ndarray,
    quality_transverse: np.ndarray,
) -> None:
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    quality = np.asarray(quality, dtype=np.float64).reshape(-1)
    quality_longitudinal = np.asarray(quality_longitudinal, dtype=np.float64).reshape(-1)
    quality_transverse = np.asarray(quality_transverse, dtype=np.float64).reshape(-1)

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("verts must be (N,3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be (F,3) triangle indices.")
    if quality.shape[0] != verts.shape[0]:
        raise ValueError("quality must have length N (same as number of vertices).")
    if quality_longitudinal.shape[0] != verts.shape[0]:
        raise ValueError("quality_longitudinal must have length N.")
    if quality_transverse.shape[0] != verts.shape[0]:
        raise ValueError("quality_transverse must have length N.")

    with Path(path).open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {verts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float quality\n")
        f.write("property float quality_longitudinal\n")
        f.write("property float quality_transverse\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for (x, y, z), q, q_long, q_trans in zip(
            verts,
            quality,
            quality_longitudinal,
            quality_transverse,
        ):
            f.write(f"{x:.9g} {y:.9g} {z:.9g} {q:.9g} {q_long:.9g} {q_trans:.9g}\n")

        for i, j, k in faces:
            f.write(f"3 {int(i)} {int(j)} {int(k)}\n")


def export_mesh(
    path: str | Path,
    mesh: pv.PolyData,
    primary_quality_name: str,
) -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".ply":
        if primary_quality_name not in mesh.point_data:
            raise KeyError(f"Primary quality array '{primary_quality_name}' was not found on the mesh.")
        if "quality_longitudinal" not in mesh.point_data:
            raise KeyError("Required point-data array 'quality_longitudinal' was not found on the mesh.")
        if "quality_transverse" not in mesh.point_data:
            raise KeyError("Required point-data array 'quality_transverse' was not found on the mesh.")

        faces_raw = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)
        faces_tri = faces_raw[:, 1:4]

        save_ply_with_dual_quality(
            path,
            np.asarray(mesh.points),
            faces_tri,
            np.asarray(mesh.point_data[primary_quality_name], dtype=np.float64),
            np.asarray(mesh.point_data["quality_longitudinal"], dtype=np.float64),
            np.asarray(mesh.point_data["quality_transverse"], dtype=np.float64),
        )
        return
    if suffix in {".vtp", ".vtk"}:
        mesh.save(path)
        return
    raise ValueError(
        f"Unsupported export extension '{suffix or '<none>'}'. "
        "Use .vtp or .vtk for native point-data arrays, or .ply for scalar vertex properties."
    )


def resolve_initial_mode(export_scalar: str | None) -> str:
    if export_scalar is None:
        return "longitudinal"

    aliases = {
        "longitudinal": "longitudinal",
        "quality_longitudinal": "longitudinal",
        "transverse": "transverse",
        "quality_transverse": "transverse",
    }
    mode = aliases.get(export_scalar.strip().lower())
    if mode is None:
        print(
            f"[WARN] --export-scalar={export_scalar!r} does not identify one of the dual quality arrays. "
            "Defaulting the viewer/export active scalars to longitudinal."
        )
        return "longitudinal"
    return mode


def add_mode_label(plotter: pv.Plotter, mode: str) -> None:
    plotter.add_text(
        f"Showing: {mode} quality    [L/1: longitudinal, T/2: transverse]",
        name="quality-mode-label",
        position="upper_left",
        font_size=11,
    )


def show_dual_quality_viewer(
    mesh: pv.PolyData,
    ctx: GeometryContext,
    results: dict[str, PlaneQualityResult],
    args: argparse.Namespace,
    initial_mode: str,
) -> None:
    cmap = "coolwarm" if args.signed_under_plane else "viridis"
    clim = (-1.0, 1.0) if args.signed_under_plane else (0.0, 1.0)
    scalar_bar_title = "Signed vertex quality" if args.signed_under_plane else "Vertex quality"
    radius = 0.01 * (ctx.diag if ctx.diag > 0 else 1.0)

    plotter = pv.Plotter()

    # Keep separate mesh instances per mode so each actor preserves its own
    # scalar mapper state when the user toggles between quality maps.
    mesh_variants: dict[str, pv.PolyData] = {}
    mesh_actors: dict[str, object] = {}
    plane_actors: dict[str, object] = {}
    band_actors: dict[str, object] = {}
    normal_actors: dict[str, object] = {}

    for mode, result in results.items():
        mesh_variant = mesh.copy(deep=True)
        mesh_variant.set_active_scalars(result.quality_name)
        mesh_variants[mode] = mesh_variant

        mesh_actors[mode] = plotter.add_mesh(
            mesh_variant,
            scalars=result.quality_name,
            cmap=cmap,
            clim=clim,
            show_edges=args.show_edges,
            opacity=1.0,
            scalar_bar_args={"title": scalar_bar_title},
            show_scalar_bar=(mode == initial_mode),
        )
        plane_actors[mode] = plotter.add_mesh(
            result.plane_patch,
            color="white",
            opacity=0.30,
            show_scalar_bar=False,
        )
        normal_actors[mode] = plotter.add_mesh(
            pv.Arrow(start=ctx.centroid, direction=result.plane_normal, scale=0.18 * (ctx.diag if ctx.diag > 0 else 1.0)),
            color="gold",
            show_scalar_bar=False,
        )
        if args.show_band:
            band_actors[mode] = plotter.add_mesh(
                pv.PolyData(result.band_points),
                color="tomato",
                render_points_as_spheres=True,
                point_size=8,
                show_scalar_bar=False,
            )

    def set_mode(mode: str) -> None:
        for key, actor in mesh_actors.items():
            actor.SetVisibility(key == mode)
        for key, actor in plane_actors.items():
            actor.SetVisibility(key == mode)
        for key, actor in normal_actors.items():
            actor.SetVisibility(key == mode)
        for key, actor in band_actors.items():
            actor.SetVisibility(key == mode)
        add_mode_label(plotter, mode)
        plotter.render()

    plotter.add_mesh(pv.Sphere(radius=radius, center=ctx.centroid), color="white", show_scalar_bar=False)
    plotter.add_mesh(pv.Sphere(radius=radius, center=ctx.closest_point), color="black", show_scalar_bar=False)
    plotter.add_mesh(
        pv.Arrow(start=ctx.centroid, direction=ctx.arrow_dir, scale=0.25 * (ctx.diag if ctx.diag > 0 else 1.0)),
        color="deepskyblue",
        show_scalar_bar=False,
    )
    plotter.add_mesh(
        pv.Arrow(start=ctx.centroid, direction=normalize(ctx.axes[:, 0]), scale=0.20 * (ctx.diag if ctx.diag > 0 else 1.0)),
        color="limegreen",
        show_scalar_bar=False,
    )

    add_mode_label(plotter, initial_mode)
    plotter.add_key_event("l", lambda: set_mode("longitudinal"))
    plotter.add_key_event("1", lambda: set_mode("longitudinal"))
    plotter.add_key_event("t", lambda: set_mode("transverse"))
    plotter.add_key_event("2", lambda: set_mode("transverse"))
    plotter.add_axes()

    set_mode(initial_mode)
    plotter.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Path to input mesh.")
    ap.add_argument("--ref", choices=["origin", "mesh_center"], default="mesh_center")
    ap.add_argument("--plane-scale", type=float, default=0.6)
    ap.add_argument("--ep", type=float, default=None)

    ap.add_argument("--distance-mode", choices=["euclidean", "diffusion", "geodesic"], default="euclidean")
    ap.add_argument("--diffusion-k", type=int, default=60)
    ap.add_argument("--diffusion-t", type=float, default=0.01)
    ap.add_argument("--geodesic-backend", choices=["auto", "scipy", "python"], default="auto")

    ap.add_argument(
        "--export-ply",
        type=str,
        default=None,
        help=(
            "Optional output mesh path. Use .vtp/.vtk to preserve native point-data arrays, "
            "or .ply to write a deterministic ASCII file with quality, "
            "quality_longitudinal, and quality_transverse vertex properties."
        ),
    )
    ap.add_argument(
        "--export-scalar",
        type=str,
        default=None,
        help=(
            "Optional initial/active scalar selection for the dual-map result. "
            "Accepted values: longitudinal, transverse, quality_longitudinal, quality_transverse."
        ),
    )

    ap.add_argument(
        "--signed-under-plane",
        action="store_true",
        help="If set, the exported/displayed quality maps are signed so the negative side of each plane is negative.",
    )
    ap.add_argument("--show-band", action="store_true")
    ap.add_argument("--show-edges", action="store_true")
    args = ap.parse_args()

    mesh = pv.read(args.input)
    ctx = compute_geometry_context(mesh, ref_mode=args.ref, ep=args.ep)

    print("=== Shared Geometry ===")
    print(f"Signed volume V = {ctx.volume_signed:.6g}   (abs(V)={abs(ctx.volume_signed):.6g})")
    print(f"Centroid c = {ctx.centroid}")
    print(
        f"Closest point p = {ctx.closest_point}  "
        f"(cell_id={ctx.closest_cell_id}, dist={np.sqrt(ctx.closest_dist2):.6g})"
    )
    print(f"Arrow direction n = {ctx.arrow_dir}")
    print(f"Principal axis u0 = {normalize(ctx.axes[:, 0])}")
    print(f"ep = {ctx.ep:.6g}")

    diffusion_data: tuple[np.ndarray, np.ndarray] | None = None
    if args.distance_mode == "diffusion":
        print("=== Shared Diffusion Embedding ===")
        print(f"Building diffusion embedding with k={args.diffusion_k}, t={args.diffusion_t} ...")
        diffusion_data = diffusion_embedding(
            ctx.points,
            ctx.faces,
            k=int(args.diffusion_k),
            t=float(args.diffusion_t),
        )
        _, evals_use = diffusion_data
        print(
            f"Computed {diffusion_data[0].shape[1]} diffusion coords. "
            f"Eigenvalue range: [{evals_use.min():.6g}, {evals_use.max():.6g}]"
        )
    elif args.distance_mode == "geodesic":
        print("=== Shared Geodesic Settings ===")
        print(f"Backend: {args.geodesic_backend}")

    results = {
        "longitudinal": compute_plane_quality(ctx, args, "longitudinal", diffusion_data),
        "transverse": compute_plane_quality(ctx, args, "transverse", diffusion_data),
    }

    for mode, result in results.items():
        A, B, Cc = result.plane_normal.tolist()
        print(f"=== {mode.capitalize()} Plane ===")
        print(f"Plane normal m = {result.plane_normal}")
        print(f"Plane equation: {A:.6g} x + {B:.6g} y + {Cc:.6g} z + {result.plane_offset:.6g} = 0")
        print(
            f"{result.quality_name}: min={float(np.min(result.quality)):.6g}, "
            f"max={float(np.max(result.quality)):.6g}, #band_vertices={result.band_idx.size}"
        )
        attach_plane_quality(ctx.mesh, result)

    initial_mode = resolve_initial_mode(args.export_scalar)
    ctx.mesh.set_active_scalars(results[initial_mode].quality_name)

    if args.export_ply:
        export_mesh(
            args.export_ply,
            ctx.mesh,
            primary_quality_name=results[initial_mode].quality_name,
        )
        print(
            f"[OK] Exported mesh with '{results['longitudinal'].quality_name}' and "
            f"'{results['transverse'].quality_name}' to: {args.export_ply}"
        )
        if Path(args.export_ply).suffix.lower() == ".ply":
            secondary_mode = "transverse" if initial_mode == "longitudinal" else "longitudinal"
            print(
                "[OK] MeshLab .ply export details:\n"
                f"  - built-in vertex quality: quality_{initial_mode}\n"
                "  - extra vertex properties in file: quality_longitudinal, quality_transverse\n"
                f"  - 'quality' mirrors quality_{initial_mode}; the other map remains available as "
                f"quality_{secondary_mode}."
            )

    show_dual_quality_viewer(ctx.mesh, ctx, results, args, initial_mode=initial_mode)


if __name__ == "__main__":
    main()
