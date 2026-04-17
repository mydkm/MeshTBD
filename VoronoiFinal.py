#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

import bpy  # noqa: F401
import pymeshlab as ml  # noqa: F401
import pyvista as pv
import numpy as np
from matplotlib.colors import rgb_to_hsv  # noqa: F401
from pathlib import Path  # noqa: F401
import bmesh  # noqa: F401
from plyfile import PlyData

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


@dataclass(frozen=True)
class PipelinePaths:
    tmp_dir: Path
    tmp1: Path
    masked_output: Path
    pyvista_cut_surface: Path
    boolean_result: Path


@dataclass(frozen=True)
class PlySummary:
    path: Path
    vertex_count: int
    face_count: int
    vertex_properties: tuple[str, ...]


QUALITY_NORMAL_LOWER_BOUND = -0.8
QUALITY_NORMAL_UPPER_BOUND = 0.4
QUALITY_TRANSVERSE_LOWER_BOUND = -0.8
QUALITY_TRANSVERSE_UPPER_BOUND = 0.6
POINT_CLOUD_SAMPLE_COUNT = 75
PYVISTA_DELETE_HUE_MAX = 45 / 360.0
PYVISTA_DELETE_SATURATION_MIN = 0.25
PRE_MASK_CROP_CONDITION = (
    "(x0 >= xmin && x0 <= xmax - 0.7*(xmax - xmin)) && "
    "(y0 >= ymin && y0 <= ymax) && "
    "(z0 >= zmin && z0 <= zmax - 0.8*(zmax-zmin))"
)


def stage_banner(title: str) -> None:
    print(f"\n=== {title} ===")


def build_pipeline_paths(tmp_dir: str | Path = "tmp") -> PipelinePaths:
    tmp_path = Path(tmp_dir)
    tmp_path.mkdir(parents=True, exist_ok=True)
    return PipelinePaths(
        tmp_dir=tmp_path,
        tmp1=tmp_path / "tmp1.ply",
        masked_output=tmp_path / "maskedoutput.ply",
        pyvista_cut_surface=tmp_path / "pyvista_cut_surface.ply",
        boolean_result=tmp_path / "tmp2.ply",
    )


def summarize_ply(path: str | Path) -> PlySummary:
    ply_path = Path(path)
    ply = PlyData.read(str(ply_path))
    vertex_properties = tuple(prop.name for prop in ply["vertex"].properties)
    face_count = int(ply["face"].count) if "face" in ply else 0
    return PlySummary(
        path=ply_path.resolve(),
        vertex_count=int(ply["vertex"].count),
        face_count=face_count,
        vertex_properties=vertex_properties,
    )


def log_ply_summary(
    path: str | Path,
    label: str,
    *,
    required_properties: tuple[str, ...] = (),
    require_faces: bool = False,
) -> PlySummary:
    summary = summarize_ply(path)
    missing = [name for name in required_properties if name not in summary.vertex_properties]
    if missing:
        raise KeyError(
            f"{label} is missing required vertex properties: {', '.join(missing)} "
            f"in {summary.path}"
        )
    if require_faces and summary.face_count == 0:
        raise SystemExit(f"{label} has no faces: {summary.path}")

    print(
        f"[{label}] {summary.path.name}: "
        f"vertices={summary.vertex_count}, faces={summary.face_count}, "
        f"vertex_properties={summary.vertex_properties}"
    )
    return summary


def ensure_file_exists(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise SystemExit(f"{label} was not created: {resolved}")
    return resolved


def _pymeshlab_mesh_counts(mesh: ml.Mesh) -> tuple[int, int]:
    vertices = int(np.asarray(mesh.vertex_matrix()).shape[0])
    try:
        faces = int(np.asarray(mesh.face_matrix()).shape[0])
    except Exception:
        faces = 0
    return vertices, faces


def log_current_pymeshlab_mesh(
    ms: ml.MeshSet,
    label: str,
    *,
    require_faces: bool = False,
) -> ml.Mesh:
    mesh = ms.current_mesh()
    vertices, faces = _pymeshlab_mesh_counts(mesh)
    if vertices == 0:
        raise SystemExit(f"{label} has no vertices in PyMeshLab.")
    if require_faces and faces == 0:
        raise SystemExit(f"{label} has no faces in PyMeshLab.")

    print(
        f"[{label}] PyMeshLab mesh_id={ms.current_mesh_id()} "
        f"vertices={vertices}, faces={faces}"
    )
    return mesh


def set_current_pymeshlab_mesh(
    ms: ml.MeshSet,
    mesh_id: int,
    label: str,
    *,
    require_faces: bool = False,
) -> ml.Mesh:
    ms.set_current_mesh(mesh_id)
    return log_current_pymeshlab_mesh(ms, label, require_faces=require_faces)


def log_pyvista_mesh(
    mesh: pv.PolyData,
    label: str,
    *,
    required_arrays: tuple[str, ...] = (),
    require_faces: bool = False,
) -> pv.PolyData:
    missing = [name for name in required_arrays if name not in mesh.point_data]
    if missing:
        raise KeyError(f"{label} is missing required PyVista point arrays: {', '.join(missing)}")
    if mesh.n_points == 0:
        raise SystemExit(f"{label} has no points in PyVista.")
    if require_faces and mesh.n_cells == 0:
        raise SystemExit(f"{label} has no cells in PyVista.")

    point_arrays = tuple(sorted(str(name) for name in mesh.point_data.keys()))
    print(
        f"[{label}] PyVista points={mesh.n_points}, cells={mesh.n_cells}, "
        f"point_arrays={point_arrays}"
    )
    return mesh


def log_surface_file(path: str | Path, label: str) -> pv.PolyData:
    mesh = load_surface_mesh(str(path))
    return log_pyvista_mesh(mesh, label, require_faces=True)


def remove_blender_object_if_exists(name: str) -> None:
    obj = bpy.data.objects.get(name)
    if obj is None:
        return

    mesh_data = obj.data if getattr(obj, "data", None) is not None else None
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh_data is not None and mesh_data.users == 0:
        bpy.data.meshes.remove(mesh_data, do_unlink=True)


def select_only_blender_object(obj) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def export_blender_object(obj, filepath: str | Path) -> None:
    filepath = Path(filepath)
    select_only_blender_object(obj)

    ext = filepath.suffix.lower()
    if ext == ".ply":
        bpy.ops.wm.ply_export(
            filepath=str(filepath),
            export_selected_objects=True,
            export_normals=True,
            export_uv=True,
            global_scale=1.0,
            forward_axis="Y",
            up_axis="Z",
            ascii_format=True,
        )
        return
    if ext == ".stl":
        bpy.ops.wm.stl_export(
            filepath=str(filepath),
            export_selected_objects=True,
            global_scale=1.0,
            forward_axis="Y",
            up_axis="Z",
        )
        return

    raise ValueError(f"Unsupported export extension for Blender export: {filepath.suffix or '<none>'}")

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


def read_ascii_ply(path: str | Path) -> dict[str, object]:
    """Read a simple ASCII PLY while preserving vertex properties and faces."""
    ply_path = Path(path)
    with ply_path.open("r", encoding="utf-8", errors="ignore") as f:
        vertex_count: int | None = None
        face_count = 0
        vertex_props: list[tuple[str, str]] = []
        face_prop_line = "property list uchar int vertex_indices"
        in_vertex_block = False

        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{ply_path}: unexpected EOF before end_header")
            line = line.rstrip("\n")
            parts = line.split()
            if not parts:
                continue

            if parts[0] == "format":
                if len(parts) < 3 or parts[1] != "ascii":
                    raise ValueError(f"{ply_path}: only ASCII PLY is supported")

            elif parts[0] == "element":
                in_vertex_block = False
                if len(parts) >= 3 and parts[1] == "vertex":
                    vertex_count = int(parts[2])
                    in_vertex_block = True
                elif len(parts) >= 3 and parts[1] == "face":
                    face_count = int(parts[2])

            elif parts[0] == "property" and in_vertex_block:
                if len(parts) != 3:
                    raise ValueError(f"{ply_path}: unsupported vertex property line: {line}")
                vertex_props.append((parts[1], parts[2]))

            elif parts[0] == "property" and "vertex_indices" in parts:
                face_prop_line = line

            elif parts[0] == "end_header":
                break

        if vertex_count is None:
            raise ValueError(f"{ply_path}: no vertex element found")
        if not vertex_props:
            raise ValueError(f"{ply_path}: no vertex properties found")

        vertices = np.empty((vertex_count, len(vertex_props)), dtype=np.float64)
        for i in range(vertex_count):
            line = f.readline()
            if not line:
                raise ValueError(f"{ply_path}: unexpected EOF while reading vertices")
            vals = line.strip().split()
            if len(vals) != len(vertex_props):
                raise ValueError(
                    f"{ply_path}: vertex line {i} has {len(vals)} values, expected {len(vertex_props)}"
                )
            vertices[i] = [float(v) for v in vals]

        faces: list[list[int]] = []
        for i in range(face_count):
            line = f.readline()
            if not line:
                raise ValueError(f"{ply_path}: unexpected EOF while reading faces")
            vals = [int(v) for v in line.strip().split()]
            if not vals:
                raise ValueError(f"{ply_path}: empty face line at index {i}")
            n = vals[0]
            idx = vals[1:]
            if len(idx) != n:
                raise ValueError(
                    f"{ply_path}: face line {i} says {n} vertices but has {len(idx)} indices"
                )
            faces.append(idx)

    return {
        "vertex_props": vertex_props,
        "face_prop_line": face_prop_line,
        "vertices": vertices,
        "faces": faces,
    }


def write_ascii_ply(
    path: str | Path,
    vertex_props: list[tuple[str, str]],
    vertices: np.ndarray,
    faces: list[list[int]],
    *,
    face_prop_line: str = "property list uchar int vertex_indices",
) -> None:
    """Write an ASCII PLY while preserving arbitrary vertex-property columns."""
    ply_path = Path(path)
    with ply_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment Created by VoronoiFinal vertex-overlap subtraction\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        for dtype_name, prop_name in vertex_props:
            f.write(f"property {dtype_name} {prop_name}\n")
        f.write(f"element face {len(faces)}\n")
        f.write(f"{face_prop_line}\n")
        f.write("end_header\n")

        for row in vertices:
            f.write(" ".join(f"{val:.9g}" for val in row) + "\n")

        for face in faces:
            f.write(f"{len(face)} " + " ".join(str(i) for i in face) + "\n")


def subtract_ply_by_vertex_overlap(
    source_ply: str | Path,
    cut_ply: str | Path,
    output_ply: str | Path,
) -> None:
    """Subtract a cut mesh by removing source vertices whose XYZ matches the cut mesh."""
    src = read_ascii_ply(source_ply)
    cut = read_ascii_ply(cut_ply)

    src_vertices = np.asarray(src["vertices"], dtype=np.float64)
    cut_vertices = np.asarray(cut["vertices"], dtype=np.float64)
    src_faces = src["faces"]
    src_vertex_props = src["vertex_props"]
    src_face_prop_line = str(src["face_prop_line"])

    src_xyz32 = src_vertices[:, :3].astype(np.float32)
    cut_xyz32 = cut_vertices[:, :3].astype(np.float32)

    cut_set = set(map(tuple, cut_xyz32.tolist()))
    remove_mask = np.array([tuple(point) in cut_set for point in src_xyz32], dtype=bool)
    keep_mask = ~remove_mask
    if not np.any(keep_mask):
        raise SystemExit("Vertex-overlap subtraction removed every vertex from the source mesh.")

    kept_vertices = src_vertices[keep_mask]

    old_to_new = np.full(src_vertices.shape[0], -1, dtype=np.int64)
    kept_old_ids = np.flatnonzero(keep_mask)
    old_to_new[kept_old_ids] = np.arange(kept_old_ids.size, dtype=np.int64)

    kept_faces: list[list[int]] = []
    dropped_faces = 0
    for face in src_faces:
        face_arr = np.asarray(face, dtype=np.int64)
        if np.all(keep_mask[face_arr]):
            kept_faces.append(old_to_new[face_arr].tolist())
        else:
            dropped_faces += 1

    if not kept_faces:
        raise SystemExit("Vertex-overlap subtraction removed every face from the source mesh.")

    write_ascii_ply(
        output_ply,
        vertex_props=src_vertex_props,
        vertices=kept_vertices,
        faces=kept_faces,
        face_prop_line=src_face_prop_line,
    )

    print(
        f"PLY overlap subtraction removed {int(remove_mask.sum())} of {src_vertices.shape[0]} source vertices "
        f"and dropped {dropped_faces} of {len(src_faces)} faces."
    )


def _load_dual_plane_quality_module():
    """Import the dual-plane quality utility from the local Utilities directory."""
    utilities_dir = Path(__file__).resolve().parent / "Utilities"
    utilities_path = str(utilities_dir)
    if utilities_path not in sys.path:
        sys.path.insert(0, utilities_path)

    import dualPlaneQuality  # type: ignore

    return dualPlaneQuality


def save_ply_with_dual_named_quality(
    path: str | Path,
    verts: np.ndarray,
    faces: np.ndarray,
    quality_normal: np.ndarray,
    quality_transverse: np.ndarray,
) -> None:
    """Write an ASCII PLY with only the normal/transverse quality arrays."""
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    quality_normal = np.asarray(quality_normal, dtype=np.float64).reshape(-1)
    quality_transverse = np.asarray(quality_transverse, dtype=np.float64).reshape(-1)

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("verts must be (N, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be (F, 3) triangle indices.")
    if quality_normal.shape[0] != verts.shape[0]:
        raise ValueError("quality_normal must have length N.")
    if quality_transverse.shape[0] != verts.shape[0]:
        raise ValueError("quality_transverse must have length N.")

    with Path(path).open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {verts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float quality_normal\n")
        f.write("property float quality_transverse\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for (x, y, z), q_normal, q_trans in zip(
            verts,
            quality_normal,
            quality_transverse,
        ):
            f.write(f"{x:.9g} {y:.9g} {z:.9g} {q_normal:.9g} {q_trans:.9g}\n")

        for i, j, k in faces:
            f.write(f"3 {int(i)} {int(j)} {int(k)}\n")


def load_ply_with_dual_named_quality(path: str | Path) -> pv.PolyData:
    """Read a PLY and preserve the normal/transverse quality vertex properties."""
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"{path} is missing a vertex element.")
    if "face" not in ply:
        raise ValueError(f"{path} is missing a face element.")

    vertices = ply["vertex"].data
    for name in ("x", "y", "z", "quality_normal", "quality_transverse"):
        if name not in vertices.dtype.names:
            raise KeyError(f"Required vertex property {name!r} was not found in {path}.")

    points = np.column_stack(
        [
            np.asarray(vertices["x"], dtype=np.float64),
            np.asarray(vertices["y"], dtype=np.float64),
            np.asarray(vertices["z"], dtype=np.float64),
        ]
    )

    face_rows = []
    for face in ply["face"].data["vertex_indices"]:
        face = np.asarray(face, dtype=np.int64)
        if face.shape != (3,):
            raise ValueError(f"{path} contains a non-triangular face; expected only triangles.")
        face_rows.append(face)

    if not face_rows:
        raise ValueError(f"{path} contains no faces.")

    faces = np.asarray(face_rows, dtype=np.int64)
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
    mesh = pv.PolyData(points, faces_pv)
    mesh.point_data["quality_normal"] = np.asarray(vertices["quality_normal"], dtype=np.float64)
    mesh.point_data["quality_transverse"] = np.asarray(vertices["quality_transverse"], dtype=np.float64)
    return mesh


def enrich_mesh_with_dual_plane_quality(path: str | Path) -> None:
    """Overwrite a mesh PLY with normal/transverse quality arrays only."""
    dual_plane_quality = _load_dual_plane_quality_module()
    mesh = load_surface_mesh(str(path))

    ctx = dual_plane_quality.compute_geometry_context(mesh, ref_mode="mesh_center", ep=None)
    args = argparse.Namespace(
        ref="mesh_center",
        plane_scale=0.6,
        ep=None,
        distance_mode="geodesic",
        diffusion_k=60,
        diffusion_t=0.01,
        geodesic_backend="auto",
        export_ply=None,
        export_scalar=None,
        signed_under_plane=True,
        show_band=False,
        show_edges=False,
    )

    results = {
        "normal": dual_plane_quality.compute_plane_quality(ctx, args, "normal", None),
        "transverse": dual_plane_quality.compute_plane_quality(ctx, args, "transverse", None),
    }

    for result in results.values():
        dual_plane_quality.attach_plane_quality(ctx.mesh, result)

    faces_raw = np.asarray(ctx.mesh.faces, dtype=np.int64).reshape(-1, 4)
    if faces_raw.size == 0:
        raise SystemExit("tmp/tmp1.ply has no faces after dual-plane quality enrichment.")

    save_ply_with_dual_named_quality(
        path,
        np.asarray(ctx.mesh.points, dtype=np.float64),
        faces_raw[:, 1:4],
        np.asarray(ctx.mesh.point_data["quality_normal"], dtype=np.float64),
        np.asarray(ctx.mesh.point_data["quality_transverse"], dtype=np.float64),
    )


def save_masked_dual_quality_surface(
    input_path: str | Path,
    output_path: str | Path,
    *,
    normal_lower_bound: float,
    normal_upper_bound: float,
    transverse_lower_bound: float,
    transverse_upper_bound: float,
) -> pv.PolyData:
    """
    Keep only cells whose vertices stay within the configured quality ranges.
    """
    if normal_lower_bound > normal_upper_bound:
        raise ValueError("normal_lower_bound cannot be greater than normal_upper_bound.")
    if transverse_lower_bound > transverse_upper_bound:
        raise ValueError("transverse_lower_bound cannot be greater than transverse_upper_bound.")

    mesh = load_ply_with_dual_named_quality(input_path).triangulate().clean(tolerance=0.0)

    qn = np.asarray(mesh.point_data["quality_normal"], dtype=np.float64)
    qt = np.asarray(mesh.point_data["quality_transverse"], dtype=np.float64)
    normal_keep = (qn >= normal_lower_bound) & (qn <= normal_upper_bound)
    transverse_keep = (qt >= transverse_lower_bound) & (qt <= transverse_upper_bound)
    keep_mask = normal_keep & transverse_keep
    if not np.any(keep_mask):
        raise SystemExit(
            "No vertices satisfied the dual-quality mask within "
            f"normal=[{normal_lower_bound:.3f}, {normal_upper_bound:.3f}] and "
            f"transverse=[{transverse_lower_bound:.3f}, {transverse_upper_bound:.3f}]."
        )

    mesh.point_data["keep"] = keep_mask.astype(np.uint8)
    masked_vol = mesh.threshold((0.5, 1.5), scalars="keep", all_scalars=True)
    masked_mesh = masked_vol.extract_surface().triangulate().clean(tolerance=0.0)

    if masked_mesh.n_points == 0 or masked_mesh.n_cells == 0:
        raise SystemExit("Dual-quality masking removed all surface cells; cannot continue.")

    log_pyvista_mesh(
        masked_mesh,
        "masked dual-quality surface",
        required_arrays=("quality_normal", "quality_transverse"),
        require_faces=True,
    )
    print(
        "Dual-quality mask kept "
        f"{masked_mesh.n_points} vertices and {masked_mesh.n_cells} faces "
        f"within normal=[{normal_lower_bound:.3f}, {normal_upper_bound:.3f}] "
        f"and transverse=[{transverse_lower_bound:.3f}, {transverse_upper_bound:.3f}]."
    )
    print(
        "Mask failures: "
        f"normal={int(np.sum(~normal_keep))}, "
        f"transverse={int(np.sum(~transverse_keep))}, "
        f"combined={int(np.sum(~keep_mask))}."
    )

    faces_raw = np.asarray(masked_mesh.faces, dtype=np.int64).reshape(-1, 4)
    if faces_raw.size == 0:
        raise SystemExit("Masked mesh has no faces after thresholding; cannot export.")

    save_ply_with_dual_named_quality(
        output_path,
        np.asarray(masked_mesh.points, dtype=np.float64),
        faces_raw[:, 1:4],
        np.asarray(masked_mesh.point_data["quality_normal"], dtype=np.float64),
        np.asarray(masked_mesh.point_data["quality_transverse"], dtype=np.float64),
    )
    return masked_mesh


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


def import_ply_object(filepath: str, object_name: str | None = None):
    if object_name is not None:
        remove_blender_object_if_exists(object_name)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.wm.ply_import(filepath=filepath)
    imported = list(bpy.context.selected_objects)
    if not imported:
        raise RuntimeError(f"Blender did not import any object from {filepath}")

    obj = imported[0]
    if object_name is not None:
        obj.name = object_name
        if obj.data is not None:
            obj.data.name = object_name
    return obj


def create_blender_mesh_object(
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
):
    remove_blender_object_if_exists(name)

    me = bpy.data.meshes.new(name)
    bm = bmesh.new()

    for vertex in np.asarray(vertices, dtype=np.float64):
        bm.verts.new(vertex)
    bm.verts.ensure_lookup_table()

    for tri in np.asarray(faces, dtype=np.int64):
        try:
            bm.faces.new([bm.verts[int(i)] for i in tri])
        except ValueError:
            pass

    bm.to_mesh(me)
    bm.free()

    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    select_only_blender_object(obj)
    return obj


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
    
    argv = sys.argv
        
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:]

    args = ap.parse_args(argv)
    
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
    
    pipeline_paths = build_pipeline_paths("tmp")

    stage_banner("Landmark Picking")
    res, _surf = pick_two_points_and_geodesic(
        inputfile,
        left_click=not args.right_click,
        auto_close=args.auto_close,
        picker=args.picker,
    )
    print(f"P0 (vertex {res.v0}): {res.p0.tolist()}")
    print(f"P1 (vertex {res.v1}): {res.p1.tolist()}")
    print(f"Geodesic distance: {res.geodesic_distance:.10g}")
    print("(Units match the input mesh coordinate units.)")

    stage_banner("Scale Calibration")
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
    print(f"Scale factor = real / mesh = {real_mm:.3f} / {geo:.3f} = {scale:.6f}")

    stage_banner("PyMeshLab Cleanup and Reconstruction")
    ms = ml.MeshSet()
    ms.load_new_mesh(inputfile)
    log_current_pymeshlab_mesh(ms, "loaded input mesh", require_faces=True)

    ms.compute_selection_by_small_disconnected_components_per_face()
    ms.meshing_remove_selected_faces()
    ms.meshing_remove_selected_vertices()
    mesh_clean(ms)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=50000, qualitythr=0.300000)
    mesh_clean(ms)
    ms.compute_matrix_from_scaling_or_normalization(axisx=scale, axisy=scale, axisz=scale)
    log_current_pymeshlab_mesh(ms, "cleaned and scaled input mesh", require_faces=True)

    if args.scaled_polydata_out:
        scaled_polydata_out = Path(args.scaled_polydata_out)
    else:
        in_path = Path(inputfile)
        scaled_polydata_out = in_path.with_name(f"{in_path.stem}_scaled_polydata.ply")
    scaled_polydata_out = scaled_polydata_out.resolve()
    print(f"[scaled-polydata] target path: {scaled_polydata_out}")

    ms.generate_surface_reconstruction_vcg(voxsize=ml.PercentageValue(0.50))
    reconstructed_surface_id = ms.current_mesh_id()
    set_current_pymeshlab_mesh(ms, reconstructed_surface_id, "reconstructed surface", require_faces=True)

    ms.meshing_surface_subdivision_loop(threshold=ml.PercentageValue(0.50))
    subdivided_surface_id = ms.current_mesh_id()
    set_current_pymeshlab_mesh(ms, subdivided_surface_id, "subdivided surface", require_faces=True)

    print("Applying legacy pre-mask crop before exporting tmp1.ply.")
    ms.compute_selection_by_condition_per_face(condselect=PRE_MASK_CROP_CONDITION)
    ms.meshing_remove_selected_vertices_and_faces()
    pre_mask_surface_id = ms.current_mesh_id()
    set_current_pymeshlab_mesh(ms, pre_mask_surface_id, "legacy pre-mask crop result", require_faces=True)

    stage_banner("Dual-Quality Enrichment and Masking")
    ms.save_current_mesh(str(pipeline_paths.tmp1))
    log_ply_summary(pipeline_paths.tmp1, "tmp1 before enrichment", require_faces=True)

    enrich_mesh_with_dual_plane_quality(pipeline_paths.tmp1)
    log_ply_summary(
        pipeline_paths.tmp1,
        "tmp1 after enrichment",
        required_properties=("quality_normal", "quality_transverse"),
        require_faces=True,
    )

    save_masked_dual_quality_surface(
        pipeline_paths.tmp1,
        pipeline_paths.masked_output,
        normal_lower_bound=QUALITY_NORMAL_LOWER_BOUND,
        normal_upper_bound=QUALITY_NORMAL_UPPER_BOUND,
        transverse_lower_bound=QUALITY_TRANSVERSE_LOWER_BOUND,
        transverse_upper_bound=QUALITY_TRANSVERSE_UPPER_BOUND,
    )
    log_ply_summary(
        pipeline_paths.masked_output,
        "masked output",
        required_properties=("quality_normal", "quality_transverse"),
        require_faces=True,
    )

    ms.load_new_mesh(str(pipeline_paths.masked_output))
    masked_surface_id = ms.current_mesh_id()
    set_current_pymeshlab_mesh(ms, masked_surface_id, "masked surface loaded into PyMeshLab", require_faces=True)

    stage_banner("Poisson Sampling and Voronoi Projection")
    set_current_pymeshlab_mesh(ms, masked_surface_id, "masked surface before Poisson sampling", require_faces=True)
    ms.generate_sampling_poisson_disk(samplenum=POINT_CLOUD_SAMPLE_COUNT, exactnumflag=True)
    pointcloud_id = ms.current_mesh_id()
    log_current_pymeshlab_mesh(ms, "Poisson point cloud", require_faces=False)

    set_current_pymeshlab_mesh(ms, masked_surface_id, "masked surface before Voronoi projection", require_faces=True)
    ms.compute_color_by_point_cloud_voronoi_projection(
        coloredmesh=masked_surface_id,
        vertexmesh=pointcloud_id,
        backward=True,
    )
    log_current_pymeshlab_mesh(ms, "masked surface after Voronoi projection", require_faces=True)

    stage_banner("PyVista Color Selection")
    csurface = ms.current_mesh()
    cvertices = csurface.vertex_matrix()
    cfaces = csurface.face_matrix()
    colors = csurface.vertex_color_matrix()
    cfaces_pv = np.hstack(
        [np.full((cfaces.shape[0], 1), 3, dtype=np.int64), cfaces]
    ).ravel()
    cmesh = pv.PolyData(cvertices, cfaces_pv)
    cmesh.point_data["RGBA"] = colors
    log_pyvista_mesh(cmesh, "PyVista projected surface", required_arrays=("RGBA",), require_faces=True)
    cmesh.plot(rgb=True)

    rgba = np.asarray(cmesh.point_data["RGBA"])
    if rgba.ndim == 1:
        rgba = rgba.reshape(cmesh.n_points, 4)
    rgb = np.clip((rgba[:, :3] * 255.0).round(), 0, 255).astype(np.uint8)
    cmesh.point_data.pop("RGBA")
    cmesh.point_data["RGB"] = rgb

    rgb_norm = cmesh["RGB"].astype(float) / 255.0
    hsv = rgb_to_hsv(rgb_norm)
    hue = hsv[:, 0]
    saturation = hsv[:, 1]

    red_like = (hue <= PYVISTA_DELETE_HUE_MAX) & (saturation >= PYVISTA_DELETE_SATURATION_MIN)
    selected_mask = ~red_like
    cmesh.point_data["selection_mask"] = selected_mask.astype(np.uint8)
    print(
        f"PyVista HSV selection inverted the red-like region: kept {int(np.sum(selected_mask))} vertices "
        f"and excluded {int(np.sum(red_like))} red-like vertices."
    )

    selected_vol = cmesh.threshold((0.5, 1.5), scalars="selection_mask", all_scalars=True)
    selected_surface = selected_vol.extract_surface().triangulate().clean(tolerance=0.0)
    log_pyvista_mesh(selected_surface, "PyVista cut surface", require_faces=True)

    selected_surface.save(str(pipeline_paths.pyvista_cut_surface), binary=False)
    log_ply_summary(pipeline_paths.pyvista_cut_surface, "pyvista cut surface", require_faces=True)

    stage_banner("PLY Overlap Subtraction")
    target_path = pipeline_paths.tmp1.resolve()
    cutter_path = pipeline_paths.pyvista_cut_surface.resolve()
    print(
        f"Overlap subtraction contract: source={target_path.name}, cut={cutter_path.name}, "
        f"output={pipeline_paths.boolean_result.name}"
    )

    subtract_ply_by_vertex_overlap(target_path, cutter_path, pipeline_paths.boolean_result)
    ensure_file_exists(pipeline_paths.boolean_result, "post-subtraction export")
    log_ply_summary(pipeline_paths.boolean_result, "post-subtraction export", require_faces=True)

    stage_banner("Post-Subtraction Remesh")
    ms.load_new_mesh(str(pipeline_paths.boolean_result))
    log_current_pymeshlab_mesh(ms, "post-subtraction mesh loaded into PyMeshLab", require_faces=True)
    mesh_clean(ms)
    ms.meshing_close_holes(maxholesize=50)
    ms.compute_selection_by_small_disconnected_components_per_face()
    ms.meshing_remove_selected_faces()
    ms.meshing_remove_selected_vertices()
    log_current_pymeshlab_mesh(ms, "post-subtraction mesh after hole close/component removal", require_faces=True)
    ms.meshing_isotropic_explicit_remeshing(
        iterations=10,
        adaptive=True,
        checksurfdist=True,
        targetlen=ml.PercentageValue(0.250),
    )
    ms.apply_coord_laplacian_smoothing(stepsmoothnum=50)
    smooth_mesh = log_current_pymeshlab_mesh(ms, "smoothed post-subtraction mesh", require_faces=True)

    stage_banner("Blender Thickening and Final Export")
    smooth_vertices = smooth_mesh.vertex_matrix()
    smooth_faces = smooth_mesh.face_matrix()
    final_obj = create_blender_mesh_object("pymlMesh", smooth_vertices, smooth_faces)

    displace = final_obj.modifiers.new(name="Displace", type="DISPLACE")
    displace.strength = 1.0
    displace.mid_level = 0.5
    displace.direction = "NORMAL"
    displace.space = "LOCAL"

    solid = final_obj.modifiers.new(name="Solidify", type="SOLIDIFY")
    solid.thickness = 2.5
    solid.offset = 1.0
    solid.use_even_offset = False

    select_only_blender_object(final_obj)
    bpy.ops.object.modifier_apply(modifier=displace.name)
    bpy.ops.object.modifier_apply(modifier=solid.name)
    print("Applied Blender displace and solidify modifiers to the final mesh.")

    export_blender_object(final_obj, outputfile)
    ensure_file_exists(outputfile, "final output")
    print(f"Final mesh exported to: {Path(outputfile).resolve()}")
    if Path(outputfile).suffix.lower() == ".ply":
        log_ply_summary(outputfile, "final output", require_faces=True)
    else:
        log_surface_file(outputfile, "final output surface")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
