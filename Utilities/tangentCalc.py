#!/usr/bin/env python3
"""
visualize_face_tangents.py

Loads a .ply triangle mesh, computes ONE tangent direction per face, and visualizes
the mesh with a tangent line drawn from each face centroid.

Note on "face tangents":
- If the mesh has UVs, a true tangent space (T,B,N) depends on the UV parameterization.
- A .ply often has no UVs, so this script defines a geometric face tangent as:
    tangent = normalized(first non-degenerate edge of the triangle) in the face plane.

Dependencies:
  pip install pyvista numpy

Usage:
  python visualize_face_tangents.py -i model.ply
  python visualize_face_tangents.py -i model.ply --max-vectors 5000 --tangent-scale 1.0
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pyvista as pv


def _to_polydata(mesh) -> pv.PolyData:
    """Read result -> single triangulated pv.PolyData."""
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    mesh = mesh.clean(tolerance=0.0)
    mesh = mesh.triangulate()
    return mesh


def _compute_face_tangent_segments(
    poly: pv.PolyData,
    max_vectors: int = 20000,
    seed: int = 0,
    tangent_scale: float = 1.0,
) -> tuple[pv.PolyData, int, float]:
    """
    Returns:
      lines_polydata: PolyData containing line segments (centroid -> centroid + L*tangent)
      used_faces: how many faces were used (possibly sampled)
      L: tangent line length used
    """
    faces = poly.faces
    if faces.size == 0:
        raise ValueError("Mesh has no faces.")

    # Faces array is [3, i0,i1,i2, 3, j0,j1,j2, ...] after triangulate().
    f = faces.reshape(-1, 4)
    if not np.all(f[:, 0] == 3):
        raise ValueError("Expected triangles after triangulation.")

    tri = f[:, 1:4]
    n_faces = tri.shape[0]

    # Sample faces if too many (visualization can become heavy).
    rng = np.random.default_rng(seed)
    if n_faces > max_vectors:
        idx = rng.choice(n_faces, size=max_vectors, replace=False)
        tri = tri[idx]
        used_faces = max_vectors
    else:
        used_faces = n_faces

    V = poly.points.astype(np.float64)
    v0 = V[tri[:, 0]]
    v1 = V[tri[:, 1]]
    v2 = V[tri[:, 2]]

    # Face centroids
    c = (v0 + v1 + v2) / 3.0

    # Geometric "tangent": pick the first non-degenerate edge direction
    e1 = v1 - v0
    e2 = v2 - v0

    e1_norm = np.linalg.norm(e1, axis=1)
    e2_norm = np.linalg.norm(e2, axis=1)

    eps = 1e-12
    use_e1 = e1_norm > eps
    t = np.where(use_e1[:, None], e1, e2)

    t_norm = np.linalg.norm(t, axis=1)
    valid = t_norm > eps
    c = c[valid]
    t = t[valid] / t_norm[valid, None]
    used_faces = int(valid.sum())

    # Choose a tangent length relative to model size and face count.
    # More faces -> shorter vectors by default.
    bounds = np.array(poly.bounds, dtype=np.float64)  # [xmin,xmax,ymin,ymax,zmin,zmax]
    diag = np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
    if not np.isfinite(diag) or diag <= 0:
        diag = 1.0

    # Base length ~ 5% of bounding-box diagonal for small meshes,
    # scaled down for dense meshes.
    density_factor = (1000.0 / max(used_faces, 1)) ** (1.0 / 3.0)
    L = diag * 0.05 * np.clip(density_factor, 0.15, 2.0) * float(tangent_scale)

    start = c
    end = c + L * t

    # Build a PolyData of line segments.
    pts = np.empty((2 * used_faces, 3), dtype=np.float64)
    pts[0::2] = start
    pts[1::2] = end

    # VTK "lines" format: [2, p0, p1, 2, p2, p3, ...]
    lines = np.empty((used_faces, 3), dtype=np.int64)
    lines[:, 0] = 2
    lines[:, 1] = np.arange(0, 2 * used_faces, 2, dtype=np.int64)
    lines[:, 2] = lines[:, 1] + 1
    lines = lines.reshape(-1)

    lines_pd = pv.PolyData(pts, lines=lines)
    return lines_pd, used_faces, float(L)


def _auto_line_width(n_vectors: int) -> float:
    """
    Pixel line width heuristic:
      - small n -> thicker
      - large n -> thin
    """
    # log-scale mapping; clamp to [1, 8]
    w = 8.0 - 2.0 * math.log10(max(n_vectors, 1))
    return float(np.clip(w, 1.0, 8.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Path to input .ply")
    ap.add_argument("--max-vectors", type=int, default=20000, help="Max face tangents to draw (sampling if needed)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for face sampling")
    ap.add_argument("--tangent-scale", type=float, default=1.0, help="Multiply tangent line length by this factor")
    ap.add_argument("--mesh-opacity", type=float, default=1.0, help="Mesh opacity (0..1)")
    ap.add_argument("--show-edges", action="store_true", help="Show mesh edges")
    ap.add_argument("--bg", default="white", help="Background color name (e.g., 'white', 'black')")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    mesh_raw = pv.read(str(path))
    mesh = _to_polydata(mesh_raw)

    tangents_pd, used, L = _compute_face_tangent_segments(
        mesh,
        max_vectors=args.max_vectors,
        seed=args.seed,
        tangent_scale=args.tangent_scale,
    )
    lw = _auto_line_width(used)

    print(f"Mesh: {mesh.n_points} points, {mesh.n_cells} faces")
    print(f"Drawing tangents for {used} faces (max_vectors={args.max_vectors}), length={L:.6g}, line_width={lw:.2f}px")

    p = pv.Plotter()
    p.set_background(args.bg)
    p.add_mesh(mesh, color="lightgray", opacity=args.mesh_opacity, show_edges=args.show_edges)
    p.add_mesh(tangents_pd, color="red", line_width=lw)
    p.add_text("Face tangents (geometric edge tangents)", font_size=12)
    p.show()


if __name__ == "__main__":
    main()