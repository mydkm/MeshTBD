#!/usr/bin/env python3
"""
plane_from_centroid_colormap.py

Pipeline:
1) Load a .ply triangle mesh (ideally closed + consistently oriented).
2) Compute volume centroid c using signed tetrahedra.
3) Find closest surface point p to c (VTK cell locator).
4) Define arrow direction n = (p - c)/||p - c||.
5) Build plane through c (choose with --plane-mode):
   - normal:       plane normal = n                 (plane ⟂ arrow)
   - longitudinal: plane contains n and axis         (plane normal = normalize(n × u_perp))
   - transverse:   plane contains n, and is ~⊥ axis  (plane normal = normalize(proj_{n^⊥}(u0)))

6) Plane band selection:
   With plane Ax+By+Cz+D=0 and ||(A,B,C)||=1, select vertices with abs(Ax+By+Cz+D) <= ep.

7) Color by distance-to-band using --distance-mode:
   - euclidean:  min Euclidean distance in R^3 to band vertices
   - diffusion:  diffusion distance via eigen-embedding of cotangent Laplacian (SciPy required)
   - geodesic:   multi-source shortest-path distance along mesh edges (SciPy optional)

8) Optional signed coloring with --signed-under-plane:
   - distances on the side where (Ax+By+Cz+D)<0 are negated
   - uses dist_signed_norm in [-1,1] for visualization

Examples:
  python plane_from_centroid_colormap.py -i input.ply
  python plane_from_centroid_colormap.py -i input.ply --distance-mode geodesic --signed-under-plane
  python plane_from_centroid_colormap.py -i input.ply --distance-mode diffusion --diffusion-k 80 --diffusion-t 0.02 --signed-under-plane
"""

from __future__ import annotations

import argparse
import heapq
import numpy as np
import pyvista as pv
import vtk


# ----------------------------
# Utilities
# ----------------------------

def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / n


def mesh_diag(mesh: pv.PolyData) -> float:
    b = np.array(mesh.bounds, dtype=np.float64)
    return float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))


def warn_if_not_watertight(mesh: pv.PolyData) -> None:
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    if edges.n_cells > 0:
        print(f"[WARN] Mesh appears to have boundary edges (count={edges.n_cells}). "
              f"Volume/centroid from signed tetrahedra may be invalid.")


def ensure_polydata(mesh: pv.DataSet) -> pv.PolyData:
    if isinstance(mesh, pv.PolyData):
        return mesh
    return mesh.extract_surface()


def normalize_01(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if abs(vmax - vmin) < eps:
        return np.zeros_like(values, dtype=np.float64)
    return (values - vmin) / (vmax - vmin)


# ----------------------------
# Step 1: signed-tetra volume centroid
# ----------------------------

def compute_volume_and_centroid_signed_tets(mesh: pv.PolyData, ref: np.ndarray) -> tuple[float, np.ndarray]:
    mesh = mesh.triangulate()

    pts = np.asarray(mesh.points, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)
    if faces.size == 0:
        raise ValueError("Mesh has no faces.")
    if not np.all(faces[:, 0] == 3):
        raise ValueError("Mesh is not purely triangles (unexpected after triangulate()).")

    tri = faces[:, 1:4]
    v0 = pts[tri[:, 0]] - ref
    v1 = pts[tri[:, 1]] - ref
    v2 = pts[tri[:, 2]] - ref

    Vi = np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0
    V = float(Vi.sum())
    if np.isclose(V, 0.0):
        raise ValueError("Signed volume is ~0. Mesh may be open or inconsistently oriented.")

    ci = ref + (v0 + v1 + v2) / 4.0
    c = (Vi[:, None] * ci).sum(axis=0) / V
    return V, c


# ----------------------------
# Step 2: closest point to centroid
# ----------------------------

def closest_point_on_surface_vtk(mesh: pv.PolyData, q: np.ndarray) -> tuple[np.ndarray, int, float]:
    mesh = mesh.triangulate()

    LocatorCls = vtk.vtkStaticCellLocator if hasattr(vtk, "vtkStaticCellLocator") else vtk.vtkCellLocator
    locator = LocatorCls()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    closest = [0.0, 0.0, 0.0]
    cell_id = vtk.mutable(0)
    sub_id = vtk.mutable(0)
    dist2 = vtk.mutable(0.0)

    locator.FindClosestPoint(q.tolist(), closest, cell_id, sub_id, dist2)
    return np.array(closest, dtype=np.float64), int(cell_id), float(dist2)


# ----------------------------
# PCA axis
# ----------------------------

def pca_axes(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    X = points - center
    C = (X.T @ X) / max(1, X.shape[0])
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    return V[:, order]


def pick_nonparallel_axis(n: np.ndarray) -> np.ndarray:
    candidates = [
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    ]
    best = candidates[0]
    best_norm = -1.0
    for a in candidates:
        a_perp = a - np.dot(a, n) * n
        an = float(np.linalg.norm(a_perp))
        if an > best_norm:
            best_norm = an
            best = a
    return best


# ----------------------------
# Plane patch for visualization
# ----------------------------

def make_plane_patch(center: np.ndarray, normal: np.ndarray, mesh: pv.PolyData, scale: float) -> pv.PolyData:
    diag = mesh_diag(mesh)
    size = max(1e-6, scale * (diag if diag > 0 else 1.0))
    return pv.Plane(center=center, direction=normal, i_size=size, j_size=size, i_resolution=1, j_resolution=1)


# ----------------------------
# Distance-to-band: Euclidean
# ----------------------------

def compute_euclidean_distance_to_band(points: np.ndarray, band_points: np.ndarray) -> np.ndarray:
    if band_points.shape[0] == 0:
        raise ValueError("Band point set is empty.")

    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(band_points)
        dists, _ = tree.query(points, k=1, workers=-1)
        return np.asarray(dists, dtype=np.float64)
    except Exception:
        pass

    # VTK fallback
    band_poly = pv.PolyData(band_points)
    LocatorCls = vtk.vtkStaticPointLocator if hasattr(vtk, "vtkStaticPointLocator") else vtk.vtkPointLocator
    locator = LocatorCls()
    locator.SetDataSet(band_poly)
    locator.BuildLocator()

    d = np.empty(points.shape[0], dtype=np.float64)
    for i, pt in enumerate(points):
        j = locator.FindClosestPoint(pt.tolist())
        d[i] = float(np.linalg.norm(pt - band_points[int(j)]))
    return d


# ----------------------------
# Distance-to-band: Geodesic (graph shortest path)
# ----------------------------

def unique_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]
    edges = np.vstack([e01, e12, e20])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges


def build_adjacency_lists(verts: np.ndarray, faces: np.ndarray) -> list[list[tuple[int, float]]]:
    n = verts.shape[0]
    edges = unique_edges_from_faces(faces)
    i = edges[:, 0]
    j = edges[:, 1]
    w = np.linalg.norm(verts[i] - verts[j], axis=1)

    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for a, b, ww in zip(i.tolist(), j.tolist(), w.tolist()):
        adj[a].append((b, ww))
        adj[b].append((a, ww))
    return adj


def multi_source_dijkstra_python(adj: list[list[tuple[int, float]]], sources: np.ndarray) -> np.ndarray:
    n = len(adj)
    dist = np.full(n, np.inf, dtype=np.float64)
    heap: list[tuple[float, int]] = []

    for s in sources.tolist():
        s = int(s)
        dist[s] = 0.0
        heapq.heappush(heap, (0.0, s))

    while heap:
        d_u, u = heapq.heappop(heap)
        if d_u != dist[u]:
            continue
        for v, w_uv in adj[u]:
            nd = d_u + w_uv
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def geodesic_distance_to_band(verts: np.ndarray, faces: np.ndarray, band_idx: np.ndarray, backend: str = "auto") -> np.ndarray:
    if band_idx.size == 0:
        raise ValueError("Band index set is empty.")
    if backend not in {"auto", "scipy", "python"}:
        raise ValueError("backend must be one of {'auto','scipy','python'}")

    if backend in {"auto", "scipy"}:
        try:
            import scipy.sparse as sp  # type: ignore
            from scipy.sparse.csgraph import dijkstra  # type: ignore

            n = verts.shape[0]
            edges = unique_edges_from_faces(faces)
            i = edges[:, 0]
            j = edges[:, 1]
            w = np.linalg.norm(verts[i] - verts[j], axis=1)

            rows = np.concatenate([i, j])
            cols = np.concatenate([j, i])
            data = np.concatenate([w, w])
            G = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

            super_node = n
            rows2 = np.concatenate([rows, band_idx, np.full(band_idx.shape[0], super_node, dtype=np.int64)])
            cols2 = np.concatenate([cols, np.full(band_idx.shape[0], super_node, dtype=np.int64), band_idx])
            data2 = np.concatenate([data, np.zeros(band_idx.shape[0]), np.zeros(band_idx.shape[0])])

            G2 = sp.coo_matrix((data2, (rows2, cols2)), shape=(n + 1, n + 1)).tocsr()
            dist_all = dijkstra(G2, directed=False, indices=super_node)
            return np.asarray(dist_all[:n], dtype=np.float64)

        except Exception as e:
            if backend == "scipy":
                raise RuntimeError(
                    "Geodesic backend 'scipy' requested but SciPy is unavailable or failed. "
                    "Install SciPy or use --geodesic-backend python."
                ) from e

    adj = build_adjacency_lists(verts, faces)
    return multi_source_dijkstra_python(adj, band_idx)


# ----------------------------
# Diffusion distance (SciPy required)
# ----------------------------

def cotangent_laplacian_and_mass(verts: np.ndarray, faces: np.ndarray, eps: float = 1e-18):
    try:
        import scipy.sparse as sp  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Diffusion distance requires SciPy (scipy.sparse). "
            "Install it or use --distance-mode euclidean/geodesic."
        ) from e

    n = verts.shape[0]
    f = faces

    v0 = verts[f[:, 0]]
    v1 = verts[f[:, 1]]
    v2 = verts[f[:, 2]]

    cross01_02 = np.cross(v1 - v0, v2 - v0)
    dblA = np.maximum(np.linalg.norm(cross01_02, axis=1), eps)
    area = 0.5 * dblA

    cot0 = np.einsum("ij,ij->i", (v1 - v0), (v2 - v0)) / dblA

    cross21_01 = np.cross(v2 - v1, v0 - v1)
    dblA1 = np.maximum(np.linalg.norm(cross21_01, axis=1), eps)
    cot1 = np.einsum("ij,ij->i", (v2 - v1), (v0 - v1)) / dblA1

    cross02_12 = np.cross(v0 - v2, v1 - v2)
    dblA2 = np.maximum(np.linalg.norm(cross02_12, axis=1), eps)
    cot2 = np.einsum("ij,ij->i", (v0 - v2), (v1 - v2)) / dblA2

    m = np.zeros(n, dtype=np.float64)
    np.add.at(m, f[:, 0], area / 3.0)
    np.add.at(m, f[:, 1], area / 3.0)
    np.add.at(m, f[:, 2], area / 3.0)
    m = np.maximum(m, eps)

    i12, j12, w12 = f[:, 1], f[:, 2], 0.5 * cot0
    i20, j20, w20 = f[:, 2], f[:, 0], 0.5 * cot1
    i01, j01, w01 = f[:, 0], f[:, 1], 0.5 * cot2

    def assemble_edge(i, j, w):
        rows_off = np.concatenate([i, j])
        cols_off = np.concatenate([j, i])
        data_off = np.concatenate([-w, -w])
        rows_diag = np.concatenate([i, j])
        cols_diag = np.concatenate([i, j])
        data_diag = np.concatenate([w, w])
        return rows_off, cols_off, data_off, rows_diag, cols_diag, data_diag

    r1, c1, d1, rd1, cd1, dd1 = assemble_edge(i12, j12, w12)
    r2, c2, d2, rd2, cd2, dd2 = assemble_edge(i20, j20, w20)
    r3, c3, d3, rd3, cd3, dd3 = assemble_edge(i01, j01, w01)

    rows = np.concatenate([r1, r2, r3, rd1, rd2, rd3])
    cols = np.concatenate([c1, c2, c3, cd1, cd2, cd3])
    data = np.concatenate([d1, d2, d3, dd1, dd2, dd3])

    import scipy.sparse as sp  # type: ignore
    L = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    L = 0.5 * (L + L.T)
    M = sp.diags(m, 0, shape=(n, n), format="csr")
    return L, M


def diffusion_embedding(verts: np.ndarray, faces: np.ndarray, k: int, t: float):
    try:
        from scipy.sparse.linalg import eigsh  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Diffusion distance requires SciPy (scipy.sparse.linalg.eigsh). "
            "Install it or use --distance-mode euclidean/geodesic."
        ) from e

    L, M = cotangent_laplacian_and_mass(verts, faces)

    k_req = int(max(2, k + 1))
    try:
        evals, evecs = eigsh(L, k=k_req, M=M, sigma=0.0, which="LM")
    except Exception:
        evals, evecs = eigsh(L, k=k_req, M=M, which="SM")

    order = np.argsort(evals)
    evals = np.asarray(evals[order], dtype=np.float64)
    evecs = np.asarray(evecs[:, order], dtype=np.float64)

    evals_nt = evals[1:]
    evecs_nt = evecs[:, 1:]

    k_use = min(k, evals_nt.shape[0])
    evals_use = evals_nt[:k_use]
    evecs_use = evecs_nt[:, :k_use]

    scales = np.exp(-evals_use * float(t))
    emb = evecs_use * scales[None, :]
    return emb, evals_use


def compute_diffusion_distance_to_band(emb: np.ndarray, band_idx: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Diffusion distance requires SciPy (scipy.spatial.cKDTree). "
            "Install it or use --distance-mode euclidean/geodesic."
        ) from e

    band_emb = emb[band_idx]
    tree = cKDTree(band_emb)
    dists, _ = tree.query(emb, k=1, workers=-1)
    return np.asarray(dists, dtype=np.float64)

def save_ply_with_vertex_quality(path: str, verts: np.ndarray, faces: np.ndarray, quality: np.ndarray) -> None:
    """
    Save an ASCII PLY with per-vertex float 'quality' usable in MeshLab as Vertex Quality.

    verts: (N,3) float
    faces: (F,3) int
    quality: (N,) float
    """
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    quality = np.asarray(quality, dtype=np.float64).reshape(-1)

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("verts must be (N,3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be (F,3) triangle indices.")
    if quality.shape[0] != verts.shape[0]:
        raise ValueError("quality must have length N (same as number of vertices).")

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {verts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float quality\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # vertices
        for (x, y, z), q in zip(verts, quality):
            f.write(f"{x:.9g} {y:.9g} {z:.9g} {q:.9g}\n")

        # faces
        for i, j, k in faces:
            f.write(f"3 {int(i)} {int(j)} {int(k)}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Path to input .ply mesh.")
    ap.add_argument("--ref", choices=["origin", "mesh_center"], default="mesh_center")
    ap.add_argument("--plane-mode", choices=["normal", "longitudinal", "transverse"], default="transverse")
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
        help="If set, export the mesh to this .ply path with the chosen scalar saved as per-vertex 'quality'.",
    )
    ap.add_argument(
        "--export-scalar",
        type=str,
        default=None,
        help="Point-data array name to export as vertex 'quality'. Default: dist_signed_norm if --signed-under-plane else dist_norm.",
    )

    ap.add_argument(
        "--signed-under-plane",
        action="store_true",
        help="If set, distances on the side where (Ax+By+Cz+D)<0 are negated and we color by dist_signed_norm.",
    )

    ap.add_argument("--show-band", action="store_true")
    ap.add_argument("--show-edges", action="store_true")
    args = ap.parse_args()

    # ----------------------------
    # Load mesh
    # ----------------------------
    mesh = pv.read(args.input)
    mesh = ensure_polydata(mesh).triangulate()
    warn_if_not_watertight(mesh)

    diag = mesh_diag(mesh)

    # Reference point for signed tetrahedra centroid
    ref = np.zeros(3, dtype=np.float64)
    if args.ref == "mesh_center":
        ref = np.array(mesh.center, dtype=np.float64)

    # ----------------------------
    # Centroid + closest point
    # ----------------------------
    V_signed, c = compute_volume_and_centroid_signed_tets(mesh, ref=ref)
    p, cell_id, dist2 = closest_point_on_surface_vtk(mesh, c)

    if np.allclose(p, c):
        raise ValueError("Closest surface point equals centroid; cannot define arrow direction.")
    n = normalize(p - c)  # arrow direction (centroid -> surface)

    # PCA axis
    pts = np.asarray(mesh.points, dtype=np.float64)
    axes = pca_axes(pts, center=c)
    u0, u1, u2 = axes[:, 0], axes[:, 1], axes[:, 2]

    # ----------------------------
    # Plane normal
    # ----------------------------
    if args.plane_mode == "normal":
        plane_normal = n

    elif args.plane_mode == "longitudinal":
        u = u0
        u_perp = u - np.dot(u, n) * n
        if np.linalg.norm(u_perp) < 1e-6:
            u = u1
            u_perp = u - np.dot(u, n) * n
        if np.linalg.norm(u_perp) < 1e-6:
            u = pick_nonparallel_axis(n)
            u_perp = u - np.dot(u, n) * n
        u_perp = normalize(u_perp)
        plane_normal = normalize(np.cross(n, u_perp))

    else:  # transverse
        u = u0
        m = u - np.dot(u, n) * n
        if np.linalg.norm(m) < 1e-6:
            u = u1
            m = u - np.dot(u, n) * n
        if np.linalg.norm(m) < 1e-6:
            u = u2
            m = u - np.dot(u, n) * n
        if np.linalg.norm(m) < 1e-6:
            u = pick_nonparallel_axis(n)
            m = u - np.dot(u, n) * n
        plane_normal = normalize(m)

    # Plane equation: A x + B y + C z + D = 0  (||(A,B,C)||=1)
    A, B, Cc = plane_normal.tolist()
    D = -float(np.dot(plane_normal, c))

    print("=== Plane + Geometry ===")
    print(f"Signed volume V = {V_signed:.6g}   (abs(V)={abs(V_signed):.6g})")
    print(f"Centroid c = {c}")
    print(f"Closest point p = {p}  (cell_id={cell_id}, dist={np.sqrt(dist2):.6g})")
    print(f"Arrow direction n = {n}")
    print(f"Principal axis u0 = {normalize(u0)}")
    print(f"Plane normal m = {plane_normal}")
    print(f"Plane equation: {A:.6g} x + {B:.6g} y + {Cc:.6g} z + {D:.6g} = 0")

    # ----------------------------
    # Plane signed distance + band selection
    # ----------------------------
    s = pts @ plane_normal + D  # signed distance since ||plane_normal||=1

    ep = args.ep if args.ep is not None else 0.01 * (diag if diag > 0 else 1.0)
    band_mask = np.abs(s) <= ep
    band_idx = np.where(band_mask)[0]
    if band_idx.size == 0:
        band_idx = np.argsort(np.abs(s))[:1]
        band_mask = np.zeros_like(s, dtype=bool)
        band_mask[band_idx] = True
        print(
            f"[WARN] No vertices satisfied abs(Ax+By+Cz+D) <= ep (ep={ep}). "
            f"Falling back to the single closest-to-plane vertex as band."
        )

    band_points = pts[band_idx]

    # ----------------------------
    # Distance-to-band
    # ----------------------------
    if args.distance_mode == "euclidean":
        d_to_band = compute_euclidean_distance_to_band(pts, band_points)
        title = "dist_norm (euclidean to band)"

    elif args.distance_mode == "diffusion":
        faces_raw = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)
        faces = faces_raw[:, 1:4]

        print("=== Diffusion distance ===")
        print(f"Building diffusion embedding with k={args.diffusion_k}, t={args.diffusion_t} ...")
        emb, evals_use = diffusion_embedding(pts, faces, k=int(args.diffusion_k), t=float(args.diffusion_t))
        print(
            f"Computed {emb.shape[1]} diffusion coords. "
            f"Eigenvalue range: [{evals_use.min():.6g}, {evals_use.max():.6g}]"
        )

        d_to_band = compute_diffusion_distance_to_band(emb, band_idx)
        title = "dist_norm (diffusion to band)"

    else:  # geodesic
        faces_raw = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)
        faces = faces_raw[:, 1:4]

        print("=== Geodesic distance (graph shortest path) ===")
        print(f"Backend: {args.geodesic_backend}")
        d_to_band = geodesic_distance_to_band(pts, faces, band_idx, backend=args.geodesic_backend)

        # Handle disconnected components
        if not np.all(np.isfinite(d_to_band)):
            finite = d_to_band[np.isfinite(d_to_band)]
            if finite.size == 0:
                print("[WARN] All geodesic distances are infinite. Setting all distances to 0.")
                d_to_band = np.zeros_like(d_to_band)
            else:
                maxf = float(np.max(finite))
                print(
                    f"[WARN] Some vertices unreachable from band (inf). "
                    f"Clamping inf to max finite ({maxf:.6g}) for normalization."
                )
                d_to_band = np.where(np.isfinite(d_to_band), d_to_band, maxf)

        title = "dist_norm (geodesic to band)"

    # ----------------------------
    # Normalize + signed-under-plane
    # ----------------------------
    dist_norm = normalize_01(d_to_band)

    side = np.where(s < 0.0, -1.0, 1.0)  # "under" == negative side of plane
    dist_signed = d_to_band * side
    dist_signed_norm = dist_norm * side  # in [-1, 1]

    # Attach scalars
    mesh.point_data["plane_signed_dist"] = s
    mesh.point_data["in_band"] = band_mask.astype(np.uint8)

    mesh.point_data["dist_to_band"] = d_to_band
    mesh.point_data["dist_norm"] = dist_norm

    mesh.point_data["dist_signed"] = dist_signed
    mesh.point_data["dist_signed_norm"] = dist_signed_norm

    print("=== Coloring ===")
    print(f"ep = {ep:.6g}   (#band_vertices={band_idx.size})")
    print(f"dist_to_band: min={float(np.min(d_to_band)):.6g}, max={float(np.max(d_to_band)):.6g}")

    # ----------------------------
    # Decide visualization scalar + title
    # ----------------------------
    if args.signed_under_plane:
        scalars_name = "dist_signed_norm"
        bar_title = title.replace("dist_norm", "dist_signed_norm") + " (negative under plane)"
    else:
        scalars_name = "dist_norm"
        bar_title = title

    # ----------------------------
    # Export PLY with vertex quality (Even safer alternative)
    # ----------------------------
    if args.export_ply:
        scalar_to_export = args.export_scalar
        if scalar_to_export is None:
            scalar_to_export = "dist_signed_norm" if args.signed_under_plane else "dist_norm"

        if scalar_to_export not in mesh.point_data:
            raise KeyError(
                f"Requested export scalar '{scalar_to_export}' not found in mesh.point_data. "
                f"Available arrays: {list(mesh.point_data.keys())}"
            )

        q = np.asarray(mesh.point_data[scalar_to_export], dtype=np.float64).reshape(-1)

        faces_raw = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)
        faces_tri = faces_raw[:, 1:4]

        save_ply_with_vertex_quality(args.export_ply, np.asarray(mesh.points), faces_tri, q)
        print(f"[OK] Exported PLY with vertex quality '{scalar_to_export}' to: {args.export_ply}")

    # ----------------------------
    # Visualization
    # ----------------------------
    r = 0.01 * (diag if diag > 0 else 1.0)
    plane_patch = make_plane_patch(center=c, normal=plane_normal, mesh=mesh, scale=args.plane_scale)

    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars=scalars_name,
        show_edges=args.show_edges,
        opacity=1.0,
        scalar_bar_args={"title": bar_title},
    )
    plotter.add_mesh(plane_patch, opacity=0.35)

    # Markers
    plotter.add_mesh(pv.Sphere(radius=r, center=c))
    plotter.add_mesh(pv.Sphere(radius=r, center=p))

    # Arrows
    plotter.add_mesh(pv.Arrow(start=c, direction=n, scale=0.25 * (diag if diag > 0 else 1.0)))
    plotter.add_mesh(pv.Arrow(start=c, direction=normalize(u0), scale=0.20 * (diag if diag > 0 else 1.0)))
    plotter.add_mesh(pv.Arrow(start=c, direction=plane_normal, scale=0.18 * (diag if diag > 0 else 1.0)))

    if args.show_band:
        band_poly = pv.PolyData(band_points)
        plotter.add_mesh(band_poly, render_points_as_spheres=True, point_size=8)

    plotter.add_axes()
    plotter.show()


if __name__ == "__main__":
    main()