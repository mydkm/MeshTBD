#!/usr/bin/env python3
"""
center_weighted_surface_sampling.py

Given an input triangle mesh (.stl), generate a surface point cloud that:
- is denser near the "middle" of the limb/part
- is sparse/empty near the ends (solid top/bottom)

Robustness + UX:
- Clean mesh + keep largest connected component (prevents singular Laplacian)
- End-set sanity checks (A/B non-empty, no overlap)
- Harmonic solve ridge regularization if L_ff is singular
- Debug prints for t / candidate counts / w stats / boundary loops
- tqdm progress bar with ETA for Poisson selection + stage timing prints

NEW (per your request):
A) "Thumb/finger hole" keepout without nuking long seams:
   - Keepout is applied ONLY to SMALL boundary loops (hole-like), excluding the two selected end loops
   - Optional explicit loop indices to keepout from

B) Optional geodesic spacing instead of Euclidean:
   - distance_metric = euclidean (fast) or geodesic (graph-geodesic; slower)
   - Geodesic mode snaps candidates to nearest mesh vertex and uses incremental multi-source Dijkstra
     to approximate distances along the surface.

Deps:
  numpy scipy pyvista pymeshlab
Optional (recommended for progress bar):
  tqdm

Install (uv):
  uv add numpy scipy pyvista pymeshlab tqdm

Examples:
  # Euclidean spacing, with thumb-hole keepout (auto-select holes):
  uv run center_weighted_surface_sampling.py -i Sabrina-revised.stl -n 500 \
    --rmin 0.42 --oversample 60 \
    --hole_keepout 0.8 --hole_perim_ratio 0.35 \
    --boundary_pick farthest_of_topk --boundary_topk 10

  # Geodesic spacing (slower), same keepout:
  uv run center_weighted_surface_sampling.py -i Sabrina-revised.stl -n 500 \
    --rmin 0.42 --oversample 60 \
    --hole_keepout 0.8 --hole_perim_ratio 0.35 \
    --distance_metric geodesic \
    --boundary_pick farthest_of_topk --boundary_topk 10

Notes:
- If mesh is open, boundary loops are used (preferred).
- If mesh is closed/no boundaries, PCA fallback end-regions are used (works, but less “intrinsic”).
- Geodesic mode is computationally heavier; for very large meshes you may want to decimate first.
"""

from __future__ import annotations

import argparse
import heapq
import time
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve, MatrixRankWarning
from scipy.spatial import cKDTree

import pyvista as pv
import pymeshlab as ml

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


# ----------------------------
# Timing helper
# ----------------------------

def timed(msg: str):
    class _T:
        def __enter__(self):
            print(f"[+] {msg} ...", flush=True)
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            dt = time.perf_counter() - self.t0
            if exc is None:
                print(f"    done in {dt:.2f}s", flush=True)

    return _T()


# ----------------------------
# Math utils
# ----------------------------

def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Classic smoothstep; vectorized."""
    if edge1 <= edge0:
        return (x >= edge1).astype(np.float64)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_weight_from_t(tP: np.ndarray, delta: float, eps: float) -> np.ndarray:
    """
    Soft falloff window:
      w(t)=smoothstep(delta,delta+eps,t)*smoothstep(delta,delta+eps,1-t)
    """
    return smoothstep(delta, delta + eps, tP) * smoothstep(delta, delta + eps, 1.0 - tP)


# ----------------------------
# Mesh utilities
# ----------------------------

def pv_polydata_to_numpy(poly: pv.PolyData) -> Tuple[np.ndarray, np.ndarray]:
    """Return (V,F) from PyVista PolyData (triangulated)."""
    poly = poly.triangulate()
    V = np.asarray(poly.points, dtype=np.float64)

    faces = np.asarray(poly.faces, dtype=np.int64).reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("Mesh is not purely triangles after triangulate().")
    F = faces[:, 1:4].astype(np.int64)
    return V, F


def face_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    vi = V[F[:, 0]]
    vj = V[F[:, 1]]
    vk = V[F[:, 2]]
    cross = np.cross(vj - vi, vk - vi)
    return 0.5 * np.linalg.norm(cross, axis=1)


def clean_and_keep_largest_component(V: np.ndarray, F: np.ndarray, area_eps: float = 1e-14) -> Tuple[np.ndarray, np.ndarray]:
    """
    - Remove degenerate triangles (repeated indices)
    - Remove near-zero-area triangles
    - Keep only the largest connected component (by face count)
    - Remove unreferenced vertices
    """
    bad = (F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 2] == F[:, 0])
    F = F[~bad]
    if F.size == 0:
        raise ValueError("All faces degenerate after cleaning.")

    A = face_areas(V, F)
    F = F[A > area_eps]
    if F.size == 0:
        raise ValueError("All faces near-zero area after cleaning.")

    n = V.shape[0]
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]).astype(np.int64)
    for a, b in E:
        union(int(a), int(b))

    roots = np.array([find(i) for i in range(n)], dtype=np.int64)

    fr0 = roots[F[:, 0]]
    fr1 = roots[F[:, 1]]
    fr2 = roots[F[:, 2]]
    same = (fr0 == fr1) & (fr1 == fr2)
    F = F[same]
    fr0 = fr0[same]
    if F.size == 0:
        raise ValueError("No faces belong to a consistent component after cleaning.")

    uniq, counts = np.unique(fr0, return_counts=True)
    keep_root = uniq[np.argmax(counts)]

    keep_faces = F[roots[F[:, 0]] == keep_root]

    used = np.unique(keep_faces.reshape(-1))
    new_index = -np.ones(n, dtype=np.int64)
    new_index[used] = np.arange(used.size, dtype=np.int64)

    V2 = V[used]
    F2 = new_index[keep_faces]
    return V2, F2


def boundary_loops(V: np.ndarray, F: np.ndarray) -> List[np.ndarray]:
    """
    Extract boundary components as vertex sets (unordered).
    Boundary edge = edge incident to exactly one triangle.
    """
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.sort(E, axis=1)

    uniqE, counts = np.unique(E, axis=0, return_counts=True)
    bE = uniqE[counts == 1]
    if bE.size == 0:
        return []

    adj: dict[int, list[int]] = {}
    for u, v in bE:
        u = int(u); v = int(v)
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    visited = set()
    loops: List[np.ndarray] = []
    for start in adj.keys():
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        comp = []
        while stack:
            a = stack.pop()
            comp.append(a)
            for b in adj.get(a, []):
                if b not in visited:
                    visited.add(b)
                    stack.append(b)
        loops.append(np.array(comp, dtype=np.int64))
    return loops


def loop_perimeter(V: np.ndarray, F: np.ndarray, loop_vertices: np.ndarray) -> float:
    """Approx perimeter by summing lengths of boundary edges inside the component."""
    lv = set(map(int, loop_vertices.tolist()))

    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.sort(E, axis=1)
    uniqE, counts = np.unique(E, axis=0, return_counts=True)
    bE = uniqE[counts == 1]

    perim = 0.0
    for u, v in bE:
        u = int(u); v = int(v)
        if u in lv and v in lv:
            perim += float(np.linalg.norm(V[u] - V[v]))
    return perim


def pick_end_sets_from_boundaries(
    V: np.ndarray,
    F: np.ndarray,
    loops: List[np.ndarray],
    mode: str = "two_largest",
    topk: int = 8,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Choose two boundary components for Dirichlet sets A (t=0), B (t=1).
    Returns: (A_vertices, B_vertices, A_loop_index, B_loop_index)
      - two_largest: by perimeter
      - farthest_of_topk: among top-K perimeters, choose farthest centroid pair
    """
    if len(loops) < 2:
        raise ValueError("Need at least 2 boundary loops for boundary-based end selection.")

    perims = np.array([loop_perimeter(V, F, lp) for lp in loops], dtype=np.float64)
    order = np.argsort(-perims)  # desc

    if mode == "two_largest":
        a_idx, b_idx = int(order[0]), int(order[1])
        return loops[a_idx], loops[b_idx], a_idx, b_idx

    if mode == "farthest_of_topk":
        k = min(int(topk), len(loops))
        cand = order[:k]
        cents = np.vstack([V[loops[int(idx)]].mean(axis=0) for idx in cand])

        best_i, best_j, best_d = 0, 1, -1.0
        for i in range(k):
            for j in range(i + 1, k):
                d = float(np.linalg.norm(cents[i] - cents[j]))
                if d > best_d:
                    best_d = d
                    best_i, best_j = i, j

        a_idx = int(cand[best_i])
        b_idx = int(cand[best_j])
        return loops[a_idx], loops[b_idx], a_idx, b_idx

    raise ValueError(f"Unknown boundary pick mode: {mode}")


def pick_end_sets_pca_fallback(V: np.ndarray, frac: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fallback for closed meshes (no boundary loops):
    pick two small vertex sets near extremes along first principal component.
    """
    X = V - V.mean(axis=0)
    _, _, VT = np.linalg.svd(X, full_matrices=False)
    axis = VT[0]
    s = X @ axis
    smin, smax = float(s.min()), float(s.max())
    band = frac * (smax - smin + 1e-12)

    A = np.where(s <= smin + band)[0].astype(np.int64)
    B = np.where(s >= smax - band)[0].astype(np.int64)

    if A.size < 3 or B.size < 3:
        raise ValueError("PCA fallback failed to find enough vertices for end constraints.")
    return A, B


# ----------------------------
# Laplace–Beltrami harmonic field
# ----------------------------

def cotan_laplacian(V: np.ndarray, F: np.ndarray, eps: float = 1e-12) -> csr_matrix:
    """Symmetric cotangent Laplacian (stiffness matrix)."""
    n = V.shape[0]
    i = F[:, 0]
    j = F[:, 1]
    k = F[:, 2]

    vi = V[i]
    vj = V[j]
    vk = V[k]

    def cotangent(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        c = np.cross(a, b)
        denom = np.linalg.norm(c, axis=1) + eps
        return (np.einsum("ij,ij->i", a, b) / denom)

    cot_i = cotangent(vj - vi, vk - vi)  # opposite (j,k)
    cot_j = cotangent(vk - vj, vi - vj)  # opposite (k,i)
    cot_k = cotangent(vi - vk, vj - vk)  # opposite (i,j)

    w_jk = 0.5 * cot_i
    w_ki = 0.5 * cot_j
    w_ij = 0.5 * cot_k

    rows = []
    cols = []
    data = []

    def add_edge(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> None:
        rows.extend([u, v])
        cols.extend([u, v])
        data.extend([w, w])
        rows.extend([u, v])
        cols.extend([v, u])
        data.extend([-w, -w])

    add_edge(j, k, w_jk)
    add_edge(k, i, w_ki)
    add_edge(i, j, w_ij)

    rows = np.concatenate(rows).astype(np.int64)
    cols = np.concatenate(cols).astype(np.int64)
    data = np.concatenate(data).astype(np.float64)

    L = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    L.sum_duplicates()
    return L


def solve_harmonic_t(V: np.ndarray, F: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Solve Δ t = 0 with Dirichlet constraints:
      t=0 on A, t=1 on B
    Adds tiny ridge regularization if L_ff is singular.
    """
    n = V.shape[0]
    L = cotan_laplacian(V, F)

    fixed = np.zeros(n, dtype=bool)
    fixed[A] = True
    fixed[B] = True

    t = np.zeros(n, dtype=np.float64)
    t[A] = 0.0
    t[B] = 1.0

    free = np.where(~fixed)[0]
    fixed_idx = np.where(fixed)[0]

    if free.size == 0:
        return t

    L_ff = L[free][:, free]
    L_fc = L[free][:, fixed_idx]
    b = -L_fc @ t[fixed_idx]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            t_free = spsolve(L_ff, b)
    except MatrixRankWarning:
        diag = L_ff.diagonal()
        scale = float(np.mean(np.abs(diag))) if diag.size else 1.0
        lam = 1e-10 * (scale if scale > 0 else 1.0)
        t_free = spsolve(L_ff + diags(lam * np.ones(L_ff.shape[0])), b)

    t[free] = t_free

    if not np.all(np.isfinite(t)):
        raise RuntimeError("Harmonic field contains NaN/Inf. Mesh/end constraints still problematic.")

    tmin, tmax = float(t.min()), float(t.max())
    if abs(tmax - tmin) > 1e-12:
        t = (t - tmin) / (tmax - tmin)
    return np.clip(t, 0.0, 1.0)


# ----------------------------
# Surface sampling (candidates)
# ----------------------------

def sample_points_on_triangles(
    V: np.ndarray,
    F: np.ndarray,
    t_vert: np.ndarray,
    num: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Area-weighted random sampling on triangles with barycentric interpolation of t."""
    A = face_areas(V, F)
    A_sum = float(A.sum())
    if A_sum <= 0.0:
        raise ValueError("Mesh has zero total area.")

    probs = A / A_sum
    tri_idx = rng.choice(len(F), size=num, replace=True, p=probs)
    tri = F[tri_idx]

    i = tri[:, 0]
    j = tri[:, 1]
    k = tri[:, 2]

    vi = V[i]
    vj = V[j]
    vk = V[k]

    ti = t_vert[i]
    tj = t_vert[j]
    tk = t_vert[k]

    r1 = rng.random(num)
    r2 = rng.random(num)
    sr1 = np.sqrt(r1)
    a = 1.0 - sr1
    b = sr1 * (1.0 - r2)
    c = sr1 * r2

    P = (a[:, None] * vi) + (b[:, None] * vj) + (c[:, None] * vk)
    tP = a * ti + b * tj + c * tk
    return P, tP


# ----------------------------
# Poisson selection config
# ----------------------------

@dataclass
class SampleConfig:
    n_points: int = 5000
    oversample_factor: int = 30

    # keep-out & weighting
    delta: float = 0.15
    eps: float = 0.05

    # variable Poisson radii
    rmin: float = 2.0
    rmax_mult: float = 6.0
    w_floor: float = 1e-3

    # selection efficiency
    rebuild_every: int = 250
    sym_dist_factor: float = 0.5  # min dist threshold = sym_dist_factor*(ri+rj)


# ----------------------------
# Poisson selection (Euclidean)
# ----------------------------

def weighted_variable_poisson(
    P: np.ndarray,
    w: np.ndarray,
    cfg: SampleConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy variable-density Poisson-ish selection (Euclidean) with progress + ETA (tqdm).
    """
    if P.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)

    w_clip = np.clip(w, cfg.w_floor, 1.0)
    r = cfg.rmin / np.sqrt(w_clip)
    rmax = cfg.rmin * cfg.rmax_mult
    r = np.clip(r, cfg.rmin, rmax)

    jitter = 1e-9 * rng.random(len(w))
    order = np.argsort(-(w + jitter))

    accepted_pts: List[np.ndarray] = []
    accepted_w: List[float] = []

    tree_pts = np.empty((0, 3), dtype=np.float64)
    tree_r = np.empty((0,), dtype=np.float64)
    tree: Optional[cKDTree] = None

    recent_pts: List[np.ndarray] = []
    recent_r: List[float] = []

    def violates(p: np.ndarray, ri: float) -> bool:
        if tree is not None and tree_pts.shape[0] > 0:
            idxs = tree.query_ball_point(p, ri + rmax)
            if idxs:
                neigh = tree_pts[idxs]
                d = np.linalg.norm(neigh - p[None, :], axis=1)
                thr = cfg.sym_dist_factor * (ri + tree_r[idxs])
                if np.any(d < thr):
                    return True

        if recent_pts:
            rp = np.vstack(recent_pts)
            rr = np.asarray(recent_r, dtype=np.float64)
            d = np.linalg.norm(rp - p[None, :], axis=1)
            thr = cfg.sym_dist_factor * (ri + rr)
            if np.any(d < thr):
                return True

        return False

    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=cfg.n_points,
            desc="Poisson selection (euclidean)",
            unit="pt",
            dynamic_ncols=True,
            smoothing=0.05,
        )

    scanned = 0
    for idx in order:
        if len(accepted_pts) >= cfg.n_points:
            break

        scanned += 1
        p = P[idx]
        wi = float(w[idx])
        ri = float(r[idx])

        if len(accepted_pts) == 0:
            accepted_pts.append(p)
            accepted_w.append(wi)
            recent_pts.append(p)
            recent_r.append(ri)
            if pbar:
                pbar.update(1)
                pbar.set_postfix(scanned=scanned, candidates=len(order), refresh=False)
            continue

        if violates(p, ri):
            if pbar and (scanned % 500 == 0):
                pbar.set_postfix(scanned=scanned, candidates=len(order), refresh=False)
            continue

        accepted_pts.append(p)
        accepted_w.append(wi)
        recent_pts.append(p)
        recent_r.append(ri)

        if pbar:
            pbar.update(1)
            pbar.set_postfix(scanned=scanned, candidates=len(order), refresh=False)

        if len(recent_pts) >= cfg.rebuild_every:
            if tree_pts.shape[0] == 0:
                tree_pts = np.vstack(recent_pts)
                tree_r = np.asarray(recent_r, dtype=np.float64)
            else:
                tree_pts = np.vstack([tree_pts, np.vstack(recent_pts)])
                tree_r = np.concatenate([tree_r, np.asarray(recent_r, dtype=np.float64)])
            tree = cKDTree(tree_pts)
            recent_pts.clear()
            recent_r.clear()

    if pbar:
        pbar.close()

    S = np.vstack(accepted_pts).astype(np.float64) if accepted_pts else np.empty((0, 3), dtype=np.float64)
    wS = np.asarray(accepted_w, dtype=np.float64) if accepted_w else np.empty((0,), dtype=np.float64)
    return S, wS


# ----------------------------
# Poisson selection (Approx Geodesic)
# ----------------------------

def build_vertex_adjacency(V: np.ndarray, F: np.ndarray) -> List[List[Tuple[int, float]]]:
    """
    Build an undirected weighted adjacency list from mesh edges.
    Edge weights are Euclidean edge lengths; geodesic is approximated by shortest path along edges.
    """
    n = V.shape[0]
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]

    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]).astype(np.int64)
    E = np.sort(E, axis=1)
    E = np.unique(E, axis=0)

    for u, v in E:
        u = int(u); v = int(v)
        w = float(np.linalg.norm(V[u] - V[v]))
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


def dijkstra_update_with_label(
    adj: List[List[Tuple[int, float]]],
    dist: np.ndarray,
    label: np.ndarray,
    sources: List[int],
    source_labels: List[int],
) -> None:
    """
    Incremental multi-source Dijkstra:
    - dist[v] stores min geodesic distance to accepted set
    - label[v] stores which accepted sample achieved that min (index into accepted arrays)
    """
    heap: List[Tuple[float, int, int]] = []

    for s, lab in zip(sources, source_labels):
        s = int(s); lab = int(lab)
        # force this vertex to be a source at dist 0 with its label
        if dist[s] > 0.0:
            dist[s] = 0.0
            label[s] = lab
        heapq.heappush(heap, (dist[s], s, label[s]))

    while heap:
        d, u, lab = heapq.heappop(heap)
        if d > dist[u] + 1e-12:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd + 1e-12 < dist[v]:
                dist[v] = nd
                label[v] = lab
                heapq.heappush(heap, (nd, v, lab))


def weighted_variable_poisson_geodesic(
    V: np.ndarray,
    F: np.ndarray,
    P: np.ndarray,
    w: np.ndarray,
    cfg: SampleConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy variable-density Poisson-ish selection using approximate geodesic spacing.

    Implementation:
    - Snap each candidate point to nearest mesh vertex.
    - Maintain dist[v] = geodesic distance from vertex v to nearest accepted sample (vertex),
      updated via incremental multi-source Dijkstra after each acceptance (safe/correct, slower).
    - Use label[v] to approximate the radius of the nearest accepted sample for symmetric threshold.
    """
    if P.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)

    # radii from weight
    w_clip = np.clip(w, cfg.w_floor, 1.0)
    r = cfg.rmin / np.sqrt(w_clip)
    rmax = cfg.rmin * cfg.rmax_mult
    r = np.clip(r, cfg.rmin, rmax)

    # process high->low weight with tiny jitter
    jitter = 1e-9 * rng.random(len(w))
    order = np.argsort(-(w + jitter))

    # snap candidates to nearest mesh vertex
    vtree = cKDTree(V)
    cand_vid = vtree.query(P, k=1)[1].astype(np.int64)

    # build adjacency once
    adj = build_vertex_adjacency(V, F)

    # dist/label fields over vertices
    dist = np.full(V.shape[0], np.inf, dtype=np.float64)
    label = np.full(V.shape[0], -1, dtype=np.int64)

    acc_pts: List[np.ndarray] = []
    acc_w: List[float] = []
    acc_r: List[float] = []
    acc_vid: List[int] = []

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=cfg.n_points, desc="Poisson selection (geodesic)", unit="pt", dynamic_ncols=True)

    for idx in order:
        if len(acc_pts) >= cfg.n_points:
            break

        vi = int(cand_vid[idx])
        ri = float(r[idx])

        if len(acc_pts) == 0:
            acc_pts.append(P[idx])
            acc_w.append(float(w[idx]))
            acc_r.append(ri)
            acc_vid.append(vi)
            dijkstra_update_with_label(adj, dist, label, [vi], [0])
            if pbar:
                pbar.update(1)
            continue

        # symmetric threshold using nearest accepted sample label at this vertex
        lab = int(label[vi])
        rj = float(acc_r[lab]) if (0 <= lab < len(acc_r)) else cfg.rmin
        thr = cfg.sym_dist_factor * (ri + rj)

        if dist[vi] < thr:
            continue

        # accept
        new_lab = len(acc_pts)
        acc_pts.append(P[idx])
        acc_w.append(float(w[idx]))
        acc_r.append(ri)
        acc_vid.append(vi)

        # update geodesic field immediately (correct, but slower)
        dijkstra_update_with_label(adj, dist, label, [vi], [new_lab])

        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()

    S = np.vstack(acc_pts) if acc_pts else np.empty((0, 3))
    wS = np.asarray(acc_w, dtype=np.float64) if acc_w else np.empty((0,))
    return S, wS


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--input", required=True, help="Input triangle mesh (.stl recommended)")
    ap.add_argument("-n", "--n_points", type=int, default=5000, help="Target number of output points")
    ap.add_argument("--oversample", type=int, default=30, help="Candidate oversample factor")
    ap.add_argument("--delta", type=float, default=0.15, help="End keep-out margin in t-space")
    ap.add_argument("--eps", type=float, default=0.05, help="Smoothstep ramp width")
    ap.add_argument("--rmin", type=float, default=2.0, help="Minimum Poisson radius in model units")
    ap.add_argument("--rmax_mult", type=float, default=6.0, help="Clamp max radius to rmax_mult*rmin")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")

    ap.add_argument("--end_mode", choices=["boundary", "pca"], default="boundary",
                    help="Choose end constraints from boundary loops (preferred) or PCA fallback")
    ap.add_argument("--boundary_pick", choices=["two_largest", "farthest_of_topk"], default="two_largest",
                    help="How to pick end loops when multiple loops exist")
    ap.add_argument("--boundary_topk", type=int, default=10,
                    help="Top-K loops to consider for farthest_of_topk")

    # NEW: hole keepout controls (small boundary loops only)
    ap.add_argument("--hole_keepout", type=float, default=0.0,
                    help="Keep samples away from selected hole boundary loops by this distance (0 disables).")
    ap.add_argument("--hole_perim_ratio", type=float, default=0.35,
                    help="Treat boundary loops with perimeter <= ratio * max(end_perimeters) as holes.")
    ap.add_argument("--hole_loop_indices", type=str, default="",
                    help="Optional explicit hole loop indices (comma-separated) to keep out from, "
                         "overrides hole_perim_ratio. Example: '3,7,9'")

    # NEW: distance metric toggle
    ap.add_argument("--distance_metric", choices=["euclidean", "geodesic"], default="euclidean",
                    help="Use euclidean (fast) or graph-geodesic (slower) distances for Poisson spacing.")

    ap.add_argument("--out_ply", default="", help="Optional output PLY file for the point cloud")
    ap.add_argument("--no_gui", action="store_true", help="Do not launch PyVista GUI")
    ap.add_argument("--no_debug", action="store_true", help="Suppress debug prints")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    cfg = SampleConfig(
        n_points=args.n_points,
        oversample_factor=args.oversample,
        delta=args.delta,
        eps=args.eps,
        rmin=args.rmin,
        rmax_mult=args.rmax_mult,
    )

    with timed("Loading mesh"):
        poly = pv.read(args.input).triangulate()
        V, F = pv_polydata_to_numpy(poly)

    with timed("Cleaning mesh (degenerates + keep largest component)"):
        V, F = clean_and_keep_largest_component(V, F)

    bbox = V.max(axis=0) - V.min(axis=0)
    diag = float(np.linalg.norm(bbox))
    area = float(face_areas(V, F).sum())
    print(f"bbox diag = {diag:g}, surface area = {area:g}, rmin = {cfg.rmin:g}", flush=True)

    other_tree: Optional[cKDTree] = None
    other_boundary = np.array([], dtype=np.int64)

    with timed("Selecting end constraints (A=t0, B=t1)"):
        loops = boundary_loops(V, F)

        perims = np.array([loop_perimeter(V, F, lp) for lp in loops], dtype=np.float64) if loops else np.array([], dtype=np.float64)

        if args.end_mode == "boundary" and len(loops) >= 2:
            A, B, a_loop_idx, b_loop_idx = pick_end_sets_from_boundaries(
                V, F, loops, mode=args.boundary_pick, topk=args.boundary_topk
            )
        else:
            A, B = pick_end_sets_pca_fallback(V)
            a_loop_idx, b_loop_idx = -1, -1

        # Sanity checks
        if A.size == 0 or B.size == 0:
            raise RuntimeError(f"End constraint set empty: |A|={A.size}, |B|={B.size}")
        overlap = np.intersect1d(A, B)
        if overlap.size > 0:
            raise RuntimeError(f"A and B overlap ({overlap.size} vertices). End selection is broken.")

        # Build keepout set ONLY from "hole" loops (small perimeters), excluding end loops
        hole_ids: List[int] = []
        if loops and args.hole_keepout > 0:
            if args.hole_loop_indices.strip():
                hole_ids = [int(x) for x in args.hole_loop_indices.split(",") if x.strip().isdigit()]
                hole_ids = [i for i in hole_ids if 0 <= i < len(loops)]
                # ensure we never treat the end loops as holes
                hole_ids = [i for i in hole_ids if i not in (a_loop_idx, b_loop_idx)]
            else:
                if a_loop_idx >= 0 and b_loop_idx >= 0 and perims.size:
                    ref = float(max(perims[a_loop_idx], perims[b_loop_idx]))
                else:
                    ref = float(np.max(perims)) if perims.size else 0.0

                cutoff = args.hole_perim_ratio * ref
                for li, p in enumerate(perims.tolist()):
                    if li in (a_loop_idx, b_loop_idx):
                        continue
                    if float(p) <= cutoff:
                        hole_ids.append(li)

            if hole_ids:
                other_boundary = np.unique(np.concatenate([loops[i] for i in hole_ids])).astype(np.int64)

        if args.hole_keepout > 0 and other_boundary.size > 0:
            other_tree = cKDTree(V[other_boundary])

        if not args.no_debug:
            print(f"    boundary loops found: {len(loops)}", flush=True)
            if perims.size:
                order = np.argsort(-perims)
                print("    top loop perimeters:", [float(perims[i]) for i in order[:min(10, len(order))]], flush=True)
            print("    end loops:", a_loop_idx, b_loop_idx, flush=True)
            print("    hole loop ids used:", hole_ids, flush=True)
            print("    hole boundary verts:", int(other_boundary.size), flush=True)
            if other_tree is not None:
                print(f"    hole_keepout active: {args.hole_keepout}", flush=True)

    with timed("Solving harmonic field (Laplace–Beltrami)"):
        t_vert = solve_harmonic_t(V, F, A, B)
        if not args.no_debug:
            print("    t stats:",
                  float(np.nanmin(t_vert)), float(np.nanmax(t_vert)),
                  "finite:", bool(np.isfinite(t_vert).all()),
                  flush=True)

    with timed("Sampling candidate points on triangles"):
        n_cand = int(cfg.oversample_factor * cfg.n_points)
        P_cand, t_cand = sample_points_on_triangles(V, F, t_vert, n_cand, rng=rng)

    with timed("Filtering + weighting candidates"):
        keep = (t_cand >= cfg.delta) & (t_cand <= 1.0 - cfg.delta) & np.isfinite(t_cand)

        # Apply hole keepout (ONLY from small loops)
        if other_tree is not None:
            d, _ = other_tree.query(P_cand, k=1)
            keep = keep & (d >= args.hole_keepout)

        P_cand = P_cand[keep]
        t_cand = t_cand[keep]

        if P_cand.shape[0] == 0:
            raise RuntimeError(
                "No candidates survived filtering. Try lowering --delta, lowering --hole_keepout, "
                "or increasing --oversample."
            )

        w_cand = build_weight_from_t(t_cand, cfg.delta, cfg.eps)

        if not args.no_debug:
            print("    candidates:", n_cand, "after keep-out:", P_cand.shape[0], flush=True)
            print("    w stats:", float(np.min(w_cand)), float(np.max(w_cand)), flush=True)

    with timed("Selecting Poisson samples"):
        if args.distance_metric == "euclidean":
            P_out, w_out = weighted_variable_poisson(P_cand, w_cand, cfg, rng=rng)
        else:
            P_out, w_out = weighted_variable_poisson_geodesic(V, F, P_cand, w_cand, cfg, rng=rng)

    if P_out.shape[0] == 0:
        raise RuntimeError(
            "No points were selected. Try lowering --delta, lowering --rmin, or increasing --oversample."
        )

    print(f"[+] Selected {P_out.shape[0]} points (target was {cfg.n_points}).", flush=True)

    with timed("Building PyMeshLab MeshSet (surface + point cloud)"):
        ms = ml.MeshSet()

        surf = ml.Mesh(vertex_matrix=V, face_matrix=F.astype(np.int32))
        ms.add_mesh(surf, "surface")

        # Store weights as grayscale RGBA for PyMeshLab (optional but handy)
        gray = np.clip(w_out, 0.0, 1.0).astype(np.float32)
        vcol = np.column_stack([gray, gray, gray, np.ones_like(gray)]).astype(np.float32)

        pc = ml.Mesh(
            vertex_matrix=P_out.astype(np.float64),
            face_matrix=np.empty((0, 3), dtype=np.int32),
            v_color_matrix=vcol,
        )
        ms.add_mesh(pc, "point_cloud")

    if args.out_ply:
        with timed(f"Saving point cloud to {args.out_ply}"):
            ms.set_current_mesh(1)  # point_cloud
            ms.save_current_mesh(args.out_ply)

    if not args.no_gui:
        with timed("Launching GUI (PyVista)"):
            plotter = pv.Plotter()
            faces_pv = np.hstack([np.full((F.shape[0], 1), 3, dtype=np.int64), F]).ravel()
            poly_clean = pv.PolyData(V, faces_pv)

            plotter.add_mesh(poly_clean, opacity=0.25, show_edges=False)

            cloud = pv.PolyData(P_out)
            cloud["w"] = w_out
            plotter.add_mesh(
                cloud,
                scalars="w",
                render_points_as_spheres=True,
                point_size=6,
            )

            # Optional debug overlay: show hole-boundary vertices used for keepout
            if (not args.no_debug) and other_boundary.size > 0:
                ob = pv.PolyData(V[other_boundary])
                plotter.add_mesh(ob, color="red", point_size=3, render_points_as_spheres=True)

            plotter.add_text("Surface + Center-weighted point cloud", font_size=12)
            plotter.show()


if __name__ == "__main__":
    main()
