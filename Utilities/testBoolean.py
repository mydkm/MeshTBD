from __future__ import annotations

import numpy as np

# Input/output paths
SOURCE_PLY = "tmp1.ply"
CUT_PLY = "pyvista_cut_surface.ply"
OUTPUT_PLY = "tmp1_minus_pyvista_cut_surface_ascii.ply"


def read_ascii_ply(path: str):
    """
    Read an ASCII PLY file with:
      - one vertex element containing scalar properties
      - an optional face element using vertex indices

    Returns a dict with:
      - vertex_props: list of (dtype_name, property_name)
      - face_prop_line: original face property line if present
      - vertices: (N, P) float64 array
      - faces: list[list[int]]
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        vertex_count = None
        face_count = 0
        vertex_props: list[tuple[str, str]] = []
        face_prop_line = "property list uchar int vertex_indices"

        in_vertex_block = False

        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: unexpected EOF before end_header")
            line = line.rstrip("\n")
            parts = line.split()
            if not parts:
                continue

            if parts[0] == "format":
                if len(parts) < 3 or parts[1] != "ascii":
                    raise ValueError(f"{path}: only ASCII PLY is supported")

            elif parts[0] == "element":
                in_vertex_block = False
                if len(parts) >= 3 and parts[1] == "vertex":
                    vertex_count = int(parts[2])
                    in_vertex_block = True
                elif len(parts) >= 3 and parts[1] == "face":
                    face_count = int(parts[2])

            elif parts[0] == "property" and in_vertex_block:
                # Example: property float x
                if len(parts) != 3:
                    raise ValueError(f"{path}: unsupported vertex property line: {line}")
                vertex_props.append((parts[1], parts[2]))

            elif parts[0] == "property" and "vertex_indices" in parts:
                face_prop_line = line

            elif parts[0] == "end_header":
                break

        if vertex_count is None:
            raise ValueError(f"{path}: no vertex element found")
        if not vertex_props:
            raise ValueError(f"{path}: no vertex properties found")

        vertices = np.empty((vertex_count, len(vertex_props)), dtype=np.float64)
        for i in range(vertex_count):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: unexpected EOF while reading vertices")
            vals = line.strip().split()
            if len(vals) != len(vertex_props):
                raise ValueError(
                    f"{path}: vertex line {i} has {len(vals)} values, expected {len(vertex_props)}"
                )
            vertices[i] = [float(v) for v in vals]

        faces: list[list[int]] = []
        for i in range(face_count):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: unexpected EOF while reading faces")
            vals = [int(v) for v in line.strip().split()]
            if not vals:
                raise ValueError(f"{path}: empty face line at index {i}")
            n = vals[0]
            idx = vals[1:]
            if len(idx) != n:
                raise ValueError(
                    f"{path}: face line {i} says {n} vertices but has {len(idx)} indices"
                )
            faces.append(idx)

    return {
        "vertex_props": vertex_props,
        "face_prop_line": face_prop_line,
        "vertices": vertices,
        "faces": faces,
    }


def write_ascii_ply(
    path: str,
    vertex_props: list[tuple[str, str]],
    vertices: np.ndarray,
    faces: list[list[int]],
    face_prop_line: str = "property list uchar int vertex_indices",
):
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment Created by tmp1 - pyvista_cut_surface vertex-difference script\n")
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
    source_ply: str = SOURCE_PLY,
    cut_ply: str = CUT_PLY,
    output_ply: str = OUTPUT_PLY,
):
    src = read_ascii_ply(source_ply)
    cut = read_ascii_ply(cut_ply)

    # Compare ONLY by x, y, z.
    # For these files, matching at float32 precision correctly identifies the overlap.
    src_xyz32 = src["vertices"][:, :3].astype(np.float32)
    cut_xyz32 = cut["vertices"][:, :3].astype(np.float32)

    cut_set = set(map(tuple, cut_xyz32.tolist()))
    remove_mask = np.array([tuple(p) in cut_set for p in src_xyz32], dtype=bool)
    keep_mask = ~remove_mask

    kept_vertices = src["vertices"][keep_mask]

    # Old vertex index -> new vertex index
    old_to_new = np.full(src["vertices"].shape[0], -1, dtype=np.int64)
    kept_old_ids = np.flatnonzero(keep_mask)
    old_to_new[kept_old_ids] = np.arange(kept_old_ids.size, dtype=np.int64)

    # Keep only faces whose vertices ALL survive.
    # Any face touching a removed vertex is discarded.
    kept_faces: list[list[int]] = []
    dropped_faces = 0
    for face in src["faces"]:
        face_arr = np.asarray(face, dtype=np.int64)
        if np.all(keep_mask[face_arr]):
            kept_faces.append(old_to_new[face_arr].tolist())
        else:
            dropped_faces += 1

    write_ascii_ply(
        output_ply,
        vertex_props=src["vertex_props"],
        vertices=kept_vertices,
        faces=kept_faces,
        face_prop_line=src["face_prop_line"],
    )

    print(f"Source vertices:  {src['vertices'].shape[0]}")
    print(f"Cut vertices:     {cut['vertices'].shape[0]}")
    print(f"Removed vertices: {remove_mask.sum()}")
    print(f"Kept vertices:    {kept_vertices.shape[0]}")
    print(f"Source faces:     {len(src['faces'])}")
    print(f"Dropped faces:    {dropped_faces}")
    print(f"Kept faces:       {len(kept_faces)}")
    print(f"Wrote:            {output_ply}")


if __name__ == "__main__":
    subtract_ply_by_vertex_overlap()