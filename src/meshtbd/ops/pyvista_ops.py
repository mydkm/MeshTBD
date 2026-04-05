from __future__ import annotations

import colorsys
import numpy as np


def _as_rows(values, width: int) -> list[list[float]]:
    rows = values.tolist() if hasattr(values, "tolist") else values
    if not rows:
        return []

    if isinstance(rows[0], (list, tuple)):
        return [list(row) for row in rows]

    if len(rows) % width != 0:
        raise ValueError(f"Expected a flat sequence divisible by {width}.")

    return [list(rows[i : i + width]) for i in range(0, len(rows), width)]


def build_colorized_polydata(vertices, faces, rgba):
    import numpy as np
    import pyvista as pv

    faces_pv = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
    ).ravel()
    mesh = pv.PolyData(vertices, faces_pv)
    mesh.point_data["RGBA"] = rgba
    return mesh


def rgba_float_to_rgb_uint8(rgba):
    rows = _as_rows(rgba, 4)
    converted = []
    for row in rows:
        converted.append(
            [
                max(0, min(255, int(round(float(row[0]) * 255.0)))),
                max(0, min(255, int(round(float(row[1]) * 255.0)))),
                max(0, min(255, int(round(float(row[2]) * 255.0)))),
            ]
        )

    try:
        import numpy as np

        return np.asarray(converted, dtype=np.uint8)
    except ModuleNotFoundError:
        return converted


def compute_red_like_mask(
    rgb,
    *,
    hue_threshold_degrees: float = 50.0,
    min_saturation: float = 0.25,
):
    rows = _as_rows(rgb, 3)
    mask = []
    for row in rows:
        hue, saturation, _ = colorsys.rgb_to_hsv(
            float(row[0]) / 255.0,
            float(row[1]) / 255.0,
            float(row[2]) / 255.0,
        )
        mask.append(hue <= hue_threshold_degrees / 360.0 and saturation >= min_saturation)

    try:
        import numpy as np

        return np.asarray(mask, dtype=bool)
    except ModuleNotFoundError:
        return mask


def preview_colorized_mesh(mesh) -> None:
    mesh.plot(rgb=True)


def attach_rgb_point_data(mesh) -> np.ndarray:
    if "RGBA" not in mesh.point_data:
        raise ValueError("Expected RGBA point data on colorized mesh.")

    rgb = rgba_float_to_rgb_uint8(mesh.point_data["RGBA"])
    mesh.point_data.pop("RGBA")
    mesh.point_data["RGB"] = rgb
    return rgb


def extract_red_surface(mesh):
    red_vol = mesh.threshold((0.5, 1.5), scalars="keep", all_scalars=True)
    print("Deleted selected vertices!")

    red_mesh = red_vol.extract_surface().triangulate()
    if red_mesh.n_points == 0 or red_mesh.n_cells == 0:
        raise SystemExit("No Voronoi cells selected by keep-mask; cannot continue.")
    return red_mesh


def triangle_arrays_from_polydata(mesh) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    return (
        mesh.points.astype(np.float64),
        mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32),
    )
