from __future__ import annotations

import math

from mesh_interlibrary_formatter.core import MeshData


def compute_uniform_scale(mesh_distance: float, actual_distance: float) -> float:
    """Return the scale that maps a measured mesh distance to a physical distance."""
    mesh_distance = float(mesh_distance)
    actual_distance = float(actual_distance)

    if not math.isfinite(mesh_distance) or mesh_distance <= 0:
        raise ValueError("mesh_distance must be a finite value greater than zero")
    if not math.isfinite(actual_distance) or actual_distance <= 0:
        raise ValueError("actual_distance must be a finite value greater than zero")

    return actual_distance / mesh_distance


def calibrate_mesh_scale(
    mesh: MeshData,
    *,
    mesh_distance: float,
    actual_distance: float,
) -> MeshData:
    """Return a uniformly scaled copy of ``mesh`` using a measured landmark pair."""
    scale = compute_uniform_scale(mesh_distance, actual_distance)
    return mesh.apply_scale(scale)
