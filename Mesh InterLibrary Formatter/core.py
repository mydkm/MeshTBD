from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MeshData:
    V: np.ndarray              # (N, 3) float32 vertices
    F: Optional[np.ndarray]    # (M, 3) int32 faces (None for point cloud)
    VN: Optional[np.ndarray]   # (N, 3) float32 vertex normals
    FN: Optional[np.ndarray]   # (M, 3) float32 face normals
    C: Optional[np.ndarray]    # (N, 3 or 4) uint8/float colors
