from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MeshData:
    V: np.ndarray              # (N, 3) float vertices
    F: Optional[np.ndarray]    # (M, 3) int faces, None for point cloud
    VN: Optional[np.ndarray]   # (N, 3) float vertex normals
    FN: Optional[np.ndarray]   # (M, 3) float face normals
    C: Optional[np.ndarray]    # (N, 3) or (N, 4) float colors in [0, 1]

    def __post_init__(self) -> None:
        self.V = np.asarray(self.V, dtype=np.float32)

        if self.F is not None:
            faces = np.asarray(self.F)
            if not np.issubdtype(faces.dtype, np.number):
                raise TypeError("F must be a numeric array of integer indices")
            if not np.isfinite(faces).all():
                raise ValueError("F contains non-finite values")
            if not np.equal(faces, np.floor(faces)).all():
                raise ValueError("F contains non-integer index values")
            self.F = faces.astype(np.int32, copy=False)

        if self.VN is not None:
            self.VN = np.asarray(self.VN, dtype=np.float32)

        if self.FN is not None:
            self.FN = np.asarray(self.FN, dtype=np.float32)

        if self.C is not None:
            colors = np.asarray(self.C)
            if not np.issubdtype(colors.dtype, np.number):
                raise TypeError("C must be a numeric array")
            colors = colors.astype(np.float32, copy=False)
            if not np.isfinite(colors).all():
                raise ValueError("C contains non-finite values")
            if colors.size and float(colors.max()) > 1.0:
                if float(colors.max()) > 255.0:
                    raise ValueError("C values must use either [0, 1] or [0, 255] range")
                colors = colors / 255.0
            self.C = colors

        self.validate()

    def validate(self) -> None:
        """
        Validate that the mesh/point cloud data has consistent shapes and indices.
        Raises ValueError or TypeError if invalid.
        """
        if self.V.ndim != 2 or self.V.shape[1] != 3:
            raise ValueError(f"V must have shape (N, 3), got {self.V.shape}")

        if len(self.V) == 0:
            raise ValueError("V cannot be empty")

        if not np.issubdtype(self.V.dtype, np.floating):
            raise TypeError("V must be a floating-point array")

        if not np.isfinite(self.V).all():
            raise ValueError("V contains non-finite values")

        if self.F is not None:
            if self.F.ndim != 2 or self.F.shape[1] != 3:
                raise ValueError(f"F must have shape (M, 3), got {self.F.shape}")

            if not np.issubdtype(self.F.dtype, np.integer):
                raise TypeError("F must be an integer array")

            if len(self.F) > 0:
                if np.any(self.F < 0):
                    raise ValueError("F contains negative indices")
                if np.any(self.F >= len(self.V)):
                    raise ValueError("F contains out-of-bounds vertex indices")

        if self.VN is not None:
            if self.VN.ndim != 2 or self.VN.shape[1] != 3:
                raise ValueError(f"VN must have shape (N, 3), got {self.VN.shape}")
            if len(self.VN) != len(self.V):
                raise ValueError("VN must have the same number of rows as V")
            if not np.isfinite(self.VN).all():
                raise ValueError("VN contains non-finite values")

        if self.FN is not None:
            if self.F is None:
                raise ValueError("FN provided but F is None")
            if self.FN.ndim != 2 or self.FN.shape[1] != 3:
                raise ValueError(f"FN must have shape (M, 3), got {self.FN.shape}")
            if len(self.FN) != len(self.F):
                raise ValueError("FN must have the same number of rows as F")
            if not np.isfinite(self.FN).all():
                raise ValueError("FN contains non-finite values")

        if self.C is not None:
            if self.C.ndim != 2:
                raise ValueError(f"C must be 2D, got shape {self.C.shape}")
            if self.C.shape[1] not in (3, 4):
                raise ValueError("C must have shape (N, 3) or (N, 4)")
            if len(self.C) != len(self.V):
                raise ValueError("C must have the same number of rows as V")
            if np.any(self.C < 0.0) or np.any(self.C > 1.0):
                raise ValueError("C must contain normalized color values in [0, 1]")

    def is_point_cloud(self) -> bool:
        return self.F is None

    def is_triangle_mesh(self) -> bool:
        return self.F is not None

    def n_vertices(self) -> int:
        return len(self.V)

    def n_faces(self) -> int:
        return 0 if self.F is None else len(self.F)

    def copy(self) -> "MeshData":
        return MeshData(
            V=self.V.copy(),
            F=None if self.F is None else self.F.copy(),
            VN=None if self.VN is None else self.VN.copy(),
            FN=None if self.FN is None else self.FN.copy(),
            C=None if self.C is None else self.C.copy(),
        )

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (min_corner, max_corner), each shape (3,)
        """
        return self.V.min(axis=0), self.V.max(axis=0)

    def centroid(self) -> np.ndarray:
        return self.V.mean(axis=0)

    def apply_scale(self, scale: float) -> "MeshData":
        """
        Return a scaled copy of the mesh.
        """
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be a finite value greater than zero")
        out = self.copy()
        out.V *= scale
        return out

    def summary(self) -> str:
        kind = "PointCloud" if self.is_point_cloud() else "TriangleMesh"
        bmin, bmax = self.bounds()
        return (
            f"{kind} | "
            f"vertices={self.n_vertices()} | "
            f"faces={self.n_faces()} | "
            f"bounds_min={bmin} | bounds_max={bmax}"
        )
