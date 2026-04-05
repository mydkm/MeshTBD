from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PickResult:
    p0: Any
    p1: Any
    v0: int
    v1: int
    geodesic_distance: float


@dataclass(frozen=True)
class CalibrationInput:
    input_path: Path
    left_click: bool = True
    auto_close: bool = False
    picker: str = "cell"
    landmark_vertices: tuple[int, int] | None = None
    real_world_distance_mm: float | None = None


@dataclass(frozen=True)
class ExportConfig:
    output_path: Path
    scaled_polydata_out: Path
    preview_color_mesh: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    calibration: CalibrationInput
    export: ExportConfig
