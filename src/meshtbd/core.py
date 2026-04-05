from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MeshData:
    V: Any
    F: Any | None
    VN: Any | None
    FN: Any | None
    C: Any | None
