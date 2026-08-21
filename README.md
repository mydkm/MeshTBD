<div align="center">

# MeshTBD

_Joshua Davidov and Ahikara Sandrasagra_ **|** Cooper Union

</div>

---

## Table of Contents
- [Overview](#overview)
- [Current Status](#current-status)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Setup](#setup)
  - [Using uv (recommended)](#using-uv-recommended)
  - [Using venv + pip](#using-venv--pip)
- [Usage](#usage)
  - [Voronoi Cast Pipeline](#voronoi-cast-pipeline)
  - [Uniform Scaling CLI](#uniform-scaling-cli)
  - [MeshData Contract](#meshdata-contract)
- [Testing](#testing)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
MeshTBD is a script-driven 3D mesh workflow focused on generating cast-like structures from scanned body-part meshes, with Voronoi-style openings and Blender post-processing.

The current primary runnable pipeline is `VoronoiFinal.py`.

---

## Current Status
- Active end-to-end script: `VoronoiFinal.py`
- Supporting interoperability package: `mesh_interlibrary_formatter/`
- Regression harness: `tests/test_phase4_regressions.py`

`VoronoiFinal.py` still uses PyVista, PyMeshLab, and Blender for their respective
operations. One active handoff now passes the Voronoi-colored PyMeshLab surface
through `MeshData` before conversion to PyVista; the remaining stages continue
to use their library-native representations.

---

## Repository Layout
- `VoronoiFinal.py`: interactive cast-generation workflow (PyVista + PyMeshLab + Blender).
- `mesh_interlibrary_formatter/`: neutral `MeshData` representation for cross-library mesh formatting.
- `mesh_interlibrary_formatter/adapters/`: adapters for PyVista, Open3D, trimesh, PyMeshLab, and Blender.
- `mesh_interlibrary_formatter/cli/scale_calibrate.py`: uniform-scale load/transform/export CLI.
- `tests/`: regression tests for key correctness and packaging behaviors.

---

## Requirements
- Python `3.11.13` (pinned in `pyproject.toml`)
- Blender-compatible Python environment for `bpy`/`bmesh` runtime usage

Core dependencies include:
- `bpy`
- `numpy`
- `open3d`
- `plyfile`
- `pymeshlab`
- `pyvista`
- `scipy`
- `trimesh`

---

## Setup
### Using uv (recommended)
1. Clone the repository:
```bash
git clone https://github.com/mydkm/MeshTBD.git
cd MeshTBD
```

2. Install dependencies:
```bash
uv sync
```

If you'd like to download uv, you could do so [here](https://github.com/astral-sh/uv).

3. Run the main script:
```bash
uv run python VoronoiFinal.py -i input_forearm.ply -o output_cast.stl
```

### Using venv + pip
1. Create and activate a virtual environment:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the script:
```bash
python VoronoiFinal.py -i input_forearm.ply -o output_cast.stl
```

---

## Usage
### Voronoi Cast Pipeline
Run:
```bash
python VoronoiFinal.py -i <input_mesh.(stl|ply)> -o <output_mesh.(stl|ply)>
```

The script will:
1. Open an interactive PyVista picker to select two landmarks on the surface.
2. Compute geodesic distance between those points.
3. Prompt for real-world distance to derive scale.
4. Save the cleaned and physically scaled intermediate mesh.
5. Compute two selected signed-geodesic plane-quality fields and apply their region mask.
6. Sample 75 points, project Voronoi colors, and pass the colored surface through the PyMeshLab -> `MeshData` -> PyVista bridge.
7. Remesh and apply Blender modifiers.
8. Export the final `.stl` or `.ply` for external fabrication and evaluation.

Useful options:
- `--right-click`: use right-click picking.
- `--auto-close`: close picker window after second pick.
- `--picker {hardware,cell,point,volume}`: choose VTK picker.
- `--scaled-polydata-out <path>`: save scaled intermediate mesh.

### Uniform Scaling CLI
Run:
```bash
python -m mesh_interlibrary_formatter.cli.scale_calibrate --help
```

Use either an explicit scale:
```bash
python -m mesh_interlibrary_formatter.cli.scale_calibrate \
  -i input.ply -o scaled.ply --scale 1.25
```

or a measured mesh distance and its corresponding physical distance:
```bash
python -m mesh_interlibrary_formatter.cli.scale_calibrate \
  -i input.ply -o calibrated.ply --mesh-distance 42 --actual-distance 105
```

The second form applies `scale = actual_distance / mesh_distance`. Landmark
picking remains part of the interactive `VoronoiFinal.py` workflow.

### MeshData Contract
The neutral representation uses a small, validated data contract:

| Field | Meaning | Canonical representation |
|---|---|---|
| `V` | Vertices | finite `float32`, shape `(N, 3)` |
| `F` | Triangle indices | `int32`, shape `(M, 3)`, or `None` for a point cloud |
| `VN` | Vertex normals | optional finite `float32`, shape `(N, 3)` |
| `FN` | Face normals | optional finite `float32`, shape `(M, 3)` |
| `C` | Vertex colors | optional normalized `float32`, shape `(N, 3)` or `(N, 4)` |

Adapters normalize integer `[0,255]` colors into the canonical `[0,1]` range.
Geometry-library imports are lazy, so core `MeshData` and calibration utilities
can be used without importing every native geometry dependency.

---

## Testing
Run the current regression suite with stdlib unittest:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The Open3D native round-trip is opt-in because its wheel requires a compatible
CPU/runtime environment:
```bash
MESHTBD_TEST_OPEN3D=1 python -m unittest \
  tests.test_adapter_roundtrips.TestAdapterRoundTrips.test_open3d_round_trip -v
```

PyMeshLab native round trips are likewise opt-in for environments where its
compiled wheel and OpenGL plugins load successfully:
```bash
MESHTBD_TEST_PYMESHLAB=1 python -m unittest \
  tests.test_adapter_roundtrips.TestAdapterRoundTrips.test_pymeshlab_round_trip -v
```

See [`docs/meshdata_validation.md`](docs/meshdata_validation.md) for the current
adapter fidelity and active-integration evidence.

---

## Known Limitations
- `VoronoiFinal.py` is interactive and GUI-dependent (PyVista + Blender context).
- End-to-end Blender pipeline validation is not fully automated.
- The active pipeline uses a fixed 75-point Poisson sample rather than quality-driven variable-density sampling.
- Signed-volume centroid calculations assume a closed, consistently oriented surface; open scans may invalidate the result.
- Hand and foot processing remain conceptual/planned extensions of the implemented general-limb baseline.
- `mesh_interlibrary_formatter` is integrated at one PyMeshLab-to-PyVista handoff; the rest of `VoronoiFinal.py` still uses direct library-native data paths.

---

## Contributing
Issues and PRs are welcome. Please keep changes reproducible and include clear validation steps (commands, sample inputs, and expected outputs).

---

## License
MIT License. See `LICENSE`.
