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
  - [Package Scaffold CLI](#package-scaffold-cli)
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
- In-progress consumable package: `mesh_interlibrary_formatter/`
- Regression harness: `tests/test_phase4_regressions.py`

---

## Repository Layout
- `VoronoiFinal.py`: interactive cast-generation workflow (PyVista + PyMeshLab + Blender).
- `mesh_interlibrary_formatter/`: package scaffold for cross-library mesh loading/formatting.
- `mesh_interlibrary_formatter/adapters/`: adapter loaders for PyVista and Open3D.
- `mesh_interlibrary_formatter/cli/scale_calibrate.py`: minimal CLI scaffold (not full calibration implementation yet).
- `tests/`: regression tests for key correctness and packaging behaviors.

---

## Requirements
- Python `3.11.13` (pinned in `pyproject.toml`)
- Blender-compatible Python environment for `bpy`/`bmesh` runtime usage

Core dependencies include:
- `bpy`
- `numpy`
- `open3d`
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
4. Run reconstruction, Voronoi color projection, thresholding, remeshing, and Blender modifiers.
5. Export final `.stl` or `.ply`.

Useful options:
- `--right-click`: use right-click picking.
- `--auto-close`: close picker window after second pick.
- `--picker {hardware,cell,point,volume}`: choose VTK picker.
- `--scaled-polydata-out <path>`: save scaled intermediate mesh.

### Package Scaffold CLI
Current scaffold command:
```bash
python -m mesh_interlibrary_formatter.cli.scale_calibrate --help
```

This CLI exists and parses arguments, but full calibration behavior is still pending.

---

## Testing
Run the current regression suite with stdlib unittest:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## Known Limitations
- `VoronoiFinal.py` is interactive and GUI-dependent (PyVista + Blender context).
- End-to-end Blender pipeline validation is not fully automated.
- `mesh_interlibrary_formatter` is importable but still functionally incomplete as a full consumable package.

---

## Contributing
Issues and PRs are welcome. Please keep changes reproducible and include clear validation steps (commands, sample inputs, and expected outputs).

---

## License
MIT License. See `LICENSE`.
