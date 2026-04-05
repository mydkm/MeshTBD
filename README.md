<div align="center">

# MeshTBD

_Joshua Davidov and Ahikara Sandrasagra_ **|** Cooper Union

</div>

---

## Overview
MeshTBD is a script-first 3D mesh workflow for turning scanned body-part meshes into cast-like shells with Voronoi openings and Blender post-processing.

The primary entrypoint is still `VoronoiFinal.py`, but the implementation now lives in a maintainable package under `src/meshtbd/`.

---

## Repository Layout
- `VoronoiFinal.py`: thin compatibility wrapper around the packaged CLI.
- `src/meshtbd/`: primary application package.
- `src/mesh_interlibrary_formatter/`: compatibility layer for the older formatter package name.
- `tests/`: regression and unit tests.
- `scripts/analysis/`: exploratory geometry utilities that are still useful but not part of the core package.
- `scripts/demo/`: helper scripts for running local demos against local-only mesh assets.
- `examples/`: lightweight demo notes.
- `docs/reports/`: prior review and remediation artifacts.
- `archive/drafts/`: archived research and draft scripts.
- `local_data/` and `local_outputs/`: ignored working directories for large meshes and generated artifacts.

---

## Setup
### Using uv
```bash
uv sync
```

### Using pip
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Usage
### Interactive workflow
```bash
python VoronoiFinal.py -i local_data/demo/Just\ forearm.stl -o local_outputs/forearm_cast.stl
```

This flow opens the PyVista picker, lets you choose two landmarks, asks for the real-world distance, and runs the full PyMeshLab + Blender pipeline.

### Less interactive workflow
```bash
python VoronoiFinal.py \
  -i local_data/demo/Just\ forearm.stl \
  -o local_outputs/forearm_cast.stl \
  --landmark-vertices 10 250 \
  --real-mm 215 \
  --no-color-preview
```

Useful options:
- `--landmark-vertices V0 V1`: skip interactive picking and reuse known mesh vertex ids.
- `--real-mm VALUE`: skip the distance prompt.
- `--no-color-preview`: disable the mid-pipeline PyVista preview window.
- `--scaled-polydata-out PATH`: save the scaled intermediate mesh explicitly.

You can also call the packaged CLI directly after installation:
```bash
meshtbd-voronoi --help
```

---

## Testing
Run the regression suite with:
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

---

## Asset Policy
Large scan inputs and generated outputs are intentionally kept out of the tracked source tree.

- Put local-only demo meshes in `local_data/`
- Put generated meshes in `local_outputs/`
- Keep only tiny text fixtures in `tests/fixtures/`

The sample meshes previously kept at the repository root and in `Demo/` have been moved into ignored local directories to keep the repository focused on source code.

---

## Known Limitations
- End-to-end automation still depends on GUI-capable PyVista and Blender environments.
- The compatibility package `mesh_interlibrary_formatter` is retained for transition purposes, but `meshtbd` is now the maintained code path.
- Review artifacts under `docs/reports/` describe the earlier remediation cycle and may not reflect every later structural change.
