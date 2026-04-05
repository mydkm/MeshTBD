# Phase 1 Scope and Inventory

## Summary
- In-scope Python files: `8`
- In-scope config/docs files: `5`
- In-scope asset sets: mesh binaries in `Meshes/` and top-level `.stl/.ply`
- Excluded subtree: `Drafts/`

## In-Scope Python Modules
- `VoronoiFinal.py`
- `mesh_interlibrary_formatter/core.py`
- `mesh_interlibrary_formatter/__init__.py`
- `mesh_interlibrary_formatter/adapters/__init__.py`
- `mesh_interlibrary_formatter/adapters/pyvista_adapter.py`
- `mesh_interlibrary_formatter/adapters/open3d_adapter.py`
- `mesh_interlibrary_formatter/cli/__init__.py`
- `mesh_interlibrary_formatter/cli/scale_calibrate.py`

## Notable Risks Identified Early
- Potential import path defect in adapters: `from meshtbd.core import MeshData`
- Empty CLI implementation file: `mesh_interlibrary_formatter/cli/scale_calibrate.py`
- README command examples may reference non-current script names/layout

## Handoff to Phase 2
- Verify import graph correctness with runtime/import checks
- Validate packaging metadata against actual module structure
- Run baseline automated checks and classify failures by severity
