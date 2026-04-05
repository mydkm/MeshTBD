# MeshTBD Comprehensive Code Review Plan (Excluding `Drafts/`)

## Phase 1 Objective
Establish a complete review scope, define a rigorous rubric, and create a prioritized checklist that will drive implementation and verification in later phases.

## In-Scope Codebase Inventory

### Python source files
- `VoronoiFinal.py`
- `mesh_interlibrary_formatter/core.py`
- `mesh_interlibrary_formatter/__init__.py`
- `mesh_interlibrary_formatter/adapters/__init__.py`
- `mesh_interlibrary_formatter/adapters/pyvista_adapter.py`
- `mesh_interlibrary_formatter/adapters/open3d_adapter.py`
- `mesh_interlibrary_formatter/cli/__init__.py`
- `mesh_interlibrary_formatter/cli/scale_calibrate.py`

### Project/config/docs
- `README.md`
- `pyproject.toml`
- `requirements.txt`
- `uv.lock`
- `LICENSE`

### Binary/data assets
- `Meshes/*` and top-level `.stl/.ply` files

## Out-of-Scope for This Review
- `Drafts/**` (explicitly excluded)
- `.venv/**`
- `.git/**`
- Notebook checkpoint artifacts (`.ipynb_checkpoints/**`)

## Review Rubric (Severity + Quality Dimensions)

### Severity model
- `S0 Critical`: data corruption, unsafe destructive behavior, invalid outputs shipped silently, or hard runtime crash on normal path
- `S1 High`: likely incorrect geometry/result in key workflow, serious reliability issue, or blocking UX failure
- `S2 Medium`: correctness edge case, maintainability issue with moderate regression risk, significant performance inefficiency
- `S3 Low`: clarity/docs/tooling quality issues and minor inconsistencies

### Review dimensions
1. Correctness and behavior parity
- Output geometry integrity, color/attribute handling, and deterministic behavior under equivalent inputs

2. Reliability and error handling
- Validation of user inputs/CLI args, dependency/runtime assumptions (GUI/Blender), and actionable failure messages

3. API/CLI design and UX
- Argument consistency, defaults, help text quality, and expected command behavior

4. Performance and memory
- Expensive mesh operations, unnecessary conversions/copies, and avoidable plotting/interactive overhead

5. Dependency and packaging integrity
- Import paths, package naming consistency, lockfile/dependency coherence, and install/run reproducibility

6. Testability and regression safety
- Existing test coverage, critical-path gaps, and ability to add automated regression checks

7. Maintainability and architecture
- Module boundaries, duplication, side effects at import time, and script-vs-library separation

8. Documentation fidelity
- README and usage examples matching actual file/module names and execution model

## Phase 1 Findings That Influence Next Phases
- Active review surface is concentrated in `VoronoiFinal.py` plus `mesh_interlibrary_formatter/*`.
- `mesh_interlibrary_formatter/adapters/*.py` import `meshtbd.core`, but the present package path is `mesh_interlibrary_formatter/core.py`; this is a likely runtime import failure risk to verify in Phase 2.
- `mesh_interlibrary_formatter/cli/scale_calibrate.py` is currently empty; CLI completeness and packaging intent need explicit assessment.
- README example paths/names may not reflect current script layout and should be validated in later phases.

## Prioritized Checklist for Upcoming Phases

### Priority A (execute first)
- Validate import graph and package name consistency (`meshtbd` vs `mesh_interlibrary_formatter`).
- Verify `VoronoiFinal.py` key pipeline stages for correctness and failure modes (interactive picking, scaling, reconstruction, color projection, extraction, Blender export).
- Confirm project install/run instructions are executable as written.

### Priority B
- Establish automated baseline (lint/type/test where feasible) and capture blockers.
- Add regression checks for previously observed Voronoi/color behavior drift.
- Identify hard GUI/runtime assumptions and gate them clearly.

### Priority C
- Evaluate adapter data contracts (`MeshData` dtypes/shapes/colors/normals) across PyVista/Open3D.
- Improve docs consistency and minimal CLI/package usability.

## Phase-by-Phase Execution Mapping
- Phase 2: automated baseline + environment/repro checks
- Phase 3: deep manual review with file/line findings
- Phase 4: test-gap implementation and regression harness
- Phase 5: targeted remediation patches
- Phase 6: full validation and final audit

## Codex Review Gate (End of Phase 1)
Request to Codex:
"Review this Phase 1 plan for missing risk areas, incorrect scope assumptions, and prioritization flaws before Phase 2 begins."
