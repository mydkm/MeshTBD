# Phase 2 Automated Baseline Report

## Scope and Method
- Scope: non-`Drafts/` code only.
- Goal: establish executable baseline for static checks, import/runtime smoke tests, and packaging/docs consistency.

## Commands Run and Outcomes
1. `.venv/bin/python --version`
- Result: `Python 3.11.13` (pass)

2. `.venv/bin/python -m py_compile <all in-scope .py files>`
- Result: pass

3. `.venv/bin/ty check VoronoiFinal.py mesh_interlibrary_formatter`
- Result: fail (`21 diagnostics`)

4. `.venv/bin/python - <<... import smoke ...>>`
- Result:
  - `mesh_interlibrary_formatter` imports: base package/core pass
  - adapter imports fail (`ModuleNotFoundError: No module named 'meshtbd'`)

5. `.venv/bin/python VoronoiFinal.py --help`
- Result: pass (CLI help renders)

6. `.venv/bin/python -m mesh_interlibrary_formatter.cli.scale_calibrate --help`
- Result: exits with no output (module is empty)

7. Import-order/native dependency smoke:
- `import bpy; import bmesh` => pass
- `import pymeshlab; import bpy` => fail (`undefined symbol: rtcGetSceneTraversable`)

8. Test tool presence:
- `.venv/bin/pytest` missing
- no in-scope `test*.py` files found

## Findings (Ordered by Severity)

### S1 High
1. Adapter modules are not importable due incorrect package path
- Evidence:
  - `mesh_interlibrary_formatter/adapters/pyvista_adapter.py:6`
  - `mesh_interlibrary_formatter/adapters/open3d_adapter.py:6`
  - both use `from meshtbd.core import MeshData`, but in-repo module is `mesh_interlibrary_formatter/core.py`.
- Runtime effect:
  - direct import fails (`ModuleNotFoundError`) and blocks adapter usage.

### S2 Medium
1. Native dependency conflict risk: `pymeshlab` import can break later `bpy` import in same process
- Evidence:
  - smoke test reproducible failure sequence: `import pymeshlab` then `import bpy` => `ImportError` (`rtcGetSceneTraversable`).
- Notes:
  - current `VoronoiFinal.py` import order (`bpy` before `pymeshlab`) avoids immediate crash, but the process is order-sensitive.

2. No runnable automated tests in repo baseline
- Evidence:
  - `pytest` executable absent in `.venv/bin`.
  - no in-scope test files found.
- Effect:
  - no regression safety net for geometry/color pipeline changes.

3. CLI module declared by layout but not implemented
- Evidence:
  - `mesh_interlibrary_formatter/cli/scale_calibrate.py` size `0`.
  - executing as module yields no command behavior.

4. README run example does not match current repository layout
- Evidence:
  - `README.md:79` references `MeshTBD/voronoi_cast.py`.
  - file does not exist in current repo; active script is `VoronoiFinal.py`.

### S3 Low
1. Static typing setup is currently noisy for dynamic C-extension APIs
- Evidence:
  - `ty` reports unresolved attributes for `bpy`/`pymeshlab` dynamic symbols and VTK interactor optionality in `VoronoiFinal.py`.
- Effect:
  - lower signal-to-noise in static analysis until tool config/typing strategy is tightened.

2. `requirements.txt` appears out-of-sync with `pyproject.toml`
- Evidence:
  - `requirements.txt` includes limited set and omits several direct dependencies listed in `pyproject.toml`.
- Effect:
  - potential environment drift for users using pip-based setup path.

## Baseline Conclusion
- Core script syntax compiles, but automated baseline is **not green**.
- The highest-priority blocker for library usage is adapter import path correctness.
- Reliability/test posture is currently weak due missing automated test harness.

## Handoff to Phase 3
- Perform deep manual review with line-level findings focused on:
  - `VoronoiFinal.py` correctness and failure-path robustness.
  - adapter contract correctness and packaging boundaries.
  - docs/config alignment with executable paths and supported runtime model.

## Codex Review Gate (End of Phase 2)
Request to Codex:
"Review the Phase 2 baseline findings for false positives, missing critical checks, and severity misclassification before Phase 3 begins."
