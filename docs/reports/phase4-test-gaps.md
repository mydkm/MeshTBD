# Phase 4 Test Gap Analysis and Regression Harness

## Context
Per your clarification, `mesh_interlibrary_formatter` is intended to be a consumable package (even though incomplete today). Phase 4 therefore includes package-consumability checks in addition to script-pipeline regressions.

## Test Gaps Identified
1. No regression tests existed for package import integrity.
2. No automated guard existed for `VoronoiFinal.py` output-selection semantics.
3. No automated guard existed for the float-RGBA to uint8-RGB conversion logic.
4. No automated guard existed for import safety (module side effects).
5. No automated guard existed for CLI/docs drift in user-facing entry points.

## Regression Harness Added
- New file: `tests/test_phase4_regressions.py`
- Framework: stdlib `unittest` (chosen because current env has no `pytest` executable)

### Tests implemented
1. `test_adapters_are_importable`
- Guards consumable package importability for:
  - `mesh_interlibrary_formatter.adapters.pyvista_adapter`
  - `mesh_interlibrary_formatter.adapters.open3d_adapter`

2. `test_cli_scale_calibrate_is_not_empty_stub`
- Guards that package CLI module is not an empty placeholder.

3. `test_has_main_guard`
- Guards import safety by requiring a `if __name__ == "__main__":` entry guard in `VoronoiFinal.py`.

4. `test_threshold_uses_all_scalars_true_for_keep_mask`
- AST-based guard for threshold semantics matching intended all-vertex selection behavior.

5. `test_rgba_to_rgb_conversion_scales_float_colors`
- AST-based guard ensuring color conversion includes scale-by-255 before `uint8` cast.

6. `test_readme_run_example_references_existing_script`
- Guards README against stale/non-existent script references.

## Baseline Execution
Command:
- `.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v`

Result:
- `Ran 6 tests`
- `FAILED (failures=5, errors=1)`

Failure/Error mapping:
- ERROR: adapter importability test (`ModuleNotFoundError: meshtbd`)
- FAIL: empty CLI stub
- FAIL: missing main guard in `VoronoiFinal.py`
- FAIL: RGBA->RGB scaling guard
- FAIL: threshold `all_scalars=True` semantic guard
- FAIL: README stale script reference guard

## Interpretation
- The suite is intentionally red at Phase 4 to capture confirmed defects from Phases 2-3.
- These tests now provide concrete regression targets for Phase 5 remediation.

## Handoff to Phase 5
Primary goal: implement fixes until this suite is green (or explicitly justified where deferred).

## Codex Review Gate (End of Phase 4)
Request to Codex:
"Review the Phase 4 test harness for coverage quality, brittleness, and whether these tests correctly encode the intended behavior before remediation begins."
