from __future__ import annotations

import ast
import importlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
VORONOI_FINAL = ROOT / "VoronoiFinal.py"
README = ROOT / "README.md"
CLI_SCALE = ROOT / "mesh_interlibrary_formatter" / "cli" / "scale_calibrate.py"


def _contains_mul_255(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.Mult):
            lhs = child.left
            rhs = child.right
            if isinstance(lhs, ast.Constant) and lhs.value in (255, 255.0):
                return True
            if isinstance(rhs, ast.Constant) and rhs.value in (255, 255.0):
                return True
    return False


class TestPackagingRegression(unittest.TestCase):
    def test_adapters_are_importable(self) -> None:
        importlib.import_module("mesh_interlibrary_formatter.adapters.pyvista_adapter")
        importlib.import_module("mesh_interlibrary_formatter.adapters.open3d_adapter")

    def test_cli_scale_calibrate_is_not_empty_stub(self) -> None:
        content = CLI_SCALE.read_text(encoding="utf-8")
        self.assertGreater(len(content.strip()), 0, "scale_calibrate.py is an empty stub")


class TestVoronoiFinalRegression(unittest.TestCase):
    def test_has_main_guard(self) -> None:
        source = VORONOI_FINAL.read_text(encoding="utf-8")
        self.assertIn(
            'if __name__ == "__main__":',
            source,
            "VoronoiFinal.py should be import-safe for testing/reuse",
        )

    def test_threshold_uses_all_scalars_true_for_keep_mask(self) -> None:
        source = VORONOI_FINAL.read_text(encoding="utf-8")
        tree = ast.parse(source)

        threshold_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "threshold":
                continue
            threshold_calls.append(node)

        self.assertGreater(len(threshold_calls), 0, "No threshold() call found")

        keep_call = None
        for call in threshold_calls:
            for kw in call.keywords:
                if kw.arg == "scalars" and isinstance(kw.value, ast.Constant) and kw.value.value == "keep":
                    keep_call = call
                    break
            if keep_call is not None:
                break

        self.assertIsNotNone(keep_call, "No threshold call found using scalars='keep'")

        all_scalars_kw = None
        for kw in keep_call.keywords:
            if kw.arg == "all_scalars":
                all_scalars_kw = kw
                break

        self.assertIsNotNone(all_scalars_kw, "threshold(..., scalars='keep') must set all_scalars")
        self.assertIsInstance(all_scalars_kw.value, ast.Constant)
        self.assertTrue(
            all_scalars_kw.value.value is True,
            "Expected all_scalars=True to require all cell vertices pass the keep mask",
        )

    def test_rgba_to_rgb_conversion_scales_float_colors(self) -> None:
        source = VORONOI_FINAL.read_text(encoding="utf-8")
        tree = ast.parse(source)

        rgb_assignments = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "rgb":
                rgb_assignments.append(node)

        self.assertGreater(len(rgb_assignments), 0, "No assignment to rgb found")

        has_scaled_conversion = any(_contains_mul_255(assign.value) for assign in rgb_assignments)
        self.assertTrue(
            has_scaled_conversion,
            "Expected RGBA float colors to be scaled by 255 before uint8 conversion",
        )


class TestDocumentationRegression(unittest.TestCase):
    def test_readme_run_example_references_existing_script(self) -> None:
        readme = README.read_text(encoding="utf-8")
        self.assertNotIn(
            "voronoi_cast.py",
            readme,
            "README references a non-existent voronoi_cast.py script",
        )


if __name__ == "__main__":
    unittest.main()
