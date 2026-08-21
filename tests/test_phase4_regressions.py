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
            operands = (child.left, child.right)
            if any(isinstance(value, ast.Constant) and value.value in (255, 255.0) for value in operands):
                return True
    return False


class TestPackagingRegression(unittest.TestCase):
    def test_public_package_exports_all_five_adapter_families(self) -> None:
        package = importlib.import_module("mesh_interlibrary_formatter")
        exported_names = set(package.__all__)
        required = {
            "from_pyvista",
            "from_open3d_triangle_mesh",
            "from_trimesh",
            "from_pymeshlab",
            "from_bpy_object",
        }
        self.assertTrue(required.issubset(exported_names), required - exported_names)

    def test_cli_scale_calibrate_is_not_empty_stub(self) -> None:
        self.assertGreater(len(CLI_SCALE.read_text(encoding="utf-8").strip()), 0)


class TestVoronoiFinalRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = VORONOI_FINAL.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_has_main_guard(self) -> None:
        self.assertIn('if __name__ == "__main__":', self.source)

    def test_scaled_polydata_target_is_written(self) -> None:
        save_calls = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "save_current_mesh" and node.args:
                    save_calls.append(ast.unparse(node.args[0]))
        self.assertIn("str(scaled_polydata_out)", save_calls)

    def test_threshold_uses_all_scalars_true_for_selection_mask(self) -> None:
        threshold_calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "threshold"
        ]
        selection_call = None
        for call in threshold_calls:
            keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
            scalar = keywords.get("scalars")
            if isinstance(scalar, ast.Constant) and scalar.value == "selection_mask":
                selection_call = call
                break
        self.assertIsNotNone(selection_call)
        keywords = {kw.arg: kw.value for kw in selection_call.keywords if kw.arg is not None}
        all_scalars = keywords.get("all_scalars")
        self.assertIsInstance(all_scalars, ast.Constant)
        self.assertIs(all_scalars.value, True)

    def test_rgba_to_rgb_conversion_scales_float_colors(self) -> None:
        rgb_assignments = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "rgb"
        ]
        self.assertTrue(any(_contains_mul_255(assign.value) for assign in rgb_assignments))

    def test_active_pipeline_contains_meshdata_bridge(self) -> None:
        self.assertIn("from mesh_interlibrary_formatter import from_pymeshlab, to_pyvista", self.source)
        self.assertIn("projected_meshdata = from_pymeshlab(csurface)", self.source)
        self.assertIn("cmesh = to_pyvista(projected_meshdata)", self.source)


class TestDocumentationRegression(unittest.TestCase):
    def test_readme_describes_bounded_meshdata_integration(self) -> None:
        readme = " ".join(README.read_text(encoding="utf-8").split())
        self.assertIn("One active handoff now passes", readme)
        self.assertNotIn("voronoi_cast.py", readme)


if __name__ == "__main__":
    unittest.main()
