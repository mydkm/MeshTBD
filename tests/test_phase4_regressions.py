from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meshtbd.calibration import compute_scale_factor, default_scaled_polydata_path
from meshtbd.cli.voronoi_cast import build_config, build_parser, default_output_path
from meshtbd.ops.pyvista_ops import compute_red_like_mask, rgba_float_to_rgb_uint8


VORONOI_FINAL = ROOT / "VoronoiFinal.py"
README = ROOT / "README.md"
FIXTURE = ROOT / "tests" / "fixtures" / "minimal_triangle.ply"


class TestPackageLayout(unittest.TestCase):
    def test_new_package_is_importable(self) -> None:
        importlib.import_module("meshtbd")
        importlib.import_module("meshtbd.cli.voronoi_cast")
        importlib.import_module("meshtbd.pipeline.cast_pipeline")

    def test_compatibility_package_is_importable(self) -> None:
        importlib.import_module("mesh_interlibrary_formatter.adapters.pyvista_adapter")
        importlib.import_module("mesh_interlibrary_formatter.adapters.open3d_adapter")
        importlib.import_module("mesh_interlibrary_formatter.cli.scale_calibrate")

    def test_readme_describes_src_layout(self) -> None:
        readme = README.read_text(encoding="utf-8")
        self.assertIn("src/meshtbd/", readme)
        self.assertIn("local_data/", readme)
        self.assertIn("tests/", readme)


class TestVoronoiWrapper(unittest.TestCase):
    def test_root_wrapper_targets_packaged_cli(self) -> None:
        source = VORONOI_FINAL.read_text(encoding="utf-8")
        self.assertIn("from meshtbd.cli.voronoi_cast import main", source)
        self.assertIn('if __name__ == "__main__":', source)


class TestCliConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = build_parser()

    def test_default_paths_and_interactive_mode(self) -> None:
        args = self.parser.parse_args(["-i", str(FIXTURE)])
        config = build_config(args, self.parser)

        self.assertEqual(config.input_path, FIXTURE)
        self.assertIsNone(config.calibration.landmark_vertices)
        self.assertEqual(config.export.output_path, default_output_path(FIXTURE))
        self.assertEqual(config.export.scaled_polydata_out, default_scaled_polydata_path(FIXTURE))
        self.assertTrue(config.export.preview_color_mesh)

    def test_noninteractive_vertex_mode(self) -> None:
        args = self.parser.parse_args(
            [
                "-i",
                str(FIXTURE),
                "--landmark-vertices",
                "1",
                "2",
                "--real-mm",
                "123.5",
                "--no-color-preview",
            ]
        )
        config = build_config(args, self.parser)

        self.assertEqual(config.calibration.landmark_vertices, (1, 2))
        self.assertEqual(config.calibration.real_world_distance_mm, 123.5)
        self.assertFalse(config.export.preview_color_mesh)

    def test_conflicting_input_paths_raise_parser_error(self) -> None:
        with self.assertRaises(SystemExit):
            args = self.parser.parse_args(["fixture_a.ply", "--input", "fixture_b.ply"])
            build_config(args, self.parser)


class TestPurePipelineHelpers(unittest.TestCase):
    def test_compute_scale_factor(self) -> None:
        self.assertAlmostEqual(compute_scale_factor(50.0, 125.0), 2.5)

    def test_rgba_float_to_rgb_uint8(self) -> None:
        rgb = rgba_float_to_rgb_uint8(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.5, 0.25, 0.0, 1.0],
            ]
        )
        self.assertEqual([list(row) for row in rgb], [[255, 0, 0], [128, 64, 0]])

    def test_compute_red_like_mask(self) -> None:
        mask = compute_red_like_mask(
            [
                [255, 20, 20],
                [255, 180, 0],
                [20, 20, 255],
            ]
        )
        self.assertEqual([bool(value) for value in mask], [True, True, False])


@unittest.skipUnless(importlib.util.find_spec("pyvista"), "pyvista is not installed")
class TestOptionalFixtureSmoke(unittest.TestCase):
    def test_fixture_mesh_loads(self) -> None:
        from meshtbd.calibration import load_surface_mesh

        mesh = load_surface_mesh(FIXTURE)
        self.assertEqual(mesh.n_points, 3)
        self.assertEqual(mesh.n_cells, 1)


if __name__ == "__main__":
    unittest.main()
