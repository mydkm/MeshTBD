from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("trimesh") is not None, "trimesh is not installed")
class TestScaleCalibrateCli(unittest.TestCase):
    def test_measurement_pair_scales_exported_geometry(self) -> None:
        import numpy as np
        import trimesh

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.ply"
            output_path = tmp_path / "output.ply"
            source = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
            source.export(input_path)
            command = [
                sys.executable,
                "-m",
                "mesh_interlibrary_formatter.cli.scale_calibrate",
                "-i",
                str(input_path),
                "-o",
                str(output_path),
                "--mesh-distance",
                "2",
                "--actual-distance",
                "10",
            ]
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertTrue(output_path.exists())
            self.assertIn("Applied uniform scale factor: 5", completed.stdout)
            result = trimesh.load(output_path, process=False)
            np.testing.assert_allclose(result.extents, source.extents * 5.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
