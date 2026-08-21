from __future__ import annotations

import unittest

import numpy as np

from mesh_interlibrary_formatter import MeshData, calibrate_mesh_scale, compute_uniform_scale


def tetra_mesh(*, colors=None) -> MeshData:
    return MeshData(
        V=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        F=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64),
        VN=np.array(
            [[-0.577, -0.577, -0.577], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=np.float64,
        ),
        FN=None,
        C=colors,
    )


class TestMeshDataValidation(unittest.TestCase):
    def test_uint8_colors_are_normalized(self) -> None:
        colors = np.array(
            [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]],
            dtype=np.uint8,
        )
        mesh = tetra_mesh(colors=colors)
        self.assertEqual(mesh.C.dtype, np.float32)
        np.testing.assert_allclose(mesh.C[0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(mesh.C[3], [1.0, 1.0, 1.0])

    def test_fractional_face_indices_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-integer"):
            MeshData(V=np.eye(3), F=np.array([[0.0, 1.5, 2.0]]), VN=None, FN=None, C=None)

    def test_nonfinite_geometry_is_rejected(self) -> None:
        vertices = np.eye(3)
        vertices[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            MeshData(V=vertices, F=np.array([[0, 1, 2]]), VN=None, FN=None, C=None)

    def test_point_cloud_contract(self) -> None:
        cloud = MeshData(V=np.eye(3), F=None, VN=None, FN=None, C=None)
        self.assertTrue(cloud.is_point_cloud())
        self.assertEqual(cloud.n_vertices(), 3)
        self.assertEqual(cloud.n_faces(), 0)


class TestScaleCalibration(unittest.TestCase):
    def test_compute_uniform_scale(self) -> None:
        self.assertAlmostEqual(compute_uniform_scale(40.0, 100.0), 2.5)

    def test_calibration_scales_copy_without_mutating_input(self) -> None:
        source = tetra_mesh()
        scaled = calibrate_mesh_scale(source, mesh_distance=2.0, actual_distance=10.0)
        np.testing.assert_allclose(scaled.V, source.V * 5.0)
        np.testing.assert_allclose(source.V[1], [1.0, 0.0, 0.0])

    def test_invalid_measurements_are_rejected(self) -> None:
        for mesh_distance, actual_distance in ((0, 1), (1, 0), (np.inf, 1), (1, np.nan)):
            with self.subTest(mesh_distance=mesh_distance, actual_distance=actual_distance):
                with self.assertRaises(ValueError):
                    compute_uniform_scale(mesh_distance, actual_distance)


if __name__ == "__main__":
    unittest.main()
