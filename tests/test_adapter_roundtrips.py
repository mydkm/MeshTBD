from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mesh_interlibrary_formatter import MeshData


def adapter_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def sample_mesh() -> MeshData:
    return MeshData(
        V=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        F=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32),
        VN=np.array(
            [[-0.577, -0.577, -0.577], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=np.float32,
        ),
        FN=None,
        C=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float32),
    )


def load_pymeshlab_adapter_with_fake_runtime():
    class Mesh:
        def __init__(
            self,
            vertex_matrix=None,
            face_matrix=None,
            v_normals_matrix=None,
            f_normals_matrix=None,
            v_color_matrix=None,
            **_kwargs,
        ):
            self._vertices = np.asarray(vertex_matrix if vertex_matrix is not None else [], dtype=np.float64)
            self._faces = np.asarray(face_matrix if face_matrix is not None else [], dtype=np.int32)
            self._vertex_normals = np.asarray(
                v_normals_matrix if v_normals_matrix is not None else [], dtype=np.float64
            )
            self._face_normals = np.asarray(
                f_normals_matrix if f_normals_matrix is not None else [], dtype=np.float64
            )
            self._colors = np.asarray(v_color_matrix if v_color_matrix is not None else [], dtype=np.float64)

        def vertex_matrix(self):
            return self._vertices

        def face_matrix(self):
            return self._faces

        def vertex_normal_matrix(self):
            return self._vertex_normals

        def face_normal_matrix(self):
            return self._face_normals

        def vertex_color_matrix(self):
            return self._colors

    class MeshSet:
        def __init__(self):
            self._mesh = None

        def add_mesh(self, mesh, _name):
            self._mesh = mesh

        def current_mesh(self):
            return self._mesh

    fake_pymeshlab = SimpleNamespace(Mesh=Mesh, MeshSet=MeshSet)
    adapter_path = (
        Path(__file__).resolve().parents[1]
        / "mesh_interlibrary_formatter"
        / "adapters"
        / "pymeshlab_adapter.py"
    )
    spec = importlib.util.spec_from_file_location("_pymeshlab_adapter_contract_test", adapter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load PyMeshLab adapter contract test")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"pymeshlab": fake_pymeshlab}):
        spec.loader.exec_module(module)
    return module


def assert_topology_and_colors(testcase: unittest.TestCase, expected: MeshData, actual: MeshData) -> None:
    np.testing.assert_allclose(actual.V, expected.V, atol=1e-6)
    np.testing.assert_array_equal(actual.F, expected.F)
    testcase.assertIsNotNone(actual.C)
    np.testing.assert_allclose(actual.C[:, :3], expected.C[:, :3], atol=1 / 255 + 1e-6)


class TestAdapterRoundTrips(unittest.TestCase):
    @unittest.skipUnless(
        adapter_available("pymeshlab")
        and adapter_available("pyvista")
        and os.environ.get("MESHTBD_TEST_PYMESHLAB") == "1",
        "set MESHTBD_TEST_PYMESHLAB=1 on a CPU-compatible PyMeshLab runtime",
    )
    def test_active_pymeshlab_to_pyvista_native_bridge(self) -> None:
        from mesh_interlibrary_formatter.adapters.pymeshlab_adapter import from_pymeshlab, to_pymeshlab_mesh
        from mesh_interlibrary_formatter.adapters.pyvista_adapter import from_pyvista, to_pyvista

        source = sample_mesh()
        bridged = from_pymeshlab(to_pymeshlab_mesh(source))
        result = from_pyvista(to_pyvista(bridged))
        assert_topology_and_colors(self, source, result)

    @unittest.skipUnless(adapter_available("pyvista"), "PyVista is not installed")
    def test_active_pymeshlab_to_pyvista_bridge_contract(self) -> None:
        from mesh_interlibrary_formatter.adapters.pyvista_adapter import from_pyvista, to_pyvista

        pymeshlab_adapter = load_pymeshlab_adapter_with_fake_runtime()
        source = sample_mesh()
        bridged = pymeshlab_adapter.from_pymeshlab(pymeshlab_adapter.to_pymeshlab_mesh(source))
        result = from_pyvista(to_pyvista(bridged))
        assert_topology_and_colors(self, source, result)

    @unittest.skipUnless(adapter_available("trimesh"), "trimesh is not installed")
    def test_trimesh_round_trip(self) -> None:
        from mesh_interlibrary_formatter.adapters.trimesh_adapter import from_trimesh, to_trimesh

        source = sample_mesh()
        result = from_trimesh(to_trimesh(source))
        assert_topology_and_colors(self, source, result)
        self.assertIsNotNone(result.VN)
        self.assertEqual(result.VN.shape, source.VN.shape)

    @unittest.skipUnless(adapter_available("pyvista"), "PyVista is not installed")
    def test_pyvista_round_trip(self) -> None:
        from mesh_interlibrary_formatter.adapters.pyvista_adapter import from_pyvista, to_pyvista

        source = sample_mesh()
        result = from_pyvista(to_pyvista(source))
        assert_topology_and_colors(self, source, result)
        np.testing.assert_allclose(result.VN, source.VN, atol=1e-6)

    @unittest.skipUnless(
        adapter_available("open3d") and os.environ.get("MESHTBD_TEST_OPEN3D") == "1",
        "set MESHTBD_TEST_OPEN3D=1 on a CPU-compatible Open3D runtime",
    )
    def test_open3d_round_trip(self) -> None:
        from mesh_interlibrary_formatter.adapters.open3d_adapter import (
            from_open3d_triangle_mesh,
            to_open3d_triangle_mesh,
        )

        source = sample_mesh()
        result = from_open3d_triangle_mesh(to_open3d_triangle_mesh(source))
        assert_topology_and_colors(self, source, result)
        np.testing.assert_allclose(result.VN, source.VN, atol=1e-6)

    def test_open3d_adapter_contract_without_native_runtime(self) -> None:
        class TriangleMesh:
            def __init__(self):
                self.vertices = []
                self.triangles = []
                self.vertex_normals = []
                self.triangle_normals = []
                self.vertex_colors = []

            def has_vertex_normals(self):
                return len(self.vertex_normals) > 0

            def has_triangle_normals(self):
                return len(self.triangle_normals) > 0

            def has_vertex_colors(self):
                return len(self.vertex_colors) > 0

        class PointCloud:
            def __init__(self):
                self.points = []
                self.normals = []
                self.colors = []

            def has_normals(self):
                return len(self.normals) > 0

            def has_colors(self):
                return len(self.colors) > 0

        fake_open3d = SimpleNamespace(
            geometry=SimpleNamespace(TriangleMesh=TriangleMesh, PointCloud=PointCloud),
            utility=SimpleNamespace(
                Vector3dVector=lambda value: np.asarray(value, dtype=np.float64),
                Vector3iVector=lambda value: np.asarray(value, dtype=np.int32),
            ),
            io=SimpleNamespace(),
        )
        adapter_path = Path(__file__).resolve().parents[1] / "mesh_interlibrary_formatter" / "adapters" / "open3d_adapter.py"
        spec = importlib.util.spec_from_file_location("_open3d_adapter_contract_test", adapter_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"open3d": fake_open3d}):
            spec.loader.exec_module(module)

        source = sample_mesh()
        result = module.from_open3d_triangle_mesh(module.to_open3d_triangle_mesh(source))
        assert_topology_and_colors(self, source, result)
        np.testing.assert_allclose(result.VN, source.VN, atol=1e-6)

    @unittest.skipUnless(
        adapter_available("pymeshlab") and os.environ.get("MESHTBD_TEST_PYMESHLAB") == "1",
        "set MESHTBD_TEST_PYMESHLAB=1 on a CPU-compatible PyMeshLab runtime",
    )
    def test_pymeshlab_round_trip(self) -> None:
        from mesh_interlibrary_formatter.adapters.pymeshlab_adapter import from_pymeshlab, to_pymeshlab_mesh

        source = sample_mesh()
        result = from_pymeshlab(to_pymeshlab_mesh(source))
        assert_topology_and_colors(self, source, result)

    def test_blender_inbound_adapter_triangulates_faces(self) -> None:
        from mesh_interlibrary_formatter.adapters.bpy_adapter import from_bpy_object

        class Vertex:
            def __init__(self, coordinate, normal):
                self.co = np.asarray(coordinate, dtype=np.float32)
                self.normal = np.asarray(normal, dtype=np.float32)

        class Polygon:
            def __init__(self, vertices, normal):
                self.vertices = vertices
                self.normal = np.asarray(normal, dtype=np.float32)

        class Mesh:
            vertices = [
                Vertex([0, 0, 0], [0, 0, 1]),
                Vertex([1, 0, 0], [0, 0, 1]),
                Vertex([1, 1, 0], [0, 0, 1]),
                Vertex([0, 1, 0], [0, 0, 1]),
            ]
            polygons = [Polygon([0, 1, 2, 3], [0, 0, 1])]

        class Object:
            type = "MESH"
            data = Mesh()

        result = from_bpy_object(Object())
        np.testing.assert_array_equal(result.F, [[0, 1, 2], [0, 2, 3]])
        self.assertEqual(result.FN.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
