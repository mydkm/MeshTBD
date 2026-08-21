# MeshData validation evidence

This evidence supports the repository's interoperability claims without
implying that the entire cast pipeline has been rewritten around `MeshData`.

## Active integration

`VoronoiFinal.py` now uses one bounded production bridge during Voronoi color
selection:

```text
PyMeshLab colored surface -> MeshData -> PyVista PolyData
```

The bridge preserves triangle topology and normalized vertex colors before the
existing PyVista HSV selection stage. The remaining pipeline stages continue to
use their library-native representations.

## Validation matrix

| Path | Automated evidence | Current status |
|---|---|---|
| Core `MeshData` | shapes, indices, finite values, point clouds, color normalization | Pass |
| Measurement calibration | scale calculation, immutable scaled copy, invalid inputs | Pass |
| Scaling CLI | measured-distance calculation and exported geometry dimensions | Pass |
| PyVista round trip | vertices, triangle topology, stored normals, colors | Pass |
| PyMeshLab adapter contract | vertices, triangle topology, normals, normalized RGBA colors | Pass |
| trimesh round trip | vertices, triangle topology, normals, normalized colors | Pass |
| Blender inbound | polygon triangulation and matching face normals | Pass |
| Active PyMeshLab -> MeshData -> PyVista bridge contract | topology and colors across the combined handoff | Pass |
| Open3D round trip | public adapter surface and opt-in native test | Runtime retest required |

The Open3D 0.19 and PyMeshLab 2025.7 wheels terminate with CPU-level bus errors
in the current container before their adapter code executes. Run the opt-in
native tests on a compatible workstation before describing those round trips—or
the full active bridge—as runtime-validated.

## Reproduction

```bash
uv sync --group dev
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

On a compatible Open3D runtime:

```bash
MESHTBD_TEST_OPEN3D=1 uv run python -m unittest \
  tests.test_adapter_roundtrips.TestAdapterRoundTrips.test_open3d_round_trip -v
```

```bash
MESHTBD_TEST_PYMESHLAB=1 uv run python -m unittest \
  tests.test_adapter_roundtrips.TestAdapterRoundTrips.test_pymeshlab_round_trip -v
```

The poster-ready vector diagram is
[`docs/figures/meshdata_bridge.svg`](figures/meshdata_bridge.svg).
