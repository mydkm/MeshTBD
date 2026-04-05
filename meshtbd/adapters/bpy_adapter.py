from __future__ import annotations

import numpy as np

from meshtbd.core import MeshData


def from_bpy_object(obj) -> MeshData:
    """
    Convert a Blender mesh object (bpy.types.Object) into MeshData.

    Notes:
    - Assumes obj is a mesh object.
    - Non-triangular faces are triangulated in the exported MeshData
      using a simple fan triangulation.
    """
    if obj.type != "MESH":
        raise TypeError(f"Expected a Blender mesh object, got type: {obj.type}")

    mesh = obj.data

    V = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)

    faces = []
    for poly in mesh.polygons:
        verts = list(poly.vertices)

        if len(verts) == 3:
            faces.append(verts)
        elif len(verts) > 3:
            # Fan triangulation: [v0,v1,v2], [v0,v2,v3], ...
            for i in range(1, len(verts) - 1):
                faces.append([verts[0], verts[i], verts[i + 1]])

    F = np.asarray(faces, dtype=np.int32) if faces else None

    VN = None
    try:
        if len(mesh.vertices) > 0:
            VN = np.array([v.normal[:] for v in mesh.vertices], dtype=np.float32)
    except Exception:
        VN = None

    FN = None
    try:
        if F is not None and len(mesh.polygons) > 0:
            poly_normals = [np.array(poly.normal[:], dtype=np.float32) for poly in mesh.polygons]

            # Need face normals to match triangulated faces
            triangulated_normals = []
            for poly, n in zip(mesh.polygons, poly_normals):
                verts = list(poly.vertices)
                if len(verts) == 3:
                    triangulated_normals.append(n)
                elif len(verts) > 3:
                    for i in range(1, len(verts) - 1):
                        triangulated_normals.append(n)

            if len(triangulated_normals) == len(F):
                FN = np.asarray(triangulated_normals, dtype=np.float32)
    except Exception:
        FN = None

    C = None
    # Minimal v1: skip Blender color attributes unless you know you need them

    return MeshData(V=V, F=F, VN=VN, FN=FN, C=C)


def to_bpy_object(meshdata: MeshData, bpy, name: str = "MILF_Mesh"):
    """
    Convert MeshData into a Blender mesh object.

    Parameters
    ----------
    meshdata : MeshData
        MILF mesh representation
    bpy : module
        Blender bpy module, passed in explicitly
    name : str
        Name for the created mesh/object

    Returns
    -------
    obj : bpy.types.Object
        Newly created Blender object
    """
    verts = np.asarray(meshdata.V, dtype=np.float32).tolist()
    faces = [] if meshdata.F is None else np.asarray(meshdata.F, dtype=np.int32).tolist()

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)

    # Optional: assign vertex normals if present
    # Keeping this minimal for v1, since Blender normal assignment can be finicky.
    return obj


def to_bpy_object_linked(meshdata: MeshData, bpy, name: str = "MILF_Mesh", collection=None):
    """
    Convert MeshData into a Blender object and link it to a collection.

    If collection is None, links to bpy.context.scene.collection.
    """
    obj = to_bpy_object(meshdata, bpy=bpy, name=name)

    if collection is None:
        collection = bpy.context.scene.collection

    collection.objects.link(obj)
    return obj


def from_bpy_mesh_data(mesh) -> MeshData:
    """
    Convert a Blender mesh datablock (bpy.types.Mesh) into MeshData.
    Useful when you already have obj.data.
    """
    class _TempObj:
        type = "MESH"

        def __init__(self, mesh):
            self.data = mesh

    return from_bpy_object(_TempObj(mesh))