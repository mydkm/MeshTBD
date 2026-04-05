import os
from pathlib import Path

import numpy as np
import bpy
import pymeshlab as pml

class DualMesh:
    """
    A mesh wrapper that holds both a PyMeshLab mesh and a Blender mesh.

    - self.pml_mesh : pymeshlab.Mesh
    - self.bpy_mesh : bpy.types.Mesh
    """

    def __init__(self, name="DualMesh", pml_mesh=None, bpy_mesh=None):
        self.name = name
        self.pml_mesh = pml_mesh
        self.bpy_mesh = bpy_mesh

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pymeshlab(cls, pml_mesh, name="FromPML"):
        """Create a DualMesh from an existing pymeshlab.Mesh."""
        dm = cls(name=name, pml_mesh=pml_mesh, bpy_mesh=None)
        dm._build_bpy_from_pml()
        return dm

    @classmethod
    def from_bpy(cls, bpy_mesh, name=None):
        """Create a DualMesh from an existing bpy.types.Mesh."""
        if name is None:
            name = bpy_mesh.name
        dm = cls(name=name, pml_mesh=None, bpy_mesh=bpy_mesh)
        dm._build_pml_from_bpy()
        return dm

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _build_bpy_from_pml(self):
        """Create a Blender mesh from self.pml_mesh."""
        if self.pml_mesh is None:
            raise ValueError("pml_mesh is None, cannot build bpy mesh.")

        v = np.array(self.pml_mesh.vertex_matrix(), dtype=float)
        f = np.array(self.pml_mesh.face_matrix(), dtype=int)

        mesh = bpy.data.meshes.new(self.name + "_bpy")
        verts = [tuple(row) for row in v]
        faces = [tuple(row) for row in f]

        mesh.from_pydata(verts, [], faces)
        mesh.update()

        self.bpy_mesh = mesh

    def _build_pml_from_bpy(self):
        """Create a PyMeshLab mesh from self.bpy_mesh."""
        if self.bpy_mesh is None:
            raise ValueError("bpy_mesh is None, cannot build pymeshlab mesh.")

        mesh = self.bpy_mesh

        verts = np.array([v.co[:] for v in mesh.vertices], dtype=float)

        faces = []
        for poly in mesh.polygons:
            if len(poly.vertices) == 3:
                faces.append(poly.vertices[:])
            else:
                # Triangulate beforehand if you need non-tris
                pass

        faces = np.array(faces, dtype=int)

        pml_mesh = pml.Mesh(v_matrix=verts, f_matrix=faces)
        self.pml_mesh = pml_mesh

    # ------------------------------------------------------------------
    # Public sync methods
    # ------------------------------------------------------------------

    def sync_to_bpy(self):
        """Overwrite the Blender mesh from the current PyMeshLab mesh."""
        if self.pml_mesh is None:
            raise ValueError("No pymeshlab mesh to sync from.")

        if self.bpy_mesh is None:
            self._build_bpy_from_pml()
            return

        v = np.array(self.pml_mesh.vertex_matrix(), dtype=float)
        f = np.array(self.pml_mesh.face_matrix(), dtype=int)

        verts = [tuple(row) for row in v]
        faces = [tuple(row) for row in f]

        bm = self.bpy_mesh
        bm.clear_geometry()
        bm.from_pydata(verts, [], faces)
        bm.update()

    def sync_to_pml(self):
        """Overwrite the PyMeshLab mesh from the current Blender mesh."""
        if self.bpy_mesh is None:
            raise ValueError("No Blender mesh to sync from.")

        self._build_pml_from_bpy()

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def to_mesh_object(self, collection=None):
        """
        Wrap the internal bpy_mesh in a Blender Object and link it to a collection.
        Returns the bpy.types.Object.
        """
        if self.bpy_mesh is None:
            raise ValueError("No bpy_mesh; call sync_to_bpy() or from_pymeshlab() first.")

        obj = bpy.data.objects.new(self.name + "_obj", self.bpy_mesh)

        if collection is None:
            collection = bpy.context.scene.collection
        collection.objects.link(obj)

        return obj


# ======================================================================
# User interaction helpers
# ======================================================================

def ask_user_for_mesh_path() -> Path:
    """
    Ask the user for a mesh file path on the console and validate it.
    You can type something like: /home/you/model.obj
    """
    while True:
        raw = input("Enter path to mesh file (.obj/.ply/.stl), or leave empty to cancel:\n> ").strip()

        if raw == "":
            raise SystemExit("No file selected; exiting.")

        path = Path(os.path.expanduser(raw))

        if not path.exists():
            print(f"File '{path}' does not exist, please try again.\n")
            continue
        if not path.is_file():
            print(f"'{path}' is not a file, please try again.\n")
            continue

        return path


def load_dualmesh_from_user() -> DualMesh:
    """
    Prompt the user for a mesh file, load it via PyMeshLab, and
    build a DualMesh instance from it.
    """
    mesh_path = ask_user_for_mesh_path()
    print(f"Loading mesh from: {mesh_path}")

    ms = pml.MeshSet()
    ms.load_new_mesh(str(mesh_path))
    pml_mesh = ms.current_mesh()

    name = mesh_path.stem
    dm = DualMesh.from_pymeshlab(pml_mesh, name=name)

    # Immediately create a Blender object so it appears in the scene
    dm.to_mesh_object()

    print(f"Created DualMesh '{dm.name}' with {pml_mesh.vertex_number()} verts "
          f"and {pml_mesh.face_number()} faces.")
    return dm


def interactive_loop(dual_mesh: DualMesh):
    """
    Simple interactive loop to keep the program alive.
    The user can type anything; typing 'exit' (case-insensitive) quits.
    """
    print("\nInteractive mode started.")
    print("Type anything to continue; type 'exit' to quit.\n")

    while True:
        try:
            cmd = input("dualmesh> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting (EOF/KeyboardInterrupt).")
            break

        if cmd.lower() == "exit":
            print("Exiting on user request.")
            break

        # At this point you could hook in real commands, e.g.:
        # if cmd == "sync_to_bpy": dual_mesh.sync_to_bpy()
        # For now, we just acknowledge the input.
        print(f"You typed: {cmd}")


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    dm = load_dualmesh_from_user()
    interactive_loop(dm)
