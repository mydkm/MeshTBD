from __future__ import annotations

from pathlib import Path


def create_object_from_triangle_mesh(vertices, faces, object_name: str = "pymlMesh"):
    import bpy
    import bmesh

    mesh = bpy.data.meshes.new(object_name)
    bm = bmesh.new()

    for vertex in vertices:
        bm.verts.new(vertex)
    bm.verts.ensure_lookup_table()

    for tri in faces:
        try:
            bm.faces.new([bm.verts[i] for i in tri])
        except ValueError:
            pass

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(object_name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def apply_cast_modifiers(
    obj,
    *,
    displacement_strength: float = 1.5,
    displacement_mid_level: float = 0.5,
    solidify_thickness: float = 2.5,
) -> None:
    import bpy

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    displace = obj.modifiers.new(name="Displace", type="DISPLACE")
    displace.strength = displacement_strength
    displace.mid_level = displacement_mid_level
    displace.direction = "NORMAL"
    displace.space = "LOCAL"

    solid = obj.modifiers.new(name="Solidify", type="SOLIDIFY")
    solid.thickness = solidify_thickness
    solid.offset = 1.0
    solid.use_even_offset = True

    bpy.ops.object.modifier_apply(modifier=displace.name)
    bpy.ops.object.modifier_apply(modifier=solid.name)
    print("Mesh thickened!")


def smooth_hole_boundaries(
    obj,
    *,
    growth_steps: int = 3,
    factor: float = 0.8,
) -> int:
    import bmesh

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    boundary_verts = set()
    for edge in bm.edges:
        if len(edge.link_faces) == 1:
            boundary_verts.add(edge.verts[0])
            boundary_verts.add(edge.verts[1])

    def grow_region(verts, n: int = 2):
        selected = set(verts)
        for _ in range(n):
            new_selected = set(selected)
            for vertex in selected:
                for edge in vertex.link_edges:
                    new_selected.add(edge.other_vert(vertex))
            selected = new_selected
        return selected

    smooth_region = grow_region(boundary_verts, n=growth_steps)
    bmesh.ops.smooth_vert(
        bm,
        verts=list(smooth_region),
        factor=factor,
        use_axis_x=True,
        use_axis_y=True,
        use_axis_z=True,
    )

    bm.to_mesh(mesh)
    bm.free()
    print(f"Localized smoothing applied to {len(smooth_region)} vertices near holes!")
    return len(smooth_region)


def export_object(output_path: Path) -> None:
    import bpy

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()
    if ext == ".ply":
        bpy.ops.wm.ply_export(
            filepath=str(output_path),
            export_selected_objects=False,
            export_normals=True,
            export_uv=True,
            global_scale=1.0,
            forward_axis="Y",
            up_axis="Z",
        )
        return

    if ext == ".stl":
        bpy.ops.wm.stl_export(
            filepath=str(output_path),
            export_selected_objects=False,
            global_scale=1.0,
            forward_axis="Y",
            up_axis="Z",
        )
        return

    raise ValueError(f"Unsupported output format: {output_path.suffix}")
