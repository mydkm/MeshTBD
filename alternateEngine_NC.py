import datoviz as dvz

OBJ = "/home/mydkm/MeshTBD/SabrinaOutput.obj"

app = dvz.App()
fig = app.figure(gui=True)
panel = fig.panel(background=True)

# Give the panel a controller & a reasonable starting camera.
panel.arcball(initial=(0.0, 0.0, 0.0))
panel.camera(initial=(0.0, 0.0, 3.0))   # camera 3 units back

# Load the OBJ into a ShapeCollection. Adjust scale if the model is huge/small.
sc = dvz.ShapeCollection()
sc.add_obj(OBJ, scale=1.0)  # try 0.01 or 100.0 if it still isn't visible

# Create the mesh visual and attach it to the panel.
visual = app.mesh(sc, lighting=True)
panel.add(visual)

# Optional helpers:
panel.gizmo()          # orientation widget
# visual.culling(False)  # if faces are wound the opposite way and appear invisible

app.run()
app.destroy()
sc.destroy()
