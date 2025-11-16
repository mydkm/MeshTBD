import datoviz as dvz
import numpy as np

OBJ = "/home/mydkm/MeshTBD/SabrinaOutput.obj"

# Load the OBJ into a ShapeCollection. Adjust scale if the model is huge/small.
sc = dvz.ShapeCollection()
sc.add_obj(OBJ, scale=1.0) 

app = dvz.App()
fig = app.figure(gui=True)
panel = fig.panel(background=True)

arcball = panel.arcball(initial=(0.0, 0.0, 0.0))
panel.camera(initial=(0.0, 0.0, 3.0))

# Create the mesh visual and attach it to the panel.
visual = app.mesh(
    sc, 
    indexed=True, 
    lighting=True
    )
panel.add(visual)

# Optional helpers:
panel.gizmo()          # orientation widget

app.run()
app.destroy()
sc.destroy()
