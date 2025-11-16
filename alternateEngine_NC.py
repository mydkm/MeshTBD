import numpy as np
import datoviz as dvz

from datoviz import Out, vec2, vec3

# Dialog width.
w = 300
OBJ = "/Users/ahikara/MeshTBD/SabrinaOutput.obj"  # <-- update to your actual OBJ path

labels = ['col0', 'col1', 'col2', '0', '1', '2', '3', '4', '5']
rows = 2
cols = 3
selected = np.array([False, True], dtype=bool)

# IMPORTANT: these values need to be defined outside of the GUI callback.
checked = dvz.Out(True)
color = dvz.vec3(0.7, 0.5, 0.3)

slider = dvz.Out(25.0)  # needs to be float
dropdown_selected = dvz.Out(1)

sc = dvz.ShapeCollection()
sc.add_obj(OBJ, scale=1.0)

app = dvz.App()
fig = app.figure(gui=True)
panel = fig.panel(background=True)

arcball = panel.arcball(initial=(0.0, 0.0, 0.0))
panel.camera(initial=(0.0, 0.0, 3.0))

@app.connect(fig)
def on_gui(ev):
    dvz.gui_pos(dvz.vec2(25, 25), dvz.vec2(0, 0))
    dvz.gui_size(dvz.vec2(w + 20, 550))

    dvz.gui_begin('My GUI', 0)

    if dvz.gui_button('Button', w, 30):
        print('button clicked')

    dvz.gui_end()

visual = app.mesh(
    sc,
    indexed=True,
    lighting=True
)
panel.add(visual)
panel.gizmo()

app.run()
app.destroy()