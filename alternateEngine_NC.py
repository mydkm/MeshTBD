import numpy as np

import datoviz as dvz
from datoviz import Out, vec2, vec3

# Dialog width.
w = 300
OBJ = "/home/mydkm/MeshTBD/SabrinaOutput.obj"

labels = ['col0', 'col1', 'col2', '0', '1', '2', '3', '4', '5']
rows = 2
cols = 3
selected = np.array([False, True], dtype=bool)

# IMPORTANT: these values need to be defined outside of the GUI callback.
checked = Out(True)
color = vec3(0.7, 0.5, 0.3)

slider = Out(25.0)  # Warning: needs to be a float as it is passed to a function expecting a float
dropdown_selected = Out(1)

# GUI callback function, called at every frame. This is using Dear ImGui, an immediate-mode
# GUI system. This means the GUI is recreated from scratch at every frame.

sc = dvz.ShapeCollection()
sc.add_obj(OBJ, scale = 1.0)


app = dvz.App()
# NOTE: at the moment, you must indicate gui=True if you intend to use a GUI in a figure
fig = app.figure(gui=True)
panel = fig.panel(background=True)

arcball = panel.arcball(initial=(0.0, 0.0, 0.0))
panel.camera(initial=(0.0, 0.0, 3.0))

@app.connect(fig)
def on_gui(ev):
    # Set the size of the next GUI dialog.
    dvz.gui_pos(vec2(25, 25), vec2(0, 0))
    dvz.gui_size(vec2(w + 20, 550))

    # Start a GUI dialog, specifying a dialog title.
    dvz.gui_begin('My GUI', 0)

    # Add a button. The function returns whether the button was pressed during this frame.
    if dvz.gui_button('Button', w, 30):
        print('button clicked')

    # End the GUI dialog.
    dvz.gui_end()

visual = app.mesh(
    sc, 
    indexed=True, 
    lighting=True
    )
panel.add(visual)

# Optional helpers:
panel.gizmo() 

app.run()
app.destroy()