import numpy as np
import tkinter as tk
from tkinter import filedialog
import datoviz as dvz
from datoviz import Out, vec2, vec3

# Create a hidden Tk root just for file dialogs
tk_root = tk.Tk()
tk_root.withdraw()

# Dialog width
w = 300

# Initial OBJ path (can be anything you like)
OBJ = "/home/mydkm/MeshTBD/SabrinaOutput.obj"

# GUI state
checked = Out(True)
color = vec3(0.7, 0.5, 0.3)
slider = Out(25.0)
dropdown_selected = Out(1)

# --- Datoviz setup -----------------------------------------------------------

# Start with a ShapeCollection and one OBJ
sc = dvz.ShapeCollection()
sc.add_obj(OBJ, scale=1.0)

app = dvz.App()
fig = app.figure(gui=True)
panel = fig.panel(background=True)

arcball = panel.arcball(initial=(0.0, 0.0, 0.0))
panel.camera(initial=(0.0, 0.0, 3.0))

visual = app.mesh(sc, indexed=True, lighting=True)
panel.add(visual)
panel.gizmo()

def load_obj(path: str):
    """Replace the current mesh with the OBJ at `path`."""
    global sc, visual

    if not path:
        return

    # Remove the old visual from the panel
    panel.remove(visual)

    # Free GPU-side data for the old ShapeCollection
    sc.destroy()

    # Build a new ShapeCollection from the selected OBJ
    sc = dvz.ShapeCollection()
    sc.add_obj(path, scale=1.0)

    # Create a new mesh visual and add it to the panel
    visual = app.mesh(sc, indexed=True, lighting=True)
    panel.add(visual)

@app.connect(fig)
def on_gui(ev):
    # Set the size of the next GUI dialog.
    dvz.gui_pos(vec2(25, 25), vec2(0, 0))
    dvz.gui_size(vec2(w + 20, 200))

    # Start a GUI dialog, specifying a dialog title.
    dvz.gui_begin('My GUI', 0)

    # Import button: open file dialog and load selected OBJ
    if dvz.gui_button('Import OBJ…', w, 30):
        file_path = filedialog.askopenfilename(
            title="Select OBJ mesh",
            filetypes=[("OBJ files", "*.obj"), ("All files", "*.*")]
        )
        if file_path:
            load_obj(file_path)

    dvz.gui_end()

app.run()
app.destroy()
sc.destroy()
