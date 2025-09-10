<div align="center">

# MeshTBD

_Joshua Davidov and Ahikara Sandrasagra_ **|** Cooper Union

</div>

---

## Table of Contents
- [Why?](#why)
- [Project Structure](#project-structure)
- [Setup](#setup)
  - [Getting Started](#getting-started)
  - [Installation (with uv)](#installation-with-uv)
  - [Installation (with venv + pip)](#installation-with-venv--pip)
- [How it Works](#how-it-works)
- [Dependencies](#dependencies)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Why?

Modern 3D processing software sucks, especially when it comes to its scripting capabiltiies. We aren't tackling a full app as of yet, we're starting with turning a 3D model of a body part into a cast with a Voronoi pattern. <br>

This project was born out of the need for **customizable, script-driven pipelines** that are easy to experiment with and adapt for biomedical prototyping, wearable devices, and fabrication. The ultimate goal is to create reproducible workflows that can generate complex casts and structures automatically from raw scans, without relying on clunky, GUI-heavy tools.

---

## Project Structure

Note that this project only works on Python 3.11.X.

- `MeshTBD/Drafts` – where we put the scripts that don't particularly work, but some of their parts may be eventually useful (currently not available).
- `MeshTBD/Meshes` – some of the intermediate meshes we've been playing around with during the project (currently not available).
- `MeshTBD` – the scripts that are actively being altered, any relevant 3D models to running these scripts. <br>

Other directories and files of interest:

- `requirements.txt` – dependency list for setting up a traditional virtual environment.
- `pyproject.toml` – the configuration file for managing dependencies via [uv](https://github.com/astral-sh/uv).
- Example `.stl` / `.ply` meshes – included as test assets to quickly verify the pipeline.
- `README.md` – you’re here!

---

# Setup

## Getting Started

Note the naming conventions for the python scripts in this repository are as follows:
`tags_currentProgress.py` (Python scripts available in `MeshTBD/Meshes` have the `tags+includedPackages_currentProgress.py` naming conventions).

The project has been migrated to work primarily with the [uv package manager](https://github.com/astral-sh/uv), however a `requirements.txt` file is still available if you'd like to set up a venv for the script.

### Installation (with uv)

1. Clone this repository:
   ```bash
   git clone https://github.com/mydkm/MeshTBD.git
   cd MeshTBD
````

2. Install dependencies:

   ```bash
   uv sync
   ```

3. Run a script (example):

   ```bash
   uv run python MeshTBD/voronoi_cast.py -i input_forearm.ply -o output_cast.stl
   ```

### Installation (with venv + pip)

1. Clone the repo and create a venv:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run scripts as above.

---

## How it Works

Here's a general overview of how the script works:

1. Input 3D model is scaled to size of body part in a MeshSet and cleaned.
2. VCG Surface Reconstruction is applied.
3. Loop Surface Subdivision is applied.
4. Poisson-Disk Sampling with an user-defined number of points is completed. (WIP)
5. 3D model is colored according to the "Voronoi Vertex Coloring" filter.
6. 3D model is converted to a PyVista mesh w/ RGB values.
7. Appropiate vertices are selected by color and deleted.
8. PyVista mesh is returned to the MeshSet.
9. Any small holes remaining from the Pyvista mesh are removed, and the mesh is cleaned.
10. Isotropic Explicit Remeshing is applied.
11. Blender "Displace" modifier is applied.
12. Blender "Thicken" modifier is applied.
13. Edge faces within holes of mesh are smoothened using a Laplacian Smooth.
14. Output 3D model is saved as either a .stl or .ply file.

A more detailed description of our workflow and the thought process for our workflow is a WIP!

---

## Dependencies

The following packages are heavily used in this project:

* [PyMeshLab](https://pymeshlab.readthedocs.io/) – mesh cleaning, reconstruction, remeshing.
* [PyVista](https://docs.pyvista.org/) – mesh visualization and manipulation with RGB values.
* [Blender bpy](https://docs.blender.org/api/current/bpy/) – displacement, thickening, smoothing modifiers.
* [NumPy](https://numpy.org/) – matrix operations and PCA.
* [Matplotlib](https://matplotlib.org/) – color conversions and debugging visualizations.

Ensure you have Blender’s Python API (`bpy`) installed and pointing to your Blender build. If compiling Blender from source, confirm the Python version matches (currently 3.11.x).

---

## Contributing

We welcome experiments and pull requests! To contribute:

1. Fork the repository.
2. Create a feature branch:

   ```bash
   git checkout -b feature/my-idea
   ```
3. Commit changes and open a PR.

Please document any new filters or workflows clearly in the script and in this README.

---

## Roadmap

* [ ] Finalize Poisson-Disk Sampling workflow.
* [ ] Automate scaling from real-world measurements.
* [ ] Expand Voronoi pattern generation to include custom seed sets.
* [ ] Add unit tests for mesh hygiene and hole-filling.
* [ ] Build CLI around the full cast-generation pipeline.
* [ ] Publish paper / demo on biomedical applications.

---

## License

This project is currently released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

* Professor Shah (Cooper Union) for guidance on research methodology.
* Open-source contributors of PyMeshLab, PyVista, and Blender APIs.
* Fellow students and collaborators who tested early workflows.

```
```
