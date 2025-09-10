<div align="center">

# MeshTBD

_Joshua Davidov and Ahikara Sandrasagra_ **|** Cooper Union 

</div>

## Why?

Modern 3D processing software sucks, especially when it comes to its scripting capabiltiies. We aren't tackling a full app as of yet, we're starting with turning a 3D model of a body part into a cast with a Voronoi pattern. <br>

## Project Structure

Note that this project only works on Python 3.11.X.

- 'MeshTBD/Drafts' - where we put the scripts that don't particularly work, but some of its parts may be eventually useful (currently not available).
- 'MeshTBD/Meshes' - some of the intermediate meshes we've been playing around with during the project (currently not available).
- 'MeshTBD' - the scripts that are actively being altered, any relevant 3D models to running these scripts. <br>

# Setup

## Getting Started

Note the naming conventions for the python scripts in this repository are as follows:
'tags_currentProgress.py' (Python scripts available in 'MeshTBD/Meshes' have the 'tags+includedPackages_currentProgress.py' naming conventions).

The project has been migrated to work primarily with the uv package manager, however a 'requirements.txt' file is still available if you'd like to set up a venv for the script.

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

TBD!
