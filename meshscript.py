import pyvista
import pymeshlab 

# setting up pymeshlab environment
ms = pymeshlab.MeshSet()
ms.load_new_mesh('Just forearm.stl')
surface_id = ms.current_mesh_id()

ms.generate_surface_reconstruction_vcg(voxsize = pymeshlab.PercentageValue(0.499991))
ms.set_current_mesh(1)
ms.save_current_mesh('plymcout.ply')
ms.load_new_mesh('plymcout.ply')

ms.meshing_surface_subdivision_loop(threshold = pymeshlab.PercentageValue(0.500009))
ms.generate_sampling_poisson_disk(samplenum = 50, exactnumflag = True)
ms.set_current_mesh(surface_id)
ms.compute_color_by_point_cloud_voronoi_projection(backward = True) 
ms.compute_selection_by_scalar_per_vertex(minq = 0.000000, maxq = 2.914720)
ms.apply_selection_inverse()
ms.meshing_remove_selected_vertices_and_faces()

# saving intermediate mesh
ms.save_current_mesh('z.ply')
mesh = pyvista.read('z.ply')
mesh.plot()

