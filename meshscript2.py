import pyvista
import pymeshlab

# setting up pymeshlab environment
ms = pymeshlab.MeshSet()
ms.load_new_mesh('plymcout.ply')

ms.meshing_surface_subdivision_loop(threshold = pymeshlab.PercentageValue(0.500009))
#ms.generate_sampling_poisson_disk(samplenum = 50, exactnumflag = True) 
#ms.compute_color_by_point_cloud_voronoi_projection(backward = True)                                                            
#ms.compute_selection_by_scalar_per_vertex(minq = 0.000000, maxq = 2.914720)
#ms.apply_selection_inverse()
#ms.meshing_remove_selected_faces()

# visualizing final mesh
ms.save_current_mesh('new.ply')
mesh = pyvista.read('new.ply')
mesh.plot()    
