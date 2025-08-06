import pyacvd, pyvista as pv #type:ignore

mesh = pv.read('a5.ply')
clus = pyacvd.Clustering(mesh)
clus.cluster(20000)
remesh = clus.create_mesh()
remesh.plot(show_edges=True)
