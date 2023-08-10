import numpy as np
from chemview import RepresentationViewer
from chemview.render import render_povray

coordinates = np.array([[0.0, 0.1, 0.1], [0.01, 0, 0]], 'float32')
colors = [0xFFFFFF, 0xFF999F]
sizes = [0.1, 0.2]

rv = RepresentationViewer(100, 100)
point_id = rv.add_representation('points', {'coordinates': coordinates, 'colors': colors, 'sizes': sizes})
rv

render_povray(rv.get_scene(), width=200, height=200, extra_opts={'radiosity':True})

# Transparent spheres
transparency = [1.0, 0.5]
rv = RepresentationViewer(100, 100)
point_id = rv.add_representation('points', {'coordinates': coordinates,
                                            'colors': colors,
                                            'sizes': sizes,
                                            'transparency': transparency})
rv

scene = rv.get_scene()
render_povray(rv.get_scene(), width=500, height=200)

coordinates = np.array([[0.0, 1.1, 0.1], [1, 0, 0]], 'float32')
colors = [0xFFFFFF, 0xFF999F]
radii = [0.1, 0.2]

rv = RepresentationViewer(100, 100)
point_id = rv.add_representation('spheres', {'coordinates': coordinates, 'colors': colors, 'radii': radii})
rv

render_povray(rv.get_scene(), width=200, height=200)

# Let's try to do surface that represent a handly little squarey
verts = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 
                  [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], 'float32')
faces = np.array([[0, 1, 3], [1, 2, 3]], 'int32')

#rv = RepresentationViewer(100, 100)
point_id = rv.add_representation('surface', { 'verts': verts, 
                                              'faces': faces, 
                                              'style': 'solid',
                                              'color': 0xff00ff})
rv

coordinates = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], 'float32')
colors = np.array([0xFF0000, 0x00FF00, 0x0000FF])
transparency = [1.0, 0.3, 1.0]

rv = RepresentationViewer()
cylinders_id = rv.add_representation('cylinders', {'startCoords': coordinates[[0, 1, 2]],
                                           'endCoords': coordinates[[1, 2, 0]],
                                           'radii': [0.2, 0.1, 0.1],
                                           'colors': colors[[0, 1, 2]].tolist(),
                                           'transparency': transparency})
rv

render_povray(rv.get_scene(), extra_opts={'radiosity': True})



