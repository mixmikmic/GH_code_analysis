from chemview import MolecularViewer

import numpy as np
coordinates = np.array([[0.00, 0.13, 0.00], [0.12, 0.07, 0.00], [0.12,-0.07, 0.00],
                       [0.00,-0.14, 0.00], [-0.12,-0.07, 0.00],[-0.12, 0.07, 0.00],
                       [ 0.00, 0.24, 0.00], [ 0.21, 0.12, 0.00], [ 0.21,-0.12, 0.00],
                       [ 0.00,-0.24, 0.00], [-0.21,-0.12, 0.00],[-0.21, 0.12, 0.00]])
atomic_types = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
bonds = [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11),
         (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]

mv = MolecularViewer(coordinates, topology={'atom_types': atomic_types,
                                            'bonds': bonds})
mv.ball_and_sticks()
mv

