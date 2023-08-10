import os
import sys
import numpy as np

sys.path.insert(1, '../src')
import z_plane

CP = 0
ZM = 1
theta = 0
print('square frame:\n')
fd = z_plane.get_complex_frame(CP,ZM,theta)
print(z_plane.complex_frame_dict_to_string(fd))

print('\n\n wide frame:\n')
n_rows = 100
n_cols = 200
print(z_plane.complex_frame_dict_to_string(z_plane.get_complex_frame(CP,ZM,theta, n_rows, n_cols)))

print('\n\n tall frame:\n')
n_rows = 200
n_cols = 100
print(z_plane.complex_frame_dict_to_string(z_plane.get_complex_frame(CP,ZM,theta, n_rows, n_cols)))

z = 1 - 1j
print(z_plane.complex_to_string(z, 12),'\n\n')

CP = 0
ZM = 1
theta = 0
fd = z_plane.get_complex_frame(CP,ZM,theta)
print(z_plane.complex_frame_dict_to_string(fd))

Z0 = np.random.random((7,7)) + np.random.random((7,7)) * 1j
z_plane.show_complex_matrix(Z0, N_DEC=3)

a_dict = {'theta': np.pi, 'center_point': -2 - 3.25j, 'zoom_factor': 0.75, 'n_rows': 5, 'n_cols': 5}
pretty_string = z_plane.get_aligned_dict_string(a_dict, N_DEC=6)
print(pretty_string)





