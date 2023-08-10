from IPython.display import Latex
Latex(r"""\begin{eqnarray} \large 
Z_{n+1} = Z_{n}^2 - p
\end{eqnarray}""")

import warnings
warnings.filterwarnings('ignore')

import os
import sys   
import numpy as np
import time

from IPython.display import display

sys.path.insert(1, '../src');
import z_plane as zp
import graphic_utility as gu;
import itergataters as ig
import numcolorpy as ncp

separable = 0.7958 + 0.1893j 
iconic = 0.7757 + 0.1234j
spirals = 0.7513 + 0.2551j
ropey = 0.81 + 0.2025j

def GastonJ(Z, p, Z0=None, ET=None):
    """ Z = bugga_bear(Z, p) 
    Args:
        Z:    a real or complex number
        p:    a real of complex number
    Returns:
        Z:    the result (complex)
    """
    return Z**2 - p

#                                        -- define parameters iterate the above equation  --  "separable" parameter p
par_set = {'n_rows': 800, 'n_cols': 800}
par_set['center_point'] = 0.0 + 0.0*1j
par_set['theta'] = np.pi / 4
par_set['zoom'] = 3/4

par_set['it_max'] = 64
par_set['max_d'] = 10 / par_set['zoom']
par_set['dir_path'] = os.getcwd()

list_tuple = [(GastonJ, (separable))]

t0 = time.time()
ET, Z, Z0 = ig.get_primitives(list_tuple, par_set)
print(time.time() - t0, '\t total time')

#                  escape time graphical norm
t0 = time.time()
Zd, Zr, ETn = ncp.etg_norm(Z0, Z, ET)
print('converstion time =\t', time.time() - t0)

H = ncp.range_norm(Zr - Zd, lo=0.5, hi=0.85)
S = ncp.range_norm(1 - ETn, lo=0.0, hi=0.9)
V = ncp.range_norm(ETn, lo=0.0, hi=1.0)
t0 = time.time()
Ihsv = ncp.rgb_2_hsv_mat(H, S, V)
print('coloring time:\t',time.time() - t0)
display(Ihsv)

#                                        -- define parameters iterate the above equation  --  "iconic" parameter p
par_set = {'n_rows': 800, 'n_cols': 800}
par_set['center_point'] = 0.0 + 0.0*1j
par_set['theta'] = np.pi / 4
par_set['zoom'] = 3/4

par_set['it_max'] = 128
par_set['max_d'] = 10 / par_set['zoom']
par_set['dir_path'] = os.getcwd()

list_tuple = [(GastonJ, (iconic))]

t0 = time.time()
ET, Z, Z0 = ig.get_primitives(list_tuple, par_set)
print(time.time() - t0, '\t total time')

#                  escape time graphical norm
t0 = time.time()
Zd, Zr, ETn = ncp.etg_norm(Z0, Z, ET)
print('converstion time =\t', time.time() - t0)

H = ncp.range_norm(Zd - Zr, lo=0.0, hi=1.0)
S = ncp.range_norm(ETn, lo=0.0, hi=1.0)
V = ncp.range_norm(1 - ETn, lo=0.9, hi=1.0)
t0 = time.time()
Ihsv = ncp.rgb_2_hsv_mat(H, S, V)
print('coloring time:\t',time.time() - t0)
display(Ihsv)

#                                        -- define parameters iterate the above equation  --  "spirals" parameter p
par_set = {'n_rows': 800, 'n_cols': 800}
par_set['center_point'] = 0.0 + 0.0*1j
par_set['theta'] = np.pi / 4
par_set['zoom'] = 3/4

par_set['it_max'] = 81
par_set['max_d'] = 10 / par_set['zoom']
par_set['dir_path'] = os.getcwd()

list_tuple = [(GastonJ, (spirals))]

t0 = time.time()
ET, Z, Z0 = ig.get_primitives(list_tuple, par_set)
print(time.time() - t0, '\t total time')

#                  escape time graphical norm
t0 = time.time()
Zd, Zr, ETn = ncp.etg_norm(Z0, Z, ET)
print('converstion time =\t', time.time() - t0)

H = ncp.range_norm(Zr * Zd, lo=0.6667, hi=1.0)
S = ncp.range_norm(ETn, lo=0.08, hi=1.0)
V = ncp.range_norm(ETn, lo=0.85, hi=1.0)
t0 = time.time()
Ihsv = ncp.rgb_2_hsv_mat(H, S, V)
print('coloring time:\t',time.time() - t0)
display(Ihsv)

#                                        -- define parameters iterate the above equation  --  "ropey" parameter p
par_set = {'n_rows': 800, 'n_cols': 800}
par_set['center_point'] = 0.0 + 0.0*1j
par_set['theta'] = np.pi / 4
par_set['zoom'] = 3/4

par_set['it_max'] = 96
par_set['max_d'] = 10 / par_set['zoom']
par_set['dir_path'] = os.getcwd()

list_tuple = [(GastonJ, (ropey))]

t0 = time.time()
ET, Z, Z0 = ig.get_primitives(list_tuple, par_set)
print(time.time() - t0, '\t total time')

#                  escape time graphical norm
t0 = time.time()
Zd, Zr, ETn = ncp.etg_norm(Z0, Z, ET)
print('converstion time =\t', time.time() - t0)

H = ncp.range_norm(Zr - Zd, lo=0.5, hi=0.85)
S = ncp.range_norm(1 - ETn, lo=0.0, hi=0.9)
V = ncp.range_norm(ETn, lo=0.0, hi=1.0)
t0 = time.time()
Ihsv = ncp.rgb_2_hsv_mat(H, S, V)
print('coloring time:\t',time.time() - t0)
display(Ihsv)

H = ncp.range_norm(Zd - Zr, lo=0.0, hi=1.0)
S = ncp.range_norm(ETn, lo=0.0, hi=1.0)
V = ncp.range_norm(1 - ETn, lo=0.9, hi=1.0)
t0 = time.time()
Ihsv = ncp.rgb_2_hsv_mat(H, S, V)
print('coloring time:\t',time.time() - t0)
display(Ihsv)

H = ncp.range_norm(Zr * Zd, lo=0.6667, hi=1.0)
S = ncp.range_norm(ETn, lo=0.08, hi=1.0)
V = ncp.range_norm(ETn, lo=0.85, hi=1.0)
t0 = time.time()
Ihsv = ncp.rgb_2_hsv_mat(H, S, V)
print('coloring time:\t',time.time() - t0)
display(Ihsv)



#                                        smaller -> view individual escape time sets -- "iconic" 
par_set = {'n_rows': 200, 'n_cols': 200}
par_set['center_point'] = 0.0 + 0.0*1j
par_set['theta'] = np.pi / 4
par_set['zoom'] = 3/4

par_set['it_max'] = 128
par_set['max_d'] = 10 / par_set['zoom']
par_set['dir_path'] = os.getcwd()

list_tuple = [(GastonJ, (iconic))]

t0 = time.time()
ET_sm, Z_sm, Z0_sm = ig.get_primitives(list_tuple, par_set)
print(time.time() - t0, '\t total time')

#                  escape time graphical norm
t0 = time.time()
Zd_sm, Zr_sm, ETn_sm = ncp.etg_norm(Z0_sm, Z_sm, ET_sm)
print('converstion time =\t', time.time() - t0)

#                  Escape Time: darker escaped sooner
t0 = time.time()
ETd = ncp.mat_to_gray(ETn_sm)
print('coloring time =\t',time.time() - t0)
display(ETd)

#                  Distance at Escape Time: lighter traveled further
t0 = time.time()
D = ncp.mat_to_gray(Zd_sm, max_v=255, min_v=64)
print('coloring time =\t',time.time() - t0)
display(D)

#                  Rotation at Escape Time: lighter rotated more
t0 = time.time()
R = ncp.mat_to_gray(Zr_sm, max_v=255, min_v=64)
print('coloring time =\t',time.time() - t0)
display(R)

#                                        view smaller - individual escape time starting points
lo_ET = 2
hi_ET = lo_ET + 6
for t in range(lo_ET, hi_ET):
    print('ET =\t',t)
    I = np.ones(ET_sm.shape)
    I[ET_sm == t] = 0
    display(ncp.mat_to_gray(I))
I = np.ones(ET_sm.shape)
I[ET_sm > hi_ET] = 0
print('ET >\t',hi_ET)
display(ncp.mat_to_gray(I))

#                                        view smaller - individual escape time frequency
for k in range(0,int(ET_sm.max())):
    print(k, (ET_sm == k).sum())
print('\nHow many never escaped:\n>',(ET_sm > k).sum())

#                           get the list of unescaped starting points and look for orbit points
Z_overs = Z0[ET_sm == ET_sm.max()]

v1 = Z_overs[0]
d = '%0.2f'%(np.abs(v1))
theta = '%0.1f'%(180*np.arctan2(np.imag(v1), np.real(v1))/np.pi)
print('Unescaped Vector:\n\tV = ', d, theta, 'degrees\n')

print('%9d'%Z_overs.size, 'total unescaped points\n')
print('%9s'%('points'), 'near V', '      (plane units)')
for denom0 in range(1,12):
    neighbor_distance = np.abs(v1) * 1/denom0
    v1_list = Z_overs[np.abs(Z_overs-v1) < neighbor_distance]
    print('%9d'%len(v1_list), 'within V/%2d  (%0.3f)'%(denom0, neighbor_distance))

#             Project equation on scaled random plane parallel
separable = 0.7958 + 0.1893j
iconic = 0.7757 + 0.1234j
spirals = 0.7513 + 0.2551j
ropey = 0.81 + 0.2025j

par_s = {'n_rows': 200, 'n_cols': 200}
par_s['center_point'] = 0.0 + 0.0*1j
par_s['theta'] = np.pi / 4
par_s['zoom'] = 3/4

par_s['it_max'] = 256
par_s['max_d'] = 10 / par_set['zoom']
par_s['dir_path'] = os.getcwd()

par_s['RANDOM_PLANE'] = True

list_tuple_s = [(GastonJ, (ropey))]

t0 = time.time()
ET_s, Z_s, Z0_s = ig.get_primitives(list_tuple_s, par_s)
print(time.time() - t0, '\t total time')
t0 = time.time()
Zd_s, Zr_s, ETn_s = ncp.etg_norm(Z0_s, Z_s, ET_s)

print('converstion time =\t',time.time() - t0,'\n\tDistance')
display(ncp.mat_to_blue(Zd_s))
print('\tRotation')
display(ncp.mat_to_red(Zr_s))
print('\tEscape Time')
display(ncp.mat_to_gray(ETn_s))



