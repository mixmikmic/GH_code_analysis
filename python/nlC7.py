from IPython.display import Latex
Latex(r"""\begin{eqnarray} \large 
Z_{n+1} = Z_{n}^{(-((((((p_{1}^{Z_{n}^{p_{2}}})^{Z_{n}^{p_{3}}})^{Z_{n}^
{p_{4}}})^{Z_{n}^{p_{5}}})^{Z_{n}^{p_{6}}})^{Z_{n}^{p_{7}}})}
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

def nlC7(Z, p, Z0=None, ET=None):
    """ Z = nlC7(Z, p) 
    Args:
        Z:    a real or complex number
        p:    a real of complex array 1 x 7
    Returns:
        Z:    the result (complex)
    """
    Z = Z**(2 * Z**( -((((((p[0]**Z**-p[1])**(Z**-p[2]))**(Z**-p[3]))**(Z**-p[4]))**(Z**-p[5]))**(Z**-p[6]))))
    return Z

par_set = {'n_rows': 800, 'n_cols': 800}
par_set['center_point'] = -1.3 + 0.0*1j
par_set['theta'] = -np.pi / 2
par_set['zoom'] = 1/4

par_set['it_max'] = 128
par_set['max_d'] = 10 / par_set['zoom']
par_set['dir_path'] = os.getcwd()

p = [0.083821, -0.2362, 0.46518, -0.91572, 1.6049, -2.3531, 3.2664]

list_tuple = [(nlC7, (p))]

t0 = time.time()
ET, Z, Z0 = ig.get_primitives(list_tuple, par_set)
print(time.time() - t0, '\t total time')

#                  escape time graphical norm
t0 = time.time()
Zd, Zr, ETn = ncp.etg_norm(Z0, Z, ET)
print('converstion time =\t', time.time() - t0)

H = ncp.range_norm(Zd + ETn, lo=0.15, hi=0.85)
S = ncp.range_norm(1 - ETn, lo=0.025, hi=0.25)
V = ncp.range_norm(1 - Zr/2 + ETn/2 + Zd/2, lo=0.25, hi=1.0)

t0 = time.time()
Ihsv = ncp.rgb_2_hsv_mat(H, S, V)
print('coloring time:\t',time.time() - t0)
display(Ihsv)

#                                        -- define parameters iterate the above equation  --  "separable" parameter p
par_set = {'n_rows': 300, 'n_cols': 900}
par_set['center_point'] = -2.35 + 0.0j
par_set['theta'] = np.pi / 2
par_set['zoom'] = 1

par_set['it_max'] = 64
par_set['max_d'] = 4 # 10 / par_set['zoom']
par_set['dir_path'] = os.getcwd()
# c = [0.083821 -0.2362 0.46518 -0.91572 1.6049 -2.3531 3.2664];
# c = [0.16565, -0.42862, 0.73984, -1.2684, 1.8704, -2.5244, 3.3187]
p = [0.16565, -0.42862, 0.73984, -1.2684, 1.8704, -2.5244, 3.3187]
list_tuple = [(nlC7, (p))]

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

#                                        smaller -> view individual escape time sets -- "iconic" 
par_set = {'n_rows': 200, 'n_cols': 600}
par_set['center_point'] = -2.35 + 0.0j
par_set['theta'] = np.pi / 2
par_set['zoom'] = 1

par_set['it_max'] = 64
par_set['max_d'] = 4 # 10 / par_set['zoom']
par_set['dir_path'] = os.getcwd()
# c = [0.083821 -0.2362 0.46518 -0.91572 1.6049 -2.3531 3.2664];
# c = [0.13197, 0.94205, 0.95613, 0.57521, 0.05978];
# c = [0.16565, -0.42862, 0.73984, -1.2684, 1.8704, -2.5244, 3.3187]
p = [0.16565, -0.42862, 0.73984, -1.2684, 1.8704, -2.5244, 3.3187]
list_tuple = [(nlC7, (p))]

t0 = time.time()
ET_sm, Z_sm, Z0_sm = ig.get_primitives(list_tuple, par_set)
print(time.time() - t0, '\t total time')

#                  escape time graphical norm
t0 = time.time()
Zd_sm, Zr_sm, ETn_sm = ncp.etg_norm(Z0_sm, Z_sm, ET_sm)
print('converstion time =\t', time.time() - t0)




t0 = time.time()
ETd = ncp.mat_to_gray(ETn_sm)
print('coloring time =\t',time.time() - t0)
display(ETd)

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



