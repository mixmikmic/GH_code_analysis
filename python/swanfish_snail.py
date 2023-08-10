from IPython.display import Latex
Latex(r"""\begin{eqnarray} \large
Z_{n+1} = Z_{n}^{-(((p_{1}^{Z_{n})^{p_{2})^{Z_{n})^{p_{3}}}}}}
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

def swanfishsnail(Z, p, Z0=None, ET=None):
    """ Z = bugga_bear(Z, p) 
    Args:
        Z:    a real or complex number
        p:    a real of complex number
    Returns:
        Z:    the result (complex)
        Z = Z.^(-exp(pi * p(2)).^Z.^(-sin(abs(p(1) + p(2)))).^Z.^(exp(p(2)-p(1))));
    """
    return Z**(-(((p[0]**Z)**p[1])**Z)**p[2])

#                                        -- define parameters iterate the above equation  --
par_set = {'n_rows': 500, 'n_cols': 700}
par_set['center_point'] = 0.2 - 0.55j
par_set['theta'] = -np.pi/4
par_set['zoom'] = 0.5

par_set['it_max'] = 100
par_set['max_d'] = 10 / par_set['zoom']
par_set['dir_path'] = os.getcwd()

p = [3.01874673, -1.00776339+0.74204475j]
p0 = np.exp(np.pi * p[1])
p1 = -np.sin(np.abs(p[0] + p[1]))
p2 = np.exp(p[1] - p[0])

list_tuple = [(swanfishsnail, ([p0, p1, p2]))]

t0 = time.time()
ET, Z, Z0 = ig.get_primitives(list_tuple, par_set)
print(time.time() - t0, '\t total time')

t0 = time.time()
Zd, Zr, ETn = ncp.etg_norm(Z0, Z, ET)
print('converstion time =\t', time.time() - t0)

#                  Escape Time: darker escaped sooner
t0 = time.time()
ETd = ncp.mat_to_gray(ETn)
print('coloring time =\t',time.time() - t0)
display(ETd)

#                  Distance at Escape Time: lighter traveled further
t0 = time.time()
D = ncp.mat_to_gray(Zd, max_v=255, min_v=64)
print('coloring time =\t',time.time() - t0)
display(D)

#                  Rotation at Escape Time: lighter rotated more
t0 = time.time()
R = ncp.mat_to_gray(Zr, max_v=255, min_v=64)
print('coloring time =\t',time.time() - t0)
display(R)

H = ncp.range_norm(Zr, lo=0.0, hi=0.6)
S = ncp.range_norm(1 - Zd, lo=0.0, hi=0.4)
V = ncp.range_norm(1 - ETn, lo=0.2, hi=0.8)
t0 = time.time()
Ihsv = ncp.rgb_2_hsv_mat(H, S, V)
print('coloring time:\t',time.time() - t0)
display(Ihsv)

#                                        smaller -> view individual escape time sets
par_set = {'n_rows': 150, 'n_cols': 250}
par_set['center_point'] = 0.2 - 0.55j
par_set['theta'] = -np.pi/4
par_set['zoom'] = 0.5

par_set['it_max'] = 512
par_set['max_d'] = 10 / par_set['zoom']
par_set['dir_path'] = os.getcwd()

p = [3.01874673, -1.00776339+0.74204475j]
p0 = np.exp(np.pi * p[1])
p1 = -np.sin(np.abs(p[0] + p[1]))
p2 = np.exp(p[1] - p[0])

list_tuple = [(swanfishsnail, ([p0, p1, p2]))]

t0 = time.time()
ET_sm, Z_sm, Z0_sm = ig.get_primitives(list_tuple, par_set)
print(time.time() - t0, '\t total time')

#                                        view smaller - individual escape time starting points
for t in range(1,7):
    print('ET =\t',t)
    I = np.ones(ET_sm.shape)
    I[ET_sm == t] = 0
    display(ncp.mat_to_gray(I))
I = np.ones(ET_sm.shape)
I[ET_sm > 7] = 0
display(ncp.mat_to_gray(I))

#                                        view smaller - individual escape time frequency
for k in range(0,int(ET_sm.max())):
    print(k, (ET_sm == k).sum())
print('\nHow many never escaped:\n>', (ET_sm > k).sum())

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



