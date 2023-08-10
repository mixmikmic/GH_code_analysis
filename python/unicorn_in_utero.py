from IPython.display import Latex
Latex(r"""\begin{eqnarray} \large
Z_{n+1} = Z_{n}^{1.5 * Z_{n}^{\sqrt{-1} * (p_{1}^{1.5 * Z_{n}^{p_{2}}})}}
\qquad \qquad \small p = [2.76544+0.997995j, 50.1518+7.53287j]
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

def unicorn_in_utero(Z, p, Z0=None, ET=None):
    """ Z = bugga_bear(Z, p) 
    Args:
        Z:    a real or complex number
        p:    a real of complex number # p = [2.76544+0.997995j, 50.1518+7.53287j]
    Returns:
        Z:    the result (complex)
    """
    Z = Z**(1.5 * Z**(1j * p[0]**(1.5 * Z**p[1])))
    return Z

#                                        
par_set = {'n_rows': 800, 'n_cols': 800}
par_set['center_point'] = -0.25 + 0.0*1j
par_set['theta'] = 0.0
par_set['zoom'] = 5/8

par_set['it_max'] = 28
par_set['max_d'] = 10 / par_set['zoom']
par_set['dir_path'] = os.getcwd()

p = [2.76544+0.997995j, 50.1518+7.53287j]

list_tuple = [(unicorn_in_utero, (p))]

t0 = time.time()
ET, Z, Z0 = ig.get_primitives(list_tuple, par_set)
tt = time.time() - t0
print(tt, '\t total time')

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

H = ncp.range_norm(1 - Zr, lo=0.0, hi=0.85)
S = ncp.range_norm(1 - ETn, lo=0.0, hi=0.15)
V = ncp.range_norm(Zd, lo=0.2, hi=1.0)
t0 = time.time()
Ihsv = ncp.rgb_2_hsv_mat(H, S, V)
print('coloring time:\t',time.time() - t0)
display(Ihsv)

mp0 = np.array([[0.2,0.2,0.2],[0.3,0.3,0.05],[0.9,0.3,0.15]])
I_mapped = ncp.mat_to_mapped(Zd, mp0)
display(I_mapped)

#                                        
par_s = {'n_rows': 200, 'n_cols': 200}
par_s['center_point'] = -0.25 + 0.0*1j
par_s['theta'] = 0.0
par_s['zoom'] = 5/8

par_s['it_max'] = 512
par_s['max_d'] = 10 / par_s['zoom']
par_s['dir_path'] = os.getcwd()

p = [2.76544+0.997995j, 50.1518+7.53287j]

list_tuple_s = [(unicorn_in_utero, (p))]

t0 = time.time()
ET_sm, Z_sm, Z0_sm = ig.get_primitives(list_tuple_s, par_s)
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



