from IPython.display import Latex
Latex(r"""\begin{eqnarray} \large 
Z_{n+1} = Z_{n}^{2*Z_{n}^{-2*a^{-2*Z_{n}^{-2*b}}}}
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

def thunderHead(Z, p, Z0=None, ET=None):
    """ Z = thunderHead(Z, p) 
    Args:
        Z:    a real or complex number
        p:    a real of complex number
    Returns:
        Z:    the result (complex)
    """
    Z = Z**(2*Z**(-2*p[0]**(-2*Z**(-2*p[1]))));
    return Z

#                                        -- define parameters iterate the above equation  --  "separable" parameter p
par_set = {'n_rows': 500, 'n_cols': 800}
par_set['center_point'] = -4.5+2.75j
par_set['theta'] = 0.0
par_set['zoom'] = 0.155

par_set['it_max'] = 64
par_set['max_d'] = 10 / par_set['zoom']
par_set['dir_path'] = os.getcwd()
p = [3.83796971, -0.09564841+0.83234946j]

list_tuple = [(thunderHead, (p))]

t0 = time.time()
ET, Z, Z0 = ig.get_primitives(list_tuple, par_set)
print(time.time() - t0, '\t total time')

#                  escape time graphical norm
t0 = time.time()
Zd, Zr, ETn = ncp.etg_norm(Z0, Z, ET)
print('converstion time =\t', time.time() - t0)

#                  Escape Time: darker escaped sooner
t0 = time.time()
ETd = ncp.mat_to_gray(ETn)
print('coloring time =\t',time.time() - t0)
display(ETd)

t0 = time.time()

H = ncp.range_norm(Zd - Zr, lo=0.05, hi=0.95)
S = ncp.range_norm(1 - ETn, lo=0.1, hi=0.65)
V = ncp.range_norm(1 - Zr, lo=0.6, hi=1.0)
Ihsv = ncp.rgb_2_hsv_mat(H, S, V)

print('coloring time:\t',time.time() - t0)
display(Ihsv)

t0 = time.time()

H = ncp.range_norm(Zd - Zr, lo=0.05, hi=0.95)
S = ncp.range_norm(1 - ETn, lo=0.1, hi=0.55)
V = ncp.range_norm(1 - ETn, lo=0.75, hi=1.0)
Ihsv = ncp.rgb_2_hsv_mat(H, S, V)

print('coloring time:\t',time.time() - t0)
display(Ihsv)

#                                        smaller -> view individual escape time sets -- "iconic" 
par_set = {'n_rows': 200, 'n_cols': 324}
par_set['center_point'] = -4.5+2.75j
par_set['theta'] = 0.0
par_set['zoom'] = 0.155

par_set['it_max'] = 64
par_set['max_d'] = 10 / par_set['zoom']
par_set['dir_path'] = os.getcwd()
p = [3.83796971, -0.09564841+0.83234946j]

list_tuple = [(thunderHead, (p))]

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

#                                        Zoom Out: show that equation is defined all over the plane
par_set = {'n_rows': 200, 'n_cols': 324}
par_set['center_point'] = -4.5+2.75j
par_set['theta'] = 0.0
par_set['it_max'] = 64

par_set['dir_path'] = os.getcwd()
p = [3.83796971, -0.09564841+0.83234946j]

list_tuple = [(thunderHead, (p))]

par_set['zoom'] = 1
par_set['max_d'] = 12 / par_set['zoom']

for xy_or_maybe_z in range(0, 7):
    par_set['zoom'] = par_set['zoom'] / 4
    par_set['max_d'] = 12 / par_set['zoom']
    print('\n\t Zoom Scale = %0.6f'%(par_set['zoom']))
    ET_sm, Z_sm, Z0_sm = ig.get_primitives(list_tuple, par_set)
    Zd_sm, Zr_sm, ETn_sm = ncp.etg_norm(Z0_sm, Z_sm, ET_sm)
    
    display(ncp.mat_to_gray(ETn_sm))

for xy_or_maybe_z in range(0, 7):
    par_set['zoom'] = par_set['zoom'] / 3
    par_set['max_d'] = 12 / par_set['zoom']
    print('\n\t Zoom Scale = %0.9f'%(par_set['zoom']))
    ET_sm, Z_sm, Z0_sm = ig.get_primitives(list_tuple, par_set)
    Zd_sm, Zr_sm, ETn_sm = ncp.etg_norm(Z0_sm, Z_sm, ET_sm)
    
    display(ncp.mat_to_gray(ETn_sm))

print('pixel increased from\n    %0.6f'%((1 / 1) / np.sqrt(200**2 + 324**2)), '\n\tto' )
print('    %0.2f'%((1 / 0.000000028) / np.sqrt(200**2 + 324**2)), '\nscale units accross' )



