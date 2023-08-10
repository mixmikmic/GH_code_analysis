from IPython.display import Latex
Latex(r"""\begin{eqnarray} \large 
Z_{n+1} = Z_{n}^{-p_{5}*{Z_{n}^{-p_{4}^{-p_{3}*Z_{n}^{p_{2}^{Z_{n}^{-p_{1}}}}}}}}
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

def RadHak(Z, p, Z0=None, ET=None):
    """ Z = RadHak(Z, p) 
    Args:
        Z:    a real or complex number
        p:    a real of complex number
    Returns:
        Z:    the result (complex)
    """
    return Z**(-p[4] * Z**(-p[3]**(-p[2]*Z**(-p[1]**(Z**(-p[0]))))))

#                                        -- define parameters iterate the above equation  --  "iconic" parameter p
par_set = {'n_rows': 300, 'n_cols': 900}
par_set['center_point'] = 0.7 + 0.0*1j
par_set['theta'] = np.pi / 2
par_set['zoom'] = 0.31

par_set['it_max'] = 64
par_set['max_d'] = 120 / par_set['zoom']
par_set['dir_path'] = os.getcwd()

p = [131.736080357263, 7.410291823809, 12.902639951370, 2.223646982174, 0.972974554764]

list_tuple = [(RadHak, (p))]

t0 = time.time()
ET, Z, Z0 = ig.get_primitives(list_tuple, par_set)
print(time.time() - t0, '\t total time')

#                  escape time graphical norm
t0 = time.time()
Zd, Zr, ETn = ncp.etg_norm(Z0, Z, ET)
print('converstion time =\t', time.time() - t0)

H = ncp.range_norm(Zd - Zr, lo=0.0, hi=1.0)
S = ncp.range_norm(ETn, lo=0.3, hi=0.6)
V = ncp.range_norm(ETn*Zd, lo=0.5, hi=1.0)
t0 = time.time()
Ihsv = ncp.rgb_2_hsv_mat(H, S, V)
print('coloring time:\t',time.time() - t0)
display(Ihsv)

H = ncp.range_norm(Zr - Zd, lo=0.25, hi=0.75)
S = ncp.range_norm(1 - ETn, lo=0.05, hi=0.1)
V = ncp.range_norm(1 - ETn*Zd, lo=0.4, hi=1.0)
t0 = time.time()
Ihsv = ncp.rgb_2_hsv_mat(H, S, V)
print('coloring time:\t',time.time() - t0)
display(Ihsv)

H = ncp.range_norm(Zd - Zr, lo=0.0, hi=1.0)
S = ncp.range_norm(ETn, lo=0.0, hi=0.15)
V = ncp.range_norm(ETn * Zd, lo=0.7, hi=1.0)
t0 = time.time()
Ihsv = ncp.rgb_2_hsv_mat(H, S, V)
print('coloring time:\t',time.time() - t0)
display(Ihsv)



#                                        smaller -> view individual escape time sets -- "iconic" 
par_set = {'n_rows': 200, 'n_cols': 600}
par_set['center_point'] = 0.7 + 0.0*1j
par_set['theta'] = np.pi / 2
par_set['zoom'] = 0.31

par_set['it_max'] = 64
par_set['max_d'] = 120 / par_set['zoom']
par_set['dir_path'] = os.getcwd()

p = [131.736080357263, 7.410291823809, 12.902639951370, 2.223646982174, 0.972974554764];

list_tuple = [(RadHak, (p))]

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

H = ncp.range_norm(Zd_sm - Zr_sm, lo=0.0, hi=1.0)
S = ncp.range_norm(ETn_sm, lo=0.3, hi=0.6)
V = ncp.range_norm(ETn_sm, lo=0.6, hi=1.0)
t0 = time.time()
Ihsv = ncp.rgb_2_hsv_mat(H, S, V)
print('coloring time:\t',time.time() - t0)
display(Ihsv)

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



