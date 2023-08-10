from IPython.display import Latex
Latex(r"""\begin{eqnarray}
\\ x = \Re(\sqrt{Z_{n}/ |Z_{n}|})
\\ y = \Im(\sqrt{Z_{n}/ |Z_{n}|})*i
\\ dv = |Z_{n} - Z_{0}|
\\ Z_{n+1} = Z_{n} - ( a*x^3 + 3*b*x^2*y + 3*c*x*y^2 + d*y^3 )^{Z_{n}*dv}
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

def T_Spake_Z(Z, p, Z0=None, ET=None):
    """ Z = T_Spake_Z(Z, p) 
    Args:
        Z:    a real or complex number
        p:    a real of complex number

    Returns:
        Z:    the result (complex)
    """
    d = np.abs(Z-Z0)
    Zxy = np.sqrt(Z/np.abs(Z))
    x = np.real(Zxy)
    y = np.imag(Zxy)*1j
    Z = Z - ( p[0]*x**3 + 3*p[1]*x**2*y + 3*p[2]*x*y**2 + p[3]*y**3 )**(Z*d)

    return Z

#                                        -- define parameters iterate the above equation  --
#                                        -- machine with 8 cores --
par_set = {'n_rows': 800, 'n_cols': 600}
par_set['center_point'] = 1/3 + 0.0j
par_set['theta'] = np.pi/2
par_set['zoom'] = 1/3

par_set['it_max'] = 38
par_set['max_d'] = 10 / par_set['zoom']
par_set['dir_path'] = os.getcwd()

list_tuple = [(T_Spake_Z, ([1.92846051108342, 2.27919841968635, 3.37327534248407, 2.17984103218476]))]

t0 = time.time()
ET, Z, Z0 = ig.get_primitives(list_tuple, par_set)
tt = time.time() - t0
print(tt, '\t total time')

#                  Escape Time Graphical norm - graphically easier data
t0 = time.time()
Zd, Zr, ETn = ncp.etg_norm(Z0, Z, ET)
print('converstion time =\t', time.time() - t0)

#                  Escape Time: darker escaped sooner
t0 = time.time()
ETd = ncp.mat_to_gray(ET)
print('coloring time =\t',time.time() - t0)
display(ETd)

#                  Distance at Escape Time: lighter traveled further
t0 = time.time()
D = ncp.mat_to_gray(Zd)
print('coloring time =\t',time.time() - t0)
display(D)

#                  Rotation at Escape Time: lighter rotated more
t0 = time.time()
R = ncp.mat_to_gray(Zr)
print('coloring time =\t',time.time() - t0)
display(R)

#                  Rotation > Hue, Distance > Saturation, Escape Time > Value (intensity-brightness) (muted)
H = ncp.graphic_norm(Zd)
S = ncp.graphic_norm(ET)
V = ncp.graphic_norm(Zr)

Hue_width = 1.0
H_min = 0.0
H_max = H_min + Hue_width

S_max = 1.0
S_min = 0.0

V_max = 1.0
V_min = 0.3
t0 = time.time()
Ihsv = ncp.normat_hsv_intrgb(H, S, V, H_max, H_min, S_max, S_min, V_max, V_min)
print('coloring time:\t',time.time() - t0)
display(Ihsv)

#                  Rotation > Hue, Escape Time > Saturation, Distance > Value (no significantly muted)
H = 1 - ncp.graphic_norm(Zd)
S = 1 - ncp.graphic_norm(ET)
V = ncp.graphic_norm(Zr)

t0 = time.time()
I = ncp.normat_hsv_intrgb(H, S, V, H_max=1.0, H_min=0.0, S_max=1.0, S_min=0.0, V_max=1.0, V_min=0.3)
print('coloring time:\t',time.time() - t0)
display(I)

H = ncp.range_norm(1 - ETn, lo=0.45, hi=1.0)
S = ncp.range_norm(Zd, lo=0.05, hi=0.1)
V = ncp.range_norm(Zr, lo=0.2, hi=0.8)
t0 = time.time()
Ihsv = ncp.rgb_2_hsv_mat(H, S, V)
print('coloring time:\t',time.time() - t0)
display(Ihsv)

#                                        smaller -> view individual escape time sets
par_set = {'n_rows': 200, 'n_cols': 200}
par_set['center_point'] = 1/3 + 0.0j
par_set['theta'] = np.pi/2
par_set['zoom'] = 1/3

par_set['it_max'] = 64
par_set['max_d'] = 10 / par_set['zoom']
par_set['dir_path'] = os.getcwd()

list_tuple = [(T_Spake_Z, ([1.92846051108342, 2.27919841968635, 3.37327534248407, 2.17984103218476]))]

t0 = time.time()
ET_sm, Z_sm, Z0_sm = ig.get_primitives(list_tuple, par_set)
print(time.time() - t0, '\t total time')

#                                        view smaller - individual escape time starting points
for t in range(2,8):
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



