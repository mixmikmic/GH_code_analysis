from IPython.display import Latex
Latex(r"""\begin{eqnarray} \large 
Z_{n+1} = (b - a*Z_{n}) / (d + c*Z_{n}^x)
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

def de_Okeeffe(Z, p, Z0=None, ET=None):
    """ Z = de_Okeeffe(Z, p) 
    Args:
        Z:    a real or complex number
        p:    a real of complex number
    Returns:
        Z:    the result (complex)
    """
    Z = (p[2] - p[1]*Z) / (p[4] + p[3]*Z**p[0]);
    return Z

#                                        -- short escape distance
par_set = {'n_rows': 800, 'n_cols': 800}
par_set['center_point'] = -0.1-0.1j
par_set['theta'] = 0.0
par_set['zoom'] = 0.3

par_set['it_max'] = 48
par_set['max_d'] = 1 / par_set['zoom']
par_set['dir_path'] = os.getcwd()

x = 1.2759;
a = 0 + 0.074647j;
b = -0.77504 + 0.007449j;
c = 1.2902 - 2.238e-18j;
d = 0.12875;
p = [x, a, b, c, d]
list_tuple = [(de_Okeeffe, (p))]

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

#                  Rotation > Hue, Distance > Saturation, Escape Time > Value (intensity-brightness) (muted)
H = Zd
S = 1 - Zr
V = ETn

Hue_width = 0.5
H_min = 0.0
H_max = H_min + Hue_width

S_max = 0.5
S_min = 0.05

V_max = 0.9
V_min = 0.2
t0 = time.time()
Ihsv = ncp.normat_hsv_intrgb(H, S, V, H_max, H_min, S_max, S_min, V_max, V_min)
print('coloring time:\t',time.time() - t0)
display(Ihsv)

#                                        smaller -> view individual escape time sets
par_set = {'n_rows': 200, 'n_cols': 200}
par_set['center_point'] = -0.1-0.1j
par_set['theta'] = 0.0
par_set['zoom'] = 0.3

par_set['it_max'] = 32
par_set['max_d'] = 1 / par_set['zoom']
par_set['dir_path'] = os.getcwd()

x = 1.2759;
a = 0 + 0.074647j;
b = -0.77504 + 0.007449j;
c = 1.2902 - 2.238e-18j;
d = 0.12875;
p = [x, a, b, c, d]
list_tuple = [(de_Okeeffe, (p))]

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



