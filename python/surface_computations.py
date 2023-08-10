get_ipython().magic('matplotlib inline')
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

from sympy import *
from sympy import init_printing
x, y, z, t = symbols('x y z t')
u, v, a, b, R = symbols('u v a b R')
k, m, n = symbols('k m n', integer=True)
init_printing()

ellipse = Matrix([[0, a*cos(u), b*sin(u)]]).T

Qx = Matrix([[1, 0, 0],
             [0, cos(n*v), -sin(n*v)],
             [0, sin(n*v), cos(n*v)]])

Qx

Qz = Matrix([[cos(v), -sin(v), 0],
             [sin(v), cos(v), 0],
             [0, 0, 1]])

Qz

trans = Matrix([[0, R, 0]]).T

trans

torobius = Qz*(Qx*ellipse + trans)
torobius

x_num = lambdify((u, v, n, a, b, R), torobius[0], "numpy")
y_num = lambdify((u, v, n, a, b, R), torobius[1], "numpy")
z_num = lambdify((u, v, n, a, b, R), torobius[2], "numpy")

u_par, v_par = np.mgrid[0:2*np.pi:50j, 0:2*np.pi:50j]

X = x_num(u_par, v_par, 2, 0.5, 1., 5.)
Y = y_num(u_par, v_par, 2, 0.5, 1., 5.)
Z = z_num(u_par, v_par, 2, 0.5, 1., 5.)

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="YlGnBu_r")
ax.set_zlim(-2, 2);
plt.show()

E = (torobius.diff(u).T * torobius.diff(u))[0]
F = (torobius.diff(u).T * torobius.diff(v))[0]
G = (torobius.diff(v).T * torobius.diff(v))[0]

def cross(A, B):
    return Matrix([[A[1]*B[2] - A[2]*B[1]],
                   [A[2]*B[0] - A[0]*B[2]],
                   [A[0]*B[1] - A[1]*B[0]]])

n_vec = cross(torobius.diff(u).T, torobius.diff(v))
n_vec = simplify(n_vec/sqrt((n_vec.T * n_vec)[0]))

L = (torobius.diff(u, 2).T * n_vec)[0]
M = (torobius.diff(u, 1, v, 1).T * n_vec)[0]
N = (torobius.diff(v, 2).T * n_vec)[0]

gauss_curvature = (L*N - M**2)/(E*G - F**2)
mean_curvature = S(1)/2*(L + N)

gauss_num = lambdify((u, v, n, a, b, R), gauss_curvature, "numpy")
mean_num = lambdify((u, v, n, a, b, R), mean_curvature, "numpy")

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
gauss_K = gauss_num(u_par, v_par, 2, 0.5, 1., 5.)
vmax = gauss_K.max()
vmin = gauss_K.min()
FC = (gauss_K - vmin) / (vmax - vmin)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=plt.cm.YlGnBu(FC))
surf.set_edgecolors("k")
ax.set_title("Gaussian curvature", fontsize=18)
ax.set_zlim(-2, 2);

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
mean_H = mean_num(u_par, v_par, 2, 0.5, 1., 5.)
vmax = mean_H.max()
vmin = mean_H.min()
FC = (mean_H - vmin) / (vmax - vmin)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=plt.cm.YlGnBu(FC))
surf.set_edgecolors("k")
ax.set_title("Mean curvature", fontsize=18)
ax.set_zlim(-2, 2);

from IPython.core.display import HTML
def css_styling():
    styles = open('./styles/custom_barba.css', 'r').read()
    return HTML(styles)
css_styling()



