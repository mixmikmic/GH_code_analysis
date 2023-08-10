get_ipython().magic('matplotlib inline')
from SimPEG import Mesh
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1)
mesh = Mesh.TensorMesh([1,1])
mesh.plotGrid(ax=ax,centers=True,faces=True)
ax.set_xlim(-0.5,1.5)
ax.set_ylim(-0.5,1.5);

# mesh._getFacePxx??

# face-x minus side and the face-y minus side
P1 = mesh._getFacePxx()('fXm','fYm')
print( P1.todense() )

P1 = mesh._getFacePxx()('fXm','fYm')
P2 = mesh._getFacePxx()('fXp','fYm')
P3 = mesh._getFacePxx()('fXm','fYp')
P4 = mesh._getFacePxx()('fXp','fYp')
fig, ax = plt.subplots(1,4, figsize=(15,3))
def plot_projection(ii, ax, P):
    ax.spy(P, ms=30)
    if P.shape[1]==4:
        ax.set_xticks(range(4))
        ax.set_xticklabels(('x-','x+','y-','y+'))
    ax.set_xlabel('P{}'.format(ii+1))
map(plot_projection, range(4), ax, (P1, P2, P3, P4));

mesh = Mesh.TensorMesh([3,4])

P1 = mesh._getFacePxx()('fXm','fYm')
P2 = mesh._getFacePxx()('fXp','fYm')
P3 = mesh._getFacePxx()('fXm','fYp')
P4 = mesh._getFacePxx()('fXp','fYp')
fig, ax = plt.subplots(1,4, figsize=(15,6))
def plot_projection(ii, ax, P):
    x = P * np.r_[mesh.gridFx[:,0], mesh.gridFy[:,0]]
    y = P * np.r_[mesh.gridFx[:,1], mesh.gridFy[:,1]]
    xx, xy = x[:mesh.nC], x[mesh.nC:]
    yx, yy = y[:mesh.nC], y[mesh.nC:]
    ax.plot(np.c_[xx, mesh.gridCC[:,0], xx*np.nan].flatten(), np.c_[yx, mesh.gridCC[:,1], yx*np.nan].flatten(), 'k-')
    ax.plot(np.c_[xy, mesh.gridCC[:,0], xy*np.nan].flatten(), np.c_[yy, mesh.gridCC[:,1], yy*np.nan].flatten(), 'k-')
    ax.plot(xx, yx, 'g>')
    ax.plot(xy, yy, 'g^')
    mesh.plotGrid(ax=ax, centers=True)
    ax.set_title('P{}'.format(ii+1))
map(plot_projection, range(4), ax.flatten(), (P1, P2, P3, P4));
plt.tight_layout()

isotropic       = np.ones(mesh.nC)*4
anisotropic_vec = np.r_[np.ones(mesh.nC)*4, np.ones(mesh.nC)]
anisotropic     = np.r_[np.ones(mesh.nC)*4, np.ones(mesh.nC), np.ones(mesh.nC)*3]
Mf_sig_i = mesh.getFaceInnerProduct(isotropic)
Mf_sig_v = mesh.getFaceInnerProduct(anisotropic_vec)
Mf_sig_a = mesh.getFaceInnerProduct(anisotropic)
clim = (
    np.min(map(np.min, (Mf_sig_i.data, Mf_sig_v.data, Mf_sig_a.data))),
    np.max(map(np.max, (Mf_sig_i.data, Mf_sig_v.data, Mf_sig_a.data)))
)
fig, ax = plt.subplots(1,3, figsize=(15,4))
def plot_spy(ax, Mf, title):
    dense = Mf.toarray()
    dense[dense == 0] = np.nan
    ms = ax.matshow(dense)#, clim=clim)
    plt.colorbar(ms, ax=ax)
    ax.set_xlabel(title)
map(plot_spy, ax, (Mf_sig_i, Mf_sig_v, Mf_sig_a), ('Isotropic', 'Coordinate Anisotropy', 'Full Anisotropy'));

import sympy
from sympy.abc import x, y

# Here we will make up some j vectors that vary in space
j = sympy.Matrix([
    x**2+y*5,
    (5**2)*x+y*5
])

# Create an isotropic sigma vector
Sig = sympy.Matrix([
    [x*y*432/1163,      0      ],
    [     0      , x*y*432/1163]
])

# Do the inner product!
jTSj = j.T*Sig*j
ans  = sympy.integrate(sympy.integrate(jTSj, (x,0,1)), (y,0,1))[0] # The `[0]` is to make it an int.

print( "It is trivial to see that the answer is {}.".format(ans) )

def get_vectors(mesh):
    """Gets the vectors sig and [jx, jy] from sympy."""
    f_jx  = sympy.lambdify((x,y), j[0], 'numpy')
    f_jy  = sympy.lambdify((x,y), j[1], 'numpy')
    f_sig = sympy.lambdify((x,y), Sig[0], 'numpy')
    jx  = f_jx(mesh.gridFx[:,0], mesh.gridFx[:,1])
    jy  = f_jy(mesh.gridFy[:,0], mesh.gridFy[:,1])
    sig = f_sig(mesh.gridCC[:,0], mesh.gridCC[:,1])
    return sig, np.r_[jx, jy]

n = 5 # get's better if you add cells!
mesh = Mesh.TensorMesh([n,n])
sig, jv = get_vectors(mesh)
Msig = mesh.getFaceInnerProduct(sig)
numeric_ans = jv.T.dot(Msig.dot(jv))
print( "Numerically we get {}.".format(numeric_ans) )

import sys
import unittest
from SimPEG.Tests import OrderTest

class Testify(OrderTest):
    meshDimension = 2
    def getError(self):
        sig, jv = get_vectors(self.M)
        Msig = self.M.getFaceInnerProduct(sig)
        return float(ans) - jv.T.dot(Msig.dot(jv))
    def test_order(self):
        self.orderTest()

# This just runs the unittest:
suite = unittest.TestLoader().loadTestsFromTestCase( Testify )
unittest.TextTestRunner().run( suite );

