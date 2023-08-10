get_ipython().run_line_magic('matplotlib', 'inline')
# plots graphs within the notebook
get_ipython().run_line_magic('config', "InlineBackend.figure_format='svg' # not sure what this does, may be default images to svg format")

from IPython.display import display,Image, Latex
from __future__ import division
from sympy.interactive import printing
printing.init_printing(use_latex='mathjax')
from IPython.display import clear_output

import time

from IPython.display import display,Image, Latex

from IPython.display import clear_output


import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.constants as sc
import h5py

import sympy as sym

    
font = {'family' : 'serif',
        #'color'  : 'black',
        'weight' : 'normal',
        'size'   : 12,
        }
fontlabel = {'family' : 'serif',
        #'color'  : 'black',
        'weight' : 'normal',
        'size'   : 16,
        }

from matplotlib.ticker import FormatStrFormatter
plt.rc('font', **font)

class PDF(object):
  def __init__(self, pdf, size=(200,200)):
    self.pdf = pdf
    self.size = size

  def _repr_html_(self):
    return '<iframe src={0} width={1[0]} height={1[1]}></iframe>'.format(self.pdf, self.size)

  def _repr_latex_(self):
    return r'\includegraphics[width=1.0\textwidth]{{{0}}}'.format(self.pdf)

class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of 
        the form [[1,2,3],[4,5,6]], and renders an HTML Table in 
        IPython Notebook. """
    
    def _repr_html_(self):
        html = ["<table>"]
        for row in self:
            html.append("<tr>")
            
            for col in row:
                html.append("<td>{0}</td>".format(col))
            
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)
    
font = {'family' : 'serif',
        #'color'  : 'black',
        'weight' : 'normal',
        'size'   : 12,
        }
fontlabel = {'family' : 'serif',
        #'color'  : 'black',
        'weight' : 'normal',
        'size'   : 16,
        }

from matplotlib.ticker import FormatStrFormatter
plt.rc('font', **font)

""" Compute the derivation matrix for derivatives on a non-uniform grid
    Step 1: Given a mesh x, compute the matrix D = nufd(x)
    Step 2: Compute derivative of u defined on mesh x: dxu = D*u
"""
from scipy.sparse import csr_matrix
def nufd(x):
    n = len(x)
    h = x[1:]-x[:n-1]
    a0 = -(2*h[0]+h[1])/(h[0]*(h[0]+h[1]))
    ak = -h[1:]/(h[:n-2]*(h[:n-2]+h[1:]))
    an = h[-1]/(h[-2]*(h[-1]+h[-2]))
    b0 = (h[0]+h[1])/(h[0]*h[1]) 
    bk = (h[1:] - h[:n-2])/(h[:n-2]*h[1:])
    bn = -(h[-1]+h[-2])/(h[-1]*h[-2])
    c0 = -h[0]/(h[1]*(h[0]+h[1]))
    ck = h[:n-2]/(h[1:]*(h[:n-2]+h[1:]))
    cn = (2*h[-1]+h[-2])/(h[-1]*(h[-2]+h[-1]))
    val  = np.hstack((a0,ak,an,b0,bk,bn,c0,ck,cn))
    row = np.tile(np.arange(n),3)
    dex = np.hstack((0,np.arange(n-2),n-3))
    col = np.hstack((dex,dex+1,dex+2))
    D = csr_matrix((val,(row,col)),shape=(n,n))
    return D

""" Compute the velocity gradient vector A[i,j,:,:] where
    i is the direction of derivation
    j is the index denoting the velocity component
    the last two indices are the z (wall normal) and x (horizontal) coordinates of computational nodes
"""
def compute_A(u,w):
    global Nx,Nz,dx,Dzu,dzpin
    A = np.zeros((2,2,Nz,Nx))
    #A[i,j,:,:] =d_iu_j
    A[0,0,:,:] = (u[1:-1,1:-1] - u[1:-1,0:-2])/dx #dxu
    A[0,1,:,:] = 0.5*(w[0:-2,1:-1] - w[0:-2,0:-2]                      +w[1:-1,1:-1] - w[1:-1,0:-2])/dx
    dzu = np.zeros((Nz+2,Nx+2))
    for i in range(Nx+2):
        dzu[:,i] = Dzu*u[:,i]
    A[1,0,:,:] = 0.5*(dzu[1:-1,1:-1]+dzu[1:-1,0:-2])
    A[1,1,:,:] = (w[1:-1,1:-1] - w[0:-2,1:-1])/dzpin[:,:]
    return A

""" Read mesh
"""
folder = "./Data/"
file = h5py.File(folder+'grid.h5','r+')
XZ = file.get('/Coor')
XZ = np.array(XZ)
file.close()
Nx = np.shape(XZ)[2]-2
Nz = np.shape(XZ)[1]-2
zu = XZ[3,:,0]
zw = XZ[4,:,0]
zp = XZ[5,:,0]
dx = XZ[0,0,1] - XZ[0,0,0]
Xp = XZ[2,1:-1,1:-1]
Zp = XZ[5,1:-1,1:-1]
Dzu = nufd(zu)
Dzw = nufd(zw)
Dzp = nufd(zp)
dzpin = XZ[5,1:-1,1:-1] - XZ[5,0:-2,1:-1]

""" Plot Temperature contours and velocity vectors
"""

# Interpolation to avoid having too many vectors
Nxnew = 64
Nznew = 33
from scipy import interpolate
xpold = np.copy(Xp[0,:])
zpold = np.copy(Zp[:,0])
xnew = np.linspace(xpold[0],xpold[-1] ,Nxnew)
znew = np.linspace(zpold[0],zpold[-1], Nznew)
Xnew, Znew = np.meshgrid(xnew,znew)
def interpolate_field(qp):
    global xpold, zpold, xnew, znew
    f = interpolate.RectBivariateSpline(zpold, xpold, qp)
    return f(znew,xnew)

# fname = "02.007.h5"
# pltname = "fig.02.007.png"
fname = "02.029.h5"
pltname = "fig.02.029.png"
file = h5py.File(folder+fname,'r')
u = file['u'][:]
w = file['w'][:]
p = file['p'][:]
T = file['T'][:]
Re = file['u'].attrs['Re']
Pr = file['u'].attrs['Pr']
tlocal = file['u'].attrs['time']
file.close()

# compute variables at cell centers
up = np.zeros((Nz,Nx))
Tp = np.copy(T[1:-1,1:-1])
pp = np.copy(p[1:-1,1:-1])
wp = np.zeros((Nz,Nx))
wp = 0.5*(w[1:-1,1:-1]+w[0:-2,1:-1])
up = 0.5*(u[1:-1,1:-1]+u[1:-1,0:-2])

fig = plt.figure(num=None, figsize=(10, 10), dpi=160, facecolor='w', edgecolor='k')
ax0 = plt.subplot2grid((1, 2), (0, 0))
ax1 = plt.subplot2grid((1, 2), (0, 1))



upnew = interpolate_field(up)
wpnew = interpolate_field(wp)
magvel = np.sqrt(np.power(upnew,2)+np.power(wpnew,2))
lev = np.linspace(0,0.4,41)
tickcmp = np.linspace(0,0.4,5)
im0 = ax0.quiver(Xnew, Znew, upnew, wpnew, magvel, pivot='mid', scale = 1.5
                 ,clim = [0, 0.4] , cmap = 'gist_ncar')
fig.colorbar(im0, ax = ax0, ticks = tickcmp, orientation = 'horizontal')
ax0.set_aspect('equal')
ax0.set_xlabel('$x$', fontdict = fontlabel)
ax0.set_ylabel('$z$', fontdict = fontlabel)
ax0.set_title(r"Velocity vectors", fontdict=fontlabel)
ax0.set_ylim(-0.5,0.5)
ax0.set_xlim(-1,1)
    

lev = np.linspace(0,1,41)
tickcmp = np.linspace(0,1,5)
im1 = ax1.contourf(Xp, Zp, Tp, levels=lev, cmap = 'gist_ncar')
ax1.set_aspect('equal')
ax1.set_xlabel('$x$', fontdict = fontlabel)
fig.colorbar(im1, ax = ax1, ticks = tickcmp, orientation = 'horizontal')
ax1.set_title(r"$T$", fontdict=fontlabel)
ax1.set_ylim(-0.5,0.5)
ax1.set_xlim(-1,1)
    

# plt.savefig(pltname, bbox_inches='tight')
plt.show()

    

# Interpolation to avoid having too many vectors
Nxnew = 32
Nznew = 16
from scipy import interpolate
xpold = np.copy(Xp[0,:])
zpold = np.copy(Zp[:,0])
xnew = np.linspace(xpold[0],xpold[-1] ,Nxnew)
znew = np.linspace(zpold[0],zpold[-1], Nznew)
Xnew, Znew = np.meshgrid(xnew,znew)
def interpolate_field(qp):
    global xpold, zpold, xnew, znew
    f = interpolate.RectBivariateSpline(zpold, xpold, qp)
    return f(znew,xnew)
fname = "02.007.h5"
pltname = "figvortp.02.007.png"
file = h5py.File(folder+fname,'r')
u = file['u'][:]
w = file['w'][:]
p = file['p'][:]
T = file['T'][:]
Re = file['u'].attrs['Re']
Pr = file['u'].attrs['Pr']
tlocal = file['u'].attrs['time']
file.close()

# compute variables at cell centers
up = np.zeros((Nz,Nx))
Tp = np.copy(T[1:-1,1:-1])
pp = np.copy(p[1:-1,1:-1])
wp = np.zeros((Nz,Nx))
wp = 0.5*(w[1:-1,1:-1]+w[0:-2,1:-1])
up = 0.5*(u[1:-1,1:-1]+u[1:-1,0:-2])

""" This is where you should compute the pressure fluctuations"""

pmean = np.mean(pp, axis = 1)

for k in range(Nz):
    pp[k,:] -= pmean[k]


fig = plt.figure(num=None, figsize=(10, 10), dpi=160, facecolor='w', edgecolor='k')
ax0 = plt.subplot2grid((1, 2), (0, 0))
ax1 = plt.subplot2grid((1, 2), (0, 1))

A = np.zeros((2,2,Nz,Nx))
A = compute_A(u,w)
""" This is where you should compute vorticity"""
omega = np.zeros((Nz,Nx))

omega = A[0,1,:,:] - A[1,0,:,:]
upnew = interpolate_field(up)
wpnew = interpolate_field(wp)

lev = np.linspace(-2,2,41)
tickcmp = np.linspace(-3,3,5)
im0 = ax0.contourf(Xp, Zp, omega, levels = lev, cmap = 'gist_ncar')
ax0.set_aspect('equal')
ax0.set_xlabel('$x$', fontdict = fontlabel)
fig.colorbar(im0, ax = ax0, ticks = tickcmp, orientation = 'horizontal')
ax0.set_title(r"$\omega$", fontdict=fontlabel)
ax0.set_ylim(-0.5,0.5)
ax0.set_xlim(-1,1)
ax0.quiver(Xnew, Znew, upnew, wpnew, pivot='mid', scale = 1.5)
    


lev = np.linspace(-0.32,0.32,41)
tickcmp = np.linspace(-0.32,0.32,5)
im1 = ax1.contourf(Xp, Zp, pp, levels = lev, cmap = 'gist_ncar')
ax1.set_aspect('equal')
ax1.set_xlabel('$x$', fontdict = fontlabel)
fig.colorbar(im1, ax = ax1, ticks = tickcmp, orientation = 'horizontal')
ax1.set_title(r"$p$", fontdict=fontlabel)
ax1.set_ylim(-0.5,0.5)
ax1.set_xlim(-1,1)
ax1.quiver(Xnew, Znew, upnew, wpnew, pivot='mid', scale = 1.5)
    

# plt.savefig(pltname, bbox_inches='tight')
plt.show()

# Interpolation to avoid having too many vectors
Nxnew = 32
Nznew = 16
from scipy import interpolate
xpold = np.copy(Xp[0,:])
zpold = np.copy(Zp[:,0])
xnew = np.linspace(xpold[0],xpold[-1] ,Nxnew)
znew = np.linspace(zpold[0],zpold[-1], Nznew)
Xnew, Znew = np.meshgrid(xnew,znew)
def interpolate_field(qp):
    global xpold, zpold, xnew, znew
    f = interpolate.RectBivariateSpline(zpold, xpold, qp)
    return f(znew,xnew)
fname = "02.007.h5"
pltname = "figvortp.02.007.png"
file = h5py.File(folder+fname,'r')
u = file['u'][:]
w = file['w'][:]
p = file['p'][:]
T = file['T'][:]
Re = file['u'].attrs['Re']
Pr = file['u'].attrs['Pr']
tlocal = file['u'].attrs['time']
file.close()

# compute variables at cell centers
up = np.zeros((Nz,Nx))
Tp = np.copy(T[1:-1,1:-1])
pp = np.copy(p[1:-1,1:-1])
wp = np.zeros((Nz,Nx))
wp = 0.5*(w[1:-1,1:-1]+w[0:-2,1:-1])
up = 0.5*(u[1:-1,1:-1]+u[1:-1,0:-2])

pmean = np.zeros(Nz)
pmean = np.mean(pp, axis = 1)

for k in range(Nz):
    pp[k,:] -= pmean[k]

fig = plt.figure(num=None, figsize=(10, 10), dpi=160, facecolor='w', edgecolor='k')
ax0 = plt.subplot2grid((1, 2), (0, 0))
ax1 = plt.subplot2grid((1, 2), (0, 1))

A = np.zeros((2,2,Nz,Nx))
A = compute_A(u,w)

omega = np.zeros((Nz,Nx))
omega = A[0,1,:,:] - A[1,0,:,:]

upnew = interpolate_field(up)
wpnew = interpolate_field(wp)

lev = np.linspace(-2,2,41)
tickcmp = np.linspace(-2,2,5)
im0 = ax0.contourf(Xp, Zp, omega, levels = lev, cmap = 'gist_ncar')
ax0.set_aspect('equal')
ax0.set_xlabel('$x$', fontdict = fontlabel)
fig.colorbar(im0, ax = ax0, ticks = tickcmp, orientation = 'horizontal')
ax0.set_title(r"$\omega$", fontdict=fontlabel)
ax0.set_ylim(-0.5,0.5)
ax0.set_xlim(-1,1)
ax0.quiver(Xnew, Znew, upnew, wpnew, pivot='mid', scale = 1.5)
    

lev = np.linspace(-0.025,0.025,41)
tickcmp = np.linspace(-0.025,0.025,5)
# lev = np.linspace(-0.32,0.32,41)
# tickcmp = np.linspace(-0.32,0.32,5)
im1 = ax1.contourf(Xp, Zp, pp, levels = lev, cmap = 'gist_ncar')
ax1.set_aspect('equal')
ax1.set_xlabel('$x$', fontdict = fontlabel)
fig.colorbar(im1, ax = ax1, ticks = tickcmp, orientation = 'horizontal')
ax1.set_title(r"$p$", fontdict=fontlabel)
ax1.set_ylim(-0.5,0.5)
ax1.set_xlim(-1,1)
ax1.quiver(Xnew, Znew, upnew, wpnew, pivot='mid', scale = 1.5)
    

# plt.savefig(pltname, bbox_inches='tight')
plt.show()

# Interpolation to avoid having too many vectors
Nxnew = 32
Nznew = 16
from scipy import interpolate
xpold = np.copy(Xp[0,:])
zpold = np.copy(Zp[:,0])
xnew = np.linspace(xpold[0],xpold[-1] ,Nxnew)
znew = np.linspace(zpold[0],zpold[-1], Nznew)
Xnew, Znew = np.meshgrid(xnew,znew)
def interpolate_field(qp):
    global xpold, zpold, xnew, znew
    f = interpolate.RectBivariateSpline(zpold, xpold, qp)
    return f(znew,xnew)
fname = "02.007.h5"
pltname = "figvortp.02.007.png"
file = h5py.File(folder+fname,'r')
u = file['u'][:]
w = file['w'][:]
p = file['p'][:]
T = file['T'][:]
Re = file['u'].attrs['Re']
Pr = file['u'].attrs['Pr']
tlocal = file['u'].attrs['time']
file.close()

# compute variables at cell centers
up = np.zeros((Nz,Nx))
Tp = np.copy(T[1:-1,1:-1])
pp = np.copy(p[1:-1,1:-1])
wp = np.zeros((Nz,Nx))
wp = 0.5*(w[1:-1,1:-1]+w[0:-2,1:-1])
up = 0.5*(u[1:-1,1:-1]+u[1:-1,0:-2])

""" This is where you should compute the pressure fluctuations"""
pmean = np.zeros(Nz)
pmean = np.mean(pp, axis = 1)

for k in range(Nz):
    pp[k,:] -= pmean[k]
fig = plt.figure(num=None, figsize=(10, 10), dpi=160, facecolor='w', edgecolor='k')
ax0 = plt.subplot2grid((1, 2), (0, 0))
ax1 = plt.subplot2grid((1, 2), (0, 1))

A = np.zeros((2,2,Nz,Nx))
A = compute_A(u,w)
""" This is where you should compute Q"""
Q = np.zeros((Nz,Nx))

for i in range(2):
    for j in range(2):
        Q += -0.5*(A[i,j,:,:]*A[j,i,:,:])

upnew = interpolate_field(up)
wpnew = interpolate_field(wp)

lev = np.linspace(-0.64,0.64,41)
tickcmp = np.linspace(-0.64,0.64,5)
im0 = ax0.contourf(Xp, Zp, Q, levels = lev, cmap = 'gist_ncar')
ax0.set_aspect('equal')
ax0.set_xlabel('$x$', fontdict = fontlabel)
fig.colorbar(im0, ax = ax0, ticks = tickcmp, orientation = 'horizontal')
ax0.set_title(r"$Q$", fontdict=fontlabel)
ax0.set_ylim(-0.5,0.5)
ax0.set_xlim(-1,1)
ax0.quiver(Xnew, Znew, upnew, wpnew, pivot='mid', scale = 1.5)
    


lev = np.linspace(-0.32,0.32,41)
tickcmp = np.linspace(-0.32,0.32,5)
im1 = ax1.contourf(Xp, Zp, pp, levels = lev, cmap = 'gist_ncar')
ax1.set_aspect('equal')
ax1.set_xlabel('$x$', fontdict = fontlabel)
fig.colorbar(im1, ax = ax1, ticks = tickcmp, orientation = 'horizontal')
ax1.set_title(r"$p$", fontdict=fontlabel)
ax1.set_ylim(-0.5,0.5)
ax1.set_xlim(-1,1)
ax1.quiver(Xnew, Znew, upnew, wpnew, pivot='mid', scale = 1.5)
    

# plt.savefig(pltname, bbox_inches='tight')
plt.show()



