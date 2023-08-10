from __future__ import division
from IPython.display import Image
from IPython.display import HTML
import math as m
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as pywid
import os,sys
get_ipython().magic('matplotlib inline')

# initial length
lA = 1.
# length after deformation
lAD = 1.2

e = (lAD - lA)/lA  
lam = (1 + e)**2
lamP = 1/lam  

print('extension is %.2f' % e)
print("quadratic extension is %.2f" % lam)
print("reciprocal quadratic extension is %.2f" % lamP)

eps = m.log(lAD/lA)
eps_lam = m.log(1+e)

psi = 63
gam = m.tan(psi*m.pi/180.)
gamP = gam/lam
print("logarithmic strain is %.3f" % eps)
print("shear strain is %.3f" % gam)
print("gamma prime is %.3f" % gamP)

def plot_rectangles(psi):
    
    gam = m.tan(psi*m.pi/180.)
    orig_rectangle = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])
    def_rectangle = np.array([[0,0],[1,0],[gam+1,1],[gam,1],[0,0]])
    
    fig = plt.figure(figsize=(15,10))
    plt.plot(orig_rectangle[:,0],orig_rectangle[:,1],'-',linewidth=2)
    plt.plot(def_rectangle[:,0],def_rectangle[:,1],'r-',linewidth=2)
    plt.xlim([-0.25, 3])
    plt.ylim([-0.25,1.25])
    plt.xlabel("x",fontsize=16)
    plt.ylabel("y",fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.gca().set_aspect('equal', adjustable='box')

pywid.interact(plot_rectangles,psi=(0.,63.))

def transform(D11,D12,D21,D22,x1,x2):
    
    x1P = D11*x1 + D12*x2
    x2P = D21*x1 + D22*x2
    
    fig = plt.figure(figsize=(10,10))
    plt.scatter(x1,x2,marker='o',s=100)
    plt.scatter(x1P,x2P,marker='o',c='r',s=100)
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.xlabel("x$_1$",fontsize=20)
    plt.ylabel("x$_2$",fontsize=20)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.gca().set_aspect('equal', adjustable='box')

pywid.interact(transform,D11=(-3,3), D12=(-3,3), D21=(-3,3), D22=(-3,3), x1=(0,5), x2=(0,5))

