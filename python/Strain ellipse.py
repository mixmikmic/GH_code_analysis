from __future__ import division
from IPython.display import Image
from IPython.display import HTML
import math as m
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as pywid
import os,sys
get_ipython().magic('matplotlib inline')

def transform(D11,D12,D21,D22):
    # rectangle
    o_r = np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])
    d_r = np.array([[D11*o_r[0,0]+D12*o_r[0,1],D21*o_r[0,0]+D22*o_r[0,1]],
                    [D11*o_r[1,0]+D12*o_r[1,1],D21*o_r[1,0]+D22*o_r[1,1]],
                    [D11*o_r[2,0]+D12*o_r[2,1],D21*o_r[2,0]+D22*o_r[2,1]],
                    [D11*o_r[3,0]+D12*o_r[3,1],D21*o_r[3,0]+D22*o_r[3,1]], 
                    [D11*o_r[4,0]+D12*o_r[4,1],D21*o_r[4,0]+D22*o_r[4,1]]])
    
    # circle, ellipse
    deg = np.linspace(0,360,360)
    r2d = np.pi/180.
    angle = deg*r2d
    circ = np.zeros([360,2])
    circ[:,0] = np.cos(angle)
    circ[:,1] = np.sin(angle)
    
    ell = np.zeros([360,2])
    ell[:,0] = circ[:,0]*D11 + circ[:,1]*D12
    ell[:,1] = circ[:,0]*D21 + circ[:,1]*D22
    
    # plot the figure
    fig = plt.figure(figsize=(10,10))
    plt.plot(o_r[:,0],o_r[:,1],'-',linewidth=2)
    plt.plot(d_r[:,0],d_r[:,1],'r-',linewidth=2)
    plt.plot(circ[:,0],circ[:,1],'-g',linewidth=1.3)
    plt.plot(ell[:,0],ell[:,1],'-c',linewidth=1.3)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.xlabel("x$_1$",fontsize=20)
    plt.ylabel("x$_2$",fontsize=20)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.gca().set_aspect('equal', adjustable='box')
    

pywid.interact(transform,D11=(-3.,3.), D12=(-3.,3.), D21=(-3.,3.), D22=(-3.,3.))

