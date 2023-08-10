from __future__ import division, unicode_literals, print_function
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from ipywidgets import interact
from math import sqrt, pi
plt.rcParams.update({'font.size': 14, 'figure.figsize': [6.0, 4.0]})

# Debye-Huckel
def wDH(r, z, D):
    lB = 7.0 # Bjerrum length, angstroms
    return lB * z**2 * np.exp(-r/D) / r

# Lennard-Jones
def wLJ(r, eps, sigma):
    return 4 * eps * ( (sigma/r)**12 - (sigma/r)**6 )

# Total potential
def w(r, z, D, eps, sigma):
    return wDH(r, z, D) + wLJ(r, eps, sigma)

def ahat(z, D, eps, sigma):
    return -2*pi*quad(lambda r: w(r, z, D, eps, sigma)*r**2, sigma, np.infty, limit=50)[0]

def Pideal(n):
    return n

def Pgvdw(n, z, D, eps, sigma):
    v0 = 2*pi*sigma**3 / 3
    v  = 1 / n
    a  = ahat(z, D, eps, sigma)
    return 1/(v-v0) - a/v**2

def plot_EOS( eps=1.0, sigma=4.0, z=0.0, Cs=0.3 ):
    D = 3.04/sqrt(Cs)
    plt.plot(n, Pideal(n),  'k--', label='ideal', lw=2)
    plt.plot(n, Pgvdw(n, z, D, eps, sigma),  'r-', label='gvdW', lw=2)
    plt.title('Equation of State')
    plt.xlabel(r'Number density, $n$')
    plt.ylabel(r'Pressure, $\beta p$')
    plt.legend(loc=0, frameon=False)
    
n = np.linspace(1e-7, 6e-3, 100)
    
i = interact(plot_EOS,
             eps=(0.0, 10.0, 0.1), sigma=(0, 5, 0.1), z=(0.0, 10, 1.0), Cs=(1e-3, 1.0, 0.1) )
 
i.widget.children[0].description=r'$\beta\varepsilon_{LJ}$'
i.widget.children[1].description=r'$\sigma_{LJ}$'
i.widget.children[2].description=r'$z$'
i.widget.children[3].description=r'$c_s$ (M)'

