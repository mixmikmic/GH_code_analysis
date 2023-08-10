get_ipython().run_line_magic('load_ext', 'base16_mplrc')
get_ipython().run_line_magic('base16_mplrc', 'dark bespin')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.ndimage import gaussian_filter1d
from scipy import signal

import seaborn as sns
from seaborn import xkcd_palette as xkcd
blue, red, green = xkcd(['cornflower','bright red','bluish green'])

get_ipython().run_line_magic('matplotlib', 'inline')

from lorenz_rhs import lorenz, get_lorenz_solution

in_0 = [5.0, 5.0, 5.0]
t_max = 100
dt = 0.01
t_steps = t_max/dt
t, [solx, soly, solz] = get_lorenz_solution(in_0, t_max, t_steps, 
                                            (10.0, 8/3, 28))

def wavelet_column(t, sol, title, z=8):
    n = len(sol)
    k = np.arange(n)
    y = np.fft.fft(sol)
    
    widths = np.arange(1, 2**z - 1)
    
    # signal.ricker is the Mexican hat wavelet
    cwtmatr = signal.cwt(sol, signal.ricker, widths)
    
    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(111)
    col = ax.imshow(cwtmatr, extent=[0, n, 2**z - 1, 1], cmap='PRGn', aspect='auto',
              vmax=abs(cwtmatr).max()/1.5, vmin=-abs(cwtmatr).max()/1.5)
    ax.set_title("Wavelet: " + title)
    ax.set_ylabel('Wavelet Bandwidth')
    
    plt.colorbar(col)
    
    return cwtmatr

cwtmatx = wavelet_column(t, solx, "x", 8)

cwtmaty = wavelet_column(t, soly, "y", 8)

cwtmatz = wavelet_column(t, solz, "z")

def multiple_wavelets(ts, solutions, titles, z=8):
    
    if(len(ts)!=4 or len(solutions)!=4):
        raise Exception("Error: need four times and four solutions")
    
    fig = plt.figure(figsize=(14,8))
    
    axes = [fig.add_subplot(2,2,i+1) for i in range(4)]
    
    for ax, t, sol, title in zip(axes, ts, solutions, titles):
    
        n = len(sol)
        k = np.arange(n)
        y = np.fft.fft(sol)
        
        widths = np.arange(1, 2**z - 1)
        cwtmatr = signal.cwt(sol, signal.ricker, widths)
        
        col = ax.imshow(cwtmatr, extent=[0, n, 2**z - 1, 1], cmap='PRGn', aspect='auto',
                        vmax=abs(cwtmatr).max()/1.5, vmin=-abs(cwtmatr).max()/1.5)
        
        ax.set_title(title)
        ax.set_ylabel('Wavelet Bandwidth')
        
        plt.colorbar(col, ax=ax)

    return cwtmatr

in_0 = [5.0, 5.0, 5.0]
t_max = 100
dt = 0.01
t_steps = t_max/dt
rz = list(np.linspace(25,40,4))

ts = []
sols = []

for r in rz:
    t, sol = get_lorenz_solution(in_0, t_max, t_steps, (10.0, 8/3, r))
    ts.append(t)
    sols.append(sol)

cw = multiple_wavelets(ts, [sol[0] for sol in sols], ["Multiple Wavelets: %0.1f"%(r) for r in rz], 9)
plt.tight_layout()

from ipywidgets import interact, interactive, fixed

def make_wavelet_column(sigma=10.0, b=8.0/3.0, r=30, z=8):
    
    in_0 = [5.0, 5.0, 5.0]
    t_max = 100
    dt = 0.01
    t_steps = t_max/dt
    t, [solx, soly, solz] = get_lorenz_solution(in_0, t_max, t_steps, (sigma, b, r))
    
    sol = solx
    n = len(sol)
    k = np.arange(n)
    y = np.fft.fft(sol)
    
    widths = np.arange(1, 2**z - 1)
    
    cwtmatr = signal.cwt(sol, signal.ricker, widths)
    
    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(111)
    col = ax.imshow(cwtmatr, extent=[0, n, 2**z - 1, 1], cmap='cool', aspect='auto',
              vmax=abs(cwtmatr).max()/1.5, vmin=-abs(cwtmatr).max()/1.5)
    ax.set_title("Wavelet")
    ax.set_ylabel('Wavelet Bandwidth')
    
    plt.colorbar(col)
    
    #return cwtmatr

sigmalims = (5.0, 20.0, 1.0)
rlims = (25.0, 40.0, 5.0)
z = (7,9,1)
w = interactive(make_wavelet_column, sigma=sigmalims, z=9)
display(w)

# Pre-compute a series of wavelets for a given signal,
# to make interaction faster.

# Sigma will be the slider variable
sigma_start  = 10.0000
sigma_end    = 10.0001
sigma_steps  = 50
sigma_ds     = (sigma_end-sigma_start)/sigma_steps
all_sigmas   = np.linspace(sigma_start, sigma_end, sigma_steps)
all_solns    = []
all_wavelets = []

in_0 = [5.0, 5.0, 5.0]
t_max = 100
dt = 0.01
t_steps = t_max/dt
b = 8/3
r = 28
z = 8

for sigma in all_sigmas:
    t, [solx, soly, solz] = get_lorenz_solution(in_0, t_max, t_steps, (sigma, b, r))
    sol = solx
    n = len(sol)
    k = np.arange(n)
    y = np.fft.fft(sol)
    widths = np.arange(1, 2**z - 1)
    cwtmat = signal.cwt(sol, signal.ricker, widths)
    all_wavelets.append(cwtmat)

def make_wavelet_sigma(ix):
    cwtmat = all_wavelets[ix]    
    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(111)
    col = ax.imshow(cwtmat, extent=[0, n, 2**z - 1, 1], 
                    cmap='cool', aspect='auto',
                    vmax=abs(cwtmat).max()/1.5, 
                    vmin=-abs(cwtmat).max()/1.5)
    
    plt.colorbar(col)
    
    ax.set_title("Wavelet")
    ax.set_ylabel('Wavelet Bandwidth')

#sigma_slider = (sigma_start, sigma_end, 1.0)
ix_slider = (0,sigma_steps,1)
w = interactive(make_wavelet_sigma, ix=ix_slider)
display(w)



