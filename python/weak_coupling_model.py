get_ipython().magic('matplotlib inline')
from itertools import product
from functools import wraps

import numpy as np
from lll import lll
import matplotlib.pyplot as plt
import plotly.plotly as py
import scipy
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, iplot_mpl
from ipywidgets import interact

init_notebook_mode()

def ham_1d(k, V, n):
    k_array = k*np.ones(n)
    r_momenta = np.linspace(-np.pi, np.pi, n + 2)
    ham_free = np.diag(np.subtract(k_array, r_momenta[1:-1])**2)
    ham_coupling = V*(np.ones((n, n)) - np.eye(n))
    ham = ham_free + ham_coupling
    eig = scipy.linalg.eigvalsh(ham)
    return eig
 
    
def ham_2d(kx, ky, V, n):
    k1 = np.outer(np.array([1, 0]), np.arange(-n, n+1))
    k2 = np.outer(np.array([0, 1]), np.arange(-n, n+1))
    k_offsets = (k1.reshape(2, 1, -1) + k2.reshape(2, -1, 1)).reshape(2, -1)
    k = np.array(([kx, ky]))
    arr = np.sum((k_offsets - k.reshape(2, 1))**2, axis=0)
    ham_free = np.diag(arr)
    ham_coupling = V*(np.ones((len(arr), len(arr))) - np.eye(len(arr)))
    ham = ham_free + ham_coupling
    eig = scipy.linalg.eigvalsh(ham)
    return eig


def energy_spectrum_1d(V, n):
    k_range = np.linspace(-1, 1, 200)
    energies = [ham_1d(k, V, n) for k in k_range]
    energy_data = np.array(energies).reshape((len(k_range), n))
    return energy_data


def energy_spectrum_2d(V, n):
    k_rangex = np.linspace(-np.pi, np.pi, 100)
    k_rangey = np.linspace(-np.pi, np.pi, 100)
    energies = []
    for kx in k_rangex:
        for ky in k_rangey:
            energies.append(ham_2d(kx, ky, V, n))
    data_2d = np.array(energies).reshape((len(k_rangex), len(k_rangey), (2*n + 1)**2))
    #if V is not 0:
    #    data_2d = np.sort(data_2d)
    return data_2d


def plot_1d_bands(data_1d, n):
    k_range = np.linspace(-1, 1, 200)
    layout = go.Layout(showlegend=False,
            autosize=False,
            width=700,
            height=500,
        xaxis=dict(
            title = 'k',
            titlefont = dict(size=20),
            showgrid=True,
            zeroline=False,
            showline=True,
            autotick=True,
            ticks='',
            showticklabels=True
        ),
        yaxis=dict(
            title = 'energy',
            titlefont = dict(size=20),
            autorange=True,
            showgrid=True,
            zeroline=False,
            showline=True,
            autotick=True,
            ticks='',
            showticklabels=True
        )
    )
    data = [go.Scatter(x=k_range, y=data_1d[:, i]) for i in range(n)]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, show_link=False)  
    
    
def plot_2d_bands(data_2d, n):
    k_rangex = np.linspace(-np.pi, np.pi, 100)
    k_rangey = np.linspace(-np.pi, np.pi, 100)
    mband = ((2*n + 1)**2)/2
    data = [go.Contour(z=data_2d[:, :, mband], x=k_rangex, y=k_rangey)]
    layout = go.Layout(showlegend=False,
                        autosize=False,
                        width=600,
                        height=600,
                        xaxis=dict(title='k_x',
                            autorange=True,
                            showgrid=False,
                            zeroline=False,
                            showline=True),
                        yaxis=dict(title='k_y',
                            autorange=True,
                            showgrid=False,
                            zeroline=False,
                            showline=True))
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, show_link=False)

n = 5
V = 0.

data_1d = energy_spectrum_1d(V, n)

plot_1d_bands(data_1d, n)

n = 1
V = 0

data_2d = energy_spectrum_2d(V, n)

plot_2d_bands(data_2d, n)









