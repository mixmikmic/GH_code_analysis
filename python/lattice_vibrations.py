from functools import wraps

import numpy as np
from scipy import linalg
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
from plotly.tools import FigureFactory as FF
from ipywidgets import interact

init_notebook_mode()

def store_last_correct(func):
    last_correct = None
    
    @wraps(func)
    def wrapped(*args, **kwargs):
        nonlocal last_correct
        try:
            last_correct = func(*args, **kwargs)
            return last_correct
        except:
            return last_correct
    
    return wrapped


# Define all the functions to compute phonon dispersion and wave modes

def spring_matrix(springs, k):
    springs = np.asanyarray(springs)
    # Add intra unit cell spring constants
    K = -np.diag(springs + 0j) - np.diag(np.roll(springs, 1))
    K += np.diag(springs[:-1], 1) + np.diag(springs[:-1], -1)
    K[0, -1] += springs[-1] * np.exp(1j * k)
    K[-1, 0] += springs[-1] * np.exp(-1j * k)
    return K


def phonon_dispersion(masses, springs):
    M = np.diag(masses)
    momenta = np.linspace(-np.pi, np.pi, 101)
    frequencies = []
    for k in momenta:
        K = spring_matrix(springs, k)
        eigvals, eigvecs = linalg.eigh(K, M) 
        eigvals, eigvecs = eigvals[::-1], eigvecs[::-1]
        frequencies.append(np.sqrt(-eigvals + 1e-13)) # Take machine precision into account
    return momenta, np.array(frequencies)


def phonon_modes(masses, springs, k, N):
    num_atoms = len(masses)
    M = np.diag(masses)
    K = spring_matrix(springs, k)
    eigvals, eigvecs = linalg.eigh(K, M)
    eigvals, eigvecs = eigvals[::-1], eigvecs[::-1]
    phase_vec = np.exp(1j*k*np.arange(N))
    eigvecs_cells = np.array([np.reshape(np.outer(eigvec, phase_vec), (N*num_atoms,)) for eigvec in eigvecs] )
    return eigvecs_cells


def visualize_phonons(masses, springs, k, N, y_only=False):
    momenta, dispersion = phonon_dispersion(masses, springs)
    data = []
    for i in range(len(masses)):
        trace = go.Scatter(
            x=momenta,
            y=dispersion[:,i],
            hoverinfo='none'
        )
        data.append(trace)
    layout = go.Layout(
            showlegend=False,
            autosize=False,
            width=600,
            height=300,
        xaxis=dict(
            title='k',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title='omega',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        shapes= [

            # Line Vertical
            {
                'type': 'line',
                'x0': k,
                'y0': min(dispersion[:, 0]),
                'x1': k,
                'y1': max(dispersion[:, -1]),
                'line': {
                    'color': 'rgb(0, 0, 0)',
                    'width': 3
                },
            }
            ]
    )
    fig = go.Figure(data=data, layout=layout)

    modes = phonon_modes(masses, springs, k, N)
    modes /= np.max(np.abs(modes), axis=1).reshape(-1, 1)

    modes_x, modes_y = 3 * modes.T.flatten().real, 3 * modes.T.flatten().imag

    x, y = np.meshgrid(np.arange(modes.shape[1], dtype=float),
                       np.arange(modes.shape[0], dtype=float))
    x, y = x.flatten(), -y.flatten()
    y += modes_x
    modes_x *= 0
    layout = go.Layout(
            autosize=False,
            width=600,
            height=300,
    xaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=False
    ),
    yaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=False)
        )
    quiver = FF.create_quiver(x, y, modes_x, modes_y)
    quiver.layout = layout
    quiver.data[0]['hoverinfo'] = 'none'
    iplot(fig, show_link=False)
    iplot(quiver, show_link=False)

#     return fig, quiver

visualize_phonons([1, 1], [2, 1], 0, 2)

@store_last_correct
def validate_input(masses, springs):
    masses, springs = np.array(eval(masses)), np.array(eval(springs))
    assert masses.shape == springs.shape and len(masses.shape) == 1
    return masses, springs


@interact(masses='[1, 1]', springs='[1, 2]', k=(-np.pi, np.pi, 0.01), N=(1, 10), y_only=False)
def plot_laue(masses, springs, k=0, N=3, y_only=False):
    masses, springs = validate_input(masses, springs)

    visualize_phonons(masses, springs, k, N, y_only)



