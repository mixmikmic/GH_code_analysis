get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')
matplotlib.rc('image', cmap='Greys_r')

from mpl_toolkits.axes_grid1 import make_axes_locatable
def colorbar(mappable):
    '''Helper for less-ugly colorbars on image data with forced-equal axes aspect ratio'''
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

import numpy as np

import os.path
import requests

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as w

import sklearn
from sklearn import decomposition
from sklearn import datasets

from astropy.io import fits

# Olivetti faces dataset
faces = datasets.fetch_olivetti_faces()  # Should take only a few seconds on a home connection

fig, axs = plt.subplots(nrows=4, ncols=10, figsize=(10, 4))
for i, axrow in enumerate(axs):
    for j, ax in enumerate(axrow):
        ax.imshow(faces['images'][10 * i  + j], cmap='Greys_r')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

@interact(i=w.IntSlider(value=252, min=0, max=faces['images'].shape[0] - 1, continuous_update=False))
def show_face(i):
    plt.imshow(faces['images'][i], cmap='Greys_r')
    plt.colorbar()

# Set up caches to ease interactivity
STRATEGIES = (decomposition.PCA, decomposition.NMF, decomposition.FastICA)
CACHE = {s: {} for s in STRATEGIES}

def decompose(strategy, n_components):
    if n_components not in CACHE[strategy]:
        CACHE[strategy][n_components] = strategy(n_components=n_components)
        CACHE[strategy][n_components].fit(faces['data'])
    return CACHE[strategy][n_components]

@interact(strategy=w.Dropdown(
    options=[('PCA', decomposition.PCA), ('FastICA', decomposition.FastICA), ('NMF', decomposition.NMF)],
    value=decomposition.PCA,
    description='Strategy:',
), n_components=w.IntSlider(value=8, min=1, max=80, continuous_update=False))
def show_components(n_components, strategy):
    cmap = 'Greys_r'
    fitter = decompose(strategy, n_components)
    components = fitter.components_.reshape((n_components,) + faces['images'].shape[1:])
#     cmin, cmax = np.min(components), np.max(components)
    n_per_row = 5
    nrows, ncols = 1 + n_components // n_per_row, min(n_per_row, n_components)
    fig = plt.figure(figsize=(2 * ncols, 1.5 * nrows))
    for i, comp in enumerate(components):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        im = ax.imshow(comp, cmap=cmap) #, vmin=cmin, vmax=cmax)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

@interact(
    n_components=w.IntSlider(value=80, min=1, max=80, continuous_update=False),
    face_id=w.IntSlider(value=252, min=0, max=faces['images'].shape[0] - 1),
)
def reconstruct(face_id, n_components):
    fig = plt.figure(figsize=(8, 3.5 * len(STRATEGIES)))
    for idx, strategy in enumerate(STRATEGIES):
        fitter = decompose(strategy, n_components)
        image = faces['images'][face_id]
        # transform
        coeffs = fitter.transform(faces['data'][face_id][np.newaxis,:])
        # invert transform
        outim = fitter.inverse_transform(coeffs)[0].reshape(faces['images'].shape[1:])
        # display
        original_ax = fig.add_subplot(len(STRATEGIES), 3, 3 * idx + 1)
        original_ax.imshow(image)
        reconstructed_ax = fig.add_subplot(len(STRATEGIES), 3, 3 * idx + 2)
        reconstructed_ax.imshow(outim)
        reconstructed_ax.set_title(fitter.__class__.__name__)
        diff_ax = fig.add_subplot(len(STRATEGIES), 3, 3 * idx + 3)
        diff_ax.imshow(outim - image)

from scipy import signal

n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

S = np.c_[s1, s2, s3]  # Translates slice objects to concatenation along the second axis.
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

fig, (mixed_ax, originals_ax) = plt.subplots(nrows=2, figsize=(6, 6))
for idx, color in enumerate(('C3', 'C4', 'C5')):
    mixed_ax.plot(X[:,idx], label='Mic {}'.format(idx + 1), color=color, alpha=0.75)
mixed_ax.legend()
for idx, label in enumerate(('sine', 'square', 'sawtooth')):
    originals_ax.plot(S[:,idx], label=label, alpha=0.75)
originals_ax.legend()

pca = decomposition.PCA(n_components=3)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

fig, pca_ax = plt.subplots(figsize=(6, 3))
for idx in range(3):
    pca_ax.plot(H[:,idx], alpha=0.75)

ica = decomposition.FastICA(n_components=3)
S_ = ica.fit_transform(X)  # Reconstruct signals

fig, ica_ax = plt.subplots(figsize=(6, 3))
for idx in range(3):
    ica_ax.plot(S_[:,idx], alpha=0.75)

from sklearn import manifold, datasets
X, color = datasets.samples_generator.make_swiss_roll(
    n_samples=1500,
    noise=0.0
)
fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

ax.set_title("Original data")

print("Computing LLE embedding")
X_r, err = manifold.locally_linear_embedding(X, n_neighbors=12,
                                             n_components=2)
print("Done. Reconstruction error: %g" % err)

#----------------------------------------------------------------------
# Plot result

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title('Projected data')
plt.show()



