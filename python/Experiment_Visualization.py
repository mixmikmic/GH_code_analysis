import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')
import ipywidgets
get_ipython().run_line_magic('matplotlib', 'notebook')
import pickle
import os
import json
import pprint
from itertools import *
import scipy.stats
from tqdm import tqdm_notebook as tqdm
import matplotlib.lines

path = "results/Rge-Vae Armarrow 10L Kde 1518449180/"
with open(os.path.join(path, "config.json")) as f:
    config = json.load(f)
pprint.pprint(config)

X = np.load(os.path.join(path, 'training_images.npy'))
fig, ax = plt.subplots(3,3, figsize=(9,9))
for i in range(9):
    ax[i%3, i/3].imshow(X[i])
    ax[i%3, i/3].axis("off")

states = np.load(os.path.join(path, 'samples_states.npy'))
n_states=states.shape[-1]
fig, ax = plt.subplots(n_states,n_states, figsize=(9,9))
for i in range(n_states**2):
    ax[i%n_states, i/n_states].scatter(states[:,i%n_states], states[:,i/n_states], s=5., alpha=.6)
    ax[i%n_states, i/n_states].axis('off')
fig.suptitle("Joint Plots of sampled states for image generation");

latents = np.load(os.path.join(path, 'training_latents.npy'))
plt.figure()
plt.scatter(latents[:,0], latents[:,1], s=1., cmap='jet', c = range(latents.shape[0]));

latents = np.load(os.path.join(path, 'training_latents.npy'))
n_latents=latents.shape[-1]
fig, ax = plt.subplots(n_latents,n_latents, figsize=(9,9))
for i in range(n_latents**2):
    ax[i%n_latents, i/n_latents].scatter(latents[:,i%n_latents], 
                                         latents[:,i/n_latents], 
                                         s=5., 
                                         c=range(latents.shape[0]),
                                         cmap='jet',
                                         marker='.',
                                         alpha=.6)
    
    ax[i%n_latents, i/n_latents].axis('off')

with open(os.path.join(path, "explored_states_history.pkl"), 'rb') as f:
    explored_states_history = pickle.load(f)
arm = scipy.misc.imread('test.png')

fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(1, 1, 1)
scatt = ax.scatter(explored_states_history[498][:,0], 
                   explored_states_history[498][:,1],
                   cmap='jet', s=10., alpha=.6)
ax.imshow(arm, extent=[-1,1, -1, 1], alpha=.3)
def update(epoch):
    scatt.set_offsets(explored_states_history[epoch*10][:,0:2])
    fig.canvas.draw()
ipywidgets.interact(update, epoch=(0, 48));

def sample_in_circle(nb_points):
    """
    This function allows to sample random points inside a circle using rejection sampling.
    """
    i = 0
    X = np.ndarray((nb_points, 2))
    while not i == nb_points:
        sample = np.random.uniform(-1, 1, 2)
        if np.linalg.norm(sample, ord=2) >1.:
            continue
        X[i] = sample
        i += 1
    return X

# We construct the histograms of achievable distribution
n_bins = 30
circ_samp = sample_in_circle(1000000)
hist_ab, _ = np.histogramdd(circ_samp, bins=n_bins, range=np.array([[-1]*2, [1]*2]).T)
hist_ab = hist_ab.astype(np.bool).astype(np.float)
hist_aa = np.tile(hist_ab.reshape(30,30,1), 30)

def kl_cov(X_s):
    n_samples, n_dim = X_s.shape
    histp , _ = np.histogramdd(X_s, bins=n_bins, range=np.array([[-1]*n_dim, [1]* n_dim]).T)
    if config['environment'] == 'armball':
        histq = hist_ab
    else:
        histq = hist_aa
    histq = histq.ravel()    
    histp = histp.ravel()
    return scipy.stats.entropy(histp, histq)

with open(os.path.join(path, "explored_states_history.pkl"), 'rb') as f:
    explored_states_history = pickle.load(f)

kls = np.zeros((49))
expl = np.zeros((49))
for i in tqdm(range(49)):   
    explored = explored_states_history[i*10]
    kls[i] = kl_cov(explored)
    if config['environment'] == 'armball':
        expl[i] = np.sum(np.linalg.norm(explored_states_history[i*10] - np.array([0.6, 0.6]), axis=1, ord=2) > 1e-3)
    else:
        expl[i] = np.sum(np.linalg.norm(explored_states_history[i*10] - np.array([0.6, 0.6, 0.6]), axis=1, ord=2) > 1e-3)

arm = scipy.misc.imread('test.png')
cmap='Blues'

fig = plt.figure(figsize=(9.5,3))
plt.title("KL-Cov for {} in {} environment.".format(config['name'].split(' ')[0], config['environment'].title()))

ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(kls, linewidth=1., label="KL-Cov", color="teal")
ax1_2 = ax1.twinx()
ax1_2.plot(expl, linewidth=1., c='darkred', label="Number of object Grasp")
ax1_2.set_ylabel("Number of object Grasp")
ax1.set_xlim(0., 50.)
ax1.set_xlabel("Exploration epochs (x100)")
ax1.set_ylabel("KL-Cov")
hi=min(kls)+(max(kls)-min(kls))/2
ax1.add_line(matplotlib.lines.Line2D([5,5],   [kls[5]+.05, hi], linewidth=1, color='teal'))
ax1.add_line(matplotlib.lines.Line2D([15,15], [kls[15]+.05, hi], linewidth=1, color='teal'))
ax1.add_line(matplotlib.lines.Line2D([25,25], [kls[25]+.05, hi], linewidth=1, color='teal'))
ax1.add_line(matplotlib.lines.Line2D([35,35], [kls[35]+.05, hi], linewidth=1, color='teal'))
ax1.add_line(matplotlib.lines.Line2D([45,45], [kls[45]+.05, hi], linewidth=1, color='teal'))
points = [5,15,25,35,45]
ax1.scatter(points, kls[points], color="teal")
ax1_2.tick_params(axis='y', colors='darkred')
ax1.tick_params(axis='y', colors='teal')

ax = fig.add_axes([0.01, 0.55, .3, .3])
ax.imshow(arm, extent=[-1,1, -1, 1], alpha=.85)
ax.scatter(explored_states_history[50][:,0], explored_states_history[50][:,1], s=.5, alpha=.2)
ax.axis("off")

ax = fig.add_axes([0.175, 0.55, .3, .3])
ax.imshow(arm, extent=[-1,1, -1, 1], alpha=.85)
ax.scatter(explored_states_history[150][:,0], explored_states_history[150][:,1], s=.5, alpha=.2)
ax.axis("off")

ax = fig.add_axes([0.35, 0.55, .3, .3])
ax.imshow(arm, extent=[-1,1, -1, 1], alpha=.85)
ax.scatter(explored_states_history[250][:,0], explored_states_history[250][:,1], s=.5, alpha=.2)
ax.axis("off")

ax = fig.add_axes([0.52, 0.55, .3, .3])
ax.imshow(arm, extent=[-1,1, -1, 1], alpha=.85)
ax.scatter(explored_states_history[350][:,0], explored_states_history[350][:,1], s=.5, alpha=.2)
ax.axis("off")

ax = fig.add_axes([0.69, 0.55, .3, .3])
ax.imshow(arm, extent=[-1,1, -1, 1], alpha=.85)
ax.scatter(explored_states_history[450][:,0], explored_states_history[450][:,1], s=.5, alpha=.2)
ax.axis("off");

plt.tight_layout()

