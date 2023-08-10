get_ipython().run_line_magic('matplotlib', 'inline')

from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import numpy as np
from sklearn.pipeline import Pipeline

from msmbuilder.example_datasets import muller, MullerPotential
from msmbuilder.preprocessing import RobustScaler
from msmbuilder.decomposition import tICA, PCA, KernelTICA
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import MarkovStateModel

import matplotlib.pyplot as pp
import msmexplorer as msme

from vde import VDE # From this package!

def simulate(x0s, KT=15000.0, random_state=None):
    M = muller.MULLER_PARAMETERS
    M['KT'] = temp
    random = np.random.RandomState(random_state)
    # propagate releases the GIL, so we can use a thread pool and
    # get a nice speedup
    tp = ThreadPool(cpu_count())
    return tp.map(lambda x0:
        muller.propagate(
            n_steps=M['N_STEPS'], x0=x0, thin=M['THIN'], kT=M['KT'],
            dt=M['DT'], D=M['DIFFUSION_CONST'], random_state=random,
            min_x=M['MIN_X'], max_x=M['MAX_X'], min_y=M['MIN_Y'],
            max_y=M['MAX_Y']), x0s)

def plot_decomp_grid(decomposition, res=100, alpha=1., cmap='magma', ylim=None,
                     obs=0, xlim=None, ax=None, n_levels=3):

    if ax is None:
        _, ax = pp.subplots(1, 1, figsize=(4, 4))
    else:
        if xlim is None:
            xlim = ax.get_xlim()
        if ylim is None:
            ylim = ax.get_ylim()

    if not xlim and not ylim:
        raise ValueError('Please supply x and y limits.')

    X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], res),
                       np.linspace(ylim[0], ylim[1], res))
    x = np.ravel(X)
    y = np.ravel(Y)
    xy = np.vstack([x, y]).T

    Z = np.vstack(decomposition.transform([xy]))[:, obs].reshape(res, res)

    levels = np.linspace(Z.min(), Z.max(), n_levels + 1)

    cnt = ax.contourf(X, Y, Z, cmap=cmap, alpha=alpha, levels=levels)
    
    for c in cnt.collections:
        c.set_edgecolor("face")
    return ax

def scatter(decomposition, cmap='plasma', stride=10, ax=None):

    if ax is None:
        _, ax = pp.subplots(1, 1, figsize=(4, 4))

    cmap = pp.get_cmap(cmap)
    w = np.concatenate(decomposition.transform(trajs))[::stride].ravel()
    w -= w.min()
    w /= w.max()
    ax.scatter(*np.vstack(trajs)[::stride].T, c=cmap(w))

    return ax

def fit_msm(traj, model=None, lagtime=1, scaled=False):
    if not scaled:
        traj = scaler.transform(traj)
    if model is not None:
        data = model.transform(traj)
    else:
        data = traj
    msm_pipe = Pipeline([('cluster', MiniBatchKMeans(n_clusters=6, batch_size=10000, random_state=256)),
                         ('msm', MarkovStateModel(lag_time=lagtime, verbose=False, n_timescales=3))])

    ass = msm_pipe.fit_transform(data)
    return msm_pipe.steps[-1][1], ass, data

def plot_FE(traj, ax, lagtime=1, model=None, scaled=False):
    msm, ass, data = fit_msm(traj, model=model, lagtime=lagtime, scaled=scaled)
    pi = msm.populations_[np.concatenate(ass)]
    pi /= pi.sum()
    msme.plot_free_energy(np.vstack(data), shade=False,
                          color=cmap(np.linspace(0., 1., 5)[i]), ax=ax, vmax=8,
                          pi=pi)

def plot_FE2D(traj, ax, lagtime=1, cmap='gray', scaled=False):
    msm, ass, data = fit_msm(traj, lagtime=lagtime, scaled=scaled)
    pi = msm.populations_[np.concatenate(ass)]
    pi /= pi.sum()
    msme.plot_free_energy(np.vstack(data), obs=(0, 1), gridsize=50,
                          n_samples=10000, pi=pi, random_state=128, n_levels=5, ax=ax)

def clean_up(ax):
    ax.axis('off')
    ax.grid('off')
    ax.set_xticks([])
    ax.set_yticks([])

trajs_raw = MullerPotential(random_state=128).get().trajectories
scaler = RobustScaler()
trajs = scaler.fit_transform(trajs_raw)

fig, ref_ax = pp.subplots(1, 1, figsize=(4, 4))

for traj in trajs:
    ref_ax.plot(*traj.T)

xlim = ref_ax.get_xlim()
ylim = ref_ax.get_ylim()

lag_time = 10

tica = tICA(n_components=1, lag_time=lag_time)


pca = PCA(n_components=1)


mdl = VDE(trajs[0].shape[-1], lag_time=lag_time,
          hidden_size=256, hidden_layer_depth=3,
          batch_size=250, n_epochs=10, cuda=True, 
          sliding_window=True, dropout_rate=0.3,
          learning_rate=1E-3)

tica.fit(trajs)

pca.fit(trajs)

mdl.fit(trajs)

fig, (ax1, ax2, ax3) = pp.subplots(1, 3, figsize=(12, 4))

scatter(mdl, cmap='plasma_r', ax=ax1)
ax1.set_title('VDE')
clean_up(ax1)

scatter(tica, cmap='plasma_r', ax=ax2)
ax2.set_title('tICA')
clean_up(ax2)

scatter(pca, cmap='plasma_r', ax=ax3)
ax3.set_title('PCA')
clean_up(ax3)

fig, (ax1, ax2, ax3) = pp.subplots(1, 3, figsize=(12, 4))

alpha = 0.6
n_levels = 50

plot_FE2D(trajs, ax1, lagtime=5, scaled=True)
plot_decomp_grid(mdl, n_levels=n_levels, cmap='plasma_r', alpha=alpha, ax=ax1)
ax1.set_title('VDE')
clean_up(ax1)

plot_FE2D(trajs, ax2, lagtime=5, scaled=True)
plot_decomp_grid(tica, n_levels=n_levels, cmap='plasma_r', alpha=alpha, ax=ax2)
ax2.set_title('tICA')
clean_up(ax2)

plot_FE2D(trajs, ax3, lagtime=5, scaled=True)
plot_decomp_grid(pca, n_levels=n_levels, cmap='plasma_r', alpha=alpha, ax=ax3)
ax3.set_title('PCA')
clean_up(ax3)

pp.savefig('muller.pdf')

from torch.autograd import Variable

def propagate(x0, scale, steps=1000):
    traj = [x0]
    for i in range(steps):
        traj.append(mdl.propagate(x0, scale=scale).ravel())
    return np.array(traj)

mdl.eval()

all_temps = np.logspace(-2, -1, 5)
fake_trajs_all = []

for temp in map(float, all_temps):
    print('Temperature set to %s' % temp)
    fake_trajs = []
    for j, traj in enumerate(trajs):
        print('making %s' % j)
        fake_trajs.append(propagate(traj[0], temp))
    fake_trajs_all.append(fake_trajs)

real_trajs_all = []

all_temps = np.logspace(4, 5, 5)
for j, temp in enumerate(all_temps):
    real_trajs_all.append(simulate([traj[0] for traj in trajs_raw], KT=temp, random_state=256))

fig, (real_ax, fake_ax) = pp.subplots(1, 2, figsize=(12, 4))

cmap = pp.get_cmap('plasma')    

for i, traj in enumerate(real_trajs_all):
    plot_FE(traj, real_ax, model=mdl, lagtime=10)
    

for i, traj in enumerate(fake_trajs_all):
    plot_FE(traj, fake_ax, model=mdl, scaled=True)   

fake_ax.set_xlim(real_ax.get_xlim())
fake_ax.set_ylim(real_ax.get_ylim())

real_ax.set_xticks([])
fake_ax.set_xticks([])

real_ax.spines["right"].set_visible(False)
real_ax.spines["top"].set_visible(False)
fake_ax.spines["right"].set_visible(False)
fake_ax.spines["top"].set_visible(False)

real_ax.set_title('Real Data')
fake_ax.set_title('Fake Data')

pp.savefig('propagator.pdf')



