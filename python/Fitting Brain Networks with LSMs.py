import os
import sys
import pickle
import datetime
sys.path.append(os.path.abspath(".."))

import numpy as np
import numpy.random as npr
from scipy.misc import logsumexp
npr.seed(0)

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable, Grid
import matplotlib.gridspec as gridspec

get_ipython().magic('matplotlib inline')

import seaborn as sns
sns.set_style("white")
sns.set_context("paper")
colors = sns.xkcd_palette(
    ["red",
     "windows blue",
     "amber",
     "faded green",
     "dusty purple",
     "orange",
     "clay",
     "midnight"
    ])

from tqdm import tqdm

from hips.plotting.layout import create_axis_at_location, remove_plot_labels
from hips.plotting.colormaps import gradient_cmap

from lsm import LatentSpaceModel, LatentSpaceModelWithShrinkage,     MixtureOfLatentSpaceModels, MixtureOfLatentSpaceModelsWithShrinkage,     FactorialLatentSpaceModel
    
from lsm.utils import random_mask, cached, white_to_color_cmap, logistic

# Override this if using an existing results directory
# results_dir = None
# results_dir = os.path.join("..", "results", "2017_06_16")
# results_dir = os.path.join("..", "results", "2017_08_02")
results_dir = os.path.join("..", "results", "2017_08_03")

if results_dir is None:
    # By default, store results in results/yyyy_mm_dd directory for today
    results_basedir = os.path.join("..", "results")
    assert os.path.exists(results_basedir), "'results' directory must exist in project root."

    today = datetime.date.today()
    results_dir = os.path.join(results_basedir, today.strftime("%Y_%m_%d"))
    if not os.path.exists(results_dir):
        print("Making directory for today's results: ", results_dir)
        os.mkdir(results_dir)

missing_frac = 0.25
N_itr = 500
Ks = np.arange(2, 21, 2, dtype=int)
H = 10
sigmasq_b = 1.0

# Load the KKI-42 dataset
datapath = os.path.join("..", "data", "kki-42-data.pkl")
assert os.path.exists(datapath)
with open(datapath, "rb") as f:
    As = pickle.load(f)

N, Vorig, _ = As.shape
assert N == 42 and Vorig == 70 and As.shape[2] == Vorig
bad_indices = [0, 35]
good_indices = np.array(sorted(list(set(np.arange(Vorig)) - set(bad_indices))))
As = As[np.ix_(np.arange(N), good_indices, good_indices)]
V = Vorig - len(bad_indices)

# Sample random masks
masks = [random_mask(V, missing_frac) for _ in range(N)]

# Compute number of train/test edges
L = np.tril(np.ones((V, V), dtype=bool), k=-1)
N_train = np.sum([mask * L for mask in masks])
N_test = np.sum([(1- mask) * L for mask in masks])

print("N: ", N)
print("V: ", V)
print("Num edges: ", N * V * (V-1) / 2)
print("Num train: ", N_train)
print("Num test:  ", N_test)

# Plot the mean network
plt.imshow(As.mean(0), vmin=0, vmax=1, interpolation="nearest")
plt.title("Average Network")

def fit(model, N_init_itr=0):
    print("Fitting ", model.name)
    lprs = []
    lls = []
    hlls = []
    ms = []
    
    # Burnin by sampling X and b only
    print("Running {} initialization iterations.".format(N_init_itr))
    for itr in tqdm(range(N_init_itr)):
        model._resample_X()
        model._resample_b()
    
    print("Running {} sampling iterations.".format(N_itr))
    for itr in tqdm(range(N_itr)):
        model.resample()
        lprs.append(model.log_prior())
        assert np.isscalar(lprs[-1])
        lls.append(model.log_likelihood())
        hlls.append(model.heldout_log_likelihood())
        ms.append(model.ms.copy())
    return model, np.array(lprs), np.array(lls), np.array(hlls), np.array(ms)

def fit_list(models, **kwargs):
    results = []
    for model in models:
        for A, mask in zip(As, masks):
            model.add_data(A, mask=mask)
        model.initialize()
        
        _fit = cached(results_dir, model.name)(fit)
        results.append(_fit(model, **kwargs))
    return results

# Baseline model
baseline_model, baseline_lprs, baseline_lls, baseline_hlls, _  =     fit_list([LatentSpaceModel(V, 0, name="bernoulli")])[0]
baseline_ll = np.mean(baseline_lls[N_itr//2:])
baseline_hll = np.mean(baseline_hlls[N_itr//2:])

# Standard model
std_models = []
for K in Ks:
    std_models.append(LatentSpaceModel(V, K, sigmasq_b=sigmasq_b))
std_results = fit_list(std_models)

# Standard model with shrinkage prior
std_shr_models = []
for K in Ks:
    std_shr_models.append(
        LatentSpaceModelWithShrinkage(V, K, sigmasq_b=sigmasq_b, 
            sigmasq_prior_prms=dict(a1=2.5, a2=3.5)))
std_shr_results = fit_list(std_shr_models)

print(std_shr_results[-1][0].sigmasq_x)

plt.figure()
plt.plot(std_shr_results[-1][1])
plt.ylabel("Log Prior")

plt.figure()
plt.plot(std_shr_results[-1][2])
plt.ylabel("Log Likelihood")

plt.figure()
plt.plot(std_shr_results[-1][1] + std_shr_results[-1][2])
plt.ylabel("Log Likelihood")

# Mixture of latent space models 
mix_models = []
for K in Ks:
    mix_models.append(MixtureOfLatentSpaceModels(V, K*H, H=H, sigmasq_b=sigmasq_b))
mix_results = fit_list(mix_models, N_init_itr=20)

# plt.imshow(np.array(mix_results[-1][0].ms))
print(mix_results[-1][0].hs)
# plt.subplot(121)
# plt.imshow(mix_results[-1][0].edge_probabilities(0))
# plt.subplot(122)
# plt.imshow(mix_results[-1][0].edge_probabilities(1))
plt.figure(figsize=(12,8))
plt.subplot(121)
plt.imshow(mix_results[-1][0].edge_probabilities(0) - mix_results[-1][0].edge_probabilities(1))
plt.subplot(122)
plt.imshow(As[0].astype(float) - As[1].astype(float))

# Mixture of latent space models with shrinkage prior
mix_shr_models = []
for K in Ks:
    mix_shr_models.append(
        MixtureOfLatentSpaceModelsWithShrinkage(
            V, K*H, H=H, sigmasq_b=sigmasq_b, 
            sigmasq_prior_prms=dict(a1=2.5, a2=3.5)))
mix_shr_results = fit_list(mix_shr_models, N_init_itr=20)

# plt.imshow(np.array(mix_results[-1][0].ms))
print(mix_shr_results[-1][0].hs)
print(mix_shr_results[-1][0].sigmasq_x)
# plt.subplot(121)
# plt.imshow(mix_results[-1][0].edge_probabilities(0))
# plt.subplot(122)
# plt.imshow(mix_results[-1][0].edge_probabilities(1))
plt.figure(figsize=(12,8))
plt.subplot(121)
plt.imshow(mix_shr_results[-1][0].edge_probabilities(0) - mix_shr_results[-1][0].edge_probabilities(1))
plt.subplot(122)
plt.imshow(As[0].astype(float) - As[1].astype(float))

# Factorial latent space models
fac_models = []
for K in Ks:
    fac_models.append(
        FactorialLatentSpaceModel(V, K, sigmasq_b=sigmasq_b, alpha=1 + K / 2.0))
fac_results = fit_list(fac_models)

# plt.imshow(np.array(mix_results[-1][0].ms))
plt.figure()
plt.imshow(np.array(fac_results[-1][0].ms))


plt.figure(figsize=(12,8))
plt.subplot(121)
plt.imshow(fac_results[-1][0].edge_probabilities(0) - fac_results[-1][0].edge_probabilities(1))
plt.subplot(122)
plt.imshow(As[0].astype(float) - As[1].astype(float))

fig = plt.figure(figsize=(5.2, 2.8))
grid = Grid(fig, (0, 0, 1, 1), 
            nrows_ncols=(2, 5),
            axes_pad=0.1,
            add_all=True,
            label_mode="L",
            )

titles = ["LSM", "LSM (MIG)", "MoLSM", "MoLSM (MIG)", "fLSM"]

# Top row: Train log joint vs iteration
for j, results in     enumerate([std_results, std_shr_results, 
               mix_results, 
               mix_shr_results,
               fac_results
              ]):

    # Plot training log joint
    axj = grid[j]
    for K, (model, lprs, lls, _, _) in zip(Ks, results):
        color = white_to_color_cmap(colors[j])(K / Ks[-1])
#         axj.plot(np.arange(N_itr), (lprs + lls - baseline_ll) / N_train, color=color)
        axj.plot(np.arange(N_itr), (lls - baseline_ll) / N_train, color=color)
        axj.set_title(titles[j])
        
        axj.spines['top'].set_visible(False)
        axj.spines['right'].set_visible(False)
        if j > 0:
            axj.spines['left'].set_visible(False)
        else:
            axj.set_ylabel("Train Log Joint", labelpad=10)

# Bottom row: Test likelihood vs iteration
for j, results in     enumerate([std_results, std_shr_results, 
               mix_results, 
               mix_shr_results,
               fac_results
              ]):

    # Plot test log likelihoods
    axj = grid[5 + j]
    for K, (model, _, _, hlls, _) in zip(Ks, results):
        color = white_to_color_cmap(colors[j])(K / Ks[-1])
        axj.plot(np.arange(N_itr), (hlls - baseline_hll) / N_test, color=color)
        
        axj.spines['top'].set_visible(False)
        axj.spines['right'].set_visible(False)
        if j > 0:
            axj.spines['left'].set_visible(False)
        else:
            axj.set_ylabel("Test Log Lkhd", labelpad=10)
        
        axj.set_xlabel("Iteration")
        
        
plt.savefig(os.path.join(results_dir, "convergence.pdf"))
plt.savefig(os.path.join(results_dir, "convergence.png"), dpi=300)

fig = plt.figure(figsize=(5.2, 2.8))
gs = gridspec.GridSpec(2, 5)

# Initialize the axes with sharing
axs = np.zeros((2, 5), dtype=object)
for j in range(5):
    sharey = None if j == 0 else axs[1,0]
    axs[1,j] = fig.add_subplot(gs[1,j], sharey=sharey)
for j in range(5):
    sharey = None if j == 0 else axs[0,0]
    axs[0,j] = fig.add_subplot(gs[0,j], sharex=axs[1,j], sharey=sharey)
    
titles = ["LSM", "LSM (MIG)", "MoLSM", "MoLSM (MIG)", "fLSM"]

# Top row: train log likelihood
for j, results in     enumerate([std_results, 
               std_shr_results, 
               mix_results, 
               mix_shr_results,
               fac_results
              ]):
        
    axj = axs[0,j]
    for K, (_, lprs, lls, _, _) in zip(Ks, results):
        color = white_to_color_cmap(colors[j])(K / Ks[-1])
        axj.bar(K, (np.mean(lls[N_itr // 2:]) - baseline_ll) / N_train, 
                yerr=np.std(lls[N_itr // 2:]) / N_train,  
                width=1.8, color=color, ecolor='k', edgecolor='k')
    
    axj.set_title(titles[j])
    
    # Remove spines
    axj.set_xticks([])
    axj.spines['top'].set_visible(False)
    axj.spines['right'].set_visible(False)
    if j > 0:
        axj.spines['left'].set_visible(False)
        plt.setp(axj.get_yticklabels(), visible=False)
    else:
        axj.set_ylabel("Train Log Lkhd")
    
    
# Bottom row: test log likelihood
for j, results in     enumerate([std_results, 
               std_shr_results, 
               mix_results, 
               mix_shr_results,
               fac_results
              ]):
    
    axj = axs[1,j]
    for K, (_, _, _, hlls, _) in zip(Ks, results):
        color = white_to_color_cmap(colors[j])(K / Ks[-1])
        axj.bar(K, (np.mean(hlls[N_itr // 2:]) - baseline_hll) / N_test,
                yerr=np.std(hlls[N_itr // 2:]) / N_test,  
                width=1.8, color=color, ecolor='k', edgecolor='k')
        
    # Plot zero line
    axj.plot([Ks[0]-2, Ks[-1]+2], [0, 0], '-k', lw=0.5)
    
    # Configure x axis
    axj.set_xlim(Ks[0]-2, Ks[-1]+2)
    axj.set_xticks(np.arange(4, 21, 4)+1)
    axj.set_xticklabels(np.arange(4, 21, 4))
    
    if titles[j].startswith("Mo"):
        axj.set_xlabel("Dimensions\nper Component")
    else:
        axj.set_xlabel("Dimensions")
    
    # Remove spines
    axj.spines['top'].set_visible(False)
    axj.spines['right'].set_visible(False)
    if j > 0:
        axj.spines['left'].set_visible(False)
        plt.setp(axj.get_yticklabels(), visible=False)
    else:
        axj.set_ylabel("Test Log Lkhd")
        
plt.tight_layout(pad=0.1)
plt.savefig(os.path.join(results_dir, "lls.pdf"))
plt.savefig(os.path.join(results_dir, "lls.png"), dpi=300)

print("Baseline train pr per edge: ", np.exp(baseline_ll / N_train))
print("Factorial train pr per edge: ", np.exp(baseline_ll / N_train + 0.08))

print("Baseline test pr per edge: ", np.exp(baseline_hll / N_test))
print("Factorial test pr per edge: ", np.exp(baseline_hll / N_test + 0.05))

# Get the fit fLSM model for K=20
fac_lsm_K20 = fac_results[-1][0]

# Sort factors by variance (for used columns only)
M = np.array(fac_lsm_K20.ms)
used = M.sum(0) > 0
N_used = np.sum(used)
lmbda = np.var(fac_lsm_K20.X, axis=0)
perm = np.argsort(lmbda * used)[::-1]

# Get the rank one factors
X = fac_lsm_K20.X[:,perm]
XXTs = np.array([np.outer(X[:,k], X[:,k].T) for k in range(fac_lsm_K20.K)])
for XXTj in XXTs:
    np.fill_diagonal(XXTj, 0)
lim = np.max(abs(XXTs[:10]))

# Get the mean network
Amean = As.mean(0)
np.fill_diagonal(Amean, 0)

# Get the bias
b_fac = logistic(fac_lsm_K20.b)

# Exaggerate the colormap by pushing values toward lim
logistic = lambda u: 1 / (1 + np.exp(-u))
logit = lambda p: np.log(p / (1-p))
class LogisticNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, clip=False, lim=1, scale=3.0):
        self.lim = lim
        self.scale = scale
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        z = (value + self.lim) / (2 * self.lim)
        zhat = logistic(self.scale * logit(z))
        return np.ma.masked_array(zhat)

fig = plt.figure(figsize=(5.2,2.8))

# Plot the factor usage
ax0 = create_axis_at_location(fig, 0.4, 0.4, 1., 2.2 )
dark_cmap = white_to_color_cmap(colors[-1])
ax0.imshow(M[:,perm[:N_used]], interpolation="nearest", aspect="auto", 
           extent=(1,N_used,N,1), cmap=dark_cmap)
ax0.set_ylabel("Network")
ax0.set_xlabel("Factor")
ax0.xaxis.set_label_position('top') 

# Plot the average network
ax1 = create_axis_at_location(fig, 1.55, 1.65, .95, .95 )
ax1.imshow(Amean, vmin=0, vmax=1, interpolation="nearest", cmap=dark_cmap)
ax1.set_title("Avg. Network")
ax1.set_xticks([])
ax1.set_yticks([])

# Plot the outer product of individual factors
factors_to_plot = [0, 1, 2, 13, 14]
cmap = gradient_cmap([colors[0], np.ones(3), colors[1]])
for j in range(2):
    axj = create_axis_at_location(fig, 2.65 + 1.1 * j, 1.65, .95, .95 )
    im = axj.imshow(XXTs[factors_to_plot[j]], 
                    interpolation="nearest", 
                    vmin=-lim, vmax=lim, cmap=cmap, 
                    norm=LogisticNormalize(lim=lim, scale=3))
    axj.set_xticks([])
    axj.set_yticks([])
    axj.set_title("Factor {}".format(factors_to_plot[j]+1))

# Plot colorbar
ax_cb = create_axis_at_location(fig, 4.75, 1.65, 0.05, .95)
plt.colorbar(im, cax=ax_cb)
    
for j in range(2, 5):
    axj = create_axis_at_location(fig, 1.55 + 1.1 * (j-2), .4, .95, .95 )
    axj.imshow(XXTs[factors_to_plot[j]], 
               interpolation="nearest", 
               vmin=-lim, vmax=lim, cmap=cmap, 
               norm=LogisticNormalize(lim=lim, scale=3))
    axj.set_xticks([])
    axj.set_yticks([])
    axj.set_title("Factor {}".format(factors_to_plot[j]+1))
                      
plt.savefig(os.path.join(results_dir, "factors.pdf"))
plt.savefig(os.path.join(results_dir, "factors.png"), dpi=300)

# Get the fit fLSM model for K=20
mix_lsm = mix_shr_results[-1][0]
mix_lsm.sigmasq_x

mix_lsm = mix_shr_results[-1][0]
# mix_lsm_ms0 = mix_results[-1][3][0]

# Sort factors by variance (for used columns only)
M = np.array(mix_lsm.ms)
# used = M.sum(0) > 0
# lmbda = np.var(mix_lsm.X, axis=0)

# # Get the rank one factors
# X = mix_lsm.X
# XXTs = np.array([np.outer(X[:,k], X[:,k].T) for k in range(mix_lsm.K)])
# for XXTj in XXTs:
#     np.fill_diagonal(XXTj, 0)
# lim = np.max(abs(XXTs[:10]))

plt.imshow(M, aspect=2.0)

