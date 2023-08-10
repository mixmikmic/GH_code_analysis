import os, sys
sys.path.append(os.path.abspath(".."))

import numpy as np
import numpy.random as npr
from scipy.misc import logsumexp
npr.seed(0)

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
get_ipython().magic('matplotlib inline')

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

from lsm import LatentSpaceModel, LatentSpaceModelWithShrinkage,     MixtureOfLatentSpaceModels, MixtureOfLatentSpaceModelsWithShrinkage,     FactorialLatentSpaceModel
from lsm.utils import random_mask, progprint_xrange, white_to_color_cmap

V = 70                  # Number of vertices
K = 2                   # Number of latent factors
N = 50                  # Number of networks in population
missing_frac = 0.1      # Fraction of data to withhold for testing
N_itr = 100              # Number of iterations of sampler
sigmasq_b = 0.0         # Prior variance of b (0 -> deterministically zero bias)

# Sample data from a model with simple 2D covariates
X = np.column_stack((np.linspace(-2, 2, V), np.zeros((V, K - 1))))
true_model = LatentSpaceModel(V, K, X=X, sigmasq_b=sigmasq_b)
masks = [random_mask(V, missing_frac) for _ in range(N)]
As = [true_model.generate(keep=True, mask=mask) for mask in masks]
true_ll = true_model.log_likelihood()
true_hll = true_model.heldout_log_likelihood()

plt.imshow(As[0], vmin=0, vmax=1, interpolation="nearest")
plt.title("First Network")

def fit(model):
    print("Fitting ", model)
    lls = []
    hlls = []
    ms = []
    for itr in progprint_xrange(N_itr):
        model.resample()
        lls.append(model.log_likelihood())
        hlls.append(model.heldout_log_likelihood())
        ms.append(model.ms)
    return np.array(lls), np.array(hlls), np.array(ms)

# Fit the data with a standard LSM
standard_lsm = LatentSpaceModel(V, K, sigmasq_b=sigmasq_b)
for A, mask in zip(As, masks):
    standard_lsm.add_data(A, mask=mask)
standard_lsm_lls, standard_lsm_hlls, standard_lsm_ms = fit(standard_lsm)

# Fit the data with a standard LSM
standard_lsm_shrink = LatentSpaceModelWithShrinkage(V, K, sigmasq_b=sigmasq_b, a1=2.5, a2=3.5)
for A, mask in zip(As, masks):
    standard_lsm_shrink.add_data(A, mask=mask)
standard_lsm_shrink_lls, standard_lsm_shrink_hlls, standard_lsm_shrink_ms =     fit(standard_lsm_shrink)

# Fit the data with a mixture of LSMs
mixture_lsm = MixtureOfLatentSpaceModels(V, 2*K, H=2, sigmasq_b=sigmasq_b)
for A, mask in zip(As, masks):
    mixture_lsm.add_data(A, mask=mask)
mixture_lsm.initialize()
mixture_lsm_lls, mixture_lsm_hlls, mixture_lsm_ms = fit(mixture_lsm)

# Fit the data with a mixture of LSMs
mixture_lsm_shrink = MixtureOfLatentSpaceModelsWithShrinkage(V, 2*K, H=2, sigmasq_b=sigmasq_b)
for A, mask in zip(As, masks):
    mixture_lsm_shrink.add_data(A, mask=mask)
mixture_lsm_shrink_lls, mixture_lsm_shrink_hlls, mixture_lsm_shrink_ms =     fit(mixture_lsm_shrink)

# Fit the data with a factorial LSM
factorial_lsm = FactorialLatentSpaceModel(V, K, sigmasq_b=sigmasq_b)
for A, mask in zip(As, masks):
    factorial_lsm.add_data(A, mask=mask)
factorial_lsm_lls, factorial_lsm_hlls, factorial_lsm_ms = fit(factorial_lsm)

# Plot the results
plt.figure(figsize=(12, 4))
plt.subplot(161)
plt.imshow(true_model.edge_probabilities(0), vmin=0, vmax=1, interpolation="nearest")
plt.title("True")

plt.subplot(162)
plt.imshow(standard_lsm.edge_probabilities(0), vmin=0, vmax=1, interpolation="nearest")
plt.title("Std LSM")

plt.subplot(163)
plt.imshow(standard_lsm_shrink.edge_probabilities(0), vmin=0, vmax=1, interpolation="nearest")
plt.title("Std LSM with Shrinkage")

plt.subplot(164)
plt.imshow(mixture_lsm.edge_probabilities(0), vmin=0, vmax=1, interpolation="nearest")
plt.title("Mixture LSM")

plt.subplot(165)
plt.imshow(mixture_lsm_shrink.edge_probabilities(0), vmin=0, vmax=1, interpolation="nearest")
plt.title("Mixture LSM with Shrinkage")

plt.subplot(166)
plt.imshow(factorial_lsm.edge_probabilities(0), vmin=0, vmax=1, interpolation="nearest")
plt.title("Factorial LSM")
plt.suptitle("Edge Probabilities")

plt.figure()
plt.plot(standard_lsm_lls, label="Standard LSM")
plt.plot(standard_lsm_shrink_lls, label="Standard LSM Shrink")
plt.plot(mixture_lsm_lls, label="Mixture of LSMs")
plt.plot(mixture_lsm_shrink_lls, label="Mixture of LSMs Shrink")
plt.plot(factorial_lsm_lls, label="Factorial LSM")
plt.plot([0, N_itr-1], true_ll * np.ones(2), ':k', label="True LSM")
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.legend(loc="lower right")

plt.figure()
plt.plot(standard_lsm_hlls, label="Standard LSM")
plt.plot(standard_lsm_shrink_hlls, label="Standard LSM Shrink")
plt.plot(mixture_lsm_hlls, label="Mixture of LSMs")
plt.plot(mixture_lsm_shrink_hlls, label="Mixture of LSMs Shrink")
plt.plot(factorial_lsm_hlls, label="Factorial LSM")
plt.plot([0, N_itr - 1], true_hll * np.ones(2), ':k', label="True LSM")
plt.xlabel("Iteration")
plt.ylabel("Heldout Log Likelihood")
plt.legend(loc="lower right")
plt.show()

