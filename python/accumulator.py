get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from datatypes import SpikeTrain, set_list_var
from pool import build_pool
from exp import (
    run_experiment,
    run_regular_neuron_experiment,
    run_poisson_neuron_experiment,
    plot_timeseries, plot_isi, plot_gamma,
    plot_hist, plot_acorr,
    compute_out_rate
)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import nengo
from nengo.utils.ensemble import tuning_curves

run_regular_neuron_experiment(weight=1, threshold=4)

run_regular_neuron_experiment(weight=3, threshold=10)

dat = run_poisson_neuron_experiment(weight=1, threshold=3)

dat = run_poisson_neuron_experiment(weight=2, threshold=5)

isi = np.diff(dat['spks_out'].times)
isi_even = isi[::2]
isi_odd = isi[1::2]

fig, axs = plt.subplots(nrows=2, figsize=(12,6), gridspec_kw={'hspace':.4})
ax = plot_acorr(isi_even, axs[0], title="Even-Indexed Output ISI ")
ax = plot_hist(
    isi_even, axs[1], normed=True,
    title="Even-Indexed Output ISI (%d ISIs)"%len(isi_even),
    ylabel="Normalized Counts"
)
ax = plot_gamma(axs[1], shape=2, scale=1./1000)
ax.legend(loc='best')

fig, axs = plt.subplots(nrows=2, figsize=(12,6), gridspec_kw={'hspace':.4})
ax = plot_acorr(isi_odd, axs[0], title="Odd-Indexed Output ISI ")
ax = plot_hist(
    isi_odd, axs[1], normed=True,
    title="Odd-Indexed Output ISI (%d ISIs)"%len(isi_odd),
    ylabel="Normalized Counts"
)
ax = plot_gamma(axs[1], shape=3, scale=1./1000)
ax.legend(loc='best')

# vary thresholds
nspikes_out = 10000
weight = 2
for threshold in [3, 5, 21]:
    nspikes_in = int(np.ceil(float(threshold)/weight*nspikes_out)) + int(np.ceil(float(threshold/weight)))
    run_poisson_neuron_experiment(weight=weight, threshold=threshold, T=None,
        nspikes=nspikes_in, timeseries_plot=False, in_isi_plot=False,
        out_isi_plot=False)

N = 100
input_rates = np.random.uniform(0, 400, size=N)
weights = np.random.randint(0, 11, size=N)
threshold = int(np.sum(input_rates*weights)/1000.)
T=10
pool = build_pool(N, input_rates=input_rates, weights=weights, threshold=threshold)
spks_in, acc_state, spks_out = run_experiment(pool, T=T)
plot_timeseries(spks_in, acc_state, spks_out, tmax=0.05, threshold=threshold)
axs = plot_isi(spks_out, bins=100)

N = 200
np.random.seed(0)
input_rates = np.random.uniform(0, 200, size=N)
weights = np.random.randint(-10, 10, size=N)
threshold = 50
T=10.
pool = build_pool(N, input_rates=input_rates, weights=weights, threshold=threshold)
spks_in, acc_state, spks_out = run_experiment(pool, T=T)
plot_timeseries(spks_in, acc_state, spks_out, tmax=0.1, threshold=threshold)
axs = plot_isi(spks_out, bins=100)
print(compute_out_rate(pool, threshold))

def threshold_weights(weights):
    """compute the threshold and rounded weights from a given weight distribution
    """
    acc_wmax = np.array([1.] + [127./2**i for i in range(7, 14)])
    acc_unit = np.array([1./64] + [1./2**i for i in range(7, 14)])
    thresholds = np.array([2**i for i in range(6,14)])

    w_max = np.max(np.abs(weights))
    idx = np.argmax(w_max >= acc_wmax)-1
    w_rounded = np.rint(weights/acc_unit[idx]).astype(int)
    threshold = thresholds[idx]   
    return threshold, w_rounded

# build a network
N = 64
net = nengo.Network()
with net:
    stim = nengo.Node(lambda t: np.sin(2*np.pi*t), size_in=0, size_out=1)
    ens = nengo.Ensemble(N, 1)
    node = nengo.Node(lambda t, x: x, size_in=1, size_out=1)
    probe_in = nengo.Probe(stim)
    probe_out = nengo.Probe(node)
    nengo.Connection(stim, ens, synapse=0)
    conn = nengo.Connection(ens, node, function=lambda x: x, transform=1000)

sim = nengo.Simulator(net)

# check the network by running a simulation
sim.run(1)

fig, axs = plt.subplots(nrows=2, figsize=(8,6), sharex=True)
# print(sim.data[probe])
# print(sim.trange())
axs[0].plot(sim.trange(), sim.data[probe_in])
axs[1].plot(sim.trange(), sim.data[probe_out])

x_test = [-1, -0.5, 0, 0.5, 1]
fig, axs = plt.subplots(nrows=2, figsize=(8,6))
x, a = tuning_curves(ens, sim)

max_weight = np.max(np.abs(sim.data[conn].weights[0,:]))

for enc, color in zip([1, -1], ['r', 'b']):
    idx = (sim.data[ens].encoders == enc)[:,0]
    lines = axs[0].plot(x, a[:, idx], color)
    line_weights = np.abs(sim.data[conn].weights[0,idx])/max_weight
    for line, line_weight in zip(lines, line_weights):
        line.set_alpha(line_weight)
        line.set_linewidth(line_weight*2)
    
n, bins, patches = axs[1].hist(sim.data[conn].weights[0,:], bins=50)
axs[0].set_xlabel('x')
axs[0].set_ylabel('Firing Rate')
axs[1].set_xlabel('Decode Weight')

for xval in x_test:
    axs[0].axvline(xval, color='k', linestyle=':')

inputs = [-1, -0.5, 0, 0.5, 1]
# inputs = [-1]
for x_in in inputs:
    x, a = tuning_curves(ens, sim, inputs=np.array([x_in]))
    threshold, weights = threshold_weights(sim.data[conn].weights[0])
    pool = build_pool(N, input_rates=a, weights=weights, threshold=threshold)
    spks_in, acc_state, spks_out = run_experiment(pool, T=1.)
    plot_timeseries(spks_in, acc_state, spks_out, tmax=1.)
    plot_isi(spks_out, bins=100)



