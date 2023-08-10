import numpy as np, GPy, pandas as pd
from mpl_toolkits.mplot3d import Axes3D
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import seaborn as sns

seeds = [8971, 3551, 3279, 5001, 5081]

from topslam.simulation import qpcr_simulation

fig = plt.figure(figsize=(15,3), tight_layout=True)

gs = plt.GridSpec(6, 5)

axit = iter([fig.add_subplot(gs[1:, i]) for i in range(5)])

for seed in seeds:
    Xsim, simulate_new, t, c, labels, seed = qpcr_simulation(seed=seed)
    ax = next(axit)

    # take only stage labels:
    labels = np.asarray([lab.split(' ')[0] for lab in labels])
    
    prevlab = None
    for lab in labels:
        if lab != prevlab:
            color = plt.cm.hot(c[lab==labels])
            ax.scatter(*Xsim[lab==labels].T, c=color, alpha=.7, lw=.1, label=lab)
            prevlab = lab
        
    ax.set_xlabel("SLS{}".format(seed))
    ax.set_frame_on(False)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

leg_hand = ax.get_legend_handles_labels()

ax = fig.add_subplot(gs[0, :])
ax.legend(*leg_hand, ncol=7, mode='expand')
ax.set_frame_on(False)
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])

fig.subplots_adjust(wspace=0, hspace=0)
fig.tight_layout()

seed = 5001
y_seed = 0

Xsim, simulate_new, t, c, labels, seed = qpcr_simulation(seed=seed)

np.random.seed(y_seed)
Y = simulate_new()

from topslam.optimization import run_methods, methods
X_init, dims = run_methods(Y, methods)

m = GPy.models.BayesianGPLVM(Y, 10, X=X_init, num_inducing=10)

m.likelihood.fix(.1)
m.kern.lengthscale.fix()
m.optimize(max_iters=500, messages=True, clear_after_finish=True)
m.likelihood.unfix()
m.kern.unfix()
m.optimize(max_iters=5e3, messages=True, clear_after_finish=False)

fig, axes = plt.subplots(2,4,figsize=(10,6))
axit = axes.flat
cols = plt.cm.hot(c)

ax = next(axit)
ax.scatter(*Xsim.T, c=cols, cmap='hot', lw=.1)
ax.set_title('Simulated')
ax.set_xticks([])
ax.set_yticks([])

ax = next(axit)
msi = m.get_most_significant_input_dimensions()[:2]
#ax.scatter(*m.X.mean.values[:,msi].T, c=t, cmap='hot')
#m.plot_inducing(ax=ax, color='w')
m.plot_magnification(resolution=20, scatter_kwargs=dict(color=cols, cmap='hot', s=20), marker='o', ax=ax)
ax.set_title('BGPLVM')
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])

for name in methods:
    ax = next(axit)
    ax.scatter(*X_init[:,dims[name]].T, c=cols, cmap='hot', lw=.1)
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
print seed
ax = next(axit)
ax.set_visible(False)
#plt.savefig('../diagrams/simulation/{}_comparison.pdf'.format(seed), transparent=True, bbox_inches='tight')

from manifold import waddington_landscape, plot_waddington_landscape

res = 120
Xgrid, wadXgrid, X, wadX = waddington_landscape(m, resolution=res)
ax = plot_waddington_landscape(Xgrid, wadXgrid, X, wadX, np.unique(labels), labels, resolution=res)
ax.view_init(elev=56, azim=-75)

from manifold import ManifoldCorrectionTree, ManifoldCorrectionKNN
import networkx as nx

msi = m.get_most_significant_input_dimensions()[:2]
X = m.X.mean[:,msi]
pos = dict([(i, x) for i, x in zip(range(X.shape[0]), X)])

mc = ManifoldCorrectionTree(m)

start = 6

pt = mc.distances_along_graph
pt_graph = mc.get_time_graph(start)

G = nx.Graph(pt_graph)

fig, ax = plt.subplots(figsize=(4,4))

m.plot_magnification(ax=ax, plot_scatter=False)
prevlab = None
for lab in labels:
    if lab != prevlab:
        color = plt.cm.hot(c[lab==labels])
        ax.scatter(*X[lab==labels].T, c=color, alpha=.9, lw=.1, label=lab)
        prevlab = lab
    
ecols = [e[2]['weight'] for e in G.edges(data=True)]
cmap = sns.cubehelix_palette(as_cmap=True, reverse=True, start=0, rot=0, dark=.2, light=.8, )

edges = nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color=ecols, edge_cmap=cmap, lw=2)
cbar = fig.colorbar(edges, ax=ax)
#cbar.set_ticks([1,13/2.,12])
#ax.set_xlim(-3,2)
#ax.set_ylim(-3,2.2)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

ax.scatter(*X[start].T, edgecolor='red', lw=1.5, facecolor='none', s=50, label='start')

ax.legend(bbox_to_anchor=(0., 1.02, 1.2, .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.)

fig.tight_layout(rect=(0,0,1,.9))
#fig.savefig('../diagrams/simulation/BGPLVMtree_{}_{}.pdf'.format(seed, y_seed), transparent=True, bbox_inches='tight')

ax = sns.jointplot(pt[start], t[:,0], kind="reg", size=4)
ax.ax_joint.set_xlabel('BGPLVM Extracted Time')
ax.ax_joint.set_ylabel('Simulated Time')
#ax.ax_joint.figure.savefig('../diagrams/simulation/BGPLVM_time_scatter_{}_{}.pdf'.format(seed, y_seed), transparent=True, bbox_inches='tight')

#fig, ax = plt.subplots(figsize=(4,4))
#msi = m.get_most_significant_input_dimensions()[:2]
#ax.scatter(*m.X.mean.values[:,msi].T, c=t, cmap='hot')
#m.plot_inducing(ax=ax, color='w')
#m.plot_magnification(resolution=20, scatter_kwargs=dict(color=cols, cmap='hot', s=20), marker='o', ax=ax)
#ax.set_title('BGPLVM')
#ax.set_xlabel('')
#ax.set_ylabel('')
#ax.set_xticks([])
#ax.set_yticks([])
#ax.figure.savefig('../diagrams/simulation/BGPLVM_magnificaton_{}_{}.pdf'.format(seed, y_seed), transparent=True, bbox_inches='tight')

print("MST spanning through the data")

fig, ax = plt.subplots(figsize=(4,4))
m.kern.plot_ARD(ax=ax)
#fig.savefig('../diagrams/simulation/BGPLVM_ARD_{}_{}.pdf'.format(seed, y_seed), transparent=True, bbox_inches='tight')

msi = m.get_most_significant_input_dimensions()[:2]
X = m.X.mean[:,msi]
pos = dict([(i, x) for i, x in zip(range(X.shape[0]), X)])

mc = ManifoldCorrectionKNN(m, 4)

start = 6

pt = mc.distances_along_graph
pt_graph = mc.get_time_graph(start)
G = nx.Graph(pt_graph)

fig, ax = plt.subplots(figsize=(4,4))

m.plot_magnification(ax=ax, plot_scatter=False)
prevlab = None
for lab in labels:
    if lab != prevlab:
        color = plt.cm.hot(c[lab==labels])
        ax.scatter(*X[lab==labels].T, c=color, alpha=.9, lw=.1, label=lab)
        prevlab = lab
    
ecols = [e[2]['weight'] for e in G.edges(data=True)]
cmap = sns.cubehelix_palette(as_cmap=True, reverse=True, start=0, rot=0, dark=.2, light=.8, )

edges = nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color=ecols, edge_cmap=cmap, lw=1.5)
cbar = fig.colorbar(edges, ax=ax)
#cbar.set_ticks([1,13/2.,12])
#ax.set_xlim(-3,2)
#ax.set_ylim(-3,2.2)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_frame_on(False)

ax.scatter(*X[start].T, edgecolor='red', lw=1.5, facecolor='none', s=50, label='start')

ax.legend(bbox_to_anchor=(0., 1.02, 1.2, .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.)

fig.tight_layout(rect=(0,0,1,.9))
#fig.savefig('../diagrams/simulation/BGPLVMknn_{}_{}.pdf'.format(seed, y_seed), transparent=True, bbox_inches='tight')

ax = sns.jointplot(pt[start], t[:,0], kind="reg", size=4)
ax.ax_joint.set_xlabel('BGPLVM Extracted Time')
ax.ax_joint.set_ylabel('Simulated Time')
#ax.ax_joint.figure.savefig('../diagrams/simulation/BGPLVM_knn_time_scatter_{}_{}.pdf'.format(seed, y_seed), transparent=True, bbox_inches='tight')

print ("3 Nearest Neighbor embedding and extracted time along it for the same Manifold embedding")

i = 0
for method in methods:
    print method, '{}:{}'.format(dims[method].start, dims[method].stop)
    i+=2

from scipy.spatial.distance import squareform, pdist

get_ipython().magic('run graph_extraction.py')

# Monocle:

X = X_init[:,dims['ICA']].copy()
pos = dict([(i, x) for i, x in zip(range(X.shape[0]), X)])

start = 6

pt, mst = extract_manifold_distances_mst(squareform(pdist(X)))
pt_graph = extract_distance_graph(pt, mst, start)
G = nx.Graph(pt_graph)

fig, ax = plt.subplots(figsize=(4,4))

prevlab = None
for lab in labels:
    if lab != prevlab:
        color = plt.cm.hot(c[lab==labels])
        ax.scatter(*X[lab==labels].T, c=color, alpha=.9, lw=.1, label=lab)
        prevlab = lab
        
ecols = [e[2]['weight'] for e in G.edges(data=True)]
cmap = sns.cubehelix_palette(as_cmap=True, reverse=True, start=0, rot=0, dark=.2, light=.8, )

edges = nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color=ecols, edge_cmap=cmap, lw=1.5)
cbar = fig.colorbar(edges, ax=ax)
#cbar.set_ticks([1,13/2.,12])
#ax.set_xlim(-3,2)
#ax.set_ylim(-3,2.2)
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

ax.scatter(*X[start].T, edgecolor='red', lw=1.5, facecolor='none', s=50, label='start')

ax.legend(bbox_to_anchor=(0., 1.02, 1.2, .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.)

fig.tight_layout(rect=(0,0,1,.9))
#fig.savefig('../diagrams/simulation/ICA_{}_{}.pdf'.format(seed, y_seed), transparent=True, bbox_inches='tight')

ax = sns.jointplot(pt[start], t[:,0], kind="reg", size=4)
ax.ax_joint.set_xlabel('Monocle Extracted Time')
ax.ax_joint.set_ylabel('Simulated Time')
#ax.ax_joint.figure.savefig('../diagrams/simulation/ICA_time_scatter_{}_{}.pdf'.format(seed, y_seed), transparent=True, bbox_inches='tight')

print("Monocle (MST on ICA embedding)")

from scipy.sparse import lil_matrix, find

# Wanderlust (without smoothing)

# take out tsne:
X = X_init[:,dims['t-SNE']].copy()
pos = dict([(i, x) for i, x in zip(range(X.shape[0]), X)])

k = 4
start = 6

_, mst = extract_manifold_distances_mst(squareform(pdist(X)))
pt, knn = extract_manifold_distances_knn(squareform(pdist(X)), knn=[k], add_mst=mst).next()
pt_graph = extract_distance_graph(pt, knn, start)
G = nx.Graph(pt_graph)

G = nx.Graph(pt_graph)

fig, ax = plt.subplots(figsize=(4,4))

prevlab = None
for lab in labels:
    if lab != prevlab:
        color = plt.cm.hot(c[lab==labels])
        ax.scatter(*X[lab==labels].T, c=color, alpha=.9, lw=.1, label=lab)
        prevlab = lab
        
ecols = [e[2]['weight'] for e in G.edges(data=True)]
cmap = sns.cubehelix_palette(as_cmap=True, reverse=True, start=0, rot=0, dark=.2, light=.8, )

edges = nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color=ecols, edge_cmap=cmap, lw=1.5)
cbar = fig.colorbar(edges, ax=ax)
#cbar.set_ticks([1,13/2.,12])
#ax.set_xlim(-3,2)
#ax.set_ylim(-3,2.2)
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

ax.scatter(*X[start].T, edgecolor='red', lw=1.5, facecolor='none', s=50, label='start')

ax.legend(bbox_to_anchor=(0., 1.02, 1.2, .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.)

fig.tight_layout(rect=(0,0,1,.9))
#fig.savefig('../diagrams/simulation/TSNE_knn_{}_{}.pdf'.format(seed, y_seed), transparent=True, bbox_inches='tight')

ax = sns.jointplot(pt[start], t[:,0], kind="reg", size=4)
ax.ax_joint.set_xlabel('t-SNE Extracted Time')
ax.ax_joint.set_ylabel('Simulated Time')

print("Wanderlust (KNN on t-SNE)")



