get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import sys
import numpy as np
import scipy.cluster.hierarchy as sch
import pylab
import scipy
import matplotlib.pyplot as plt

# Add the ptdraft folder path to the sys.path list
sys.path.append('../src/')
import data_config as dc
reload(dc)

wd = dc.kato.data()

import katotools as kt
derivs = kt.integrated('deltaFOverF')

# Recalculating everything again is messy, I know. I'll change it
cross_correlations = derivs.cross_correlations(global_ns=True)
groupings = derivs.neuron_pairings
cross_correlations

linkages = {}
indexes = {}
for i in groupings:
    D = cross_correlations[i]
    Y = sch.linkage(D, method='centroid')
    linkages[i] = Y  
    indexes[i] = sch.leaves_list(Y)

get_ipython().magic('matplotlib inline')
import os
if not os.path.exists('./Dendrogram'):
    os.mkdir('./Dendrogram')

for i in [(n,n) for n in range(5)]: #in [(j,j) for j in xrange(5)]:
    D=cross_correlations[i]
    
    # Compute and plot dendrogram.
    fig = plt.figure()
    fig.suptitle('Dataset {0}'.format(i))

    axdendro = fig.add_axes([0.00,0.15,0.19,0.8])
    Y = sch.linkage(D, method='centroid')
    Z = sch.dendrogram(Y, orientation='right', labels = derivs.global_neurons, ax=axdendro)
    
    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.15,0.6,0.8])
    index = Z['leaves']
    D = D[index][:,index]
    im = axmatrix.matshow(D, aspect='auto', origin='lower')
    axdendro.set_yticks([])
    axdendro.set_xticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.9,0.15,0.02,0.8])
    pylab.colorbar(im, cax=axcolor)

    ticks = np.array(derivs.global_neurons)[index]
    xticks = np.arange(0,len(ticks))
    yticks = np.arange(0,len(ticks))

    axmatrix.set_yticks(yticks, minor=False)
    axmatrix.set_yticklabels(ticks, minor=False)
    axmatrix.yaxis.set_ticks_position('left') 



    axmatrix.set_xticks(xticks, minor=False)
    axmatrix.set_xticklabels(ticks, minor=False, rotation='vertical')
    axmatrix.xaxis.set_ticks_position('bottom') 

    # Display and save figure.
    fig.savefig('Dendrogram/dendrogram-{0}.png'.format(i))

data = kt.integrated('deltaFOverF')

get_ipython().magic('matplotlib inline')

import dimensions_kit as dk
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import graphing as graph
import numpy as np

dimensions=(5,5)


from itertools import *
pairings = [ i for i in product(range(5), range(5))]

def plot_crosscorrelations(title, dataset,global_ns=False):
    data = kt.integrated(dataset)
    cross_correlations = data.cross_correlations(global_ns=global_ns)
    
    def draw((x,y), axes, f, plot):
        correl = cross_correlations[(x,y)]
        Z = sch.linkage(correl, method='centroid')
        indexs = sch.leaves_list(Z)
        
        axis = axes[x][y]
                        
        neurons = data.local_neurons[(x,y)]
        ticks = neurons
        
        xticks = np.arange(0,len(ticks))+0.5
        yticks = np.arange(0,len(ticks))+0.5
        
        axis.set_yticks(yticks, minor=False)
        axis.set_yticklabels(ticks, minor=False)
        
        axis.set_xticks(xticks, minor=False)
        axis.set_xticklabels(ticks, minor=False, rotation='vertical')
        
        pc = axis.pcolormesh(correl[indexs][:,indexs])
        
        div = make_axes_locatable(axis)
        cax = div.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(pc, cax=cax)
        
        axis.set_title('Dataset ${0}$ vs ${1}$ Crosscorrelation'.format(x,y))
        axis.axis('tight')
    
    f, axes = graph.plot_grid(draw, dims=dimensions, include=groupings)
    f.suptitle(title, fontsize=15,x=0.5,y=1)
    f.tight_layout()
    f.set_size_inches(20, 20, forward=True)


    
plot_crosscorrelations('CrossCorr across globally shared neurons', 
                       'deltaFOverF', global_ns=True)
plt.savefig('/Users/akivalipshitz/Desktop/CrossCorr-Global.jpg')

plot_crosscorrelations('CrossCorr across locally shared neurons', 'deltaFOverF')
plt.savefig('/Users/akivalipshitz/Desktop/CrossCorr-Local.jpg')

import katotools as kt
wormdata = kt.integrated('deltaFOverF')
gt = wormdata.global_timeseries()


get_ipython().magic('matplotlib inline')
import data_config as dc
wormData = dc.kato.data()
from mpl_toolkits.axes_grid1 import make_axes_locatable

import dimensions_kit as dk
dimensions=(3,2)
def plot_shared_datasets(dataset):
    data = kt.integrated(dataset).global_timeseries()
    f, axes = plt.subplots(*dimensions,figsize=(20,20))
    f.suptitle('Clustered {0} Timeseries'.format(dataset), fontsize=20)

    for n in range(len(data)):
        x,y = dk.transform(dimensions,n)

        axis = axes[x][y]
        
        indexs = np.array(indexes[(n,n)])
        
        plotting = data[n].timeseries[indexs]
        
        pc = axis.pcolormesh(plotting, vmin=-0.2, vmax=1.5)
        
        # Dealing with the colorbar
        div = make_axes_locatable(axis)
        cax = div.append_axes("right", size="2%", pad=0.05)
        cbar = plt.colorbar(pc, cax=cax)
        
        # Setting yticks for easy identification
        yticks = np.array(data[n].nnames)[indexs]
        axis.set_yticks(np.arange(len(yticks))+0.5, minor=False)
        axis.set_yticklabels(yticks, minor=False)
        
        # Setting xticks
        xticks = wormData[n]['tv'][0]
        axis.set_xticks(np.linspace(0, 3500,num=10))
        axis.set_xticklabels(np.around(np.array(axis.get_xticks())*0.34460753/60,decimals=2))
        
        axis.set_title('Dataset ${0}$ {1}'.format(n,dataset))
        axis.set_ylabel('Neuron')
        axis.set_xlabel('Time (minutes)')
        axis.axis('tight')
        print 'generating'
    axes[2,1].axis('off')
    
plot_shared_datasets('deltaFOverF')
plt.savefig('/Users/akivalipshitz/Desktop/shared_datasets.jpg',dpi=200)

adjacency = cross_correlations[(1,1)]
adjacency[np.abs(adjacency)<0.015]=0

# We can represent neuron connectivity as a graph, with edge weights being the strength
# of correlation between two neurons. We can change the metric to cross_correlation, etc. 

# It would be nice to have a library of arbitary graph clustering algorithms implemented
# for clustering any adjacency matrix. 
import networkx as nx
get_ipython().magic('pylab')
get_ipython().magic('matplotlib inline')
global_neurons = derivs.global_shared()
# def normalized(a, axis=-1, order=2):
#     l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
#     l2[l2==0] = 1
#     return a / np.expand_dims(l2, axis)

adjacency = cross_correlations[(1,1)]
adjacency==0
matshow(adjacency)
colorbar()
figure()
G = nx.Graph()
G.add_nodes_from(global_neurons)
for i in range(adjacency.shape[0]):
    for j in range(adjacency.shape[1]):
        G.add_edge(global_neurons[i], global_neurons[j], weight=adjacency[i,j])

        
colors=range(len(G.edges()))
plt.figure(figsize=(10,10))
pos=nx.spring_layout(G,scale=2)

x = nx.draw(G,
       pos=pos,
       with_labels=True, 
       node_size=700,
       font_size=8,
       edge_cmap=plt.cm.magma,
       edge_color=colors,
       node_color='w')

points = np.array([v for k,v in pos.iteritems()]).T

plt.figure(figsize=(10,10))
nnames = pos.keys()
plt.scatter(points[0],points[1])

for n in range(len(nnames)):
    plt.annotate('%s'%nnames[n], [points.T[n][0], points.T[n][1]], textcoords='offset points')


get_ipython().magic('matplotlib tk')

# Now we need to generate a set of points in 3d space for one crosscorrelation
data = cross_correlations[(3,3)]

xx = np.arange(data.shape[0])
yy = np.arange(data.shape[1])

points = np.array([ [x,y,data[x,y]] for x in xx for y in yy])
X = points
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(points[:,0],points[:,1],points[:,2])
plt.show()

# from sklearn.cluster import AffinityPropagation
# af = AffinityPropagation(preference=-50).fit(points)
# cluster_centers_indices = af.cluster_centers_indices_
# n_clusters_ = len(cluster_centers_indices)
# labels = af.labels_

# import matplotlib.pyplot as plt
# from itertools import cycle

# plt.close('all')
# plt.figure(1)
# plt.clf()

# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     class_members = labels == k
#     cluster_center = X[cluster_centers_indices[k]]
#     plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
#     for x in X[class_members]:
#         plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()



#lets do this ft thinga majing
import scipy.fftpack

timeseries = wormData[0]['deltaFOverF'][34]
x = wormData[0]['tv'][0]

N=x.shape[0]

plt.figure(figsize=(20,5))
plt.title('DeltaFOverF for a random neuron')
plt.ylabel('Ca+ Fluorescence')
plt.xlabel('Time (ms)')
plt.plot(x,timeseries)


w = scipy.fftpack.rfft(timeseries)
f = scipy.fftpack.rfftfreq(N, x[1]-x[0])
spectrum = w**2

y2 = scipy.fftpack.irfft(w)

plt.figure(figsize=(20,5))
plt.title('Adding fourier waves')
plt.ylim((-1.5,1.5))
plt.plot(x,w)






