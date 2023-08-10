import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import os
from os import getcwd
from os.path import dirname
from time import gmtime, strftime
import scipy as sp
import cvxpy as cvx
import pywt
from sklearn.preprocessing import binarize
from sklearn.linear_model import lars_path
get_ipython().magic('matplotlib inline')
import sys

cwd = getcwd()
dir_root = dirname(cwd)
filepath = os.path.join(dir_root, 'src')
sys.path.append(filepath) #('/home/tianpei/Dropbox/Codes/Python/LatNet/src/')
print(filepath)
get_ipython().magic('load_ext Cython')

from latent_signal_network import latent_signal_network as lsn 

savefigure = False
seed = 1000
choice = 'imported'
d = 90 #600
size = 88 #584
prob = 0
time = 8
option= {'seed': seed, 'node_dim': d, 'model': choice}
option['k-NN'] = 2

from scipy.io import loadmat
from load_gene_network import load_mat_data, sparse_adjmat_oneshot, detect_isolates

RNA_seq, RNAseq_info = load_mat_data("../data/RNAseq_genes_allT_norm.mat", "RNAseq_genes_allT_norm")

At_Rsimt_resricted_allT, _ = load_mat_data("../data/At_Rsimt_resricted_allT.mat", "At_Rsimt_resricted_allT")

At_Rsimt_resricted_0 = sparse_adjmat_oneshot(At_Rsimt_resricted_allT, time)

n1, n2 = [85, 60]
#n1 = 85  #90    #85  #size of total network
#n2 = 60  #70   #60  #size of a connected subgraph 

LSN = lsn(size, prob, option)
G0, G1, node_lists, node_sets = LSN.graph_from_sparse_adjmat(At_Rsimt_resricted_0, node_range=np.arange(n1), subset_range=np.arange(n2))
#G0.remove_nodes_from(nx.isolates(G0)) #remove isolates

n1 = len(G0)
n2 = len(G1)
print("size of total graph %d, size of sub-network %d" % (n1, n2))

#draw with spectral position, i.e. eigenvectors 
nx.draw_spectral(G0, scale=100, node_size=30)

seed = 30
scale = 100
np.random.seed(seed)
pos_init = dict(zip(G0.nodes(), scale*np.random.rand(len(G0),2)))

option['draw_scale'] = 2000
pos = nx.shell_layout(G0, [node_lists[0], node_lists[1]]) #nx.nx_pydot.graphviz_layout(G0,  prog='dot')  #nx.circular_layout(G0,  scale=option['draw_scale'])
pos= LSN.draw(G0, option, node_lists=node_lists, pos_init=pos_init, pos=pos, node_size=250, 
              with_labels=True, fontsize=3, font_color='w')

degree_sequence=sorted(nx.degree(G0).values(),reverse=True) # degree sequence
#print "Degree sequence", degree_sequence
dmax=max(degree_sequence)

fig3 = plt.figure(3)
#plt.loglog(degree_sequence,'b-',marker='o')
plt.plot(degree_sequence,'b-',marker='o')
#plt.title("Degree rank plot")
plt.ylabel("degree")
plt.xlabel("rank")

# draw graph in inset
plt.axes([0.45,0.45,0.45,0.45])
Gcc=sorted(nx.connected_component_subgraphs(G0), key = len, reverse=True)[0]
#pos=nx.spring_layout(Gcc)
plt.axis('off')
nx.draw_networkx_nodes(Gcc,pos,node_size=20)
nx.draw_networkx_edges(Gcc,pos,alpha=0.4)

filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_degree_rank_plot.eps"
#filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_eigenvalue_adjMat.eps"
if savefigure : fig3.savefig(filename)
plt.show()

#Plot the eigenvalue of Laplacian matrix
Laplacian = nx.normalized_laplacian_matrix(G0, weight=None).todense()
#Sigma, U = np.linalg.eigh(abs(adjMat))
Sigma, U = np.linalg.eigh(Laplacian)

index_sig = np.argsort(Sigma)
Sigma = Sigma[index_sig[::-1]]
U = U[:,index_sig[::-1]]

fig3 =plt.figure(3)
ax = plt.gca()
(markerline, stemlines, baseline) = plt.stem(np.arange(len(Sigma)), Sigma, 'b', basefmt='k-')
#plt.plot(np.arange(len(Sigma)), np.ones((len(Sigma, ))), 'r')
plt.xlabel('rank of eigenvalue')
plt.ylabel('eigenvalue')
ax.grid(True)
plt.show()
filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_eigenvalue_laplacian.eps"
#filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_eigenvalue_adjMat.eps"
if savefigure : fig3.savefig(filename)

option['draw_scale'] = scale
pos1 = dict(zip(G1.nodes(), [pos[key] for key in G1.nodes()]))
_ = LSN.draw(G1, option, pos=pos1, pos_init=pos_init, with_labels=True, node_size=250, save_fig=savefigure)

Laplacian1 = nx.normalized_laplacian_matrix(G1, weight=None).todense()

import pywt
fig2= plt.figure(2)
ax = fig2.add_subplot(111)
cax = ax.matshow(sp.sign(abs(Laplacian)), cmap=plt.cm.gray)
fig2.colorbar(cax)


plt.show()
filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_cov_all.eps"
if savefigure : fig2.savefig(filename)

G0_Nodes = G0.nodes()
G1_Nodes = G1.nodes()
G1_complement_Nodes = list(set(G0_Nodes).difference(set(G1_Nodes)))

n3 = len(G1_complement_Nodes)
n3

n2

G1_complement = G0.subgraph(G1_complement_Nodes)

pos1_complement = dict(zip(G1_complement_Nodes, [pos[key] for key in G1_complement_Nodes]))
_ = LSN.draw(G1_complement, option, pos=pos1_complement, pos_init=pos_init, with_labels=True, node_size=200, save_fig=savefigure)

nodeIdx = [{'node': idx, 'loc': i} for i, idx in enumerate(G0.nodes_iter())]
observed_idx = [item['loc'] for item in nodeIdx if item['node'] in node_sets[0]]
hidden_idx = [item['loc'] for item in nodeIdx if item['node'] in node_sets[1]]

x = RNA_seq[G0.nodes(),time]
x -= np.mean(x)
x /= np.linalg.norm(x)
r = np.random.randn(d) #a Gaussian scale vector for each 
r /= np.linalg.norm(r)
n = len(x)
scale_factor = np.sqrt(n)

sigma_0 = 0.5/np.sqrt(d)
power = 2

X0 = power*scale_factor*np.outer(x, r)
noise = sigma_0*scale_factor*np.random.randn(X0.shape[0], X0.shape[1])
X0.shape

print("SNR = %.3f dB" % (10*np.log10(power**2/(sigma_0**2))))

X0 += noise
#X0 -= np.mean(X0, axis=0)
#X0 /= X0.std(axis=0)

np.mean(X0[0,:])

n, m = X0.shape
emp_cov = np.cov(X0)

plt.stem(sorted(np.linalg.eigvals(emp_cov)))

fig2= plt.figure(2, figsize=(15,6))
ax = fig2.add_subplot(121)
cax = ax.matshow(emp_cov)
fig2.colorbar(cax)

ax = fig2.add_subplot(122)
adjMatSparse =  nx.adjacency_matrix(G0, weight=None)
adjMat = adjMatSparse.todense()
cax = ax.matshow(adjMat, cmap=plt.cm.gray)
fig2.colorbar(cax)
#cbar.ax.set_yticklabels(['< -1', '0', '> 1'])
plt.show()
filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_cov_adjmat.eps"
if savefigure : fig2.savefig(filename)


fig2= plt.figure(2)
ax = fig2.add_subplot(111)
cax = ax.matshow(np.sign(abs(Laplacian)), cmap=plt.cm.gray)
fig2.colorbar(cax)


plt.show()
filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_cov_all.eps"
if savefigure : fig2.savefig(filename)

X_o = X0[observed_idx,:]
X_h = X0[hidden_idx,:]

n, m = X0.shape
emp_cov_o = np.cov(X_o)
alpha = 0.4
lambda_s = 1
fig2= plt.figure(2, figsize=(15,6))
ax = fig2.add_subplot(121)
cax = ax.matshow(emp_cov_o)
fig2.colorbar(cax)

ax = fig2.add_subplot(122)
adjMatSparse =  nx.adjacency_matrix(G1, weight=None)
adjMat = adjMatSparse.todense()
cax = ax.matshow(adjMat, cmap=plt.cm.gray)
fig2.colorbar(cax)
#cbar.ax.set_yticklabels(['< -1', '0', '> 1'])
plt.show()
filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_cov_adjmat.eps"
if savefigure : fig2.savefig(filename)

from graphical_lasso import sparse_inv_cov_glasso 

alpha = 7.5e-2

covariance_all, precision_all = sparse_inv_cov_glasso(X0, alpha=alpha, max_iter = 100)

# plot the precision matrix and the support of Laplacian matrix
fig2= plt.figure(2, figsize=(15,6))
ax = fig2.add_subplot(121)
cax = ax.matshow(sp.sign(abs(precision_all)), cmap=plt.cm.gray)
fig2.colorbar(cax)

ax = fig2.add_subplot(122)
cax = ax.matshow(sp.sign(abs(Laplacian)), cmap=plt.cm.gray)
fig2.colorbar(cax)
#cbar.ax.set_yticklabels(['< -1', '0', '> 1'])

plt.show()
filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_precision_laplacian.eps"
if savefigure : fig2.savefig(filename)

from generalized_Laplacian_estimate import generalized_Laplacian_estimate

gen_Laplacian = generalized_Laplacian_estimate(X0, max_iter = 2000)

# plot the precision matrix and the support of Laplacian matrix
fig2= plt.figure(2, figsize=(15,6))
ax = fig2.add_subplot(121)
cax = ax.matshow(sp.sign(abs(gen_Laplacian)), cmap=plt.cm.gray)
fig2.colorbar(cax)

ax = fig2.add_subplot(122)
cax = ax.matshow(sp.sign(abs((Laplacian))), cmap=plt.cm.gray)
fig2.colorbar(cax)
#cbar.ax.set_yticklabels(['< -1', '0', '> 1'])

plt.show()
filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_precision_laplacian_genLap.eps"
if savefigure : fig2.savefig(filename)



alpha = 6.5e-2

from graphical_lasso import sparse_inv_cov_glasso 

covariance, precision = sparse_inv_cov_glasso(X_o, alpha=alpha, max_iter = 100)

# plot the precision matrix and the support of Laplacian matrix
fig2= plt.figure(2, figsize=(15,6))
ax = fig2.add_subplot(121)
cax = ax.matshow(sp.sign(abs(precision)), cmap=plt.cm.gray)
fig2.colorbar(cax)

ax = fig2.add_subplot(122)
Laplacian1 = nx.normalized_laplacian_matrix(G1, weight=None).todense()
cax = ax.matshow(sp.sign(abs(Laplacian1)), cmap=plt.cm.gray)
fig2.colorbar(cax)
#cbar.ax.set_yticklabels(['< -1', '0', '> 1'])

plt.show()
filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_precision_laplacian.eps"
if savefigure : fig2.savefig(filename)

from latent_graphical_lasso import latent_variable_gmm_cvx, latent_variable_glasso_data

#alpha = 0.005
#sparse_cvx_o, low_rank_cvx_o = latent_variable_gmm_cvx(X_o, alpha=alpha, lambda_s=1, verbose=True)

#import pywt
#fig2= plt.figure(2, figsize=(15,6))
#ax = fig2.add_subplot(121)
#cax = ax.matshow(sp.sign(abs(pywt.threshold(sparse_cvx_o, 1e-4, 'hard'))), cmap=plt.cm.gray)
#fig2.colorbar(cax)

#ax = fig2.add_subplot(122)
#Laplacian1 = nx.normalized_laplacian_matrix(G1, weight=None).todense()
#cax = ax.matshow(sp.sign(abs(Laplacian1)), cmap=plt.cm.gray)
#fig2.colorbar(cax)
#cbar.ax.set_yticklabels(['< -1', '0', '> 1'])


#plt.show()
#filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_precision_laplacian.eps"
#if savefigure : fig2.savefig(filename)

mask = np.zeros((len(G0), len(G0)))
mask[np.ix_(observed_idx[0:40],hidden_idx)] = np.ones((len(observed_idx[0:40]), len(hidden_idx)))
mask[np.ix_(hidden_idx, observed_idx[0:40])] = np.ones(( len(hidden_idx), len(observed_idx[0:40])))
mask[np.ix_(observed_idx, observed_idx)] = np.ones((len(node_sets[0]), len(node_sets[0])))

alpha = 0.03
covariance_em_o, precision_em_o, _, prec_all_list_em, dsol_list =                     latent_variable_glasso_data(X_o, X_h, alpha=alpha, max_iter_out = 400, 
                                                verbose=False, threshold=5e-3, return_hists=True)

plt.semilogy(dsol_list)
plt.xlabel('iteration')
plt.ylabel('difference of solutions')
plt.show()

# plot the precision matrix and the support of Laplacian matrix
fig2= plt.figure(2, figsize=(15,6))
ax = fig2.add_subplot(121)
cax = ax.matshow(sp.sign(abs(precision_em_o)), cmap=plt.cm.gray)
fig2.colorbar(cax)

ax = fig2.add_subplot(122)



cax = ax.matshow(sp.sign(abs(Laplacian1)), cmap=plt.cm.gray)
fig2.colorbar(cax)
#cbar.ax.set_yticklabels(['< -1', '0', '> 1'])

plt.show()
filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_precision_laplacian_em.eps"
if savefigure : fig2.savefig(filename)





