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
d = 12
size = 584
prob = 0
option= {'seed': seed, 'node_dim': d, 'model': choice}
option['k-NN'] = 2

from scipy.io import loadmat
from load_gene_network import load_mat_data, sparse_adjmat_oneshot, detect_isolates

X0, RNAseq_info = load_mat_data("../data/RNAseq_genes_allT_norm.mat", "RNAseq_genes_allT_norm")

At_Rsimt_resricted_allT, _ = load_mat_data("../data/At_Rsimt_resricted_allT.mat", "At_Rsimt_resricted_allT")

At_Rsimt_resricted_0 = sparse_adjmat_oneshot(At_Rsimt_resricted_allT, 0)

LSN = lsn(size, prob, option)
G0 = LSN.graph_from_sparse_adjmat(At_Rsimt_resricted_0)
G0.remove_nodes_from(nx.isolates(G0)) #remove isolates

#draw with spectral position, i.e. eigenvectors 
nx.draw_spectral(G0, scale=100, node_size=30)

seed = 30
scale = 100
np.random.seed(seed)
pos_init = dict(zip(G0.nodes(), scale*np.random.rand(len(G0),2)))

option['draw_scale'] = 100
pos= LSN.draw(G0, option, pos_init=pos_init)

Gcc=sorted(nx.connected_component_subgraphs(G0), key = len, reverse=True)[0]

len(G0)

len(Gcc)

option['draw_scale'] = scale
pos = LSN.draw(Gcc, option, pos_init=pos_init, save_fig=savefigure)

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
Laplacian = nx.normalized_laplacian_matrix(Gcc, weight=None).todense()
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

X0 -= np.mean(X0, axis=0)
X0 /= X0.std(axis=0)

n, m = X0.shape
emp_cov = np.cov(X0)
alpha = 0.4
lambda_s = 1
fig2= plt.figure(2, figsize=(15,6))
ax = fig2.add_subplot(121)
cax = ax.matshow(np.cov(X0))
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

n1 = 300
G1=sorted(nx.connected_component_subgraphs(G0.subgraph(np.arange(n1))), key = len, reverse=True)[0]
n1= len(G1)
n1



option['draw_scale'] = scale
pos1 = dict(zip(G1.nodes(), [pos[key] for key in G1.nodes()]))
_ = LSN.draw(G1, option, pos=pos1, pos_init=pos_init, node_size=50, save_fig=savefigure)

len(G1)

X1 = X0[G1.nodes(),:]

n, m = X0.shape
emp_cov = np.cov(X1)
alpha = 0.4
lambda_s = 1
fig2= plt.figure(2, figsize=(15,6))
ax = fig2.add_subplot(121)
cax = ax.matshow(emp_cov)
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

Laplacian1 = nx.normalized_laplacian_matrix(G1, weight=None).todense()

import pywt
fig2= plt.figure(2)
ax = fig2.add_subplot(111)
cax = ax.matshow(sp.sign(abs(Laplacian1)), cmap=plt.cm.gray)
fig2.colorbar(cax)


plt.show()
filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_cov_all.eps"
if savefigure : fig2.savefig(filename)

n2 = 70
G2=sorted(nx.connected_component_subgraphs(G1.subgraph(np.arange(n2))), key = len, reverse=True)[0]
n2= len(G2)
n2

pos2 = dict(zip(G2.nodes(), [pos[key] for key in G2.nodes()]))
_ = LSN.draw(G2, option, pos=pos2, pos_init=pos_init, node_size=50, save_fig=savefigure)

Laplacian2 = nx.normalized_laplacian_matrix(G2, weight=None).todense()
fig2= plt.figure(2)
ax = fig2.add_subplot(111)
cax = ax.matshow(sp.sign(abs(Laplacian2)), cmap=plt.cm.gray)
fig2.colorbar(cax)


plt.show()
filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_cov_all.eps"
#if savefigure : fig2.savefig(filename)

G1_Nodes = G1.nodes()
G2_Nodes = G2.nodes()
G2_complement_Nodes = list(set(G1_Nodes).difference(set(G2_Nodes)))

n3 = len(G2_complement_Nodes)
n3

n2

G2_complement = G1.subgraph(G2_complement_Nodes)

pos2_complement = dict(zip(G2_complement_Nodes, [pos[key] for key in G2_complement_Nodes]))
_ = LSN.draw(G2_complement, option, pos=pos2_complement, pos_init=pos_init, node_size=50, save_fig=savefigure)

X2 = X0[G2.nodes(),:]
alpha = 0.5

from graphical_lasso import sparse_inv_cov_glasso 

covariance, precision = sparse_inv_cov_glasso(X2, alpha=alpha, max_iter = 100)

# plot the precision matrix and the support of Laplacian matrix
fig2= plt.figure(2, figsize=(15,6))
ax = fig2.add_subplot(121)
cax = ax.matshow(sp.sign(abs(precision)), cmap=plt.cm.gray)
fig2.colorbar(cax)

ax = fig2.add_subplot(122)
Laplacian2 = nx.normalized_laplacian_matrix(G2, weight=None).todense()
cax = ax.matshow(sp.sign(abs(Laplacian2)), cmap=plt.cm.gray)
fig2.colorbar(cax)
#cbar.ax.set_yticklabels(['< -1', '0', '> 1'])

plt.show()
filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_precision_laplacian.eps"
if savefigure : fig2.savefig(filename)

from latent_graphical_lasso import latent_variable_gmm_cvx, latent_variable_glasso_data

#sparse_cvx_o, low_rank_cvx_o = latent_variable_gmm_cvx(X2, alpha=alpha, lambda_s=1, verbose=True)

#fig2= plt.figure(2, figsize=(15,6))
#ax = fig2.add_subplot(121)
#cax = ax.matshow(sp.sign(abs(sparse_cvx_o)), cmap=plt.cm.gray)
#fig2.colorbar(cax)

#ax = fig2.add_subplot(122)
#Laplacian2 = nx.normalized_laplacian_matrix(G2, weight=None).todense()
#cax = ax.matshow(sp.sign(abs(Laplacian2)), cmap=plt.cm.gray)
#fig2.colorbar(cax)
##cbar.ax.set_yticklabels(['< -1', '0', '> 1'])


#plt.show()
#filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_precision_laplacian.eps"
#if savefigure : fig2.savefig(filename)

X_o = X2
X_h = X0[G2_complement.nodes(),:]

covariance_em_o, precision_em_o, _, prec_all_list_em, _ =                     latent_variable_glasso_data(X_o, X_h, alpha=0.008, max_iter_out = 100, 
                                               verbose=True, return_hists=True)

# plot the precision matrix and the support of Laplacian matrix
fig2= plt.figure(2, figsize=(15,6))
ax = fig2.add_subplot(121)
cax = ax.matshow(sp.sign(abs(precision_em_o)), cmap=plt.cm.gray)
fig2.colorbar(cax)

ax = fig2.add_subplot(122)



cax = ax.matshow(sp.sign(abs(Laplacian2)), cmap=plt.cm.gray)
fig2.colorbar(cax)
#cbar.ax.set_yticklabels(['< -1', '0', '> 1'])

plt.show()
filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_precision_laplacian_em.eps"
if savefigure : fig2.savefig(filename)





