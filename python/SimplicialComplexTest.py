import neuraltda.simpComp as sc
import neuraltda.topology2 as tp2
import neuraltda.spectralAnalysis as sa
import pickle
import glob
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
reload(sa)
reload(sc)

blockPath = '/Volumes/gentner/btheilma/experiments/B1235/phy051316/Pen02_Lft_AP200_ML800__Site03_Z3000__B1235_cat_P02_S03_1/'
bdf = '/Volumes/gentner/btheilma/experiments/B1235/phy051316/Pen02_Lft_AP200_ML800__Site03_Z3000__B1235_cat_P02_S03_1/binned_data/20170128T005306Z/20170128T005306Z-10.0-5.0.binned'
scgf = '/Volumes/gentner/btheilma/experiments/B1235/phy051316/Pen02_Lft_AP200_ML800__Site03_Z3000__B1235_cat_P02_S03_1/scg/20170128T005306Z-10.0-5.0.scg'

with open(scgf, 'r') as scgff:
    scg = pickle.load(scgff)

bp1 = '/Volumes/gentner/btheilma/experiments/B1235/phy051316/Pen02_Lft_AP200_ML800__Site01_Z3000__B1235_cat_P02_S01_1/'
#bp1 = '/Volumes/gentner/btheilma/experiments/B1075/phy041216/Pen01_Lft_AP300_ML700__Site03_Z2700__B1075_cat_P01_S03_1/'
bps = [bp1]

winSize = 25.0 #ms
segmentInfo = {'period': 1}
ncellsperm = 0
nperms = 0
nshuffs = 0
thresh = 10.0
propOverlap = 0.5
dtovr = propOverlap*winSize

for blockPath in bps:
    bfdict = tp2.dag_bin(blockPath, winSize, segmentInfo, ncellsperm, nperms, nshuffs, dtOverlap=dtovr)
    bdf = glob.glob(os.path.join(bfdict['raw'], '*.binned'))[0]
    sa.computeChainGroups(blockPath, bdf, thresh)

bdf = glob.glob(os.path.join(bfdict['raw'], '*.binned'))[0]
sa.computeChainGroups(blockPath, bdf, thresh)

blockPath = '/Volumes/gentner/btheilma/experiments/B1235/phy051316/Pen02_Lft_AP200_ML800__Site01_Z3000__B1235_cat_P02_S01_1/'

blockPath = '/mnt/cube/btheilma/emily/B604/klusta/phy103116/Pen01_Lft_AP2300_ML1250__Site15_Z2780__B604_cat_P01_S15_1/'
#Load simplicial chain generator dictionary
scgfs = glob.glob(os.path.join(blockPath, 'scg/*.scg'))[0]
print(scgfs)
with open(scgfs, 'r') as scgf:
    scg = pickle.load(scgf)

scg.keys()


stimtrials = [(s, t) for s in scg.keys() for t in range(20)]
stimtrials = [(s, t) for s in scg.keys()[0:5] for t in range(5)]
stimtrials
labs = [str(s[0])+ str(s[1]) for s in stimtrials]

stimtrials = [(s, t) for s in ['T40S40D3', 'T40S70D3', 'T-1S-1D40'] for t in range(5)]
stimtrials = [(s, t) for s in scg.keys() for t in range(5)]

print(stimtrials)

beta = 0.15
divsave = []
d = 1
for ind in range(len(stimtrials)):
    print(ind)
    for ind2 in range(ind, len(stimtrials)):
        
        stim1 = stimtrials[ind][0]
        t1 = stimtrials[ind][1]
        stim2 = stimtrials[ind2][0]
        t2 = stimtrials[ind2][1]
        #print(stim1, t1, stim2, t2)
        scg1 = scg[stim1][t1]
        scg2 = scg[stim2][t2]
        
        scgTot = sc.simplexUnion(scg1, scg2)
        mE = sc.maxEnt(scgTot, d)
        
        D1 = sc.maskedBoundaryOperatorMatrix(scgTot, scg1)
        D2 = sc.maskedBoundaryOperatorMatrix(scgTot, scg2)
        
        #rhos1 = sc.densityMatrices(D1, beta*np.ones(len(D1)))
        #rhos2 = sc.densityMatrices(D2, beta*np.ones(len(D2)))
        L1 = sc.laplacian(D1, d)
        L2 = sc.laplacian(D2, d)
        
        rho1 = sc.densityMatrix(L1, beta)
        rho2 = sc.densityMatrix(L2, beta)
        #div = sc.JSdivergences(rhos1, rhos2)
        div = sc.JSdivergence(rho1, rho2)
        divsave.append(((stim1, t1), (stim2, t2), div))

len(divsave)

d = 2

mat = np.zeros((len(stimtrials), len(stimtrials)))
iu1 = np.triu_indices(len(stimtrials))
for ind, x in enumerate(divsave):

    mat[iu1[0][ind], iu1[1][ind]] = x[2]

mat = mat + mat.T
pickle.dump(mat, open('B604_P01S15_Lap1_0.15_dmat.pkl', 'w'))
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.set_cmap('hot')
plt.figure(figsize=(11,11))
plt.imshow(1.0/mat, interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(0, len(stimtrials), 5))
plt.yticks(np.arange(0, len(stimtrials),5))
#plt.savefig('/Users/brad/B1235_P02S01_1LaplacianComparison.pdf', format='pdf')
#plt.clim(vmin=3)

plt.figure(figsize=(11,11))
plt.imshow(mat[:, :])
plt.clim(0, 0.5)

plt.figure(figsize=(11,11))
normmat = mat/np.max(np.max(mat))
plt.imshow(1.0/normmat)
plt.grid(True)
plt.xticks(np.arange(0, 100, 10))
plt.yticks(np.arange(0,100, 10))

from scipy.cluster import hierarchy as ha

stimlabels = [str(s[0]) for s in stimtrials]
print(stimlabels)
# First define the leaf label function.
n = len(stimtrials)
def llf(id):
    if id < n:
        return stimlabels[id]
    else:
        return '[%d]' % (id)

linmat = mat[np.triu_indices(len(stimtrials))]
Z = ha.linkage(linmat, method='complete')
# calculate full dendrogram
plt.figure(figsize=(25, 25))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
ha.dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=18.,  # font size for the x axis labels
    leaf_label_func=llf
)
plt.show()

import scipy
import pylab
import scipy.cluster.hierarchy as sch

# Generate features and distance matrix.
D = mat
method = 'complete'
# Compute and plot dendrogram.
fig = pylab.figure(figsize=(15,15))
ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
Y = sch.linkage(D, method=method)
Z1 = sch.dendrogram(Y, orientation='left')
ax1.set_xticks([])
ax1.set_yticks([])


# Compute and plot second dendrogram.
ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
Y = sch.linkage(D, method=method)
Z2 = sch.dendrogram(Y)
ax2.set_xticks([])
ax2.set_yticks([])

# Plot distance matrix.
axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
idx1 = Z1['leaves']
idx2 = Z2['leaves']
D = D[idx1,:]
D = D[:,idx2]
im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
axmatrix.set_xticks([])
axmatrix.set_yticks([])


# Plot colorbar.
axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
plt.colorbar(im, cax=axcolor)

# Display and save figure.
fig.show()
fig.savefig('/home/brad/B604_P01S15_Lap1_AllStims_dendrogram.png')

from sklearn.manifold import MDS

matMDS = MDS(n_components=2, dissimilarity='precomputed')
matMDS.fit(mat)
pts = matMDS.embedding_
#pts = np.log(pts)

plt.plot(pts[:, 0], pts[:, 1], 'b.')

r = np.diag(np.sqrt(np.dot(pts, pts.T)))
theta = np.arctan2(pts[:, 1], pts[:, 0])

lr = np.log(r)
ptsx = lr*np.cos(theta)
ptsy = lr*np.sin(theta)

plt.plot(ptsx, ptsy, '.', pts[:, 0], pts[:, 1], 'r.')

from scipy.cluster import hierarchy as ha

linmat = mat[np.triu_indices(len(stimtrials))]
z = ha.linkage(linmat)

divsavefile = './B1235_P02S01_divsave.pkl'
with open(divsavefile, 'w') as f:
    pickle.dump(divsave, f)

scgTot

E = sc.simplicialChainGroups([(1,2,3)])
(t, basis) = sc.stimSpaceGraph(E, sc.boundaryOperatorMatrix(E))
print(t)
print(basis)

gms = sc.adjacency2maxsimp(t)

Egraph = sc.simplicialChainGroups(gms)
Dgraph = sc.boundaryOperatorMatrix(Egraph)
print(Dgraph)
print(Egraph)
sc.laplacian(sc.boundaryOperatorMatrix(Egraph), 0)

sc.graphLaplacian(t)

np.dot(Dgraph[1], Dgraph[1].T)
np.dot(Dgraph[0].T, Dgraph[0])

(t, basis) = sc.stimSpaceGraph(scgTot, sc.boundaryOperatorMatrix(scgTot))
print(t)
print(basis)
gms = sc.adjacency2maxsimp(t)

Egraph = sc.simplicialChainGroups(gms)
Dgraph = sc.boundaryOperatorMatrix(Egraph)
print(Dgraph)
print(Egraph)
sc.laplacian(sc.boundaryOperatorMatrix(Egraph), 0)

ntrials = 5
stimtrials = [(s, t) for s in scg.keys() for t in range(20)]
stimtrials = [(s, t) for s in scg.keys()[0:2] for t in range(ntrials)]
stimtrials

s1 = sc.simplicialChainGroups([(1,2,3)])
s2 = sc.simplicialChainGroups([(4,5,6)])
stimtrials = (s1, s2)

beta = 1
divsave = []
d = 0
for ind in range(len(stimtrials)):
    print(ind)
    for ind2 in range(ind, len(stimtrials)):
        
        stim1 = stimtrials[ind][0]
        t1 = stimtrials[ind][1]
        stim2 = stimtrials[ind2][0]
        t2 = stimtrials[ind2][1]
        #print(stim1, t1, stim2, t2)
        scg1 = scg[stim1][t1]
        scg2 = scg[stim2][t2]
        
        scgTot = sc.simplexUnion(scg1, scg2)
        (adjTot, basisTot) = sc.stimSpaceGraph(scgTot, sc.boundaryOperatorMatrix(scgTot))
        (adj1, basis1) = sc.stimSpaceGraph(scg1, sc.boundaryOperatorMatrix(scg1))
        (adj2, basis2) = sc.stimSpaceGraph(scg2, sc.boundaryOperatorMatrix(scg2))
        
        EgraphTot = sc.simplicialChainGroups(sc.adjacency2maxsimp(adjTot, basisTot))
        Egraph1 = sc.simplicialChainGroups(sc.adjacency2maxsimp(adj1, basis1))
        Egraph2 = sc.simplicialChainGroups(sc.adjacency2maxsimp(adj2, basis2))
        D1 = sc.maskedBoundaryOperatorMatrix(EgraphTot, Egraph1)
        D2 = sc.maskedBoundaryOperatorMatrix(EgraphTot, Egraph2)
        
        #rhos1 = sc.densityMatrices(D1, beta*np.ones(len(D1)))
        #rhos2 = sc.densityMatrices(D2, beta*np.ones(len(D2)))
        
        L1 = sc.laplacian(D1, d)
        L2 = sc.laplacian(D2, d)
        
        rho1 = sc.densityMatrix(L1, beta)
        rho2 = sc.densityMatrix(L2, beta)
        #div = sc.JSdivergences(rhos1, rhos2)
        div = sc.JSdivergence(rho1, rho2)
        divsave.append(((stim1, t1), (stim2, t2), div))
        
mat = np.zeros((len(stimtrials), len(stimtrials)))
iu1 = np.triu_indices(len(stimtrials))
for ind, x in enumerate(divsave):
    mat[iu1[0][ind], iu1[1][ind]] = x[2]
mat = mat + mat.T

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.set_cmap('hot')
plt.figure(figsize=(11,11))
plt.imshow(1.0/mat, interpolation='none')
plt.grid(True)
plt.xticks(np.arange(0, len(stimtrials), ntrials))
plt.yticks(np.arange(0, len(stimtrials),ntrials))
#plt.clim(vmin=3)

plt.figure(figsize=(11,11))
plt.imshow(mat[:, :])
#plt.clim(0, 0.5)


from sklearn.manifold import MDS

matMDS = MDS(n_components=2, dissimilarity='precomputed')
matMDS.fit(mat)
pts = matMDS.embedding_
#pts = np.log(pts)

plt.plot(pts[:, 0], pts[:, 1], 'b.')

mat = np.zeros((len(stimtrials), len(stimtrials)))
iu1 = np.triu_indices(len(stimtrials))
for ind, x in enumerate(divsave):
    mat[iu1[0][ind], iu1[1][ind]] = x[2]
mat = mat + mat.T

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.set_cmap('hot')
plt.figure(figsize=(11,11))
plt.imshow(1.0/mat, interpolation='none')
plt.grid(True)
plt.xticks(np.arange(0, len(stimtrials), ntrials))
plt.yticks(np.arange(0, len(stimtrials),ntrials))
#plt.clim(vmin=3)

plt.figure(figsize=(11,11))
plt.imshow(mat[:, :])
plt.clim(0, 0.5)

plt.figure(figsize=(11,11))
normmat = mat/np.max(np.max(mat))
plt.imshow(1.0/normmat)
plt.grid(True)
plt.xticks(np.arange(0, 100, 10))
plt.yticks(np.arange(0,100, 10))

scg1 = sc.simplicialChainGroups([(1,2,3)])
scg2 = sc.simplicialChainGroups([(4,5,6)])
scgTot = sc.simplexUnion(scg1, scg2)
(adjTot, basisTot) = sc.stimSpaceGraph(scgTot, sc.boundaryOperatorMatrix(scgTot))
(adj1, basis1) = sc.stimSpaceGraph(scg1, sc.boundaryOperatorMatrix(scg1))
(adj2, basis2) = sc.stimSpaceGraph(scg2, sc.boundaryOperatorMatrix(scg2))
        
EgraphTot = sc.simplicialChainGroups(sc.adjacency2maxsimp(adjTot, basisTot))
Egraph1 = sc.simplicialChainGroups(sc.adjacency2maxsimp(adj1, basis1))
Egraph2 = sc.simplicialChainGroups(sc.adjacency2maxsimp(adj2, basis2))
D1 = sc.maskedBoundaryOperatorMatrix(EgraphTot, Egraph1)
D2 = sc.maskedBoundaryOperatorMatrix(EgraphTot, Egraph2)
print(D1)
print(D2)

s1 = sc.simplicialChainGroups([(1,2,3)])
s2 = sc.simplicialChainGroups([(4,5,6)])
stimtrials = []

ntrials = 5
for ind in range(ntrials):
    n = np.random.rand(15, 200)
    n = (n > 0.9).astype(int)
    maxSimpList = sorted(sc.binarytomaxsimplex(n, rDup=True))
    E = sc.simplicialChainGroups(maxSimpList)
    stimtrials.append(E)

beta = 1
divsave = []
d = 0
for ind in range(len(stimtrials)):
    print(ind)
    for ind2 in range(ind, len(stimtrials)):
        
        #stim1 = stimtrials[ind][0]
        #t1 = stimtrials[ind][1]
        #stim2 = stimtrials[ind2][0]
        #t2 = stimtrials[ind2][1]
        #print(stim1, t1, stim2, t2)
        #scg1 = scg[stim1][t1]
        #scg2 = scg[stim2][t2]
        stim1 = ind
        stim2 = ind2
        t1 = stim1
        t2 = stim2
        scg1 = stimtrials[ind]
        scg2 = stimtrials[ind2]
        
        scgTot = sc.simplexUnion(scg1, scg2)
        (adjTot, basisTot) = sc.stimSpaceGraph(scgTot, sc.boundaryOperatorMatrix(scgTot))
        (adj1, basis1) = sc.stimSpaceGraph(scg1, sc.boundaryOperatorMatrix(scg1))
        (adj2, basis2) = sc.stimSpaceGraph(scg2, sc.boundaryOperatorMatrix(scg2))
        
        EgraphTot = sc.simplicialChainGroups(sc.adjacency2maxsimp(adjTot, basisTot))
        Egraph1 = sc.simplicialChainGroups(sc.adjacency2maxsimp(adj1, basis1))
        Egraph2 = sc.simplicialChainGroups(sc.adjacency2maxsimp(adj2, basis2))
        D1 = sc.maskedBoundaryOperatorMatrix(EgraphTot, Egraph1)
        D2 = sc.maskedBoundaryOperatorMatrix(EgraphTot, Egraph2)
        
        #rhos1 = sc.densityMatrices(D1, beta*np.ones(len(D1)))
        #rhos2 = sc.densityMatrices(D2, beta*np.ones(len(D2)))
        
        L1 = sc.laplacian(D1, d)
        L2 = sc.laplacian(D2, d)
        
        rho1 = sc.densityMatrix(L1, beta)
        rho2 = sc.densityMatrix(L2, beta)
        #div = sc.JSdivergences(rhos1, rhos2)
        div = sc.JSdivergence(rho1, rho2)
        divsave.append(((stim1, t1), (stim2, t2), div))
 


mat = np.zeros((len(stimtrials), len(stimtrials)))
iu1 = np.triu_indices(len(stimtrials))
for ind, x in enumerate(divsave):
    mat[iu1[0][ind], iu1[1][ind]] = x[2]
mat = mat + mat.T

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.set_cmap('hot')
plt.figure(figsize=(11,11))
plt.imshow(1.0/mat, interpolation='none')
plt.grid(True)
plt.xticks(np.arange(0, len(stimtrials), ntrials))
plt.yticks(np.arange(0, len(stimtrials),ntrials))
#plt.clim(vmin=3)

plt.figure(figsize=(11,11))
plt.imshow(mat[:, :], interpolation='none')
plt.clim(0, 0.5)

ds = mat[np.triu_indices(len(stimtrials))]

dontcare = plt.hist(ds, bins=100)

from sklearn.manifold import MDS

matMDS = MDS(n_components=2, dissimilarity='precomputed')
matMDS.fit(mat)
pts = matMDS.embedding_
#pts = np.log(pts)

plt.plot(pts[:, 0], pts[:, 1], 'b.')

stimSimplexes = []
for stim in scg.keys():
    stimTotalSimplex = [[]]
    stimData = scg[stim]
    for trial in stimData.keys():
        #print((stim, trial))
        trialData = stimData[trial]
        stimTotalSimplex = sc.simplexUnion(stimTotalSimplex, trialData)
    stimSimplexes.append((stim, stimTotalSimplex))

beta = 0.25
divsave = []
d = 0
for ind in range(len(stimSimplexes)):
    print(ind)
    for ind2 in range(ind, len(stimSimplexes)):
        
        stim1 = stimSimplexes[ind][0]
        stim2 = stimSimplexes[ind2][0]
        scg1 = stimSimplexes[ind][1]
        scg2 = stimSimplexes[ind2][1]
        scgTot = sc.simplexUnion(scg1, scg2)
        D1 = sc.maskedBoundaryOperatorMatrix(scgTot, scg1)
        D2 = sc.maskedBoundaryOperatorMatrix(scgTot, scg2)
        
        #rhos1 = sc.densityMatrices(D1, beta*np.ones(len(D1)))
        #rhos2 = sc.densityMatrices(D2, beta*np.ones(len(D2)))
        
        L1 = sc.laplacian(D1, d)
        L2 = sc.laplacian(D2, d)
        
        rho1 = sc.densityMatrix(L1, beta)
        rho2 = sc.densityMatrix(L2, beta)
        #div = sc.JSdivergences(rhos1, rhos2)
        div = sc.JSdivergence(rho1, rho2)
        divsave.append(((stim1, stim2), div))

mat = np.zeros((len(stimSimplexes), len(stimSimplexes)))
iu1 = np.triu_indices(len(stimSimplexes))
for ind, x in enumerate(divsave):

    mat[iu1[0][ind], iu1[1][ind]] = x[1]

mat = mat + mat.T

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.set_cmap('hot')
plt.figure(figsize=(11,11))
plt.imshow(1.0/mat, interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(0, len(stimSimplexes), 1))
plt.yticks(np.arange(0, len(stimSimplexes),1))
plt.savefig('/Users/brad/B1235_P02S01_0LaplacianComparisonAllTrialsUnion.pdf', format='pdf')
#plt.clim(vmin=3)

for ind, stim in enumerate(scg.keys()):
    print((ind, stim))

beta = 0.25
divsave = []
d = 1
for ind in range(len(stimtrials)):
    print(ind)
    for ind2 in range(ind, len(stimtrials)):
        
        stim1 = stimtrials[ind][0]
        t1 = stimtrials[ind][1]
        stim2 = stimtrials[ind2][0]
        t2 = stimtrials[ind2][1]
        #print(stim1, t1, stim2, t2)
        scg1 = scg[stim1][t1]
        scg2 = scg[stim2][t2]
        
        scgTot = sc.simplexUnion(scg1, scg2)
        mE = sc.maxEnt(scgTot, d)
        
        D1 = sc.maskedBoundaryOperatorMatrix(scgTot, scg1)
        D2 = sc.maskedBoundaryOperatorMatrix(scgTot, scg2)
        
        #rhos1 = sc.densityMatrices(D1, beta*np.ones(len(D1)))
        #rhos2 = sc.densityMatrices(D2, beta*np.ones(len(D2)))
        divtot = 0
        for d in range(3):
            
            try:
                L1 = sc.laplacian(D1, d)
                L2 = sc.laplacian(D2, d)
        
                rho1 = sc.densityMatrix(L1, beta)
                rho2 = sc.densityMatrix(L2, beta)
        #div = sc.JSdivergences(rhos1, rhos2)
                div = sc.JSdivergence(rho1, rho2)
                divtot = divtot + div
            except:
                divtot = divtot+0
            
        divsave.append(((stim1, t1), (stim2, t2), divtot))

d = 2

mat = np.zeros((len(stimtrials), len(stimtrials)))
iu1 = np.triu_indices(len(stimtrials))
for ind, x in enumerate(divsave):

    mat[iu1[0][ind], iu1[1][ind]] = x[2]

mat = mat + mat.T

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.set_cmap('hot')
plt.figure(figsize=(11,11))
plt.imshow(1.0/mat, interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(0, len(stimtrials), 5))
plt.yticks(np.arange(0, len(stimtrials),5))
#plt.savefig('/Users/brad/B1235_P02S01_1LaplacianComparison.pdf', format='pdf')
#plt.clim(vmin=3)

plt.figure(figsize=(11,11))
plt.imshow(mat[:, :])
plt.clim(0, 0.5)

plt.figure(figsize=(11,11))
normmat = mat/np.max(np.max(mat))
plt.imshow(1.0/normmat)
plt.grid(True)
plt.xticks(np.arange(0, 100, 10))
plt.yticks(np.arange(0,100, 10))

binmat = 1.0*(np.random.rand(10, 20) > 0.8)

clus = np.zeros((10, 1))
clus= np.array(range(10))+8


t = np.multiply(clus, binmat)

np.nonzero(binmat)

t[:, 2]

[tuple(clus[list(np.nonzero(t)[0])]) for t in binmat.T]

import neuraltda.simpComp as sc
reload(sc)

x = sc.binarytomaxsimplex(binmat, rDup=True)

[len(t) for t in x]

sc.simplicialChainGroups(x)

def binarytomaxsimplex(binMat, rDup=False, clus=None):
    '''
    Takes a binary matrix and computes maximal simplices according to CI 2008

    Parameters
    ----------
    binMat : numpy array
        An Ncells x Nwindows array
    '''
    if rDup:
        lexInd = np.lexsort(binMat)
        binMat = binMat[:, lexInd]
        diff = np.diff(binMat, axis=1)
        ui = np.ones(len(binMat.T), 'bool')
        ui[1:] = (diff != 0).any(axis=0)
        binMat = binMat[:, ui]

    Ncells, Nwin = np.shape(binMat)
    if not clus:
        clus = np.arange(Ncells)
    MaxSimps = []
    MaxSimps = [tuple(clus[list(np.nonzero(t)[0])]) for t in binMat.T]
    #for win in range(Nwin):
    #    if binMat[:, win].any():
    #        verts = np.arange(Ncells)[binMat[:, win] == 1]
    #        verts = np.sort(verts)
    #        MaxSimps.append(tuple(verts))
    return MaxSimps

def oldbinarytomaxsimplex(binMat, rDup=False, clus=None):
    '''
    Takes a binary matrix and computes maximal simplices according to CI 2008

    Parameters
    ----------
    binMat : numpy array
        An Ncells x Nwindows array
    '''
    if rDup:
        lexInd = np.lexsort(binMat)
        binMat = binMat[:, lexInd]
        diff = np.diff(binMat, axis=1)
        ui = np.ones(len(binMat.T), 'bool')
        ui[1:] = (diff != 0).any(axis=0)
        binMat = binMat[:, ui]

    Ncells, Nwin = np.shape(binMat)
    if not clus:
        clus = np.arange(Ncells)
    MaxSimps = []
    #MaxSimps = [tuple(clus[list(np.nonzero(t)[0])]) for t in binMat.T]
    for win in range(Nwin):
        if binMat[:, win].any():
            verts = np.arange(Ncells)[binMat[:, win] == 1]
            verts = np.sort(verts)
            MaxSimps.append(tuple(verts))
    return MaxSimps

binmat = 1.0*np.random.randn(87, 400) > 0.8
get_ipython().magic('timeit x=oldbinarytomaxsimplex(binmat, rDup=True)')
get_ipython().magic('timeit x=binarytomaxsimplex(binmat, rDup=True)')

import profile

reload(sa)
reload(sc)
poptens = 1.0*np.random.randn(10, 20, 57) > 0.8
profile.run('sa.computeChainGroup(poptens, 0, 1)')
E = sa.computeChainGroup(poptens, 0, 1)

q = [1,2,3,4]
q.extend(q[:4-2])

q = tuple(range(26))

def is_sub_simplex(E, q):
    
    k = len(q)
    return (q in E[k])

is_sub_simplex(E, (1,2,5,9))

(1,2,3) in (1,2,3,4)

(1,2,5) < (1,2,3,4)

q = [(1,2,3), (4,5,6,7), (4,5,6), (1,2,3,4), (1,2,3,4,5,6,18), (1,2,3,4,5,6,7,9,11)]
qs = sorted(q, key=len)
print(qs)

newq  = qs
r = 1
while(r < len(newq)):
    print(r)
    newq = [t for t in newq if not set(t) < set(newq[-r])]
    print(newq)
    r += 1

newq

import h5py
thresh = 13.0
with h5py.File('/mnt/cube/btheilma/emily/B604/klusta/phy103116/Pen01_Lft_AP2300_ML1250__Site15_Z2780__B604_cat_P01_S15_1/binned_data/win-10.0_dtovr-5.0_cg-Good-MUA_seg-0--2500.0-Target/20170418T173600Z-10.0-5.0.binned', 'r') as f:
    print(f.keys())
    probstim = f['T3S3D40']
    poptens = probstim['pop_tens']
    (ncell, nwin, ntrial) = np.shape(poptens)
    print(ntrial)
    
    for trial in [37, 29]:
        popmat = poptens[:, :, trial]
        popmatbinary = sc.binnedtobinary(popmat, thresh)
        maxsimps = sc.binarytomaxsimplex(popmatbinary, rDup=True)
        maxsimps = sorted(maxsimps, key=len)
        newms = maxsimps
        r = 1
        while(r < len(newms)):
            newms = [t for t in newms if not set(t) < set(newms[-r])]
            r+=1
        print([len(s) for s in newms])

reload(sc)
t = sc.simplicialChainGroups([(1,)])

t

d = sc.boundaryOperatorMatrix(t)

d

l = sc.laplacian(d, 10)
l.shape

reload(sa)
scgA = t
scgB = t

sa.compute_JS_expanded(scgA, scgB, 1, 1)

a = sc.simplicialChainGroups([(1,2,3,4), (4,5,6), (5,6), (6,7,8,9)])
b = sc.simplicialChainGroups([(1,4,5), (5,6,7,8), (2,3), (1,2,3)])

reload(sa)

betas = np.linspace(-1, 1, 100)
ds = []
for beta in betas:
    ds.append(sa.compute_JS_expanded_negativeL(a, b, 1, beta))
    
plt.plot(betas, ds)

D = sc.boundaryOperatorMatrix(a)

L = sc.laplacian(D, 1)
np.linalg.eig(L)
beta = 5.0
rho = sc.densityMatrix(L, beta)

rhom = sc.densityMatrix(-L, beta)
ent1 = sc.Entropy(rho)
ent2 = sc.Entropy(rhom)
print(np.linalg.eig(rhom)[0])
print(np.linalg.eig(rho)[0])

np.linalg.eig(L)

np.linalg.eig(-L)



