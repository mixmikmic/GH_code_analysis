#Import packages
import numpy as np
#Import slow algorithm, ssvd_work,s from ssvd_original.py source code
from ssvd_original import ssvd_works 
get_ipython().magic('load_ext line_profiler')
#Load in genes data
X = np.loadtxt('data.txt')

get_ipython().magic('timeit -r1 -n1 [u,v,iters] = ssvd_works(X)')

get_ipython().magic('lprun -s -f ssvd -T ssvd_slow_results.txt ssvd_works(X)')
get_ipython().magic('cat ssvd_slow_results.txt')

#Import optimized algorithm, ssvd, from ssvd_fast.py source code file
from ssvd_fast import ssvd

get_ipython().magic('timeit -r1 -n1 [u,v,iters] = ssvd(X)')

get_ipython().magic('lprun -s -f ssvd -T ssvd_results.txt ssvd(X)')
get_ipython().magic('cat ssvd_results.txt')

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from sklearn.datasets import make_checkerboard
from sklearn.datasets import samples_generator as sg
from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.metrics import consensus_score

#Rank-one biclustering of genes data with SSVD
[u1,v1,iters] = ssvd(X)

Xstar1 = np.outer(u1, v1.T)
X = X-Xstar1
v1sort = np.sort(v1)[::-1] #sort descending
u1sort = np.sort(u1)[::-1] #sort descending
x = np.outer(u1sort, v1sort.T)
x = x/np.max(np.abs(X))
plt.matshow(x.T, cmap=plt.cm.coolwarm_r, aspect='auto');
plt.colorbar()
plt.title('SSVD Algorithm on Genes Data Set', y=1.15)
pass

#Rank-one biclustering of genes data with Spectral Biclustering
model = SpectralBiclustering(n_clusters=4, method='log',
                             random_state=0)
model.fit(X)
fit_data = X[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]
fit_data = np.sort(fit_data)
plt.matshow(fit_data.T, cmap=plt.cm.coolwarm_r, aspect='auto');
plt.colorbar()
plt.title('SKLearn Algorithm on Genes Data Set', y=1.15)
pass

#Make simulated data with a checkerboard structure
n_clusters = (2, 2)
data, rows, columns = make_checkerboard(
    shape=(12000, 60), n_clusters=n_clusters, noise=4,
    shuffle=False, random_state=0)

#Implement SSVD algorithm on non-sparse synthesized data
[u1,v1,iters] = ssvd(data)

Xstar1 = np.outer(u1, v1.T)
X = data-Xstar1
xmax = np.max(np.abs(X))
v1sort = np.sort(v1)[::-1] #sort descending
u1sort = np.sort(u1)[::-1] #sort descending
x = np.outer(u1sort, v1sort.T)
xfake = x/xmax
#Plots
plt.matshow(data.T, cmap=plt.cm.coolwarm_r, aspect='auto')
plt.title("Original Synthesized Data Set", y=1.15)

#Shuffled data
datashuff, row_idx, col_idx = sg._shuffle(data, random_state=0)
plt.matshow(datashuff.T, cmap=plt.cm.coolwarm_r, aspect='auto')
plt.title("Shuffled Synthesized Data Set", y=1.15)

plt.matshow(xfake.T, cmap=plt.cm.coolwarm_r, aspect='auto')
plt.title("After SSVD Biclustering", y=1.15)
pass

plt.matshow(data.T, cmap=plt.cm.coolwarm_r, aspect='auto')
plt.title("Original Synthesized Data Set", y=1.15)

#Shuffled data
datashuff, row_idx, col_idx = sg._shuffle(data, random_state=0)
plt.matshow(datashuff.T, cmap=plt.cm.coolwarm_r, aspect='auto')
plt.title("Shuffled Synthesized Data Set", y=1.15)

#Spectral Biclustering on synthesized data
model = SpectralBiclustering(n_clusters=n_clusters, method='log',
                             random_state=0)
model.fit(datashuff)

fit_data = datashuff[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

plt.matshow(fit_data.T, cmap=plt.cm.coolwarm_r, aspect='auto')
plt.title("After Sklearn Biclustering", y=1.15)
pass

#u is a unit vector of length 100
#with ui = 1/√50 for i = 1,...,50, and ui = 0 otherwise
ufirst = (1/np.sqrt(50))*np.ones((50,1))
ulast = np.zeros((50,1))
u = np.hstack((ufirst.flatten(), ulast.flatten()))

#v is a unit vector of length 50
#with vj = 1/5 for j = 1,...,25, and vj = 0 otherwise.
vfirst = (1/5)*np.ones((25,1))
vlast = np.zeros((25,1))
v = np.hstack((vfirst.flatten(), vlast.flatten()))

#X* = suvT is a rank one matrix with uniform nonzero entries
#s is set to 30
s = 30
Xstar = s*np.outer(u, v.T)
noise = np.random.standard_normal(size=Xstar.shape)

#The input matrix X is the summation of the true signal, Xstar
#and noise from the standard normal distribution
X = Xstar + noise

#Plots
plt.matshow(Xstar.T, cmap=plt.cm.coolwarm_r, aspect='auto')
plt.title("Ground Truth Sparse Synthesized Data Set", y=1.15)

plt.matshow(X.T, cmap=plt.cm.coolwarm_r, aspect='auto')
plt.title("Ground Truth Sparse Synthesized Data Set Plus Noise",
          y=1.15)
pass

#Shuffled data
datashuff, row_idx, col_idx = sg._shuffle(X, random_state=0)
plt.matshow(datashuff.T, cmap=plt.cm.coolwarm_r, aspect='auto')
plt.title("Shuffled Sparse Synthesized Data Set", y=1.15)

#Implement SSVD algorithm on synthesized sparse data
[u1,v1,iters] = ssvd(datashuff)
Xstar1 = np.outer(u1, v1.T)
Xbi = X-Xstar1
xmax = np.max(np.abs(Xbi))
v1sort = np.sort(v1)[::-1] #sort descending
u1sort = np.sort(u1)[::-1] #sort descending
x = np.outer(u1sort, v1sort.T)
Xbi = x/xmax

plt.matshow(Xbi.T, cmap=plt.cm.coolwarm_r, aspect='auto')
plt.title("After SSVD Biclustering", y=1.15)
plt.colorbar()
pass

#Plots
plt.matshow(Xstar.T, cmap=plt.cm.coolwarm_r, aspect='auto')
plt.title("Ground Truth Sparse Synthesized Data Set", y=1.15)

plt.matshow(X.T, cmap=plt.cm.coolwarm_r, aspect='auto')
plt.title("Ground Truth Sparse Synthesized Dataset Plus Noise",
          y=1.15)

datashuff, row_idx, col_idx = sg._shuffle(X, random_state=0)
plt.matshow(datashuff.T, cmap=plt.cm.coolwarm_r, aspect='auto')
plt.title("Shuffled Sparse Synthesized Dataset", y=1.15)

#Run Spectral Biclustering
model = SpectralBiclustering(n_clusters=2, method='log',
                             random_state=0)
model.fit(datashuff)

fit_data = datashuff[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

plt.matshow(fit_data.T, cmap=plt.cm.coolwarm_r, aspect='auto')
plt.title("After Sklearn Biclustering", y=1.15)
pass

