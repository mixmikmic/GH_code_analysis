import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from msmbuilder.example_datasets import AlanineDipeptide
trajs = AlanineDipeptide().get().trajectories

from msmbuilder.featurizer import AtomPairsFeaturizer
pairs = []
for i in range(22):
    for j in range(i):
        pairs.append((i,j))
apf = AtomPairsFeaturizer(pairs)
X = apf.fit_transform(trajs)

X[0].shape

from msmbuilder.decomposition import tICA

ticas = dict()
times = list(range(1,100)) + list(range(100,1000,10)) + list(range(1000,10000,1000))

for t in times:
    tica = tICA(lag_time=t)
    tica.fit(X)
    
    ticas[t] = tica

plt.plot(np.cumsum(tica.eigenvalues_))

plt.plot(tica.eigenvectors_[0])

# compare leading eigenvector

dots = np.zeros((len(times),len(times)))
cos = np.zeros((len(times),len(times)))
evec = 0

from scipy.spatial.distance import cosine

for i in range(len(times)):
    for j in range(len(times)):
        evec_i = ticas[times[i]].eigenvectors_[evec]
        evec_j = ticas[times[j]].eigenvectors_[evec]
        
        dots[i,j] = np.dot(evec_i,evec_j)
        cos[i,j] = cosine(evec_i,evec_j)

plt.imshow(dots,interpolation='none',cmap='Blues')
plt.colorbar()

plt.imshow(cos,interpolation='none',cmap='Blues')
plt.colorbar()

for t in times:
    plt.plot(ticas[t].eigenvectors_[0])

for t in times:
    plt.plot(ticas[t].eigenvectors_[1])

for t in times:
    plt.plot(ticas[t].eigenvectors_[2])

for i in range(10):
    plt.plot(times,[ticas[t].timescales_[i] for t in times])
    
plt.figure()
for i in range(10):
    plt.plot(times,[ticas[t].eigenvalues_[i] for t in times])

for i in range(10):
    plt.plot(times,[ticas[t].timescales_[i] for t in times])
plt.yscale('log')

for i in range(10):
    plt.plot(times,[ticas[t].timescales_[i] for t in times])
plt.yscale('log')
plt.xlim(0,100)
plt.ylim(0,400)

for i in range(50):
    plt.plot(times,[ticas[t].timescales_[i] for t in times])
plt.yscale('log')
plt.xlim(0,50)
plt.ylim(0,400)

plt.plot(times,[ticas[t].eigenvalues_[0] for t in times])



