import neuraltda.simpComp as sc
import neuraltda.topology2 as tp2
import neuraltda.spectralAnalysis as sa
from ephys import rasters
import pickle
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import seaborn as sns
sns.set_context('poster')
sns.set_style('white')
get_ipython().magic('matplotlib inline')
reload(sa)
reload(sc)

# generate a random graph:
plink = 0.09
N = 200
adj = np.random.rand(N, N)
adj = (adj + adj.T)/2.0
adjData = (adj < plink).astype(int)

def loss(p, beta):
    adj = np.random.rand(N, N)
    adj = (adj + adj.T)/2.0
    adj = (adj < p).astype(int)
    Esample = sc.simplicialChainGroups(sc.adjacency2maxsimp(adj, range(N)))
    Edata = sc.simplicialChainGroups(sc.adjacency2maxsimp(adjData, range(N)))
    Lsamp = sc.laplacian(sc.boundaryOperatorMatrix(Esample), 0)
    Ldata = sc.laplacian(sc.boundaryOperatorMatrix(Edata), 0)
    rsamp = sc.densityMatrix(Lsamp, beta)
    rdata = sc.densityMatrix(Ldata, beta)
    return sc.KLdivergence(rdata, rsamp)

kl = []
ps = np.linspace(0.025, 0.1, 25)
for p in ps:
    kl.append(loss(p, 0.25))
    
plt.plot(ps, kl)
plt.xlabel('Link Probability')
plt.ylabel('KL Divergence')
plt.title('MLE Model Fitting of Erdos Renyi')

def loss(p, beta):
    
    adj = np.random.rand(N, N)
    adj = (adj + adj.T)/2.0
    adj = (adj < p).astype(int)
    Esample = sc.simplicialChainGroups(sc.adjacency2maxsimp(adj, range(N)))
    Edata = sc.simplicialChainGroups(sc.adjacency2maxsimp(adjData, range(N)))
    Lsamp = sc.laplacian(sc.boundaryOperatorMatrix(Esample), 0)
    Ldata = sc.laplacian(sc.boundaryOperatorMatrix(Edata), 0)
    rsamp = sc.densityMatrix(Lsamp, beta)
    rdata = sc.densityMatrix(Ldata, beta)
    return sc.KLdivergence(rdata, rsamp)

kl = []
kl_p = []
nsamples = 10
ps = np.linspace(0.025, 0.1, 50)
for p in ps:
    print(p)
    for n in range(nsamples):
        kl_p.append(loss(p, 0.25))
    kl.append(np.mean(kl_p))
    
plt.plot(ps, kl)
plt.xlabel('Link Probability')
plt.ylabel('KL Divergence')
plt.title('MLE Model Fitting of Erdos Renyi')

beta = 0.1
c = 0.25
Edata = sc.simplicialChainGroups(sc.adjacency2maxsimp(adjData, range(N)))
Ldata = sc.laplacian(sc.boundaryOperatorMatrix(Edata), 0)
rdata = sc.densityMatrix(Ldata, beta)
entData = sc.Entropy(rdata, beta)

#determine beta
f = lambda b: -sc.Entropy(rdata, b) / N - c*b
betastar = brentq(f, 0.001, 1.5)
betastar

def loss(p, beta):
    N = 200
    adj = np.random.rand(N, N)
    adj = (adj + adj.T)/2.0
    adj = (adj < p).astype(int)
    Esample = sc.simplicialChainGroups(sc.adjacency2maxsimp(adj, range(N)))
    Lsamp = sc.laplacian(sc.boundaryOperatorMatrix(Esample), 0)
    rsamp = sc.densityMatrix(Lsamp, beta)
    return sc.KLdivergence(rdata, rsamp)

kl = []
kl_p = []
nsamples = 10
ps = np.linspace(0.025, 0.1, 50)
for p in ps:
    print(p)
    for n in range(nsamples):
        kl_p.append(loss(p, betastar))
    kl.append(np.mean(kl_p))
    
plt.plot(ps, kl)
plt.xlabel('Link Probability')
plt.ylabel('KL Divergence')
plt.title('MLE Model Fitting of Erdos Renyi')

# Generate binary matrix with given probabilities for each "cell"
ncells = 20
nwin = 1000
a = 0.02
b = 0.15
probs = (a*np.ones((ncells, 1)))
nsamples = 1
samples = np.random.rand(ncells, nwin, nsamples)
probmat = np.tile(probs,  (1, nwin))[:, :, np.newaxis]
probmat = np.tile(probmat, (1, 1, nsamples))
binMatsamples = np.greater(probmat, samples).astype(int)

# Compute SCG for each sample
SCGs = []
for ind in range(nsamples):
    msimps = sc.binarytomaxsimplex(binMat=binMatsamples[:, :, ind], rDup=True)
    E = sc.simplicialChainGroups(msimps)
    SCGs.append(E)
Edata = SCGs[0]

nsamples2 = 10
def loss(a, beta):
    # take a set of probabilities, generate random configurations, measure KL divergence to data, report loss
    probs = (a*np.ones((ncells, 1)))
    KLsave=[]
    samples = np.random.rand(ncells, nwin, nsamples2)
    probmat = np.tile(probs,  (1, nwin))[:, :, np.newaxis]
    probmat = np.tile(probmat, (1, 1, nsamples2))
    binMatsamples = np.greater(probmat, samples).astype(int)
    SCGs = []
    for ind in range(nsamples2):
        msimps = sc.binarytomaxsimplex(binMat=binMatsamples[:, :, ind], rDup=True)
        Emodel = sc.simplicialChainGroups(msimps)

        Ecombined = sc.simplexUnion(Emodel, Edata)
        
        Lsamp = sc.laplacian(sc.boundaryOperatorMatrix(Emodel), 1)
        Ldata = sc.laplacian(sc.boundaryOperatorMatrix(Edata), 1)
        rsamp = sc.densityMatrix(Lsamp, beta)
        
        rdata = sc.densityMatrix(Ldata, beta)
        
        KLsave.append(sc.KLdivergence(rdata, rsamp))
    return np.mean(KLsave)

KL=[]
X = np.linspace(0.01, 0.1, 30)

for x in X:
    KL.append(loss(x, 0.15))

plt.plot(X, KL)
plt.xlabel('Firing Probability')
plt.ylabel('KL Divergence')
plt.title('Random Spiking Population Model Fit')

ez = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
ey = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
ex = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])

plink = 0.5
N = 500
adjMats= []
lapMats = []
for n in range(3):
    adj = np.random.rand(N, N)
    w = np.random.randn(N, N)
    adj = (adj + adj.T)/2.0
    w = (w+w.T)/2.0
    adjData = (adj < plink).astype(int)
    adjData = np.multiply(adjData, w)
    adjMats.append(adjData)
    lapMats.append(sc.graphLaplacian(adjData))

def mat(i, j):
    return adjMats[0][i,j]*ex + adjMats[1][i,j]*ey + adjMats[2][i,j]*ez
def lap(i, j):
    return lapMats[0][i,j]*ex + lapMats[1][i,j]*ey + lapMats[2][i,j]*ez

nsamples2 = 25
d = 1
def loss(a, beta):
    # take a set of probabilities, generate random configurations, measure KL divergence to data, report loss
    probs = (a*np.ones((ncells, 1)))
    KLsave=[]
    samples = np.random.rand(ncells, nwin, nsamples2)
    probmat = np.tile(probs,  (1, nwin))[:, :, np.newaxis]
    probmat = np.tile(probmat, (1, 1, nsamples2))
    binMatsamples = np.greater(probmat, samples).astype(int)
    SCGs = []
    for ind in range(nsamples2):
        msimps = sc.binarytomaxsimplex(binMat=binMatsamples[:, :, ind], rDup=True)
        Emodel = sc.simplicialChainGroups(msimps)

        Ecombined = sc.simplexUnion(Emodel, Edata)
        Ddata = sc.maskedBoundaryOperatorMatrix(Ecombined, Edata)
        Dsamp = sc.maskedBoundaryOperatorMatrix(Ecombined, Emodel)
        
        Lsamp = sc.laplacian(Dsamp, d)
        Ldata = sc.laplacian(Ddata, d)
        rsamp = sc.densityMatrix(Lsamp, beta)
        
        rdata = sc.densityMatrix(Ldata, beta)
        
        KLsave.append(sc.KLdivergence(rdata, rsamp))
    m = np.mean(KLsave)
    std = np.std(KLsave)
    stderr = std / np.sqrt(nsamples2)
    return (m, stderr)

def loss_new(a, beta):
    # take a set of probabilities, generate random configurations, measure KL divergence to data, report loss
    probs = (a*np.ones((ncells, 1)))
    KLsave=[]
    samples = np.random.rand(ncells, nwin, nsamples2)
    probmat = np.tile(probs,  (1, nwin))[:, :, np.newaxis]
    probmat = np.tile(probmat, (1, 1, nsamples2))
    binMatsamples = np.greater(probmat, samples).astype(int)
    SCGs = []
    for ind in range(nsamples2):
        msimps = sc.binarytomaxsimplex(binMat=binMatsamples[:, :, ind], rDup=True)
        Emodel = sc.simplicialChainGroups(msimps)

        Ddata = sc.boundaryOperatorMatrix(Edata)
        Dsamp = sc.boundaryOperatorMatrix(Emodel)
        Lsamp = sc.laplacian(Dsamp, d)
        Ldata = sc.laplacian(Ddata, d)
        if (np.size(Lsamp) > np.size(Ldata)):
            (Ldata, Lsamp) = sc.reconcile_laplacians(Ldata, Lsamp)
        else:
            (Lsamp, Ldata) = sc.reconcile_laplacians(Lsamp, Ldata)
        rsamp = sc.densityMatrix(Lsamp, beta)
        
        rdata = sc.densityMatrix(Ldata, beta)
        
        KLsave.append(sc.KLdivergence(rdata, rsamp))
    m = np.mean(KLsave)
    std = np.std(KLsave)
    stderr = std / np.sqrt(nsamples2)
    return (m, stderr)

# Generate binary matrix with given probabilities for each "cell"
ncells = 200
nwin = 1000
a = 0.001
probs = (a*np.ones((ncells, 1)))
nsamples = 1
samples = np.random.rand(ncells, nwin, nsamples)
probmat = np.tile(probs,  (1, nwin))[:, :, np.newaxis]
probmat = np.tile(probmat, (1, 1, nsamples))
binMatsamples = np.greater(probmat, samples).astype(int)

# Compute SCG for each sample
SCGs = []
for ind in range(nsamples):
    msimps = sc.binarytomaxsimplex(binMat=binMatsamples[:, :, ind], rDup=True)
    E = sc.simplicialChainGroups(msimps)
    SCGs.append(E)
Edata = SCGs[0]


est_save = []
ntrials = 1
X = np.linspace(0.0005, 0.007, 25)

for t in range(ntrials):
    print(t)
    KL=[]
    KLerr = []
    for x in X:
        print(x)
        (m, stderr) = loss_new(x, 0.15)
        KL.append(m)
        KLerr.append(stderr)
        
    
    index_min = np.argmin(KL)
    est_save.append(X[index_min])

print(np.mean(est_save))
np.std(est_save)

sns.set_style('white')
plt.figure(figsize=(11,8))
plt.errorbar(X, KL, yerr=KLerr, linewidth=3, capsize=6, capthick=2, fmt='b')
plt.ylim(0, 0.46)
plt.plot(a*np.ones((20, 1)), np.linspace(0, 0.45, 20), 'k--')
plt.xlabel('Firing Probability')
plt.ylabel('KL Divergence')
plt.title('Poisson Spiking Population Model Fit')
#plt.savefig('/Users/brad/PoissonFit-{}.pdf'.format(a), format='pdf')

sns.set_style('white')
plt.figure(figsize=(11,8))
plt.errorbar(X, KL, yerr=KLerr, linewidth=3, capsize=6, capthick=2, fmt='b')
plt.ylim(0, 0.46)
plt.plot(a*np.ones((20, 1)), np.linspace(0, 0.45, 20), 'k--')
plt.xlabel('Firing Probability')
plt.ylabel('KL Divergence')
plt.title('Poisson Spiking Population Model Fit')
#plt.savefig('/Users/brad/PoissonFit-{}.pdf'.format(a), format='pdf')

sns.set_style('white')
plt.figure(figsize=(11,8))
plt.errorbar(X, KL, yerr=KLerr, linewidth=3, capsize=6, capthick=2, fmt='b')
plt.ylim(0, 0.46)
plt.plot(a*np.ones((20, 1)), np.linspace(0, 0.45, 20), 'k--')
plt.xlabel('Firing Probability')
plt.ylabel('KL Divergence')
plt.title('Poisson Spiking Population Model Fit')
#plt.savefig('/Users/brad/PoissonFit-{}.pdf'.format(a), format='pdf')

# make rasters of the original and best fit.
spikesorig = np.squeeze(binMatsamples)
def convert_to_raster_data(binmat):
    (ncells, nt) = np.shape(binmat)
    raster_data = []
    sp = np.transpose(np.nonzero(binmat))
    for cell in range(ncells):
        raster_data.append(sp[sp[:, 0]==cell][:, 1])
    return raster_data
rasterdatorig = convert_to_raster_data(spikesorig)
plt.figure()
rasters.do_raster(rasterdatorig, [-100, 1100], [0, 1000])

a = 0.015
probs = (a*np.ones((ncells, 1)))
KLsave=[]
nsamples2 = 5
for n in range(5):
    samples = np.random.rand(ncells, nwin, nsamples2)
    probmat = np.tile(probs,  (1, nwin))[:, :, np.newaxis]
    probmat = np.tile(probmat, (1, 1, nsamples2))
    binMatsamples2 = np.greater(probmat, samples).astype(int)
    newrast = convert_to_raster_data(np.squeeze(binMatsamples2[:, :, n]))
    plt.figure(figsize = (11, 8))
    rasters.do_raster(rasterdatorig, [-100, 1100], [0, 1000])
    rasters.do_raster(newrast, [-100, 1100], [0, 1000], spike_color='b--')
    plt.savefig('/home/brad/SpikeFitting_{}.pdf'.format(n), format='pdf')

#mean squared difference
nsamples2 = 25
def lossMSE(a):
    # take a set of probabilities, generate random configurations, measure KL divergence to data, report loss
    probs = (a*np.ones((ncells, 1)))
    MSEsave=[]
    samples = np.random.rand(ncells, nwin, nsamples2)
    probmat = np.tile(probs,  (1, nwin))[:, :, np.newaxis]
    probmat = np.tile(probmat, (1, 1, nsamples2))
    binMatsamples2 = np.greater(probmat, samples).astype(int)
    for ind in range(nsamples2):
        mse = np.sum(np.square(binMatsamples2[:, :, ind] - spikesorig))
        MSEsave.append(mse)
    m = np.mean(MSEsave)
    std = np.std(MSEsave)
    stderr = std / np.sqrt(nsamples2)
    return (m, stderr)

MSE=[]
MSEerr = []
X = np.linspace(0.01, 0.1, 50)

for x in X:
    
    (m, stderr) = lossMSE(x)
    MSE.append(m)
    MSEerr.append(stderr)
    #$print(x)
    
sns.set_style('white')
plt.figure(figsize=(11,8))
plt.errorbar(X, MSE, yerr=MSEerr, linewidth=3, capsize=6, capthick=2, fmt='b')
plt.ylim(0, max(MSE)+10)
plt.plot(a*np.ones((20, 1)), np.linspace(0, 0.45, 20), 'k--')
plt.xlabel('Firing Probability')
plt.ylabel('MSE')
plt.title('Poisson Spiking Population Model Fit')

MSE

# Generate binary matrix with given probabilities for each "cell"
ncells = 20
nwin = 1000
a = 0.05
b = 0.05
probs = (a*np.ones((ncells, 1)))
nsamples = 2
samples = np.random.rand(ncells, nwin, nsamples)
probmat = np.tile(probs,  (1, nwin))[:, :, np.newaxis]
probmat = np.tile(probmat, (1, 1, nsamples))
binMatsamples = np.greater(probmat, samples).astype(int)

# Compute SCG for each sample
SCGs = []
for ind in range(nsamples):
    msimps = sc.binarytomaxsimplex(binMat=binMatsamples[:, :, ind], rDup=True)
    E = sc.simplicialChainGroups(msimps)
    SCGs.append(E)
Edata = SCGs[0]
Emodel = SCGs[1]


Ecombined = sc.simplexUnion(Emodel, Edata)
Ddata = sc.maskedBoundaryOperatorMatrix(Ecombined, Edata)
Dsamp = sc.maskedBoundaryOperatorMatrix(Ecombined, Emodel)
        
Lsamp = sc.laplacian(Dsamp, 0)
Ldata = sc.laplacian(Ddata, 0)
rsamp = sc.densityMatrix(Lsamp, beta)
        
rdata = sc.densityMatrix(Ldata, beta)

# Generate binary matrix with given probabilities for each "cell"
ncells = 10
nwin = 1000
a = 0.02
b = 0.02
probs = np.vstack((a*np.ones((ncells, 1)), b*np.ones((ncells, 1))))
nsamples = 1
samples = np.random.rand(2*ncells, nwin, nsamples)
probmat = np.tile(probs,  (1, nwin))[:, :, np.newaxis]
probmat = np.tile(probmat, (1, 1, nsamples))
binMatsamples = np.greater(probmat, samples).astype(int)
np.random.shuffle(binMatsamples)



# Compute SCG for each sample
SCGs = []
for ind in range(nsamples):
    mat = binMatsamples[:, :, ind]
    mat = np.random.permutation(mat.T)
    mat = mat.T
    print(mat.shape)
    msimps = sc.binarytomaxsimplex(binMat=mat, rDup=True)
    E = sc.simplicialChainGroups(msimps)
    SCGs.append(E)
Edata = SCGs[0]

d = 1
nsamples2 = 10
def loss(a, b, beta):
    # take a set of probabilities, generate random configurations, measure KL divergence to data, report loss
    probs = np.vstack((a*np.ones((ncells, 1)), b*np.ones((ncells, 1))))
    KLsave=[]
    samples = np.random.rand(2*ncells, nwin, nsamples2)
    probmat = np.tile(probs,  (1, nwin))[:, :, np.newaxis]
    probmat = np.tile(probmat, (1, 1, nsamples2))
    binMatsamples = np.greater(probmat, samples).astype(int)
    SCGs = []
    for ind in range(nsamples2):
        msimps = sc.binarytomaxsimplex(binMat=binMatsamples[:, :, ind], rDup=True)
        Emodel = sc.simplicialChainGroups(msimps)

        Ecombined = sc.simplexUnion(Emodel, Edata)
        Ddata = sc.maskedBoundaryOperatorMatrix(Ecombined, Edata)
        Dsamp = sc.maskedBoundaryOperatorMatrix(Ecombined, Emodel)
        
        Lsamp = sc.laplacian(Dsamp, d)
        Ldata = sc.laplacian(Ddata, d)
        rsamp = sc.densityMatrix(Lsamp, beta)
        
        rdata = sc.densityMatrix(Ldata, beta)
        
        KLsave.append(sc.KLdivergence(rdata, rsamp))
    return np.mean(KLsave)

# Generate binary matrix with given probabilities for each "cell"
ncells = 10
nwin = 1000
a = 0.02
b = 0.02
probs = np.vstack((a*np.ones((ncells, 1)), b*np.ones((ncells, 1))))
nsamples = 1
samples = np.random.rand(2*ncells, nwin, nsamples)
probmat = np.tile(probs,  (1, nwin))[:, :, np.newaxis]
probmat = np.tile(probmat, (1, 1, nsamples))
binMatsamples = np.greater(probmat, samples).astype(int)
np.random.shuffle(binMatsamples)



# Compute SCG for each sample
SCGs = []
for ind in range(nsamples):
    mat = binMatsamples[:, :, ind]
    mat = np.random.permutation(mat.T)
    mat = mat.T
    print(mat.shape)
    msimps = sc.binarytomaxsimplex(binMat=mat, rDup=True)
    E = sc.simplicialChainGroups(msimps)
    SCGs.append(E)
Edata = SCGs[0]

KL=[]
nKL = 40
X = np.linspace(0.005, 0.05, nKL)

for x in X:
    print(x)
    for y in X:
        KL.append(loss(x,y, 0.15))
KL = np.reshape(KL, (nKL,nKL))

x, y = np.meshgrid(np.linspace(0.005, 0.05, 20), np.linspace(0.005, 0.05, 20))
levels = np.arange(0, 1, 0.025)
plt.contour(x, y, KL, levels=levels)

plt.figure(figsize=(11,11))

plt.plot([a], [b], 'w.')
plt.plot([b], [a], 'w.')
im = plt.imshow(KL, interpolation='bilinear', origin='lower',
                cmap='hot', extent=(0.005, 0.05, 0.005, 0.05))
levels = np.arange(0, 0.4, 0.005)
CS = plt.contour(KL, origin='lower', levels=levels, linewidths=2,extent=(0.005, 0.05, 0.005, 0.05), cmap='jet')
plt.title('Heterogeneous Poisson Population Model Fit')
plt.xlabel('Population A Rate')
plt.ylabel('Population B Rate')
#plt.savefig('/Users/brad/twopopulationpoisson.pdf', format='pdf')

# try to reproduce given structure

targetMaxSimps = [(1,2), (2,3), (3,4), (4,5), (1,5)]

initweights = np.random.randn(5,5)/np.sqrt(5)

## setup parameters and state variables
T       = 50                  # total time to simulate (msec)
dt      = 0.125               # simulation time step (msec)
time    = np.arange(0, T+dt, dt) # time array
t_rest  = 0                   # initial refractory time
ncells = 5

## LIF properties
Vm      = np.zeros((ncells, len(time)))    # potential (V) trace over time
Rm      = 1                   # resistance (kOhm)
Cm      = 10                  # capacitance (uF)
tau_m   = Rm*Cm               # time constant (msec)
tau_ref = 0                   # refractory period (msec)
Vth     = 1                   # spike threshold (V)
V_spike = 0                 # spike reset (V)

## Stimulus
I       = 1.5                 # input current (A)

def spike(W, T):
    ## iterate over each time step
    dt      = 0.125               # simulation time step (msec)
    time    = np.arange(0, T+dt, dt) # time array
    Vm      = np.zeros((ncells, len(time)))    # potential (V) trace over time
    binmat = np.zeros((ncells, len(time)))
    for i, t in enumerate(time):
        if t > 0:
            Vm[:, i] = Vm[:, i-1] + (-Vm[:, i-1] + np.random.randn(5, 1)) / tau_m * dt
        if (Vm[:, i] >= Vth).any():
            binmat[:, i] = (Vm[:, i] >= Vth).astype(int)
            Vm[:, i] = np.dot(W, binmat[:, i]*Rm)
            Vm[:, i] += np.multiply(V_spike, (Vm[:, i] >= Vth).astype(int))
    return binmat, Vm

ps, Vm = spike(initweights, 50)
Vm

np.random.randn(ncells, 1)

plt.plot(Vm[3, :])

# Generate binary matrix with given probabilities for each "cell"
ncells = 10
nwin = 1000
a = 0.02
b = 0.05
probs = np.vstack((a*np.ones((ncells, 1)), b*np.ones((ncells, 1))))
nsamples = 1
samples = np.random.rand(2*ncells, nwin, nsamples)
probmat = np.tile(probs,  (1, nwin))[:, :, np.newaxis]
probmat = np.tile(probmat, (1, 1, nsamples))
binMatsamples = np.greater(probmat, samples).astype(int)
np.random.shuffle(binMatsamples)



# Compute SCG for each sample
SCGs = []
for ind in range(nsamples):
    mat = binMatsamples[:, :, ind]
    mat = np.random.permutation(mat.T)
    mat = mat.T
    print(mat.shape)
    msimps = sc.binarytomaxsimplex(binMat=mat, rDup=True)
    E = sc.simplicialChainGroups(msimps)
    SCGs.append(E)
Edata = SCGs[0]

# Find beta of model in each dimension


d = 1
maxd = 3
nsamples2 = 5
def loss(a, b, beta):
    # take a set of probabilities, generate random configurations, measure KL divergence to data, report loss
    probs = np.vstack((a*np.ones((ncells, 1)), b*np.ones((ncells, 1))))
    KLsave=[]
    samples = np.random.rand(2*ncells, nwin, nsamples2)
    probmat = np.tile(probs,  (1, nwin))[:, :, np.newaxis]
    probmat = np.tile(probmat, (1, 1, nsamples2))
    binMatsamples = np.greater(probmat, samples).astype(int)
    SCGs = []
    for ind in range(nsamples2):
        msimps = sc.binarytomaxsimplex(binMat=binMatsamples[:, :, ind], rDup=True)
        Emodel = sc.simplicialChainGroups(msimps)

        Ecombined = sc.simplexUnion(Emodel, Edata)
        Ddata = sc.maskedBoundaryOperatorMatrix(Ecombined, Edata)
        Dsamp = sc.maskedBoundaryOperatorMatrix(Ecombined, Emodel)
        diver = 0
        for d in range(maxd):
            try:
                Lsamp = sc.laplacian(Dsamp, d)
                Ldata = sc.laplacian(Ddata, d)
                rsamp = sc.densityMatrix(Lsamp, beta)
                rdata = sc.densityMatrix(Ldata, beta)
                diver = diver + sc.KLdivergence(rdata, rsamp)
            except:
                diver = diver +0
            
        
        KLsave.append(diver)
    return np.mean(KLsave)

KL=[]
X = np.linspace(0.01, 0.1, 20)

for x in X:
    print(x)
    for y in X:
        KL.append(loss(x,y, 0.15))
KL = np.reshape(KL, (20,20))

plt.figure(figsize=(11,11))

plt.plot([a], [b], 'w.')
plt.plot([b], [a], 'w.')
im = plt.imshow(KL, interpolation='bilinear', origin='lower',
                cmap='hot', extent=(0.01, 0.1, 0.01, 0.1))
#levels = np.arange(0, 0.1, 0.01)
CS = plt.contour(KL, origin='lower', levels=levels, linewidths=2,extent=(0.01, 0.1, 0.01, 0.1), cmap='jet')
plt.title('Heterogeneous Poisson Population Model Fit')
plt.xlabel('Population A Rate')
plt.ylabel('Population B Rate')
#plt.savefig('/Users/brad/twopopulationpoisson.pdf', format='pdf')

x, y = np.meshgrid(np.linspace(0.01, 0.1, 20), np.linspace(0.01, 0.1, 20))
levels = np.arange(0, 1, 0.025)
plt.contour(x, y, KL, levels=levels)

ms1 = [(1,2,8), (4,8)]
ms2 = [(1,2,4,5)]



def maskedLaplacians(ms1, ms2,d):

    Ems1 = sc.simplicialChainGroups(ms1)
    Ems2 = sc.simplicialChainGroups(ms2)
    Ecombined = sc.simplexUnion(Ems1, Ems2)
    Dms1 = sc.maskedBoundaryOperatorMatrix(Ecombined, Ems1)
    Dms2 = sc.maskedBoundaryOperatorMatrix(Ecombined, Ems2)
    print(Dms1[d+1])
    print(Dms2[d+1])
    Lms1 = sc.laplacian(Dms1, d)
    Lms2 = sc.laplacian(Dms2, d)
    return (Lms1, Lms2)

rms1 = sc.densityMatrix(Lms1, beta)
rms2 = sc.densityMatrix(Lms2, beta)
print(rms1)
print(rms2)

sv = []
b = np.linspace(0, 2, 100)
for beta in b:
    rms1 = sc.densityMatrix(Lms1, beta)
    rms2 = sc.densityMatrix(Lms2, beta)
    sv.append(sc.JSdivergence(rms1, rms2))
plt.plot(b, sv)

print(np.dot(Dms1[2], Dms1[2].T))
np.dot(Dms1[1].T, Dms1[1])

Lms1

import pygraphviz as pg
import itertools

def build_graph_recursive(graph, cell_group, parent_name):

    cell_group_name = ''.join(cell_group)
    graph.add_node(cell_group_name)
    n_cells_in_group = len(cell_group)

    graph.add_edge(cell_group_name, parent_name)
    #graph.edge[cell_group_name][parent_name]['name'] = cell_group_name+parent_name
    
    if n_cells_in_group > 1:
        for subgrp in itertools.combinations(cell_group, n_cells_in_group-1):
            build_graph_recursive(graph, subgrp, cell_group_name)

    return graph


def build_graph_from_cell_groups(cell_groups):

    graph = pg.AGraph()
    prev='A'
    for group in cell_groups:
        group_s = [str(s)+'-' for s in sorted(group)]
        cell_group_name = ''.join(group_s)
        graph = build_graph_recursive(graph, group_s, 'A')
        graph.add_edge(prev, cell_group_name)
        prev=cell_group_name

    return graph


# Generate binary matrix with given probabilities for each "cell"
ncells = 20
nwin = 100
a = 0.05
probs = (a*np.ones((ncells, 1)))
nsamples = 1
samples = np.random.rand(ncells, nwin, nsamples)
probmat = np.tile(probs,  (1, nwin))[:, :, np.newaxis]
probmat = np.tile(probmat, (1, 1, nsamples))
binMatsamples = np.greater(probmat, samples).astype(int)
binMat = binMatsamples[:, :, 0]
maxsimps = sc.binarytomaxsimplex(binMat)
test = build_graph_from_cell_groups(maxsimps)
test.layout(prog='dot')
test.draw('test.png')

from brian2 import *
eqs = '''
dv/dt  = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms)                : volt
dgi/dt = -gi/(10*ms)               : volt
'''
P = NeuronGroup(400, eqs, threshold='v>-50*mV', reset='v=-60*mV')
P.v = -60*mV
Pe = P[:320]
Pi = P[320:]
Ce = Synapses(Pe, P, on_pre='ge+=1.62*mV')
Ce.connect(p=0.02)
Ci = Synapses(Pi, P, on_pre='gi-=9*mV')
Ci.connect(p=0.02)
M = SpikeMonitor(P)
run(0.25*second)
plot(M.t/ms, M.i, '.')
show()

M.t




