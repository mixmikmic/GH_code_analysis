# Importations
get_ipython().magic('matplotlib inline')
import pySPIRALTAP
import numpy as np
import matplotlib.pyplot as plt
import scipy.io # to import.mat files

# ==== Load example data: 
# f = True signal
# A = Sensing matrix
# y ~ Poisson(Af)
rf=scipy.io.loadmat('./demodata/canonicaldata.mat')
f,y,Aorig = (rf['f'], rf['y'], rf['A']) # A Stored as a sparse matrix

## Setup function handles for computing A and A^T:
AT = lambda x: Aorig.transpose().dot(x)
A = lambda x: Aorig.dot(x)

# ==== Set regularization parameters and iteration limit:
tau   = 1e-6
maxiter = 100
tolerance = 1e-8
verbose = 10

# ==== Simple initialization:  
# AT(y) rescaled to a least-squares fit to the mean intensity
finit = y.sum()*AT(y).size/AT(y).sum()/AT(np.ones_like(y)).sum() * AT(y)

# ==== Run the algorithm:
## Demonstrating all the options for our algorithm:
resSPIRAL = pySPIRALTAP.SPIRALTAP(y,A,tau,
                                  AT=AT,
                                  maxiter=maxiter,
                                  miniter=5,
                                  stopcriterion=3,
                                  tolerance=tolerance,
                                  alphainit=1,
                                  alphamin=1e-30,
                                  alphamax=1e30,
                                  alphaaccept=1e30,
                                  logepsilon=1e-10,
                                  saveobjective=True,
                                  savereconerror=True,
                                  savesolutionpath=False,
                                  truth=f,
                                  verbose=verbose, savecputime=True)
## Deparse outputs
fhatSPIRAL = resSPIRAL[0]
parSPIRAL = resSPIRAL[1]
iterationsSPIRAL = parSPIRAL['iterations']
objectiveSPIRAL = parSPIRAL['objective']
reconerrorSPIRAL = parSPIRAL['reconerror']
cputimeSPIRAL = parSPIRAL['cputime']

## ==== Display Results:
## Problem Data:
plt.figure(1, figsize=(18,10))
plt.subplot(311)
plt.plot(f)
plt.title('True Signal (f), Nonzeros = {}, Mean Intensity = {}'.format((f!=0).sum(), f.mean()))
plt.ylim((0, 1.24*f.max()))

plt.subplot(312)
plt.plot(A(f))
plt.title('True Detector Intensity (Af), Mean Intensity = {}'.format(A(f).mean()))

plt.subplot(313)
plt.plot(y)
plt.title('Observed Photon Counts (y), Mean Count = {}'.format(y.mean()))

## Reconstructed Signals:
plt.figure(2, figsize=(24,12))
plt.plot(f, color='blue')
plt.plot(fhatSPIRAL, color='red')
plt.xlabel('Sample number')
plt.ylabel('Amplitude')
plt.title('SPIRAL Estimate, RMS error = {}, Nonzero Components = {}'.format(np.linalg.norm(f-fhatSPIRAL)/np.linalg.norm(f), (fhatSPIRAL!=0).sum()))

## RMS Error:
plt.figure(3, figsize=(18,6))
plt.subplot(211)
plt.plot(range(iterationsSPIRAL), reconerrorSPIRAL, color='blue')
plt.xlabel('Iteration')
plt.ylabel('RMS Error')

plt.subplot(212)
plt.plot(cputimeSPIRAL, reconerrorSPIRAL, color='blue')
plt.xlabel('CPU Time')
plt.ylabel('RMS Error')
plt.title('RMS Error Evolution (CPU Time)')

## Objective:
plt.figure(4, figsize=(18,6))
plt.subplot(211)
plt.plot(range(iterationsSPIRAL), objectiveSPIRAL)
plt.xlabel('Iteration')
plt.ylabel('Objective')
plt.subplot(212)
plt.plot(cputimeSPIRAL, objectiveSPIRAL)
plt.xlabel('CPU Time')
plt.ylabel('Objective')
plt.title('Objective Evolution (CPU Time)')

