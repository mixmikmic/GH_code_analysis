import os
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from math import factorial

detComponentDir = '../components/camera/detector/'
intermediateDir = '../intermediateFiles/components/camera/'
ven1QEFile = os.path.join(detComponentDir, 'vendor1/vendor1_QE.dat')
ven1LossesFile = os.path.join(detComponentDir, 'vendor1/vendor1_Losses/vendor1_losses.dat')
ven2QEFile = os.path.join(detComponentDir, 'vendor2/vendor2_QE.dat')
ven2LossesFile = os.path.join(detComponentDir, 'vendor2/vendor2_Losses/vendor2_losses.dat')
detectorFile = os.path.join(intermediateDir, 'detThroughput.dat')

ven1QE = np.loadtxt(ven1QEFile)
ven1Losses = np.loadtxt(ven1LossesFile)
ven2QE = np.loadtxt(ven2QEFile)
ven2Losses = np.loadtxt(ven2LossesFile)

_ = plt.ylim([0., 1.002])
_ = plt.plot(ven1QE[:,0], ven1QE[:,1], label='Vendor 1 measured QE')
_ = plt.plot(ven2QE[:,0], ven2QE[:,1], label='Vendor 2 measured QE')
_ = plt.legend(loc='lower left')
_ = plt.xlabel('wavlength (nm)')
_ = plt.ylabel('Fractional Throughput Response')

wavelen = np.arange(300, 1101, 1)
extrapolator = UnivariateSpline(ven1Losses[:,0], ven1Losses[:,1], k=1)
ven1DetLoss = extrapolator(wavelen)
extrapolator = UnivariateSpline(ven2Losses[:,0], ven2Losses[:,1], k=1)
ven2DetLoss = extrapolator(wavelen)

ven1DetQE = ven1QE[:,1] * ven1DetLoss
ven2DetQE = ven2QE[:,1] * ven2DetLoss

_ = plt.ylim([0., 1.002])
_ = plt.plot(wavelen, ven1DetQE, label='Vendor 1 QE with losses')
_ = plt.plot(wavelen, ven2DetQE, label='Vendor 2 QE with losses')
_ = plt.legend(loc='lower left')
_ = plt.xlabel('wavlength (nm)')
_ = plt.ylabel('Fractional Throughput Response')

minDetQE = np.minimum(ven1DetQE, ven2DetQE)

_ = plt.ylim([0., 1.002])
_ = plt.plot(wavelen, minDetQE, label='Minimum QE with losses')
_ = plt.legend(loc='lower left')
_ = plt.xlabel('wavlength (nm)')
_ = plt.ylabel('Fractional Throughput Response')

np.savetxt(detectorFile, zip(wavelen, minDetQE))



