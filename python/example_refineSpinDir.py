### Import everything we need
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

from diffpy.mpdf import *
from diffpy.Structure import loadStructure

import time # we will be doing some speed testing

get_ipython().magic('matplotlib notebook')

### Create the structure from our cif file, update the lattice params
structureFile = "MnO_R-3m.cif"
mnostructure = loadStructure(structureFile)
lat = mnostructure.lattice
lat.a, lat.b, lat.c = 3.1505626, 3.1505626, 7.5936979 # refined values from PDFgui
print mnostructure

### Set up the magSpecies
mn2p=MagSpecies(struc=mnostructure, label='Mn2+')
mn2p.magIdxs = [0,1,2]
mn2p.kvecs = np.array([0,0,1.5])
mn2p.ffparamkey = 'Mn2'

# start with a randomly oriented spin basis vector
theta0 = np.arccos(np.random.uniform(-1,1))
phi0 = np.random.uniform(-np.pi,np.pi)
mn2p.basisvecs = 2.5*np.array([[np.sin(theta0)*np.cos(phi0),
                                np.sin(theta0)*np.sin(phi0), np.cos(theta0)]])

### Create and prep the magStructure
mstr = MagStructure()
mstr.loadSpecies(mn2p)
mstr.makeAll()

### Set up the mPDF calculator
mc = MPDFcalculator(magstruc=mstr)

### Load the data
PDFfitFile = 'MnOfit_PDFgui.fgr'
rexp,Drexp = getDiffData([PDFfitFile])
mc.rmin = rexp.min()
mc.rmax = rexp.max()

# Do the refinement
def residual1(p,yexp):
    mc.paraScale, mc.ordScale, theta, phi = p
    mn2p.basisvecs = 2.5*np.array([np.sin(theta)*np.cos(phi),
                                   np.sin(theta)*np.sin(phi), np.cos(theta)])
    mstr.makeSpins()
    return yexp - mc.calc(both=True)[2]

p0 = [5.0, 3.0, theta0, phi0] # initial parameter values

start = time.time()
pOpt = leastsq(residual1, p0, args=(Drexp))
end = time.time()

print 'Refined parameters: ', pOpt[0]
print 'Duration of refinement: ', end-start
thetaF, phiF = pOpt[0][2], pOpt[0][3]
Sfinal = 2.5*np.array([np.sin(thetaF)*np.cos(phiF),
                       np.sin(thetaF)*np.sin(phiF),np.cos(thetaF)])
print 'Refined spin: ', Sfinal

fit=mc.calc(both=True)[2]

# Plot the results
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(rexp,Drexp,marker='o',mfc='none',mec='b',linestyle='none')
ax.plot(rexp,fit,'r-',lw=2)
ax.set_xlim(xmin=mc.rmin,xmax=mc.rmax)
ax.set_xlabel('r ($\AA$)')
ax.set_ylabel('d(r) ($\AA^{-2}$)')

plt.show()

### Reset the magnetic structure to how it was originally
mn2p.basisvecs = 2.5*np.array([np.sin(theta0)*np.cos(phi0),
                               np.sin(theta0)*np.sin(phi0), np.cos(theta0)])
mstr.makeSpins()

### Find the locations of the up and down spins (we are creating boolean masks)
upSpins = np.all(mstr.spins==mn2p.basisvecs, axis=1)
downSpins = np.all(mstr.spins==-1.0*mn2p.basisvecs, axis=1)
# Warning! If you define your basisvecs as a nested array, i.e. np.array([[x,y,z]]),
# you will want to use basisvecs[0] instead of basisvecs in the two preceding lines.

### Modify the residual function
def residual2(p,yexp):
    mc.paraScale, mc.ordScale, theta, phi = p
    Svec = 2.5*np.array([np.sin(theta)*np.cos(phi),
                         np.sin(theta)*np.sin(phi), np.cos(theta)])
    mstr.spins[upSpins] = Svec
    mstr.spins[downSpins] = -Svec
    return yexp - mc.calc(both=True)[2]

p0=[5.0, 3.0, theta0, phi0] # initial parameter values

start = time.time()
pOpt = leastsq(residual2, p0, args=(Drexp))
end=time.time()

print 'Refined parameters: ', pOpt[0]
print 'Duration of refinement: ', end-start
thetaF, phiF = pOpt[0][2], pOpt[0][3]
Sfinal = 2.5*np.array([np.sin(thetaF)*np.cos(phiF),
                       np.sin(thetaF)*np.sin(phiF), np.cos(thetaF)])
print 'Refined spin: ', Sfinal

### Update the magnetic structure with the refined spin
#mn2p.basisvecs = np.array([Sfinal])
#mstr.makeSpins()
fit = mc.calc(both=True)[2]

### Plot the results
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(rexp, Drexp, marker='o', mfc='none', mec='b', linestyle='none')
ax.plot(rexp, fit, 'r-', lw=2)
ax.set_xlim(xmin=mc.rmin, xmax=mc.rmax)
ax.set_xlabel('r ($\AA$)')
ax.set_ylabel('d(r) ($\AA^{-2}$)')

plt.show()

plt.close('all')

