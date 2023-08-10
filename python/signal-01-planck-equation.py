from scipy import constants
h = constants.h
print " h = %.3e Joule second"%h
c = constants.c
print " c = %.3e meters/second"%c
k = constants.k
print " k = %.3e Joule/K"%k
sb = constants.Stefan_Boltzmann
print "sb = %.3e W/m^2/K^4"%sb

from ipy_table import *  # import a python package that makes tables look nice
import numpy as np
nus = np.logspace(6, 15, 10) # Define 10 frequencies, from 10^6 to 10^15, logarithmically spaced
lines = [['Frequency (GHz)', "Wavelength (cm)"]]
for nu in nus:
    lam = c/nu
    nug = nu/1e9
    lamcm =lam*1e2
    lines.append([nug, lamcm])
make_table(lines)
apply_theme('basic')
set_global_style(float_format='%.1e')

import math 
t = 213          # Kelvin
lmbda = 10e-6 # 10 microns
nu = c/lmbda
bLambda = (2*h*c**2/lmbda**5)/(math.exp(h*c/(lmbda*k*t)))
print "bLambda=%.3e Watts/m^3/sr"%bLambda
#$$B_\nu(T) = \frac{2h\nu^3/c^2}{e^{h\nu/kT}-1}$$
bNu = (2*h*nu**3/c**2)/(math.exp(h*nu/(k*t))-1)
print "    bNu=%.3e Watts/Hz/m^2/sr"%bNu

get_ipython().magic('matplotlib inline')

import warnings
import matplotlib.pyplot as plt
nus = np.logspace(5, 16, 10000)
dnus = nus[1:]-nus[:-1]
bNu = {}
bNuSum = {}
tnameList = [[5778, "Sun"],[300, "room"], [77, "Liquid Nitrogen"], [3, "CMB"]]
for t,name in tnameList:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bNu[name] = (2*h*np.power(nus,3)/c**2)/(np.exp(h*nus/(k*t))-1)
    plt.loglog(nus/1e9,bNu[name],label="%s (%d)"%(name,t))
    bNuSum[name] = (bNu[name][1:] * dnus).sum()
plt.ylim(ymin=1e-26)
plt.xlim(xmax=1e7)
plt.legend(loc='best')
plt.xlabel("Frequency (GHz)")
l = plt.ylabel(" $B_\\nu$ [Watts/Hz/m^2/sr]")

lam0 = c/nus[0]
lam1 = c/nus[-1]
lams = np.logspace(math.log10(lam1), math.log10(lam0), 10000)
dlams = lams[1:]-lams[:-1]
bLambda = {}
bLambdaSum = {}
for t,name in tnameList:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bLambda[name] = (2*h*c**2/np.power(lams,5))/(np.exp(h*c/(lams*k*t))-1)
    plt.loglog(nus/1e9,bLambda[name],label="%s (%d)"%(name,t))
    bLambdaSum[name] =  (bLambda[name][1:]*dlams).sum()
plt.ylim(ymin=1e-16)
#plt.xlim(xmax=1e7)
plt.legend(loc='best')
plt.xlabel("Wavelength (m)")
l = plt.ylabel("$B_\lambda$ [Watts/m^3/sr]")

lines = [['Object', "T (Kelvin)","$\sigma_{SB} T^4/\pi$","$\sum B_\nu \Delta\nu$", "$\sum B_{\lambda} \Delta\lambda$"]]

for t,name in tnameList:
    jStar = sb*t**4
    radiance = jStar/math.pi
    line = [name, t, radiance, bNuSum[name], bLambdaSum[name]]
    lines.append(line)
make_table(lines)
apply_theme('basic')
set_global_style(float_format='%.3e')



