import sys
import os

#Tell python where Mantid is installed.
#The official packages put this information in an environment variable called "MANTIDPATH"
sys.path.append(os.environ['MANTIDPATH'])

#Also adds the folder where 

#We can now import Mantid's Python API
from mantid.simpleapi import *

#Import matplotlib's pyplot interface under the name 'plt'
import matplotlib.pyplot as plt

#Some magic to tell matplotlib how to behave in IPython Notebook
get_ipython().magic('matplotlib inline')

from MARIReduction_2016_5 import *

wbvan = 21943
sum_runs = False
remove_bkg = True
sam_mass = 19
sam_rmm = 2*173.05 + 2*47.867 + 7*16
monovan = 22034
runno = 22038
ei = 200
iliad_mari(runno,ei,wbvan,monovan,sam_mass,sam_rmm,sum_runs,check_background=remove_bkg, 
           energy_bins=[-100,0.5,195])

wsname = 'MAR{0}Reduced#{1:4.2f}'.format(runno,ei)

Rebin2D(wsname+'_SQW', '0,5,5', '-50,0.5,150', True, True, OutputWorkspace=wsname+'_ecut_lowQ')
Rebin2D(wsname+'_SQW', '13,5,18', '-50,0.5,150', True, True, OutputWorkspace=wsname+'_ecut_highQ')
Rebin2D(wsname+'_SQW','2,0.1,16','10,40,50', True, False, OutputWorkspace=wsname+'_qcut_phon')
Rebin2D(wsname+'_SQW','2,0.1,16','60,40,100', True, False, OutputWorkspace=wsname+'_qcut_cf')

f = [plt.subplots(1, 2, sharey=True, figsize=(20, 5)), plt.subplots(1, 2, sharey=True, figsize=(20, 5))]
for i, wssuffix in enumerate(['_ecut_lowQ', '_ecut_highQ', '_qcut_phon', '_qcut_cf']):  
    row = int(i/2)
    col = i-row*2
    ws = mtd[wsname+wssuffix]
    f[row][1][col].errorbar(ws.readX(0)[1:], ws.readY(0), yerr=ws.readE(0), fmt='.k')
    f[row][1][col].set_title(wssuffix)
    if row==0:
        f[row][1][col].set_ylim([0, 100])
        f[row][1][col].set_xlabel('Energy Transfer (meV)', fontsize=18)
    else:
        f[row][1][col].set_ylim([0, 60])
        f[row][1][col].set_xlabel('$|Q|$ ($\AA$$^{-1}$)', fontsize=18)

import numpy as np

Fit(Function='name=UserFunction,Formula=A1*exp( -x*x*u)*x*x+A0,A1=0.01,u=0.001,A0=0.5', 
    InputWorkspace=wsname+'_qcut_phon', Output=wsname+'_qcut_phon', OutputCompositeMembers=True)
parvals = mtd[wsname+'_qcut_phon_Parameters'].column(1)
pars = {pnames:parvals[id] for id,pnames in enumerate(mtd[wsname+'_qcut_phon_Parameters'].column(0))}
q1=2.5
q2=15.5
fac1 = pars['A1']*np.exp(-q1*q1*pars['u'])*q1*q1+pars['A0']
fac2 = pars['A1']*np.exp(-q2*q2*pars['u'])*q2*q2+pars['A0']
scal = fac1/fac2
print fac1,fac2,scal

Scale(wsname+'_ecut_highQ', scal, OutputWorkspace=wsname+'_ecut_highQ')
Minus(wsname+'_ecut_lowQ',wsname+'_ecut_highQ', OutputWorkspace=wsname+'_cf')

ws = mtd[wsname+'_cf']
plt.errorbar(ws.readX(0)[1:], ws.readY(0), yerr=ws.readE(0), fmt='.k')
plt.ylim([0,60])

# Fit a Gaussian function to the elastic line
Fit(Function='name=Gaussian,Height=55,PeakCentre=0,Sigma=5', 
    InputWorkspace=wsname+'_cf', Output=wsname+'_cf', OutputCompositeMembers=True)
parvals = mtd[wsname+'_cf_Parameters'].column(1)
parsG = {pnames:parvals[id] for id,pnames in enumerate(mtd[wsname+'_cf_Parameters'].column(0))}
print parsG

from PyChop import PyChop2
from CrystalField import ResolutionModel

# Set up a resolution model for MARI
mari = PyChop2('MARI', 'S', 600)
mari.setEi(200)
resmod = ResolutionModel(mari.getResolution, xstart=-200, xend=199)

from CrystalField import CrystalField, CrystalFieldFit, Background, Function

# Generates a random set of parameters
nonzero_parameters = ['B20', 'B40', 'B60', 'B43', 'B63', 'B66']
Blm = {}
for pname in nonzero_parameters:
    Blm[pname] = np.random.rand()*2-1

# Set up a CrystalField model
cf = CrystalField('Yb', 'D3d', Temperature=100, ResolutionModel=resmod, **Blm)
#    B20=1.135, B40=-0.0615, B43=0.315, B60=0.0011, B63=0.037, B66=0.005)
# Define a background function for the CrystalField Model
cf.background = Background(peak=Function('Gaussian', Height=parsG['Height'], Sigma=parsG['Sigma']),
    background=Function('LinearBackground', A0=0, A1=0))
# Fixes the background parameters, so that estimate_parameters only varies the CF parameters
cf.background.peak.ties(Height=parsG['Height'], Sigma=parsG['Sigma'])
cf.background.background.ties(A0=0, A1=0)
cf.IntensityScaling=1
cf.ties(IntensityScaling=1)

# Runs the estimate_parameters algorithm to find a decent set of initial parameters
cffit = CrystalFieldFit(cf, InputWorkspace=wsname+'_cf')
cffit.estimate_parameters(EnergySplitting=100, Parameters=nonzero_parameters, NSamples=1000)
print 'Returned', cffit.get_number_estimates(), 'sets of parameters.'
cffit.select_estimated_parameters(1)
print 'Best guess parameters:'
for pname in nonzero_parameters:
    print '%s = %5.3g' % (pname, cf[pname])
    
# Reruns the fit on these parameters
cffit.fit()
Blmfit = {pname:cf[pname] for pname in nonzero_parameters}
print 'Fitted parameters:'
for pname in nonzero_parameters:
    print '%s = %5.3g' % (pname, cf[pname])

# Calculates the resulting crystal field spectrumInfo
cf = CrystalField('Yb', 'D3d', Temperature=100, ResolutionModel=resmod, **Blmfit)
# Define a background function for the CrystalField Model
cf.background = Background(peak=Function('Gaussian', Height=parsG['Height'], Sigma=parsG['Sigma']),
    background=Function('LinearBackground', A0=0, A1=0))
cf.IntensityScaling=1    
x,y = cf.getSpectrum()
ws_fit = CreateWorkspace(x,y)
ws = mtd[wsname+'_cf']
plt.errorbar(ws.readX(0)[1:], ws.readY(0), yerr=ws.readE(0), fmt='.k')
plt.plot(x,y,'-r')
plt.ylim([0,60])

T1,invchi = cf.getSusceptibility(Temperature=np.linspace(1,30,300), Unit='cgs', Inverse=True, Hdir='powder')
T2,chipar = cf.getSusceptibility(Temperature=np.linspace(1,300,300), Unit='bohr', Inverse=False, Hdir=[0,0,1])
T3,chiperp = cf.getSusceptibility(Temperature=np.linspace(1,300,300), Unit='bohr', Inverse=False, Hdir=[1,0,0])

plt.subplot(121)
plt.plot(T1,invchi)
plt.xlabel('T(K)')
plt.ylabel('Powder inverse susceptibility (mole Yb/emu)')
plt.subplot(122)
plt.plot(T2,chipar,label='$\chi_z$')
plt.plot(T3,chiperp,label='$\chi_{\perp}$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('T(K)')
plt.ylabel('Susceptibility ($\mu_B/Yb$)')
plt.legend(loc='upper center', shadow=True)

H1,ws_magx = cf.getMagneticMoment(Hmag=np.linspace(0,30,300), Unit='bohr', Inverse=False, Hdir=[1,0,0])
H2,ws_magy = cf.getMagneticMoment(Hmag=np.linspace(0,30,300), Unit='bohr', Inverse=False, Hdir=[0,1,0])
H3,ws_magz = cf.getMagneticMoment(Hmag=np.linspace(0,30,300), Unit='bohr', Inverse=False, Hdir=[0,0,1])

plt.plot(H1,ws_magx,label='H||x')
plt.plot(H2,ws_magy,label='H||y')
plt.plot(H3,ws_magz,label='H||z')
plt.legend()
plt.xlabel('B (Tesla)')
plt.ylabel('Magnetisation ($\mu_B$/Yb)')

np.set_printoptions(precision=4, linewidth=140, suppress=True)
print cf.getEigenvectors()

