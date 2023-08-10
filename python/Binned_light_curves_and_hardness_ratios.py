get_ipython().magic('pylab inline')

get_ipython().magic('system fv VELAX-1_steps_lc.fits')

from OccView import OccView
import scipy.signal
import numpy
import matplotlib.pyplot as plt

ov = OccView("VELAX-1_steps_lc.fits")

print ov.eneEdge
print ov.eneEdge[1]
testedge = ov.eneEdge[1]
tests="%.1f-%.1f keV"%(testedge[0],testedge[1])
print tests

ov.PlotBinnedFluxes(56000,57890,8.9643680,[0,1,2],quality="good",save = "binnedFlux.pdf")

#first channel (~12-25 keV)
h0=ov.GetBinnedFluxes(56000,57890,89.643680,0)
#second channel (~25-50 keV)
h1=ov.GetBinnedFluxes(56000,57890,89.643680,1)
#get time, flux, and error from first channel
t = h0["time"]
f0 = h0["flux"]
e0 = h0["error"]
#get flux and error from second channel
f1 = h1["flux"]
e1 = h1["error"]

ov.PlotBinnedFluxes(56000,57890,89.643680,[0,1],quality="good",save = "binnedFlux10orb.pdf")

sr=divide(f1,f0) 
srerr=sr*sqrt(divide(e0,f0)**2+divide(e1,f1)**2)
plt.errorbar(t,sr,srerr,fmt='o')
plt.xlabel("Time (MJD)")
plt.ylabel("Softness (25-50 keV)/(12-25 keV)")
plt.title('VELAX-1')



