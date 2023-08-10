get_ipython().magic('pylab inline')

# The SCAT reader class will read in an SCAT file
# and process it so that you can easily access the 
# fitted parameters. Additionally, there are hooks 
# from other programs that read these files.

from spectralTools.scatReader import scatReader

# Often we need to compute the energy/photon fluxes 
# from our spectral fits. The following utility allows
# you to do this with a few bonuses. When doing multi-component
# spectral fitting, it is important to properly propoagte your
# errors when computing the fluxes. This code will handle that 
# for you.

from spectralTools.temporal.fluxLightCurve import fluxLightCurve

# Create and SCAT object
s=scatReader("batchFit.fit")

# If you made multiple fits by hand,
# scatReader objects add
# s = s1 + s2

# Printing the scatReader object tells
# you the models loaded and the time
# bins that are included
print s

# The GetParamArray(model,param) member 
# returns an array of the values and the 
# errors. The 0th column is the values
# and the final two are the symmetric or
# asymmetric errors

Ep = s.GetParamArray("Band's GRB, Epeak","Epeak")[:,0]
EpErr = s.GetParamArray("Band's GRB, Epeak","Epeak")[:,1]

# Get the mean of the time bins for plotting

time = s.meanTbins

# And plot

errorbar(time,Ep,yerr=EpErr,fmt=',',color='k')
ylabel(r"$E_{\rm p}$")
xlabel("Time")
xscale('log')
yscale('log')

alpha = s.GetParamArray("Band's GRB, Epeak","alpha")[:,0]
alphaErr = s.GetParamArray("Band's GRB, Epeak","alpha")[:,1]

# Get the mean of the time bins for plotting

time = s.meanTbins

# And plot

errorbar(time,alpha,yerr=alphaErr,fmt=',',color='k')
ylabel(r"$\alpha$")
xlabel("Time")

# Feed in he scat object and select an energy range\
# Optionally you can  input a redshift to perform
# k-corrections

flc = fluxLightCurve(s, 8., 40000.)

# Calcualte the fluxes and
# The errors
flc.CreateEnergyLightCurve()
flc.EnergyLightCurveErrors()
flc.SaveEnergy() # Save out these possibly lengthy calculations

bandEflux    = flc.energyFluxes["Band's GRB, Epeak"]
bandEfluxErr = flc.energyFluxErrors["Band's GRB, Epeak"]

errorbar(Ep,bandEflux,xerr=EpErr,yerr=bandEfluxErr,fmt='o',color='k')
xscale('log')
yscale('log')
ylim(bottom = 5E-8)
xlim(left = 50)
xlabel(r"$E_{\rm p}$")
ylabel(r"$F_{\rm E}$")

s2 = scatReader("batchFit_BB.fit")

print s2

flc2 = fluxLightCurve(s2, 8., 40000.)
flc2.CreateEnergyLightCurve()
flc2.EnergyLightCurveErrors()
flc2.SaveEnergy() # Save out these possibly lengthy calculations

from spectralTools.step import Step

bandFlux2 = flc2.energyFluxes["Band's GRB, Epeak"]
bbFlux2 = flc2.energyFluxes["Black Body"]

fig = figure(123)
ax = fig.add_subplot(111)
Step(ax,flc2.tBins,bandFlux2,col="blue")
Step(ax,flc2.tBins,bbFlux2,col="red")
ylabel(r"$F_{\rm E}$")
xlabel("Time")
ylim(bottom=0)
legend(["Band","Blackbody"])

Ep = s2.GetParamArray("Band's GRB, Epeak","Epeak")[:,0]
EpErr = s2.GetParamArray("Band's GRB, Epeak","Epeak")[:,1]

kT = s2.GetParamArray("Black Body","kT")[:,0]
kTErr = s2.GetParamArray("Black Body","kT")[:,1]


# Get the mean of the time bins for plotting

time = s.meanTbins


errorbar(time,Ep,yerr=EpErr,fmt=',',color='k')
errorbar(time,kT,yerr=kTErr,fmt=',',color='g')
#ylabel(r"$E_{\rm p}$")
xlabel("Time")
xscale('log')
yscale('log', nonposy='clip')

legend(["Ep","kT"])



