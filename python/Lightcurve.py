get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from fermipy.gtanalysis import GTAnalysis

gta = GTAnalysis.create('data/fit2_sed.npy')

get_ipython().magic('pinfo gta.lightcurve')

lc = gta.lightcurve('3C 279',nbins=6,make_plots=True)

list(lc.keys())

plt.errorbar(lc['tmax_mjd'],lc['flux'],yerr=lc['flux_err'],fmt='o')

plt.xlabel('Time [MJD]')
plt.ylabel(r'dN/dE [MeV cm$^{-2}$ s$^{-1}$]')
plt.show()
plt.show()

plt.plot(lc['flux']/lc['flux_err'],lc['npred']/np.sqrt(lc['npred']),'o')

plt.xlabel('Flux/Flux_err')
plt.ylabel('Npred/sqrt(Npred)')
plt.show()



