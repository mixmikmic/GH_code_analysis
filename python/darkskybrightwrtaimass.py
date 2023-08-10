get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
import numpy as np
import lsst.sims.skybrightness as sb

sm = sb.SkyModel(observatory='LSST', mags=True, moon=False, zodiacal=False, twilight=False)
alt = np.arange(30.,100.,10.)
az = alt*0

sm.setRaDecMjd(az,alt,59000., degrees=True, azAlt=True)

mags = sm.returnMags()

mags

sm.airmass

plt.plot(sm.airmass, mags[:,1])

filterNames = ['u','g','r','i','z','y']
fits = [np.polyfit(sm.airmass,mags[:,i], 1) for i,ack in enumerate(filterNames)]

fits



