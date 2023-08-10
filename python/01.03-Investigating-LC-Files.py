from astropy.io import fits
h = fits.open('data/kplr006922244-2010078095331_llc.fits')
h[1].data.names

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(h[1].data['TIME'],h[1].data['SAP_FLUX'],label='SAP Flux')
plt.plot(h[1].data['TIME'],h[1].data['PDCSAP_FLUX'],label='PDCSAP Flux')
plt.legend()
plt.xlabel('Time BJD-2454833 (days)')
plt.ylabel('Counts (e$^-$/s)')

