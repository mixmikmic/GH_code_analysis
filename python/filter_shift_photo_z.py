import numpy as np
filter_edges= np.array([400., 522., 691., 818., 922., 948., 1060.])
bright_lines = {'Ha':656.3, 'OIII':500.7, 'Hb':486.1, 'OII':372.7}
for line in bright_lines:
    print line, (filter_edges-bright_lines[line])/bright_lines[line]

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits
from lsst.sims.photUtils import Sed, Bandpass
import os

# Should read redshift from header and un-shift all the spectra to start
files = {'starburst':'spec-1850-53786-0466.fits', 'spiral':'spec-2143-54184-0503.fits','elliptical':'spec-2599-54234-0283.fits'}
galaxies = {}
for key in files.keys():
    hdul = fits.open(files[key])
    flux = hdul[1].data['flux'].copy()
    wave = 10.**hdul[1].data['loglam'].copy()/10.
    add_wave = np.arange(100,wave.min(), 1)
    add_flux = add_wave*0+flux[0]
    galaxies[key] = Sed(wavelen=np.hstack((add_wave,wave)), 
                        flambda=np.hstack((add_flux,flux))*1e-16)
    galaxies[key].redshiftSED(-.1)
    hdul.close()



plt.plot(galaxies['spiral'].wavelen, galaxies['spiral'].flambda)



shift_size = 10  # nm shift in the r-band filter

throughPath = os.getenv('LSST_THROUGHPUTS_BASELINE')
keys = ['u','g','r','r_shifted', 'i','z','y']
filters = {}
for filtername in keys:
    bp = np.loadtxt(os.path.join(throughPath, 'filter_'+filtername[0]+'.dat'),
                    dtype=zip(['wave','trans'],[float]*2 ))
    #good = np.where(bp['trans'] > 1e-3)
    #bp = bp[good]
    tempB = Bandpass()
    if filtername == 'r_shifted':
        over = np.where(bp['wave'] > 650)
        bp['wave'][over] += shift_size
    tempB.setBandpass(bp['wave'],bp['trans'])
    filters[filtername] = tempB

galaxies['spiral'].calcMag(filters['i'])

# Need array of galaxy x filter x redshift.  Loop over redshifts, 
redshifts = np.arange(0,0.5,.01)
mags = np.zeros(redshifts.size, dtype=zip(['g','r','r_shifted', 'i'],[float]*4))
gtype = 'starburst'
for i,z in enumerate(redshifts):
    temp_wave, temp_flam = galaxies[gtype].redshiftSED(z, wavelen=galaxies[gtype].wavelen,
                                                          flambda=galaxies[gtype].flambda)
    temp_sed = Sed(wavelen=temp_wave, flambda=temp_flam)
    for key in mags.dtype.names:
        temp_bp = Bandpass()
        temp_bp.setBandpass(filters[key].wavelen, filters[key].sb, wavelen_min = temp_sed.wavelen.min()+1, 
                            wavelen_max = temp_sed.wavelen.max()-1)
        mags[i][key] = temp_sed.calcMag(temp_bp)

plt.plot(filters['r'].wavelen, filters['r'].sb)
plt.plot(filters['r_shifted'].wavelen, filters['r_shifted'].sb)
plt.xlim([660,750])

plt.plot(redshifts, mags['r']-mags['i'])
plt.ylabel(r'$r-i$')
plt.xlabel('z')

plt.plot(redshifts, mags['r']-mags['r_shifted'])
plt.ylabel('$r-r^{shifted}$')
plt.xlabel('z')

z=.1
temp_wave, temp_flam = galaxies[gtype].redshiftSED(z, wavelen=galaxies[gtype].wavelen,
                                                          flambda=galaxies[gtype].flambda)
plt.plot(temp_wave, temp_flam)
plt.plot(filters['r'].wavelen, filters['r'].sb*temp_flam.max())
plt.plot(filters['r_shifted'].wavelen, filters['r_shifted'].sb*temp_flam.max())

z=.10
temp_wave, temp_flam = galaxies[gtype].redshiftSED(z, wavelen=galaxies[gtype].wavelen,
                                                          flambda=galaxies[gtype].flambda)
plt.plot(temp_wave, temp_flam)
plt.plot(filters['r'].wavelen, filters['r'].sb*temp_flam.max())
plt.plot(filters['r_shifted'].wavelen, filters['r_shifted'].sb*temp_flam.max())

plt.plot(galaxies[gtype].wavelen, galaxies[gtype].flambda)
plt.xlim(600,800)





