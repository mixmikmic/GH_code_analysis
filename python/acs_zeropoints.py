import pysynphot as pyS
import numpy as np

# Set up bandpass and source spectrum. Perform a 
# synthetic observation of the source. Note that the
# term 'counts' in PySynphot is variable for each instrument.
# For ACS, 'counts' refers to electrons.
bp = pyS.ObsBandpass('acs,wfc1,f555w,mjd#57754')
spec_bb = pyS.BlackBody(10000)
spec_bb_norm = spec_bb.renorm(1, 'counts', bp)
obs = pyS.Observation(spec_bb_norm, bp)

# Get photometric calibration information.
photflam = obs.effstim('flam') 
photplam = bp.pivot() 

# Get the magnitudes of the source spectrum in the
# bandpass. Because the source was normalized to
# 1 electron per second, the magnitudes are the 
# zeropoints in their respective systems.
# e.g. m_vega = -2.5*log10(counts) + zpt_vega
zp_vega = obs.effstim('vegamag')
zp_st = obs.effstim('stmag')
zp_ab = obs.effstim('abmag')

print(bp.pivot(),obs.efflam(), obs.pivot())
# Print the results.
print('PHOTFLAM = {}'.format(photflam))
print('PHOTPLAM = {}'.format(photplam))
print('')
print('VegaMag_ZP = {}'.format(zp_vega))
print('STMag_ZP = {}'.format(zp_st))
print('ABMag_ZP = {}'.format(zp_ab))

import pysynphot as pyS
import numpy as np
from astropy.table import Table

# Create some fake fluxes in electrons per second.
instrumental_flux = np.array([5.2393, 7.2935, 3.2355, 4.9368])

# Get and apply the aperture correction from 0.2" to 0.5". Use
# the blackbody defined in the previous example to be our source
# and measure the count rate in a 0.2" and 0.5" aperture.
band02 = pyS.ObsBandpass('acs,wfc1,f555w,mjd#57754,aper#0.2')
band05 = pyS.ObsBandpass('acs,wfc1,f555w,mjd#57754,aper#0.5')

obs02 = pyS.Observation(spec_bb, band02)
obs05 = pyS.Observation(spec_bb, band05)

correction_05 = obs02.countrate()/obs05.countrate()
print('Aperture correction 0.2 -> 0.5 = {}'.format(correction_05))

# Apply the aperture correction from 0.2" to 0.5" to measured 
# fluxes. Then apply the correction from 0.5" to infinity.
# The correction from 0.5" to infinity for the ACS/WFC camera
# in F555W is 0.915.
correction_inf = 0.915

flux05 = instrumental_flux / correction_05
flux_inf = flux05 / correction_inf

# Let's recalculate the zeropoints assuming we only know
# PHOTFLAM and PHOTPLAM rather than having PySynphot tell
# us the values. We will re-use the zeropoint for VegaMag
# from the previous example as we would have used PySynphot 
# to get that value in any case.
zp_stmag = -2.5 * np.log10(photflam) - 21.10
zp_abmag = -2.5 * np.log10(photflam) - (5 * np.log10(photplam)) - 2.408

# Now convert instrumental fluxes to physical fluxes and magnitudes.
# f_lambda is the flux density in units of erg/sec/cm^2/Angstrom.
f_lambda = flux_inf * photflam
stmags = -2.5 * np.log10(flux_inf) + zp_stmag
abmags = -2.5 * np.log10(flux_inf) + zp_abmag
vegamags = -2.5 * np.log10(flux_inf) + zp_vega

# Assemble the values into an Astropy Table. Note that we could
# attach units to these columns, however advanced Astropy
# Tables use is outside the scope of this example.
phot_table = Table({'Measured Flux': instrumental_flux, 'F_lambda': f_lambda,
                    'ST Mag': stmags, 'AB Mag': abmags, 'Vega Mag': vegamags}, 
                   names=['Measured Flux', 'F_lambda', 'ST Mag', 'AB Mag', 'Vega Mag'])

phot_table

