from astropy.io import fits
import numpy as np

datdir = '/Users/rfinn/research/NSA/'

nsatab = datdir+'nsa_v0_1_2.fits'

nsa = fits.getdata(nsatab)

nsa.columns

nsadict = dict((a,b) for a,b in zip(nsa.NSAID,np.arange(len(nsa.NSAID))))

i = nsadict[56434]
print i

print 'sersic index = ',nsa.SERSIC_N[i]
print 'sersic B/A = ',nsa.SERSIC_BA[i]
print 'sersic pos ang = ',nsa.SERSIC_PHI[i]
print 'sersic Re (pix) = ',nsa.SERSIC_TH50[i]/.396
print 'r mag = ',5.*np.log10(nsa.ZDIST[i]*3.e5/70.*1.e6) -5+nsa.ABSMAG[i][4]
print 'extinction @ r = ',nsa.EXTINCTION[i][4]

print 'run = ',nsa.RUN[i] 
print 'camcol = ',nsa.CAMCOL[i]
print 'field = ',nsa.FIELD[i]
print 'rerun = ',nsa.RERUN[i]



