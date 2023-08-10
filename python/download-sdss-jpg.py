import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#from astropy.cosmology.funcs import distmod
from astropy.cosmology import WMAP9 as cosmo
from astropy.constants import c
from astropy import units as u
import warnings 
warnings.filterwarnings('ignore')
from pydl.pydlutils.spheregroup import spheregroup

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

try:
    # Python 3.x
    from urllib.parse import urlencode
    from urllib.request import urlretrieve
except ImportError:
    # Python 2.x
    from urllib import urlencode
    from urllib import urlretrieve

import IPython.display

table_path = '/Users/rfinn/Dropbox/Research/APPSS/SDSSphot/run_sep14/'
latest_run = 'a100.code12.SDSSvalues170914.csv'
agc_cross = 'a100.sdsscross.code12.170914.csv'
HI = 'a100.code12.tab1.170914.csv'

infile = table_path+latest_run
sdss = np.recfromcsv(infile)

keepflag = (sdss['ra']> 150.) & (sdss['ra'] < 225.) & (sdss['objid'] > 12)
ra = sdss['ra'][keepflag]
dec = sdss['dec'][keepflag]

linking_length = 10. # arcmin

get_ipython().run_line_magic('time', 'grp, mult, frst, nxt = spheregroup(ra, dec, linking_length/60.)')

print grp[0:20]
print mult[0:20]
print frst[0:20]
print nxt[0:20]

plt.figure()
t = plt.hist(grp,bins=len(grp))

npergrp, _ = np.histogram(grp, bins=len(grp), range=(0, len(grp)))
nbiggrp = np.sum(npergrp > 1).astype('int')
nsmallgrp = np.sum(npergrp == 1).astype('int')
ngrp = nbiggrp + nsmallgrp

print('Found {} total groups, including:'.format(ngrp))
print('  {} groups with 1 member'.format(nsmallgrp))
print('  {} groups with 2-5 members'.format(np.sum( (npergrp > 1)*(npergrp <= 5) ).astype('int')))
print('  {} groups with 5-10 members'.format(np.sum( (npergrp > 5)*(npergrp <= 10) ).astype('int')))
print('  {} groups with >10 members'.format(np.sum( (npergrp > 10) ).astype('int')))

# try getting picture for first group with more than 10 members

ragroup = ra[npergrp > 10]
decgroup = dec[npergrp > 10]

print ragroup

testcoord = SkyCoord(ragroup[0]*u.deg, decgroup[0]*u.deg, frame='icrs')

impix = 1024
imsize = 20*u.arcmin
cutoutbaseurl = 'http://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx'
query_string = urlencode(dict(ra=testcoord.ra.deg, 
                                     dec=testcoord.dec.deg, 
                                     width=impix, height=impix, 
                                     scale=imsize.to(u.arcsec).value/impix))
url = cutoutbaseurl + '?' + query_string

# this downloads the image to your disk
urlretrieve(url, 'group0_SDSS_cutout.jpg')

IPython.display.Image('group0_SDSS_cutout.jpg')



