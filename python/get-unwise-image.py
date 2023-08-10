import numpy as np
from astroquery.sdss import SDSS
from astroquery.ned import Ned
from astropy import coordinates as coords
import astropy.units as u
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from astropy.io import fits
import wget
from scipy.stats import scoreatpercentile
import tarfile

galaxy_name = 'NGC5320'
result_table = Ned.query_object(galaxy_name)

pos = coords.SkyCoord(ra=result_table['RA(deg)'][0]*u.deg,dec=result_table['DEC(deg)'][0]*u.deg, frame='icrs')
print pos.ra.deg
print pos.dec.deg

baseurl = 'http://unwise.me/cutout_fits?version=allwise'
ra = pos.ra.deg
dec = pos.dec.deg
wisefilter = '3' # 12 micron
imsize = '100' # max size = 256 pixels
bands='34'
#version=neo1&ra=41&dec=10&size=100&bands=12
imurl = baseurl+'&ra=%.5f&dec=%.5f&size=%s&bands=%s'%(ra,dec,imsize,bands)

print imurl

# this will download a tar file
wisetar = wget.download(imurl)

tartemp = tarfile.open(wisetar,mode='r:gz')

wnames = tartemp.getnames()
wmembers = tartemp.getmembers()

print filelist

tartemp.extract(wmembers[0])
# to extract all files use
# tartemp.extractall()
# see https://docs.python.org/2/library/tarfile.html for more details

wisedat = fits.getdata(filelist[0])

plt.imshow(wisedat,origin='lower',vmin=scoreatpercentile(wisedat,2.5),vmax=scoreatpercentile(wisedat,99.9))#,cmap='gray')



