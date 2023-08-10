import healpy as hp # for working with HEALPix files
import numpy as np # needed for vector operations
from matplotlib import pyplot as plt # plotting skymaps
from scipy.stats import norm # probability functions

from astropy.utils.data import download_file
url = ('https://dcc.ligo.org/public/0146/G1701985/001/LALInference_v2.fits.gz')
# This is the publication LALInference localization
filename = download_file(url, cache=True)

prob, distmu, distsigma, distnorm = hp.read_map(filename, field=range(4))

npix = len(prob)
nside = hp.npix2nside(npix)
npix, nside

# Area per pixel in steradians
pixarea = hp.nside2pixarea(nside)

from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1 # This gets the complete catalog
cat1, = Vizier.get_catalogs('J/ApJS/199/26/table3') # Downloading the 2MRS Galaxy Catalog

from scipy.special import gammaincinv
completeness = 0.5
alpha = -1.0
MK_star = -23.55
MK_max = MK_star + 2.5*np.log10(gammaincinv(alpha + 2, completeness))
MK_max

from astropy.cosmology import WMAP9 as cosmo
from astropy.table import Column
import astropy.units as u
import astropy.constants as c
z = (u.Quantity(cat1['cz'])/c.c).to(u.dimensionless_unscaled)
MK = cat1['Ktmag']-cosmo.distmod(z)
keep = (z > 0) & (MK < MK_max)
cat1 = cat1[keep]
z = z[keep]

r = cosmo.luminosity_distance(z).to('Mpc').value
theta = 0.5*np.pi - cat1['DEJ2000'].to('rad').value
phi = cat1['RAJ2000'].to('rad').value
ipix = hp.ang2pix(nside, theta, phi)

dp_dV = prob[ipix] * distnorm[ipix] * norm(distmu[ipix], distsigma[ipix]).pdf(r)/pixarea

top50 = cat1[np.flipud(np.argsort(dp_dV))][:50]
top50['RAJ2000', 'DEJ2000', 'Ktmag']

catalog_list = Vizier.find_catalogs('GLADE')
{k:v.description for k,v in catalog_list.items()}

catalogs = Vizier.get_catalogs(catalog_list.keys())
catalogs

Vizier.ROW_LIMIT = 50000
# Note, the GLADE catalog is 1,918,147 rows long thus we will get a memory error if we set the row limit to -1
cat2, = Vizier.get_catalogs('VII/275/glade1') # Downloading the GLADE catalog (Galaxy List for the Advanced Detector Era)

completeness = 0.5
alpha = -1.07
MK_star = -20.47
MK_max = MK_star + 2.5*np.log10(gammaincinv(alpha + 2, completeness))
MK_max

dist = u.Quantity(cat2['Dist']) # Distance in Mpc
z = (u.Quantity(cat2['zph2MPZ'])).to(u.dimensionless_unscaled)
MK = cat2['Kmag2']-cosmo.distmod(z)
keep = (z > 0) & (MK < MK_max)
cat2 = cat2[keep]
dist = dist[keep]

r = dist.value
theta = 0.5*np.pi - cat2['DEJ2000'].to('rad').value
phi = cat2['RAJ2000'].to('rad').value
ipix = hp.ang2pix(nside, theta, phi)

dp_dV = prob[ipix] * distnorm[ipix] * norm(distmu[ipix], distsigma[ipix]).pdf(r)/pixarea

top50 = cat2[np.flipud(np.argsort(dp_dV))][:50]
top50['RAJ2000', 'DEJ2000', 'Kmag2']

