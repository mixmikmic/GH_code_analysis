get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pylab as plt
from lsst.all_sky_phot.wcs import Fisheye, load_fisheye
from lsst.all_sky_phot import forced_phot, readYBC, readcr2, lsst_earth_location, radec2altaz
import healpy as hp
import glob
import photutils as phu
from astropy.table import Table, hstack, vstack

# Load up the WCS we fit in Full_WCS_fit (generated in notebook Full_WCS_fit)
wcs = load_fisheye('fisheye_wcs.npz')
# Load the Yale Bright Star catalog
ybc = readYBC()

dirs = ['ut012516', 'ut012616', 'ut012716']
zp = 0.
final_table = None
hpmaps = []
mjds = []
files = []
for direc in dirs:
    files.extend(glob.glob(direc +'/*.long.cr2'))

for filename in files:
    im, header = readcr2(filename)
    sum_image = np.sum(im, axis=2).astype(float)

    alt_cat, az_cat = radec2altaz(ybc['RA'], ybc['Dec'], header['mjd'])
    above = np.where(alt_cat > 15.)
    phot_table, hpmap = forced_phot(sum_image, wcs, zp, alt_cat[above], az_cat[above],
               ybc['Vmag'].values[above], ybc['HR'].values[above], return_table=True,
                                   mjd=header['mjd'])
    phot_table.remove_columns(['xcenter', 'ycenter'])
    if final_table is None:
        final_table = phot_table.copy()
    else:
        final_table = vstack([final_table, phot_table])
    hpmaps.append(hpmap)
    mjds.append(header['mjd'])

# convert things to numpy arrays
mjds = np.array(mjds)
hpmaps = np.array(hpmaps)

np.savez('forced_phot_results/trans_maps.npz', mjds=mjds, hpmaps=hpmaps, files=files)
final_table.write('3_day_photometry.hdf5', path='forced_phot_results', format='hdf5')

back = Table.read('3_day_photometry.hdf5', format='hdf5', path='forced_phot_results')

len(back), len(final_table)

len(hpmaps)

hp.mollview(hpmaps[61], rot=[0,90,0])





