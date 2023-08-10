import SpecFlattener
import glob
import StarData
from astropy.io import fits
import SpectralTypeRelations
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

#hdf5_lib = '/media/ExtraSpace/Kurucz_FullGrid/CHIRON_grid_full.hdf5'
hdf5_lib = '/Volumes/DATADRIVE/Kurucz_Grid/IGRINS_grid_full.hdf5'
star_list = [f for f in glob.glob('../201*/*corrected.fits') if 'flattened' not in f and 'oph' not in f.lower() and 'HER' not in f]
print(len(star_list))
#star_list.index('../20131019/HIP_22913.fits')
for s in star_list:
    print s

# Guess stellar properties
MS = SpectralTypeRelations.MainSequence()
def guess_teff_logg(fname):
    header = fits.getheader(fname)
    data = StarData.GetData(header['OBJECT'])
    spt = data.spectype
    teff = MS.Interpolate('Temperature', spt)
    logg = 3.5 if 'I' in spt else 4.0
    return teff, logg

teff, logg = guess_teff_logg(star_list[0])
print(teff, logg)

# Read in flat lamp spectrum (it is not flat!)
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import numpy as np
out = np.loadtxt('../plp/flat_lamp.txt', unpack=True)
wvl, fl = out[:, 0], out[:, 1]
flat = spline(wvl, fl)

import HelperFunctions
orders = HelperFunctions.ReadExtensionFits(star_list[2])

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib notebook')
#nums = tuple(range(5, 16)) + tuple(range(18, 26))
nums = range(3, 18)
#for order in orders[4:25]:
n_left, n_right = 250, 100
good_orders = [o[n_left:-n_right].copy() for i, o in enumerate(orders) if i in nums]
for order in good_orders:
    plt.plot(order.x, order.y, 'k-', alpha=0.5)
    plt.plot(order.x, order.y*flat(order.x), 'r-', alpha=0.5)

reload(SpecFlattener)
print(len(nums))
output = SpecFlattener.flatten_spec(star_list[2], hdf5_lib, teff=teff, logg=logg, normalize_model=False,
                                    ordernums=nums, x_degree=4, orders=good_orders)
final_orders, flattened, shifted_orders, mcf = output


get_ipython().magic('matplotlib notebook')
for order in final_orders:
    plt.plot(order.x, order.y, 'k-', alpha=0.5)



