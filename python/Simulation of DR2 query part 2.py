import numpy as np
import time
import matplotlib as mp
import matplotlib.pyplot as plt
import pandas as pd
import healpy as hp
get_ipython().run_line_magic('matplotlib', 'inline')

from astroquery.vizier import Vizier

import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.units import Quantity

from context import gaiapix
import gaiapix.gaiapix as hp

mp.rcParams['figure.figsize'] = (12, 8)

HE0435 =  SkyCoord.from_name("HE 0435-1223")

tables = Vizier.query_object("HE 0435-1223")

Vizier.ROW_LIMIT = 3000000

get_ipython().run_line_magic('time', "c = Vizier.get_catalogs(['J/ApJS/221/12/table1'])")

r=c.values()[0]

len(r)

r[:10]

d = r.to_pandas()

d['alpha'] = d['RAJ2000']*u.deg.to(u.rad)-np.pi
d['delta'] = d['DEJ2000']*u.deg.to(u.rad)

s = d.sample(frac=0.1)
plt.subplot(221)
d.gmag.plot.kde(label="g")
s.W1mag.plot.kde(label="W1")
plt.legend()
plt.xlim(10,25)
plt.grid()
plt.xlabel("[mag]")

plt.subplot(222)
d.z.plot.kde()
plt.xlim(0,6)
plt.xlabel('z')

plt.hexbin(s['W1-W2'],s['W2-W3'])

d.groupby(np.floor(d.gmag)).WISEA.count()

d[np.isnan(d.gmag)].count()

plt.scatter(d.gmag,d.W1mag,c=d.z,s=d.z)
plt.colorbar(label="z")
plt.xlabel("g [mag]")
plt.ylabel("W1 [mag]")

hpX = hp.gaiapix(6)
d['hp'] = hpX.angle2pixel(d.RAJ2000,d.DEJ2000)
hpX.setHpCount(d)
f = plt.figure()
hpX.plot(f,vmin=0,vmax=100,cmap=mp.cm.viridis,coord='G')

hpX = hp.gaiapix(6)
d['hp'] = hpX.angle2pixel(d.RAJ2000,d.DEJ2000)
hpX.setHpValues(d,keyValue='z')
f = plt.figure()
hpX.plot(f,vmin=0,vmax=2,cmap=mp.cm.viridis,coord='G')

hpX = hp.gaiapix(6)
d['hp'] = hpX.angle2pixel(d.RAJ2000,d.DEJ2000)
hpX.setHpValues(d,keyValue='gmag')
f = plt.figure()
hpX.plot(f,vmin=18,vmax=20,cmap=mp.cm.viridis,coord='G')

hpX = hp.gaiapix(6)
d['hp'] = hpX.angle2pixel(d.RAJ2000,d.DEJ2000)
hpX.setHpValues(d,keyValue='W1mag')
f = plt.figure()
hpX.plot(f,vmin=14,vmax=17,cmap=mp.cm.viridis,coord='G')

d['l'] = SkyCoord(d.RAJ2000.values,d.DEJ2000.values,unit=u.deg).galactic.l.deg
d['b'] = SkyCoord(d.RAJ2000.values,d.DEJ2000.values,unit=u.deg).galactic.b.deg

columns = ['ra','ra_error','dec','dec_error','ra_dec_corr',
           'parallax','parallax_error',
           'pmra','pmra_error','pmdec','pmdec_error',
           'phot_g_mean_mag']

def DR1error(g): 
    if g <= 15 :
        return 0.05 
    return 0.05*np.exp(0.2*(g-15)*(g-15))

DR1error = np.vectorize(DR1error)

def pm_error_DR2(g): 
    if g <= 15 :
        return 0.06
    return 0.06*np.exp(0.6*(g-15))

pm_error_DR2 = np.vectorize(pm_error_DR2)

def parallax_error_DR2(g): 
    if g <= 15 :
        return 0.04
    return 0.04*np.exp(0.5*(g-15)+0.01*(g-15)*(g-15))

parallax_error_DR2 = np.vectorize(parallax_error_DR2)

def randomFromData(data,n,range=(-1,1),bins=1000) : 
    """generate n random points following data distribution"""
    hist, bins = np.histogram(data, bins=bins,range=range)
    bin_midpoints = bins[:-1] + np.diff(bins)/2
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    values = np.random.rand(n)
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = bin_midpoints[value_bins]
    return random_from_cdf

data = np.random.normal(size=1000)
plt.subplot(121)
plt.hist(d[d.gmag<21].gmag, 50,range=(15,22))
plt.subplot(122)
plt.hist(randomFromData(d[d.gmag<21].gmag,1000,range=(15,22)), bins=50)
plt.show()

d['ra'] =  d['RAJ2000']
d['dec'] = d['DEJ2000']
d['phot_g_mean_mag'] = randomFromData(d[d.gmag<21].gmag,len(d),range=(15,22))
d['ra_error'] = DR1error(d.phot_g_mean_mag)
d['dec_error'] = d['ra_error']
d['pmra_error'] = pm_error_DR2(d.phot_g_mean_mag)
d['pmdec_error'] = d['pmra_error']
d['pmra'] = np.random.normal(scale=d.pmra_error,size=len(d))
d['pmdec'] = np.random.normal(scale=d.pmdec_error,size=len(d))
d['parallax_error'] = parallax_error_DR2(d.phot_g_mean_mag)
d['parallax'] = np.random.normal(scale=d.parallax_error,size=len(d))

plt.subplot(221)
d.pmra.hist(bins=100)
plt.yscale('log')
plt.subplot(222)
d.pmdec.hist(bins=100)
plt.yscale('log')
plt.subplot(223)
d.parallax.hist(bins=100)
plt.yscale('log')

d[np.abs(np.sin(d.b*u.deg.to(u.rad)))>0.1].b.hist(bins=100)

dsim = d[(np.abs(np.sin(d.b*u.deg.to(u.rad)))>0.1)].sample(frac=0.3)

dsim.ra.count()

hpX.setHpCount(dsim)
f = plt.figure()
hpX.plot(f,vmin=0,vmax=20,cmap=mp.cm.viridis,coord='G')

get_ipython().run_line_magic('time', 'dsim.to_csv("../data/DR2simALLwiseQSO.csv")')

dsim.info(verbose=False)



