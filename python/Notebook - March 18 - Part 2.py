get_ipython().run_cell_magic('bash', '', 'pip install aplpy\npip install https://github.com/ericmandel/pyds9/archive/master.zip')

get_ipython().run_cell_magic('bash', '', 'curl -O https://astropy.stsci.edu/data/galactic_center/gc_bolocam_gps.fits\ncurl -O https://astropy.stsci.edu/data/galactic_center/gc_2mass_k.fits')

get_ipython().magic('matplotlib inline')
import pylab as pl

from astropy.io import fits

# load the data (no headers, first extension by default)
# if there are extensions, fits.getdata('file.fits', ext=extension_number)
# if you want many extension, use fits.open('file.fits'), then access each independently
stellardata = fits.getdata('gc_2mass_k.fits')

# show the image: vmax sets the brightest displayed pixel
# cmap can be any of the valid matplotlib colormaps (pl.cm....)
pl.imshow(stellardata, cmap='viridis', vmax=1000)

dustdata = fits.getdata('gc_bolocam_gps.fits')

pl.contour(dustdata)

dustdata.shape, dustdata.flatten().shape

np.any(np.isnan(dustdata))

# subset of the data that is not nan
# implicitly flattens
non_nan_dustdata = dustdata[~np.isnan(dustdata)]
non_nan_dustdata = dustdata[np.isfinite(dustdata)]
non_nan_dustdata = np.compress(np.isfinite(dustdata.flatten()), dustdata.flatten())
len(non_nan_dustdata)

pl.hist(dustdata[~np.isnan(dustdata)], bins=np.linspace(0,2,50))

pl.contour(dustdata, levels=np.linspace(0.2, 10, 10))

pl.figure(figsize=(12,12))
pl.imshow(stellardata, cmap='gray')
pl.contour(dustdata[100:-100, 100:-100], levels=np.linspace(0.2, 10, 10))

import aplpy

get_ipython().magic('matplotlib nbagg')
FF = aplpy.FITSFigure('gc_2mass_k.fits')
FF.show_grayscale(vmax=1000)

get_ipython().magic('matplotlib nbagg')
FF = aplpy.FITSFigure('gc_2mass_k.fits')
FF.show_grayscale(vmax=1000)
# convention not generally needed, only for specific (CAR) FITS projections
FF.show_contour('gc_bolocam_gps.fits', convention='calabretta')

get_ipython().magic('matplotlib nbagg')
FF = aplpy.FITSFigure('gc_2mass_k.fits')
FF.show_grayscale(vmax=1000)
# convention not generally needed, only for specific (CAR) FITS projections
FF.show_contour('gc_bolocam_gps.fits', convention='calabretta')
scalebar = FF.add_scalebar(0.1, label='0.1$^\circ$', color='orange')
FF.scalebar.set_corner('top right')
FF.scalebar.set_font_size(40)
FF.scalebar.set_font_weight('bold')
FF.scalebar.set_linewidth(4)
FF.scalebar.set_label('0.1$^\circ$')

import astroquery
from astropy import units as u

from astroquery.irsa import Irsa
from astroquery.vizier import Vizier
from astroquery.eso import Eso

Eso.ROW_LIMIT = 500

Eso.query_instrument('naco', help=True)

tbl = Eso.query_instrument('naco', target='Sgr A*')
tbl

rslt = Irsa.query_region('Sgr A*', radius=10*u.arcmin, catalog='pt_src_cat')
#rslt

bright = rslt[rslt['k_m'] < 9]

get_ipython().magic('matplotlib nbagg')
FF = aplpy.FITSFigure('gc_2mass_k.fits')
FF.show_grayscale(vmax=1000, invert=True)
# convention not generally needed, only for specific (CAR) FITS projections
FF.show_contour('gc_bolocam_gps.fits', convention='calabretta', colors=['r'])
scalebar = FF.add_scalebar(0.1, label='0.1$^\circ$', color='orange')
FF.scalebar.set_corner('top right')
FF.scalebar.set_font_size(40)
FF.scalebar.set_font_weight('bold')
FF.scalebar.set_linewidth(4)
FF.scalebar.set_label('0.1$^\circ$')
FF.show_markers(bright['ra'], bright['dec'])

get_ipython().magic('run file.py')
get_ipython().magic('run -i file.py')
execfile('file.py') # is equivalent to %run -i ...

import pyds9
from astropy.io import fits

DD = pyds9.DS9('mine')

DD.set('frame 1')

DD.set_pyfits(fits.open('gc_2mass_k.fits'))

DD.set('frame lock wcs')
DD.set('frame 2')
DD.set_pyfits(fits.open('gc_bolocam_gps.fits'))

DD.set('single')

