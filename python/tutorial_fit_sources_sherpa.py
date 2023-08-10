get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u
from gammapy.image import SkyImage, IACTBasicImageEstimator
from gammapy.background import RingBackgroundEstimator
from gammapy.data import DataStore

from regions import CircleSkyRegion
import numpy as np

# Create reference image: 6x6 deg FOV centered on (l,b) = (300,0)
target_position = SkyCoord(299, 0, unit='deg', frame='galactic')

xref = target_position.galactic.l.value
yref = target_position.galactic.b.value

ref_image = SkyImage.empty(
    nxpix=200, nypix=150, binsz=0.02,
    xref=xref, yref=yref,
    proj='TAN', coordsys='GAL')

on_region = CircleSkyRegion(center=SkyCoord(300, 0, unit='deg',
                            frame='galactic'), radius=0.5 * u.deg)
excl_mask = ref_image.region_mask(on_region)
excl_mask.data = np.invert(excl_mask.data)

# excl_mask = ref_image
# excl_mask.data = ref_image.data == 0

print(excl_mask)
excl_mask.plot(add_cbar=True)

bkg_estimator = RingBackgroundEstimator(
    r_in=0.5 * u.deg,
    width=0.2 * u.deg)

image_estimator = IACTBasicImageEstimator(
    reference=ref_image,
    emin=0.3 * u.TeV,
    emax=100 * u.TeV,
    offset_max=4 * u.deg,
    background_estimator=bkg_estimator,
    exclusion_mask=excl_mask)

# Point to data challenge simulation directory
INDX_DIR = '../handson/index/gps'

data_store = DataStore.from_dir(INDX_DIR)

ras = data_store.obs_table['RA_PNT'] * u.deg
decs = data_store.obs_table['DEC_PNT'] * u.deg
pointings = SkyCoord(ras, decs, frame='icrs' )
dists = pointings.separation(target_position)
obs_id = data_store.obs_table['OBS_ID'][(dists < 4 * u.deg)]
print((obs_id), "observations within radius 4 deg")

obslist = data_store.obs_list(obs_id)
images = image_estimator.run(obslist)

for i,im in enumerate(images):
    print(images.names[i])
    im.smooth(radius=3).plot() ; plt.show()

images['counts']    .write("G300-0_test_counts.fits", clobber=True)
images['exposure']  .write("G300-0_test_exposure.fits", clobber=True)
images['background'].write("G300-0_test_background.fits", clobber=True)

#As psf is an array of quantities we cannot use the images['psf'].write() function
#all the other arrays do not have quantities. To investigate
fits.writeto("G300-0_test_psf.fits",images['psf'].data.value,overwrite=True)

import sherpa.astro.ui as sh
from gammapy.image import SkyImage
from regions.core.pixcoord import PixCoord
from regions.shapes.circle import CirclePixelRegion

sh.set_stat("cash")
sh.set_method("simplex")
sh.load_image("G300-0_test_counts.fits")
sh.set_coord("logical")

sh.load_table_model("expo", "G300-0_test_exposure.fits")
sh.load_table_model("bkg",  "G300-0_test_background.fits")
sh.load_psf        ("psf",  "G300-0_test_psf.fits")

sh.set_full_model(bkg)
bkg.ampl = 1
sh.freeze(bkg)

data = sh.get_data_image().y -  sh.get_model_image().y
resid = SkyImage(data=data, wcs=ref_image.wcs)

resid_table=[]  #Keep residual images in a list to show them later
resid_smo6 = resid.smooth(radius = 6)
resid_smo6.plot()
resid_table.append(resid_smo6)

maxcoord = resid_smo6.lookup_max()
maxpix = resid_smo6.wcs_skycoord_to_pixel(maxcoord[0])
sh.set_full_model(bkg + psf(sh.gauss2d.g0) * expo) # creates g0 as a gauss2d instance
g0.xpos = maxpix[0]
g0.ypos = maxpix[1]
sh.freeze(g0.xpos, g0.ypos) # fix the position in the initial fitting step

expo.ampl = 1e-9 # fix exposure amplitude so that typical exposure is of order unity
sh.freeze(expo)
sh.thaw(g0.fwhm, g0.ampl) # in case frozen in a previous iteration

g0.fwhm = 10 # give some reasonable initial values
g0.ampl = maxcoord[1]
sh.fit() # Performs the fit; this takes a little time.

sh.thaw(g0.xpos, g0.ypos)
sh.fit()
sh.freeze(g0)

data = sh.get_data_image().y -  sh.get_model_image().y
resid = SkyImage(data=data, wcs=ref_image.wcs)

resid_smo6 = resid.smooth(radius = 6)
resid_smo6.show(vmin = -0.5, vmax = 1)
resid_table.append(resid_smo6)

for i in range(1,6) :
    sh.create_model_component('gauss2d', 'g'+str(i))
gs = [g0, g1, g2, g3, g4, g5]
sh.set_full_model(bkg + psf(g0+g1+g2+g3+g4+g5) * expo)

for i in range(1, len(gs)) :
    gs[i].ampl = 0   # initialize components with fixed, zero amplitude
    sh.freeze(gs[i])

for i in range(1, len(gs)) :
    maxcoord = resid_smo6.lookup_max()
    maxpix = resid_smo6.wcs_skycoord_to_pixel(maxcoord[0])
    gs[i].xpos = maxpix[0]
    gs[i].ypos = maxpix[1]
    gs[i].fwhm = 10
    gs[i].fwhm = maxcoord[1]

    sh.thaw(gs[i].fwhm)
    sh.thaw(gs[i].ampl)
    sh.fit()

    sh.thaw(gs[i].xpos)
    sh.thaw(gs[i].ypos)
    sh.fit()
    sh.freeze(gs[i])

    data = sh.get_data_image().y -  sh.get_model_image().y
    resid = SkyImage(data=data, wcs=ref_image.wcs)

    resid_smo6 = resid.smooth(radius = 6)
    resid_smo6.show(vmin = -0.5, vmax = 1)
    resid_table.append(resid_smo6)

from astropy.stats import gaussian_fwhm_to_sigma
from astropy.table import Table

pix_scale = ref_image.wcs_pixel_scale()[0].deg


rows = []
for idx, g in enumerate(gs):
    ampl = g.ampl.val
    g.ampl = 0
    stati = sh.get_stat_info()[0].statval
    g.ampl = ampl
    statf = sh.get_stat_info()[0].statval
    delstat = stati - statf
    
    coord = resid.wcs_pixel_to_skycoord(g.xpos.val, g.ypos.val)
    pix_scale = resid.wcs_pixel_scale()[0].deg
    sigma = g.fwhm.val * pix_scale * gaussian_fwhm_to_sigma
    rows.append(dict(
        idx=idx,
        delstat=delstat,
        glon=coord.l.deg,
        glat=coord.b.deg,
        sigma=sigma ,
    ))

table = Table(rows=rows, names=rows[0])
table[table['delstat'] > 25]

from ipywidgets.widgets.interaction import interact

def plot_resid(i):
    resid_table[i].plot(vmin=-0.5, vmax=1)
    ax=plt.gca()
    ax.set_title('CStat=%.2f'%(table['delstat'][i]))
    ax.scatter(
    table['glon'][i], table['glat'][i],
    transform=ax.get_transform('galactic'),
    color='none', edgecolor='azure', marker='o', s=400)
    plt.show()
interact(plot_resid,i=(0,5))



