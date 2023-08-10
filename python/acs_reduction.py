from astroquery.mast import Observations
import shutil
import glob
import os

# Retrieve the observation information.
obs_table = Observations.query_criteria(obs_id='J9L960010')

# Download only the j9l960010_asn.fits, *_raw.fits, and *_spt.fits files.
Observations.download_products(obs_table['obsid'], mrp_only=False, 
                               productSubGroupDescription=['RAW', 'ASN', 'SPT'], 
                               extension='fits')

# Move the files from the mastDownload directory to the current working
# directory and make a backup of the files.
fits_files = glob.glob('mastDownload/HST/J*/*.fits')
for fil in fits_files:
    try:
        shutil.move(fil, '.')
    except:
        os.remove(fil.split('/')[-1])
        shutil.move(fil, '.')
    
# Delete the mastDownload directory and all subdirectories it contains.
shutil.rmtree('mastDownload')

from astropy.io import fits

with fits.open('j9l960010_asn.fits', mode='update') as asn_hdu:
    asn_tab = asn_hdu[1].data
    asn_tab = asn_tab[asn_tab['MEMTYPE'] == 'EXP-DTH']
    asn_hdu[1].data = asn_tab

cte_correct = False

import os

os.environ['CRDS_SERVER_URL'] = 'https://hst-crds.stsci.edu'
os.environ['CRDS_SERVER'] = 'https://hst-crds.stsci.edu'
os.environ['CRDS_PATH'] = './crds_cache'
os.environ['JREF'] = './crds_cache/references/hst/acs/'
os.environ['jref'] = './crds_cache/references/hst/acs/'

from astropy.io import fits
import glob
import os

raw_files = glob.glob('*_raw.fits')

for fil in raw_files:
    with fits.open(fil, mode='update') as hdu:
        if cte_correct:
            hdu[0].header['PCTECORR'] = 'PERFORM'
        else:
            hdu[0].header['PCTECORR'] = 'OMIT'

for fil in raw_files:
    get_ipython().system('crds bestrefs --files $fil --sync-references=1 --update-bestrefs')

get_ipython().system('calacs.e j9l960010_asn.fits')

from ginga.web.pgw import ipg
from code import notebook_tools

gserver = ipg.make_server(host='localhost', port=9916, use_opencv=True)
gserver.start(no_ioloop=True)

if cte_correct:
    img2 = 'j9l960a7q_flc.fits'
else:
    img2 = 'j9l960a7q_flt.fits'

notebook_tools.side2(gserver, 'j9l960a7q_raw.fits', img2)

from stwcs import updatewcs

# Note that the wildcards here will exclude the j9l960a7q dataset.
if cte_correct:
    img_files = 'j9l9*a[9-f]q_flc.fits'
else:
    img_files = 'j9l9*a[9-f]q_flt.fits'

updatewcs.updatewcs(img_files, use_db=False)

from drizzlepac import tweakreg

tweakreg.TweakReg(img_files, shiftfile=True, interactive=False, 
                  expand_refcat=False, refimage='j9l960abq_flt.fits', 
                  updatehdr=True, runfile='tweak.log',
                  imagefindcfg={'dqbits': 0, 'threshold': 1000, 
                                'peakmax': 50000, 'conv_width': 3.5},
                  refimagefindcfg={'dqbits': 0, 'threshold': 4000, 
                                   'peakmax': 50000, 'conv_width': 3.5})

from drizzlepac import astrodrizzle

astrodrizzle.AstroDrizzle(img_files, output='47tuc_f606w_1.0', 
                          driz_sep_bits='16, 64, 256', final_bits='16, 64, 256', 
                          final_wht_type='EXP', final_rot=0, final_scale=0.03, 
                          final_pixfrac=1.0)

import numpy

# Create an array of final_pixfrac values ranging from 0.1
# to 0.9 in steps of 0.1.
pixfracs = numpy.linspace(0.1, 0.9, 9)

# Loop over each final_pixfrac value, creating drizzled
# products with root file name "47tuc_f606w_XX" where
# XX is the final_pixfrac value.
for frac in pixfracs:
    astrodrizzle.AstroDrizzle(img_files, 
                              output='47tuc_f606w_' + str(frac), resetbits=0,
                              driz_sep_bits='16, 64, 256', final_bits='16, 64, 256',
                              final_wht_type='EXP', final_rot=0, final_scale=0.03, 
                              final_pixfrac=frac, skysub=False, driz_separate=False,
                              median=False, blot=False, driz_cr=False)

from ginga.web.pgw import ipg
from code import notebook_tools
import warnings

warnings.filterwarnings('ignore')

gserver = ipg.make_server(host='localhost', port=9917, use_opencv=True)
gserver.start(no_ioloop=True)

if cte_correct:
    driz_02 = '47tuc_f606w_0.2_drc_sci.fits'
else:
    driz_02 = '47tuc_f606w_0.2_drz_sci.fits'
    
v1 = gserver.get_viewer('v1')
v1.load(driz_02)
v1.embed(height=650, width=650)

from astropy.io import fits
import glob

drz_files = glob.glob('47tuc_f606w_*_drz_wht.fits')

rms_med = []
pixfrac = []

# Now for each drizzled mosaic, compute the RMS/median
# statistic.
for fil in drz_files:
    with fits.open(fil) as hdu:
        box1 = hdu[0].data[2250: 2750, 1000: 1500]
        box2 = hdu[0].data[7250: 7750, 2250: 2750]
        box3 = hdu[0].data[6000: 6500, 7250: 7750]
        box4 = hdu[0].data[1250: 1750, 5750: 6250]
        
        data = np.vstack((box1, box2, box3, box4))
        
        rms_med.append(np.std(data)/np.median(data))
        pixfrac.append(float(fil.split('_')[2]))

from plotly.offline import init_notebook_mode, iplot
from code import plots

init_notebook_mode(connected=True)

figure = plots.drizzle_rms_plot(pixfrac, rms_med)
iplot(figure)

from ginga.web.pgw import ipg
from code import notebook_tools
import warnings

warnings.filterwarnings('ignore')

gserver = ipg.make_server(host='localhost', port=9927, use_opencv=True)
gserver.start(no_ioloop=True)

if cte_correct:
    driz_08 = '47tuc_f606w_0.8_drc_sci.fits'
else:
    driz_08 = '47tuc_f606w_0.8_drz_sci.fits'
    
ystars = [6823, 2937, 1722]
xstars = [7340, 2139, 6129]
    
notebook_tools.side3(gserver, driz_08, driz_08, driz_08, ycen=ystars, xcen=xstars)

from plotly.offline import init_notebook_mode, iplot
from src import plots

ystars = [6823, 2937, 1722]
xstars = [7340, 2139, 6129]

driz_08 = '47tuc_f606w_0.8_drz_sci.fits'

for i, _ in enumerate(ystars):
    if i < 1:
        fig = plots.star_2d(driz_08, ystars[i], xstars[i], size=10, high=150, spacing=20)
    else:
        fig = plots.star_2d(driz_08, ystars[i], xstars[i], size=10, high=500, spacing=70)
    iplot(fig)

