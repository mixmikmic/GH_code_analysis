from astroquery.mast import Observations
from astropy.io import fits

import os
import glob
import shutil

obs_table = Observations.query_criteria(obs_id='J6ME13QHQ')

Observations.download_products(obs_table['obsid'], mrp_only=False, productSubGroupDescription=['FLT'])

files = glob.glob('mastDownload/HST/*qhq/*.fits')
for f in files:
    try:
        shutil.move(f, '.')
    except:
        os.remove(f.split('/')[-1])
        shutil.move(f, '.')
    
shutil.rmtree('mastDownload')

from stsci.skypac import pamutils

# The pamutils.pam_from_file syntax is (input_file, extension, output_file).
pamutils.pam_from_file('j6me13qhq_flt.fits', 1, 'j6me13qhq_wfc2_pam.fits')

