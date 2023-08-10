import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import requests, io, astropy
from IPython.display import Image, display
from html import unescape

## For handling ordinary astropy Tables
from astropy.table import Table, vstack

## For reading FITS files
import astropy.io.fits as apfits

## There are a number of relatively unimportant warnings that 
## show up, so for now, suppress them:
import warnings
warnings.filterwarnings("ignore")

# Use the NASA_NAVO/astroquery
from navo_utils.registry import Registry

# Use the astroquery TapPlus library.
from astroquery.utils.tap.core import TapPlus

tap_services_CAOM=Registry.query(keyword='caom',service_type='table', publisher='Space Telescope')
print('Found {} results:'.format(len(tap_services_CAOM)))
tap_url = unescape(tap_services_CAOM[0]['access_url'])
print(tap_url) 

CAOM_service = TapPlus(url=tap_url)

job = CAOM_service.launch_job("""
    SELECT * FROM ivoa.Obscore 
    WHERE CONTAINS(POINT('ICRS', 16.0, 40.0),s_region)=1
  """)
CAOM_results = job.get_results()
print(CAOM_results)

job = CAOM_service.launch_job("""
    SELECT top 10 s_ra, s_dec, access_estsize, access_url FROM ivoa.Obscore 
    WHERE CONTAINS(POINT('ICRS', 16.0, 40.0),s_region)=1
    AND obs_collection = "GALEX" AND dataproduct_type = 'image'
  """)
CAOM_results = job.get_results()

print(CAOM_results[1])

