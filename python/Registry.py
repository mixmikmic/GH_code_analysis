import requests, io, astropy

## For handling ordinary astropy Tables
from astropy.table import Table, vstack

## There are a number of relatively unimportant warnings that 
## show up, so for now, suppress them:
import warnings
warnings.filterwarnings("ignore")

## our stuff
import sys

# Use the NASA_NAVO/astroquery
from navo_utils.registry import Registry

results = Registry.query(source='ned', service_type='cone', verbose=True)
print('Found {} results:'.format(len(results)))
print(results[:]['access_url'])
print(results[1]['ivoid'])
print(results.columns)

results = Registry.query_counts('publisher', 15, verbose=True)
print(results)

results = Registry.query(source='ned', publisher='Extragalactic Database', service_type='cone', verbose=True)
print('Found {} results:'.format(len(results)))
print(results[:]['access_url'])

from html import unescape

for result in results:
    print(unescape(result['access_url']))

