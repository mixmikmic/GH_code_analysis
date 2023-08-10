from extcats import CatalogPusher
import pandas as pd
import numpy as np
import concurrent.futures
from healpy import ang2pix

import importlib
importlib.reload(CatalogPusher)

# build the pusher object and point it to the raw files.
ps1p = CatalogPusher.CatalogPusher(
    catalog_name = 'ps1_test',                    # short name of the catalog
    data_source = '../testdata/PS1DR1_test/',     # where to find the data (other options are possible)
    file_type = '*.csv.gz'                        # filter files (there is col definition file in data_source)
    )

# define the reader for the raw files (import column names from file.)
headfile = '../testdata/PS1DR1_test/column_headings.csv'
with open(headfile, 'r') as header:
    catcols=[c.strip() for c in header.readline().split(',')]

# skimm out some columns
bad = ['projectionID', 'skyCellID']
usecols = [c for c in catcols if (not c in bad) or ('gNpt' in c)]

# specify some data types to save up on the storage
# See https://outerspace.stsci.edu/display/PANSTARRS/PS1+MeanObject+table+fields
types = {}
for c in usecols:
    types[c] = np.float16
    if c == 'objID':
        types[c] = np.int32
    if 'Flags' in c:
        types[c] = np.int16
    if ('ra' in c) or ('dec' in c):
        types[c] = np.float32

ps1p.assign_file_reader(
    reader_func = pd.read_csv,           # callable to use to read the raw_files. 
    read_chunks = True,                  # weather or not the reader process each file into smaller chunks.
    names=catcols,                       # All other arguments are passed directly to this function.
    usecols=usecols,
    dtype = types,
    na_values = -999,
    chunksize=50000,
    engine='c')

# define modifier. This time the healpix grid is finer (an orer 16 corresponds to 3")
hp_nside16=2**16
def ps1_modifier(srcdict):
    srcdict['hpxid_16']=int(
        ang2pix(hp_nside16, srcdict['raMean'], srcdict['decMean'], lonlat = True, nest = True))
    return srcdict
ps1p.assign_dict_modifier(ps1_modifier)

# wrap up the file pushing function so that we can 
# use multiprocessing to speed up the catalog ingestion

def pushfiles(filerange):
    # push stuff
    ps1p.push_to_db(
        coll_name = 'srcs',
        index_on = ['hpxid_16'],
        filerange = filerange,
        overwrite_coll = False,
        dry = False, 
        fillna_val = None)
    # add metadata to direct queries
    ps1p.healpix_meta(
        healpix_id_key = 'hpxid_16', 
        order = 16, is_indexed = True, nest = True)

# each job will run on a subgroup of all the files
file_groups = ps1p.file_groups(group_size=1)
with concurrent.futures.ProcessPoolExecutor(max_workers = 2) as executor:
    executor.map(pushfiles, file_groups)   
print ("done! Enjoy your PS1_test database.")

