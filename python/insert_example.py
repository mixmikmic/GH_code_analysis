from extcats import CatalogPusher

# build the pusher object and point it to the raw files.
mqp = CatalogPusher.CatalogPusher(
    catalog_name = 'milliquas',             # short name of the catalog
    data_source = '../testdata/milliquas/', # where to find the data
    file_type = 'tdat.gz'
    )


# define the reader for the raw files. In this case the formatting or the raw file 
# is pretty ugly so we have to put quite a lot of information here.
# the pandas package provides very efficient ways to read flat files and its use
# is recommended. For this specific example see:
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_table.html
catcols=['name', 'ra', 'dec', 'lii', 'bii', 'broad_type', 'rmag', 
         'bmag', 'optical_flag', 'red_psf_flag', 'blue_psf_flag',
         'redshift', 'ref_name', 'ref_redshift', 'qso_prob', 
         'radio_name', 'xray_name', 'alt_name_1', 'alt_name_2', 'class']
import pandas as pd
mqp.assign_file_reader(
        reader_func = pd.read_table,         # callable to use to read the raw_files. 
        read_chunks = True,                  # weather or not the reader process each file into smaller chunks.
        names=catcols,                       # All other arguments are passed directly to this function.
        chunksize=50000,
        engine='c',
        skiprows=65,
        sep='|',
        index_col=False,
        comment='<')


# now we have to define a modifier function that acts on the single documents
# (dictionaries) and format them in the way they have to appear in the database.
# in this case we format coordinates in the geoJSON type (this enables mongo to
# support queries in spherical cooridnates), and we assign to each source its
# healpix index on a grid of order 16, corresponding to ~3" resolution.
from healpy import ang2pix
def mqc_modifier(srcdict):
    
    # format coordinates into geoJSON type (commented version uses 'legacy' pair):
    # unfortunately mongo needs the RA to be folded into -180, +180
    ra=srcdict['ra'] if srcdict['ra']<180. else srcdict['ra']-360.
    srcdict['pos']={
            'type': 'Point', 
            'coordinates': [ra, srcdict['dec']]
                    }
    #srcdict['pos']=[srcdict['ra'], srcdict['dec']] # This is the legacy coordinate format
    
    # add healpix index
    srcdict['hpxid_16']=int(
        ang2pix(2**16, srcdict['ra'], srcdict['dec'], lonlat = True, nest = True))
    
    return srcdict

mqp.assign_dict_modifier(mqc_modifier)

# fill in the database, creting indexes on the position and healpix ID.
import pymongo
mqp.push_to_db(
    coll_name = 'srcs', 
    index_on = ['hpxid_16', [('pos', pymongo.GEOSPHERE)] ] ,
    index_args = [{}, {}], # specify arguments for the index creation
    overwrite_coll = False, 
    append_to_coll = False)

# now print some info on the database
#mqp.info()

# define the funtion to test coordinate based queries:
import numpy as np
from healpy import ang2pix, get_all_neighbours
from astropy.table import Table
from astropy.coordinates import SkyCoord
from math import radians
# define your search radius
rs_arcsec = 10.

def test_query_healpix(ra, dec, coll):
    """query collection for the closest point within 
    rs_arcsec of target ra, dec. It uses healpix ID
    to perform the search.
    
    The results as returned as an astropy Table. """
    
    # find the index of the target pixel and its neighbours 
    target_pix = int( ang2pix(2**16, ra, dec, nest = True, lonlat = True) )
    neighbs = get_all_neighbours(2*16, ra, dec, nest = True, lonlat = True)

    # remove non-existing neigbours (in case of E/W/N/S) and add center pixel
    pix_group = [int(pix_id) for pix_id in neighbs if pix_id != -1] + [target_pix]
    
    # query the database for sources in these pixels
    qfilter = { 'hpxid_16': { '$in': pix_group } }
    qresults = [o for o in coll.find(qfilter)]
    if len(qresults)==0:
        return None
    
    # then use astropy to find the closest match
    tab = Table(qresults)
    target = SkyCoord(ra, dec, unit = 'deg')
    matches_pos = SkyCoord(tab['ra'], tab['dec'], unit = 'deg')
    d2t = target.separation(matches_pos).arcsecond
    match_id = np.argmin(d2t)

    # if it's too far away don't use it
    if d2t[match_id]>rs_arcsec:
        return None
    return tab[match_id]


def test_query_2dsphere(ra, dec, coll):
    """query collection for the closest point within 
    rs_arcsec of target ra, dec. It uses mondod spherical
    geometry queries.
    
    The results as returned as an astropy Table. """
    
    
    # fold the RA between -180 and 180.
    if ra > 180:
        ra = ra - 360.
    
    # query and return
    geowithin={"$geoWithin": { "$centerSphere": [[ra, dec], radians(rs_arcsec/3600.)]}}
    qresults = [o for o in coll.find({"pos": geowithin})]
    if len(qresults)==0:
        return None
    
    # then use astropy to find the closest match
    tab = Table(qresults)
    target = SkyCoord(ra, dec, unit = 'deg')
    matches_pos = SkyCoord(tab['ra'], tab['dec'], unit = 'deg')
    d2t = target.separation(matches_pos).arcsecond
    match_id = np.argmin(d2t)

    # if it's too far away don't use it
    if d2t[match_id]>rs_arcsec:
        return None
    return tab[match_id]

# run the test. Here we compare queries based on the 
# healpix index with those based on the 2dsphere mongod support.
mqp.run_test(test_query_healpix, npoints = 1000)
mqp.run_test(test_query_2dsphere, npoints = 1000)

mqp.healpix_meta(healpix_id_key = 'hpxid_16', order = 16, is_indexed = True, nest = True)
mqp.sphere2d_meta(sphere2d_key = 'pos', is_indexed = True, pos_format = 'geoJSON')
mqp.science_meta(
    contact =  'C. Norris', 
    email = 'chuck.norris@desy.de', 
    description = 'compilation of AGN and Quasar',
    reference = 'http://quasars.org/milliquas.htm')



