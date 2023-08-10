get_ipython().magic('matplotlib inline')
from __future__ import print_function, division
import sys, os
import json

from shapely.geometry import Polygon
import shapely.wkt
import fiona.crs
import numpy as np
import pandas as pd
import geopandas as gp

import viirstools as vt

ALT1 = False
ALT2 = False 

basedir = '/Volumes/cwdata1/VIIRS/GINA/dds.gina.alaska.edu/NPP/viirs/'
outdir = '/Volumes/SCIENCE_mobile_Mac/Fire/DATA_BY_PROJECT/2015VIIRSMODIS/rasterout/'
if ALT1:
    basedir = '/Volumes/SCIENCE_mobile_Mac/Fire/DATA_BY_PROJECT/2015VIIRSMODIS/VIIRS/'
elif ALT2:
    basedir = '/Volumes/SCIENCE/Fire/DATA_BY_AREA/2015/VIIRS/'

if os.path.isdir(basedir):
    print(basedir, "exists")
else:
    print("Please check directory {}: cannot access it.".format(basedir))

granulefn = 'viirsgranulecatalog.json'

with open(os.path.join(basedir, granulefn), 'rU') as src:
    granuledir = json.load(src)

crs = fiona.crs.from_epsg('4326')

granuleDF = pd.DataFrame.from_dict(granuledir, orient='index')
granuleDF = granuleDF.loc[granuleDF['iband_complete']]
granuleDF['geometry'] = granuleDF['edgepolygon_I'].apply(shapely.wkt.loads)

granuleDF = gp.GeoDataFrame(
    granuleDF, 
    crs=crs,
    geometry=granuleDF.geometry)

granuleDF.head()

granuleDF.shape

roi = gp.GeoDataFrame.from_file('BorealAKForUAFSmoke.json')

roi.to_crs(epsg=3338)

borealroifootprint = roi.to_crs(epsg=3338)['geometry'][0]
granuleDF = granuleDF.to_crs(epsg=3338)

borealroifootprint.area

granuleDF['borealoverlap'] = granuleDF['geometry'].apply(lambda x: borealroifootprint.intersection(x).area/borealroifootprint.area)

granuleDF['archived'] = granuleDF['borealoverlap'].apply(lambda x: x < 0.01)

granuleDF.head(20)

granuleDF[granuleDF['borealoverlap'] < 0.01].head(100)['geometry'].plot()
roi.to_crs(epsg=3338)['geometry'].plot()

datasettypes = [u'GITCO',
    u'GMTCO',
    u'SVM16',
    u'SVM14',
    u'SVM15',
    u'SVM12',
    u'SVM13',
    u'SVM10',
    u'SVM11',
    u'SVI01',
    u'SVI03',
    u'SVI02',
    u'SVI05',
    u'SVI04',
    u'GDNBO',
    u'SVM05',
    u'SVM04',
    u'SVM07',
    u'SVM06',
    u'SVM01',
    u'SVM03',
    u'SVM02',
    u'SVM09',
    u'SVM08',
    u'SVDNB']
datasettypes.sort()
print(len(datasettypes))

counter = 0
with open(os.path.join(basedir, 'viirsfilesfordeletion.txt'), 'w') as dest:
    for idx, row in granuleDF[granuleDF['borealoverlap'] < 0.01].iterrows():
        for colname in datasettypes:
            if row[colname]:
                try:
                # catching issues with NaN written in empty string slots
                    dest.write(os.path.join(row['dir'], row[colname]) + '\n')
                    counter += 1
                except AttributeError:
                    pass
print("Files marked for deletion: {}.".format(counter))

granuleDF['archived'].value_counts()

granuleDF.drop(['geometry'], axis=1, inplace=True)

granuleDF.head()

granule_dict = pd.DataFrame(granuleDF).to_dict(orient='index')

with open(os.path.join(basedir, 'viirsgranulecatalog_post_archive.json'), 'w') as dest:
    dest.write(json.dumps(granule_dict, indent=2))

goodgranule_dict = pd.DataFrame(granuleDF[~granuleDF['archived']]).to_dict(orient='index')

with open(os.path.join(basedir, 'viirsgranulecatalog_in_ROI.json'), 'w') as dest:
    dest.write(json.dumps(goodgranule_dict, indent=2))

pd.__version__



