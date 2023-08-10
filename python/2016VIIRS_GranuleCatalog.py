get_ipython().magic('matplotlib inline')
from __future__ import print_function, division
import sys, os
import json

from shapely.geometry import Polygon

from pygaarst import raster
import viirstools as vt
import viirsswathtools as vst

reload(vt)

ALT1 = True
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

overpasses = [
    u'2015_06_14_165_1148',
    u'2015_06_14_165_2144',
    u'2015_06_14_165_2325',
]

myviirsfiles = vt.getfilesbygranule(basedir, overpasses)

allviirsfiles = vt.getfilesbygranule(basedir)

BANDFILES = {
    u'dnb': ['SVDNB', u'GDNBO'],
    u'iband': [u'SVI01', u'SVI02', u'SVI03', u'SVI04', u'SVI05', u'GITCO'],
    u'mband': [u'SVM01', u'SVM02', u'SVM03', u'SVM04', u'SVM05', 
               u'SVM06', u'SVM07', u'SVM08', u'SVM09', u'SVM10', 
               u'SVM11', u'SVM12', u'SVM13', u'SVM14', u'SVM15', 
               u'SVM16', u'GMTCO'],
}

def checkviirsganulecomplete(granuledict, dataset='iband'):
    dataset = dataset.lower()
    complete = True
    if dataset not in BANDFILES.keys():
        print("Unknown band type '{}' for viirs granule. Valid values are: {}.".format(
            dataset, ', '.join(BANDFILES.keys())))
        return
    complete = True
    for bandname in BANDFILES[dataset]:
        try:
            if not granuledict[bandname]:
                complete = False
                print("detected missing band {}".format(bandname))
                return complete
        except KeyError:
            complete = False
            print("detected missing key for band {}".format(bandname))
            return complete
    return complete

checkviirsganulecomplete(myviirsfiles['2015_06_14_165_1148']['20150614_1152377'], 'mband')

def getgranulecatalog(basedir, overpassdirlist=None):
    intermediary = vt.getfilesbygranule(basedir, scenelist=overpassdirlist)
    catalog = {}
    for overpass in intermediary:
        for granule in intermediary[overpass]:
            if granule in ['dir', 'message']: continue
            print(granule)
            catalog[granule] = intermediary[overpass][granule]
            catalog[granule][u'dir'] = intermediary[overpass]['dir']
            for datasettype in BANDFILES:
                catalog[granule][datasettype + u'_complete'] = checkviirsganulecomplete(catalog[granule])
            if catalog[granule][u'iband_complete']:
                try:
                    viirs = raster.VIIRSHDF5(os.path.join(
                            catalog[granule][u'dir'], 
                            catalog[granule][u'SVI01']))
                except IOError:
                    print("cannot access data file for I-band in {}".format(granule))
                    catalog[granule][u'iband_complete'] = False
                    continue
                catalog[granule][u'granuleID'] = viirs.meta[u'Data_Product'][u'AggregateBeginningGranuleID']
                catalog[granule][u'orbitnumber'] = viirs.meta[u'Data_Product'][u'AggregateBeginningOrbitNumber']
                try:
                    catalog[granule][u'ascending_node'] = viirs.ascending_node
                    edgelons, edgelats = vt.getedge(viirs)
                except IOError:
                    print("cannot access geodata file for I-band in {}".format(granule))
                    catalog[granule][u'iband_complete'] = False
                    continue
                catalog[granule][u'edgepolygon_I'] = Polygon(zip(edgelons, edgelats)).wkt
                viirs.close()
    return catalog 

singlecata = getgranulecatalog(basedir, ['2015_06_14_165_1148'])

cata = vt.getgranulecatalog(basedir)

with open(os.path.join(basedir, 'viirsgranulecatalog.json'), 'w') as dest:
    dest.write(json.dumps(cata, indent=2))

