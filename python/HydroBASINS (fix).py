get_ipython().magic('matplotlib inline')

import logging
import sys
import os
import glob

import math
import ogr
import shapely.geometry, shapely.wkt
import pylab
import matplotlib.pyplot as plt
import shapely as sl
import fiona
import numpy as np

pylab.rcParams['figure.figsize'] = (17.0, 15.0)
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

def convert(input_path):
    filename = os.path.splitext(os.path.basename(input_path))[0]

    output_path = '../output/' + filename + '.shp'

    print(output_path)
    
    dst_ds = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource(output_path)
    dst_lyr = dst_ds.CreateLayer(filename)

    # create fields using OGR
    src_ds = ogr.Open(input_path)
    src_lyr = src_ds.GetLayerByIndex(0)
    f = src_lyr.GetFeature(0)
    [dst_lyr.CreateField(f.GetFieldDefnRef(i)) for i in range(f.GetFieldCount())]

    for feat in src_lyr:
        try:
            geom = shapely.wkt.loads(feat.GetGeometryRef().ExportToWkt())
        except Exception as e: 
            print('Error({0}), skipping geometry.'.format(e))
            continue

        #id = feat.GetField('HYBAS_ID')
        #count = get_geom_coord_count(geom)

        # geom coord count
        # print('{0}: {1}'.format(id, count))

        #if count > 10000:
        #    print ('simplifying ...')
        #    geom = geom.simplify(0.004)
        
        # uncomment in case of messed-up geometry
        #
        #if not geom.is_valid:
        geom = geom.buffer(0.0)
        
        print(geom.area)

        # geom = geom.simplify(0.004)

        # if id == 6030021870:
        #    print ('simplifying ...')
        #    geom = geom.simplify(0.004)

        f = ogr.Feature(dst_lyr.GetLayerDefn())

        # set field values
        for i in range(feat.GetFieldCount()):
            fd = feat.GetFieldDefnRef(i)
            f.SetField(fd.GetName(), feat.GetField(fd.GetName()))

        # set geometry    
        f.SetGeometry(ogr.CreateGeometryFromWkt(geom.to_wkt()))
        
        # create feature
        dst_lyr.CreateFeature(f)
        
        f.Destroy() 
        
    src_ds.Destroy()

    dst_ds.Destroy()

files = glob.glob(r'..\data\HydroBASINS\*.shp')
# files = glob.glob(r'D:\gis\HydroBASINS\without_lakes\hybas_sa_lev*_v1c.shp')
# files = glob.glob(r'D:\gis\HydroBASINS\without_lakes\simplified\hybas_au_lev08_v1c.shp')

# files = glob.glob(r'D:\gis\GRanD\GRanD_dams_v1_1.shp')
# files = glob.glob(r'D:\gis\GRanD\GRanD_reservoirs_v1_1.shp')

# files = glob.glob(r'D:\gis\HydroBASINS\rivers\*_riv_15s.shp')
# files = [r'D:\gis\OpenStreetMaps\Asia\Australia\rivers_lines.shp',
#          r'D:\gis\OpenStreetMaps\Asia\Australia\rivers_multipolygons.shp']

# files = [r'D:\gis\OpenStreetMaps\simplified-land-polygons-complete-3857\simplified_land_polygons.shp']
    
# convert(r'D:\src\GitHub\data-utils\notebooks\shapefile.shp')

# convert(r'D:\gis\HydroBASINS\without_lakes\hybas_au_lev07_v1c.shp')

for f in files:
    convert(f)

