import numpy as np
import matplotlib.path as mplPath

def indexZones(shapeFilename):
    import rtree
    import fiona.crs
    import geopandas as gpd
    index = rtree.Rtree()
    zones = gpd.read_file(shapeFilename).to_crs(fiona.crs.from_epsg(2263))
    for idx,geometry in enumerate(zones.geometry):
        index.insert(idx, geometry.bounds)
    return (index, zones)

def findBlock(p, index, zones):
    match = index.intersection((p.x, p.y, p.x, p.y))
    for idx in match:
        z = mplPath.Path(np.array(zones.geometry[idx].exterior))
        if z.contains_point(np.array(p)):
            return zones['OBJECTID'][idx]
    return -1

def mapToZone(parts):
    import pyproj
    import shapely.geometry as geom
    proj = pyproj.Proj(init="epsg:2263", preserve_units=True)    
    index, zones = indexZones('datasets/block-groups-polygons.geojson')
    for line in parts:
        if (line['pickup_longitude'] and line['pickup_latitude']):
            pickup_location  = geom.Point(proj(float(line['pickup_longitude']), float(line['pickup_latitude'])))
            pickup_block = findBlock(pickup_location, index, zones)
            if pickup_block >= 0:
                print (pickup_block)

import pandas as pd
data_pd = pd.read_csv("datasets/yellow_tripdata_2012-01.csv")
data_pd.head()

# testing
mapToZone(data_pd.head(20).T.to_dict().values())



