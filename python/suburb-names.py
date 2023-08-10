from osgeo import ogr, osr
import pandas as pd
import numpy as np
import collections

data_dir = '../data/'
driver = ogr.GetDriverByName('ESRI Shapefile')

node_source = driver.Open(data_dir + 'SA2_2011_AUST.shp', 0)
layer = node_source.GetLayer()
layer.GetFeatureCount()

layer.ResetReading()
feature = layer.GetNextFeature()
feature.keys()

layer.ResetReading()
feature = layer.GetNextFeature()
sla2 = []
name = []
while feature:
    # Victoria is state number 2
    if feature.GetField('STE_CODE11') == '2':
        sla2.append(feature.GetField('SA2_MAIN11'))
        name.append(feature.GetField('SA2_NAME11'))
    feature = layer.GetNextFeature()
print(len(sla2))

sla_name_dict = {}
for ix, sla in enumerate(sla2):
    sla_name_dict[sla] = name[ix]

def find_points(locations, sla_name, pt, polygon):
    """
    locations : the dataframe containing the locations
    sla_name : the name of the SLA region
    pt : a handle that determines coordinates
    polygon : the polygon defining the SLA
    """
    for idx, row in locations.iterrows():
        lat, long = row['poiLat'], row['poiLon']
        if np.isinf(lat):
            continue
        pt.SetPoint(0, long, lat)
        try:
            inside = pt.Within(polygon)
        except ValueError:
            inside = False
            print(long, lat)
            print('Unable to solve inside polygon')
            return
        if inside:
            locations.loc[idx, 'suburb'] = sla_name

spatial_ref = osr.SpatialReference()
spatial_ref.SetWellKnownGeogCS("WGS84")

pt = ogr.Geometry(ogr.wkbPoint)
pt.AssignSpatialReference(spatial_ref)

layer.ResetReading()
poi = pd.read_csv(data_dir + 'poi-Melb-all.csv')
poi['suburb'] = ''
sla = layer.GetNextFeature()
num_sla = 1
while sla:
    # Victoria is state number 2
    if sla.GetField('STE_CODE11') == '2':
        sla_id = sla.GetField('SA2_MAIN11')
        sla_name = sla_name_dict[sla_id]
        polygon = sla.GetGeometryRef()
        find_points(poi, sla_name, pt, polygon)
        
        # progress bar
        if num_sla % 100 == 0:
            print(num_sla)
        num_sla += 1
        
    sla = layer.GetNextFeature()

poi.head()    

poi.to_csv(data_dir + 'poi-Melb-all-suburb.csv', index=False)

print(len(poi))
c_theme = collections.Counter(poi['poiTheme'])
c_suburb = collections.Counter(poi['suburb'])
print(c_theme.most_common(10))
print(c_suburb.most_common(10))



