from osgeo import ogr

# Open the shapefile
dataset = ogr.Open('..Shapefile1.shp')
print dataset.name

## get the layer
layer = dataset.GetLayerByIndex()
## count the feayures
feature_count = layer.GetFeatureCount()
## print the count
print feature_count
## get the geometry - this will be a number so...
geometry = layer.GetGeomType()
## convert that number to the associated geometry type
geometry_name = ogr.GeometryTypeToName(geometry)
## print the geometry type
print geometry_name 

import os
count = 0
lsshape = []
for root, dirs, files in os.walk("D:/"):
    for file in files:
        if file.endswith(".shp"):
            shapefile = (os.path.join(root, file))
            lsshape.append(shapefile)
            
            count +=1
print count

def infoShp(path):

    # Open the shapefile
    dataset = ogr.Open(path)

    ## get the layer
    layer = dataset.GetLayerByIndex()
    ## count the feayures
    feature_count = layer.GetFeatureCount()
    ## print the count
    #print feature_count
    ## get the geometry - this will be a number so...
    geometry = layer.GetGeomType()
    ## convert that number to the associated geometry type
    geometry_name = ogr.GeometryTypeToName(geometry)
    ## print the geometry type
    #print geometry_name 
 
    return dataset.name, geometry_name, feature_count



import pandas as pd
df = pd.DataFrame(columns=['File', 'Geometry', 'features'])
for i in range(0, len(lsshape)):
    name,geoName, featcount = (infoShp(lsshape[i]))
    df.loc[i] = name,geoName,featcount
    
print df.head()



df.to_csv('.../out_shp.csv', encoding='utf-8', index=False)

