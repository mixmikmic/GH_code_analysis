import pygplates

# The magnetic picks are the 'features to partition'
# Since they are already in OGR GMT format, gplates can read them directly 
mag_picks = pygplates.FeatureCollection('Data/GSFML.Gaina++_2009_JGeolSoc.picks.gmt')

# static polygons are the 'partitioning features'
static_polygons = pygplates.FeatureCollection('Data/Seton_etal_ESR2012_StaticPolygons_2012.1.gpmlz')

# The partition_into_plates function requires a rotation model, since sometimes this would be
# necessary even at present day (for example to resolve topological polygons)
rotation_model=pygplates.RotationModel('Data/Seton_etal_ESR2012_2012.1.rot')

# partition features
partitioned_mag_picks = pygplates.partition_into_plates(static_polygons,
                                                       rotation_model,
                                                       mag_picks)

# Write the partitioned data set to a file
output_feature_collection = pygplates.FeatureCollection(partitioned_mag_picks)
output_feature_collection.write('tmp/GSFML.Gaina++_2009_JGeolSoc.picks.partitioned.gmt')

import pandas as pd

df = pd.read_csv('Data/Boucot_etal_Map24_Paleocene_v4.csv',sep=',')

df

# put the points into a feature collection, using Lat,Long coordinates from dataframe
point_features = []
for index,row in df.iterrows():
    point = pygplates.PointOnSphere(float(row.LAT),float(row.LONG))
    point_feature = pygplates.Feature()
    point_feature.set_geometry(point)
    point_features.append(point_feature)
    
# The partition points function can then be used as before
partitioned_point_features = pygplates.partition_into_plates(static_polygons,
                                                       rotation_model,
                                                       point_features) 

# Reconstruct the points to 60 Ma (in the Paleocene)
#reconstructed_point_features = []
pygplates.reconstruct(partitioned_point_features,
                      rotation_model,
                      'tmp/reconstructed_points.shp',
                     60.0)  

coastlines_filename = 'Data/Seton_etal_ESR2012_Coastlines_2012.1_Polygon.gpmlz'
pygplates.reconstruct(coastlines_filename,
                      rotation_model,
                      'tmp/reconstructed_coastlines.shp',
                     60.0) 


import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

get_ipython().magic('matplotlib inline')

fig = plt.figure(figsize=(14,10))
ax_map = fig.add_axes([0,0,0.9,1.0])
m = Basemap(projection='robin', lon_0=0, resolution='c', ax=ax_map)

shp_info = m.readshapefile('tmp/reconstructed_coastlines','shp',drawbounds=True,color='w')
    
for nshape,seg in enumerate(m.shp):
    poly = Polygon(seg,facecolor='khaki',edgecolor='k',alpha=0.7)
    ax_map.add_patch(poly)
    
m.readshapefile('tmp/reconstructed_points', 'reconstructed_points')
 
for p in m.reconstructed_points:
    m.plot(p[0], p[1], marker='o', color='m', markersize=8, markeredgewidth=2)
    
plt.show()



