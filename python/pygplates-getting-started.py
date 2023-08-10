import pygplates

# Names of input rotation file
input_rotation_filename = 'Data/Seton_etal_ESR2012_2012.1.rot'

# Input parameters to define how the reconstruction is made
reconstruction_time = 40.
anchor_plate = 0

# Define a list of (lat,long,plateid) for each point we want to reconstruct 
points = []
points.append((-30.,110.,801))
points.append((-30.,120.,801))

point_features = []
for lat, lon, plate_id in points:
    point_feature = pygplates.Feature()
    point_feature.set_geometry(pygplates.PointOnSphere(lat, lon))
    point_feature.set_reconstruction_plate_id(plate_id)
    point_features.append(point_feature)
    
# Reconstruct the point features.
reconstructed_feature_geometries = []
pygplates.reconstruct(point_features, input_rotation_filename, reconstructed_feature_geometries, reconstruction_time)
    
# Each reconstructed geometry should be a point - return a list of all reconstructed points.
for reconstructed_feature_geometry in reconstructed_feature_geometries:
    print 'Coordinates of the reconstructed point:',reconstructed_feature_geometry.get_reconstructed_geometry().to_lat_lon() 

# Names of input files
input_feature_filename = 'Data/Seton_etal_ESR2012_Coastlines_2012.1_Polygon.gpmlz'
input_rotation_filename = 'Data/Seton_etal_ESR2012_2012.1.rot'

# Input parameters to define how the reconstruction is made
reconstruction_time = 120.6
anchor_plate = 0

# Name of ouput file
output_reconstructed_feature_filename = 'tmp/tmp.shp'

# Use pygplates to carry out the reconstruction 
pygplates.reconstruct(input_feature_filename, input_rotation_filename, output_reconstructed_feature_filename, reconstruction_time, anchor_plate)

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

get_ipython().magic('matplotlib inline')

fig = plt.figure(figsize=(14,10))
ax_map = fig.add_axes([0,0,0.9,1.0])
m = Basemap(projection='robin', lon_0=0, resolution='c', ax=ax_map)

shp_info = m.readshapefile('tmp/tmp','shp',drawbounds=True,color='w')
    
for nshape,seg in enumerate(m.shp):
    poly = Polygon(seg,facecolor='blue',edgecolor='k',alpha=0.7)
    ax_map.add_patch(poly)
 
plt.show()

anchor_plate = 801
pygplates.reconstruct(input_feature_filename, input_rotation_filename, output_reconstructed_feature_filename, reconstruction_time, anchor_plate)

fig = plt.figure(figsize=(14,10))
ax_map = fig.add_axes([0,0,0.9,1.0])
m = Basemap(projection='robin', lon_0=0, resolution='c', ax=ax_map)

shp_info = m.readshapefile('tmp/tmp','shp',drawbounds=True,color='w')
    
for nshape,seg in enumerate(m.shp):
    poly = Polygon(seg,facecolor='blue',edgecolor='k',alpha=0.7)
    ax_map.add_patch(poly)
 
plt.show()



