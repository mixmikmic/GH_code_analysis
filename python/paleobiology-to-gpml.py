import requests
from io import StringIO
import pandas as pd
import pygplates

# this is the string that defines the request - you can get this string from the pbdb download generator
# listed above
# note.1, assumes that you select 'csv' as your output format
# note.2, the download generator will also control which fields are available in the output. This
#       example is fairly minimal, many more fields are available
url = 'https://paleobiodb.org/data1.2/occs/list.csv?base_name=Bryozoa&max_ma=200&min_ma=0&show=coords'

# send the request to the server - the entire output is contained within the object 'r'
r = requests.get(url)

# uncomment this line to see the entire output message and data
#print r.text

# this line reads the text part of the output (in this case, csv-formatted text) into a pandas dataframe.
# note that the 'StringIO' is necessary because pandas is used to reading files - r.text is not a file,
# but 'StringIO(r.text)' makes the data readable by pandas 
df = pd.read_csv(StringIO(r.text))

# print the columns in the data table to see what we have
df.columns

# put the points into a feature collection, using Lat,Long coordinates from dataframe
point_features = []
for index,row in df.iterrows():
    point = pygplates.PointOnSphere(float(row.lat),float(row.lng))
    point_feature = pygplates.Feature()
    point_feature.set_geometry(point)
    point_feature.set_valid_time(row.max_ma,row.min_ma)
    point_features.append(point_feature)

    

# static polygons are the 'partitioning features'
static_polygons = pygplates.FeatureCollection('../Data/Seton_etal_ESR2012_StaticPolygons_2012.1.gpmlz')

# The partition_into_plates function requires a rotation model, since sometimes this would be
# necessary even at present day (for example to resolve topological polygons)
rotation_model=pygplates.RotationModel('../Data/Seton_etal_ESR2012_2012.1.rot')
    
# The partition points function can then be used as before
partitioned_point_features = pygplates.partition_into_plates(static_polygons,
                                                       rotation_model,
                                                       point_features)

output_features = pygplates.FeatureCollection(partitioned_point_features)

# Note that the output format could be shp, gmt, gpml or gpmlz - the extension controls the format
output_features.write('FossilData.gpmlz')



