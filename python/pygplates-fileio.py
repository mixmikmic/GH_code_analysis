import pygplates

# specify the file containing the global coastline geometries
input_feature_filename = "Data/Seton_etal_ESR2012_Coastlines_2012.1_Polygon.gpmlz"

# load the features into an object called 'features'
features = pygplates.FeatureCollection(input_feature_filename)

# make an empty object in which to store the selected features
selected_features = []

# iterate over every feature in the coastline file, and append the features that match our criteria
for feature in features:
    if feature.get_reconstruction_plate_id() == 801:
        selected_features.append(feature)

# Write the feature to a new file
output_feature_collection = pygplates.FeatureCollection(selected_features)
output_feature_collection.write('tmp/Aus_Coastlines.gpmlz')
print 'selected features successfully written to gpmlz'

# List of files to merge - this can be as long as you want
filename_list = ['Data/Seton_etal_ESR2012_Isochrons_2012.1.gpmlz',
                 'Data/Seton_etal_ESR2012_Ridges_2012.1.gpmlz']

# Create an empty feature collection
merge_features = []

# iterate over each file, append the features into the merged feature collection
for filename in filename_list:
    print filename
    features = pygplates.FeatureCollection(filename)
    merge_features.extend(features)

# Write the merged features to a file
output_feature_collection = pygplates.FeatureCollection(merge_features)
output_feature_collection.write('tmp/ridges_and_isochrons.gpmlz')
print 'merged features successfully written to gpmlz'

# Create a list of tuples defining arbitrary points on the Australian plate (plate id 801)
points = [
(-30.,110.,801),
(-30.,120.,801),
]

# Create an unclassified point feature
point_features = []
for lat, lon, plate_id in points:
    point_feature = pygplates.Feature()
    point_feature.set_geometry(pygplates.PointOnSphere(lat, lon))
    point_feature.set_reconstruction_plate_id(plate_id)
    point_features.append(point_feature)

# Write the point feature to a file
output_feature_collection = pygplates.FeatureCollection(point_features)
output_feature_collection.write('tmp/points.gpmlz')
print 'point features successfully written to gpmlz'



