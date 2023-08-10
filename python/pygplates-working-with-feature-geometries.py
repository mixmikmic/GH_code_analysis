import pygplates

input_feature_filename = "Data/LIPs_2014.gpmlz"

features = pygplates.FeatureCollection(input_feature_filename)

for feature in features:
    print feature.get_reconstruction_plate_id()

for feature in features:
    print feature.get_geometry()

for feature in features:
    if feature.get_reconstruction_plate_id() == 901:
        print feature.get_geometry().get_area()

for feature in features:
    if feature.get_reconstruction_plate_id() == 101:
        polygon = feature.get_geometry()
        point = polygon.get_boundary_centroid()
        print point.to_lat_lon()

for feature in features:
    if feature.get_reconstruction_plate_id() == 101:
        polygon = feature.get_geometry()
        points = polygon.get_points()
        print 'Polygon vertices are:'
        print points.to_lat_lon_list()

# Load a rotation file
input_rotation_filename = 'Data/Seton_etal_ESR2012_2012.1.rot'

rotation_model=pygplates.RotationModel(input_rotation_filename)

# Get finite rotation for plate 701 relative to spin axis at 52 Ma
rotation = rotation_model.get_rotation(52.,701, 0., 1)
print rotation

# For each LIP polygon feature with plateid 701, get the centroid location and reconstruct it 
# based on the rotation pole we just got from the rotation file
for feature in features:
    if feature.get_reconstruction_plate_id() == 701:
        polygon = feature.get_geometry()
        reconstructed_point = rotation * polygon.get_boundary_centroid()
        print 'Centroid Location = ',polygon.get_boundary_centroid().to_lat_lon()
        print 'Reconstructed Centroid Location = ',reconstructed_point.to_lat_lon()
        



