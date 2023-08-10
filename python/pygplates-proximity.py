import pygplates


rotation_filename = '../Data/Seton_etal_ESR2012_2012.1.rot'

input_topology_filename = '../Data/Seton_etal_ESR2012_PP_2012.1.gpmlz'

topology_features = pygplates.FeatureCollection(input_topology_filename)
rotation_model = pygplates.RotationModel(rotation_filename)

time = 100
resolved_topologies = []
shared_boundary_sections = []
pygplates.resolve_topologies(topology_features, rotation_model, resolved_topologies, time, shared_boundary_sections)


# Reconstruct features to 10Ma.
reconstruction_time = 10

# All features have their distance calculated relative to this point.
point_latitude = 0
point_longitude = 0
point = pygplates.PointOnSphere(point_latitude, point_longitude)


# The minimum distance to all features and the nearest feature.
min_distance_to_all_features = None
nearest_feature = None

# Iterate over the shared boundary sections.
for shared_boundary_section in shared_boundary_sections:

    for shared_sub_segment in shared_boundary_section.get_shared_sub_segments():        
        
        # Get the minimum distance from point to the current reconstructed geometry.
        min_distance_to_feature = pygplates.GeometryOnSphere.distance(
                point,
                shared_sub_segment.get_geometry(),
                min_distance_to_all_features)

        # If the current geometry is nearer than all previous geometries then
        # its associated feature is the nearest feature so far.
        if min_distance_to_feature is not None:
            min_distance_to_all_features = min_distance_to_feature
            nearest_feature = shared_sub_segment

print nearest_feature.get_feature().get_feature_type()
print nearest_feature.get_feature().get_name()



