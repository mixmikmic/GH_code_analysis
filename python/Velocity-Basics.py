import pygplates
import numpy as np

rotation_filename = '../Data/Seton_etal_ESR2012_2012.1.rot'

rotation_model = pygplates.RotationModel(rotation_filename)

timeFrom = 11.
timeTo = 10.
MovingPlate = 101

# Get the rotation from 11Ma to 10Ma, and the feature's reconstruction plate ID.
equivalent_stage_rotation = rotation_model.get_rotation(
    timeTo, MovingPlate, timeFrom)

velocity_point = pygplates.PointOnSphere((20,-60))

# Calculate a velocity for each reconstructed point over the 1My time interval.
velocity_vector = pygplates.calculate_velocities(
    velocity_point,
    equivalent_stage_rotation,
    1,
    pygplates.VelocityUnits.cms_per_yr)

velocity_magnitude_azimuth = pygplates.LocalCartesian.convert_from_geocentric_to_magnitude_azimuth_inclination(
                        velocity_point,
                        velocity_vector)

print velocity_vector
print velocity_magnitude_azimuth

print 'Velocity: magnitude = %0.4f cm/yr, azimuth = %0.4f' %             (velocity_magnitude_azimuth[0][0],np.degrees(velocity_magnitude_azimuth[0][1]))

# Create for each point we want to reconstruct 
points = []
points.append((-30.,110.,801))

point_features = []
for lat, lon, plate_id in points:
    point_feature = pygplates.Feature()
    point_feature.set_geometry(pygplates.PointOnSphere(lat, lon))
    point_feature.set_reconstruction_plate_id(plate_id)
    point_features.append(point_feature)

max_time = 100.
delta_time = 10.

fixed_plate = 802

for time in np.arange(0,max_time+1.,delta_time):    
    # Reconstruct the point features.
    reconstructed_feature_geometries = []
    pygplates.reconstruct(point_features, rotation_model, reconstructed_feature_geometries, time)
    
    # Get the rotation from 'time+delta' to 'time', and the feature's reconstruction plate ID.
    equivalent_stage_rotation = rotation_model.get_rotation(
        time, plate_id, time+delta_time)
    
    for reconstructed_feature_geometry in reconstructed_feature_geometries:      
        # Calculate a velocity for each reconstructed point over the delta time interval.
        velocity_vector = pygplates.calculate_velocities(
            reconstructed_feature_geometry.get_reconstructed_geometry(),
            equivalent_stage_rotation,
            delta_time,
            pygplates.VelocityUnits.cms_per_yr)

        velocity_magnitude_azimuth = pygplates.LocalCartesian.convert_from_geocentric_to_magnitude_azimuth_inclination(
                                reconstructed_feature_geometry.get_reconstructed_geometry(),
                                velocity_vector)
    
        print 'Time = %0.2f Ma' % time
        print 'Reconstructed Point Lat, Long = %s, %s' %             reconstructed_feature_geometry.get_reconstructed_geometry().to_lat_lon()
        print 'Velocity: magnitude = %0.4f cm/yr, azimuth = %0.4f' %             (velocity_magnitude_azimuth[0][0],np.degrees(velocity_magnitude_azimuth[0][1]))



