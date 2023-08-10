import pygplates

rotation_filename = 'Data/Seton_etal_ESR2012_2012.1.rot'

input_topology_filename = 'Data/Seton_etal_ESR2012_PP_2012.1.gpmlz'

topology_features = pygplates.FeatureCollection(input_topology_filename)
rotation_model = pygplates.RotationModel(rotation_filename)

# Specify time at which to create resolved topological plate polygons
time=100.
resolved_topologies = []
pygplates.resolve_topologies(topology_features, rotation_model, resolved_topologies, time)

# the number of plates is the length of the resolved topologies object 
num_plates = len(resolved_topologies)
print 'Number of Plates is ',num_plates

# Create empty lists of plate_id and area to which we will append these values for each resolved polygon
plate_ids = []
plate_names = []
plate_areas = []

for topology in resolved_topologies:
    # Get the plate_id and name
    plate_feature = topology.get_resolved_feature()
    plate_ids.append(plate_feature.get_reconstruction_plate_id())
    plate_names.append(plate_feature.get_name())
    # Get the plate area - note we use the built in pygplates Earth radius to get 
    plate_geometry = topology.get_resolved_geometry()
    plate_areas.append(plate_geometry.get_area()*pygplates.Earth.mean_radius_in_kms*pygplates.Earth.mean_radius_in_kms)

    

import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

label_index = np.arange(0,len(plate_areas))

plt.figure(figsize=(12,6))
plt.bar(label_index,plate_areas)
plt.xticks(label_index+0.3,plate_names,rotation=80)
plt.ylabel('Plate Area in sq.km')
plt.show()

max_time = 100
time_step = 10

for time in np.arange(0,max_time + 1,time_step):
    resolved_topologies = []
    pygplates.resolve_topologies(topology_features, rotation_model, resolved_topologies, time)
    print 'number of plate polygons at %s Ma is %s' % (time,len(resolved_topologies))

RotFile_List = 'Data/Seton_etal_ESR2012_2012.1.rot'
topology_features = 'Data/Seton_etal_ESR2012_PP_2012.1.gpmlz'

rotation_model = pygplates.RotationModel(RotFile_List)

max_time = 200
time_step = 2

SZ_length = []
RT_length = []

# 'time' = 0, 1, 2, ... , 140
for time in np.arange(0,max_time + 1,time_step):

    # Resolve our topological plate polygons (and deforming networks) to the current 'time'.
    # We generate both the resolved topology boundaries and the boundary sections between them.
    resolved_topologies = []
    shared_boundary_sections = []
    pygplates.resolve_topologies(topology_features, rotation_model, resolved_topologies, time, shared_boundary_sections)

    # We will accumulate the total ridge and subduction zone lengths for the current 'time'.
    total_ridge_length = 0
    total_subduction_zone_length = 0

    # Iterate over the shared boundary sections.
    for shared_boundary_section in shared_boundary_sections:

        # Skip sections that are not ridges or subduction zones.
        if (shared_boundary_section.get_feature().get_feature_type() != pygplates.FeatureType.create_gpml('SubductionZone') and
            shared_boundary_section.get_feature().get_feature_type() != pygplates.FeatureType.create_gpml('MidOceanRidge')):
            continue

        # Iterate over the shared sub-segments to accumulate their lengths.
        shared_sub_segments_length = 0
        for shared_sub_segment in shared_boundary_section.get_shared_sub_segments():

            # Each sub-segment has a polyline with a length.
            shared_sub_segments_length += shared_sub_segment.get_geometry().get_arc_length()

        # The shared sub-segments contribute either to the ridges or to the subduction zones.
        if shared_boundary_section.get_feature().get_feature_type() == pygplates.FeatureType.create_gpml('MidOceanRidge'):
            total_ridge_length += shared_sub_segments_length            
            
        else:
            total_subduction_zone_length += shared_sub_segments_length

    # The lengths are for a unit-length sphere so we must multiple by the Earth's radius.
    total_ridge_length_in_kms = total_ridge_length * pygplates.Earth.mean_radius_in_kms
    total_subduction_zone_length_in_kms = total_subduction_zone_length * pygplates.Earth.mean_radius_in_kms

    #print "At time %dMa, total ridge length is %f kms and total subduction zone length is %f kms." % (
    #        time, total_ridge_length_in_kms, total_subduction_zone_length_in_kms)
    SZ_length.append(total_subduction_zone_length_in_kms)
    RT_length.append(total_ridge_length_in_kms)


plt.figure(figsize=(6,8))
plt.subplot(211)
plt.plot(np.arange(0,max_time + 1,time_step),SZ_length)
plt.gca().invert_xaxis()
plt.title('Subduction Zone Length (km)')
plt.subplot(212)
plt.plot(np.arange(0,max_time + 1,time_step),RT_length)
plt.xlabel('Time (Ma)')
plt.gca().invert_xaxis()
plt.title('Ridge-Transform Length (km)')
plt.show()



