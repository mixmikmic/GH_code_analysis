import pygplates
import math
import numpy as np
import matplotlib.pyplot as plt
import glob

get_ipython().magic('matplotlib inline')


RotFile_List = ['../Data/Global_EarthByte_230-0Ma_GK07_AREPS.rot']
GPML_List = ['../Data/Global_EarthByte_230-0Ma_GK07_AREPS_PlateBoundaries.gpmlz',
             '../Data/Global_EarthByte_230-0Ma_GK07_AREPS_Topology_BuildingBlocks.gpmlz']


min_time = 0
max_time = 200
time_step = 1.

times = np.arange(min_time,max_time + 1,time_step)


rotation_model = pygplates.RotationModel(RotFile_List)
topology_features = pygplates.FeatureCollection()
for file in GPML_List:
    topology_feature = pygplates.FeatureCollection(file)
    topology_features.add(topology_feature)
#'''


total_area_list = []

times = np.arange(min_time,max_time + 1,time_step)
# 'time' = 0, 1, 2, ... , max time
for time in times:

    total_area = 0
    
    # Resolve our topological plate polygons (and deforming networks) to the current 'time'.
    resolved_topologies = []
    pygplates.resolve_topologies(topology_features, rotation_model, resolved_topologies, time)

    for resolved_topology in resolved_topologies:
        total_area += resolved_topology.get_resolved_boundary().get_area()

    total_area_list.append(total_area)
    
    print 'total area at %0.2f Ma = %0.2f' % (time,total_area)

    
plt.figure()
plt.plot(times,total_area_list)
plt.ylabel('Total Area of all Polygons (normalised sphere)',fontsize=12)
plt.xlim((0,200))
plt.xlabel('Age (Ma)',fontsize=12)
plt.gca().yaxis.grid(True,which='major')
plt.show()




