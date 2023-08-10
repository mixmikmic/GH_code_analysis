get_ipython().magic('matplotlib inline')

from cmt.components import Waves
waves = Waves()

waves.get_output_var_names()

waves.get_input_var_names()

angle_name = 'sea_surface_water_wave__azimuth_angle_of_opposite_of_phase_velocity'

print "Data type: %s" % waves.get_var_type(angle_name)
print "Units: %s" % waves.get_var_units(angle_name)
print "Grid id: %d" % waves.get_var_grid(angle_name)
print "Number of elements in grid: %d" % waves.get_grid_size(0)
print "Type of grid: %s" % waves.get_grid_type(0)

waves.initialize(None)

waves.set_value('sea_shoreline_wave~incoming~deepwater__ashton_et_al_approach_angle_asymmetry_parameter', .25)
waves.set_value('sea_shoreline_wave~incoming~deepwater__ashton_et_al_approach_angle_highness_parameter', .7)

waves.update()

print 'Current model time: %f' % waves.get_current_time()
val = waves.get_value(angle_name)
print 'The current wave angle is: %f' % val[0]

import numpy as np

number_of_time_steps = 400
angles = np.empty(number_of_time_steps)
for time in xrange(number_of_time_steps):
    waves.update()
    angles[time] = waves.get_value(angle_name)

import matplotlib.pyplot as plt

plt.plot(np.array(angles) * 180 / np.pi)
plt.xlabel('Time (days)')
plt.ylabel('Incoming wave angle (degrees)')

plt.hist(np.array(angles) * 180 / np.pi, bins=25)
plt.xlabel('Incoming wave angle (degrees)')
plt.ylabel('Number of occurences')

