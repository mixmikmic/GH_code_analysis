get_ipython().magic('matplotlib inline')

from cmt.components import Cem
cem = Cem()

print cem.get_output_var_names()

cem.get_input_var_names()

angle_name = 'sea_surface_water_wave__azimuth_angle_of_opposite_of_phase_velocity'

print "Data type: %s" % cem.get_var_type(angle_name)
print "Units: %s" % cem.get_var_units(angle_name)
print "Grid id: %d" % cem.get_var_grid(angle_name)
print "Number of elements in grid: %d" % cem.get_grid_size(0)
print "Type of grid: %s" % cem.get_grid_type(0)

cem.initialize(None)

import numpy as np

cem.set_value("sea_surface_water_wave__height", 2.)
cem.set_value("sea_surface_water_wave__period", 7.)
cem.set_value("sea_surface_water_wave__azimuth_angle_of_opposite_of_phase_velocity", 0. * np.pi / 180.)

grid_id = cem.get_var_grid('sea_water__depth')

grid_type = cem.get_grid_type(grid_id)
grid_rank = cem.get_grid_rank(grid_id)
print 'Type of grid: %s (%dD)' % (grid_type, grid_rank)

spacing = np.empty((grid_rank, ), dtype=float)

shape = cem.get_grid_shape(grid_id)
cem.get_grid_spacing(grid_id, out=spacing)

print 'The grid has %d rows and %d columns' % (shape[0], shape[1])
print 'The spacing between rows is %f and between columns is %f' % (spacing[0], spacing[1])

z = np.empty(shape, dtype=float)
cem.get_value('sea_water__depth', out=z)

def plot_coast(spacing, z):
    import matplotlib.pyplot as plt
    
    xmin, xmax = 0., z.shape[1] * spacing[0] * 1e-3
    ymin, ymax = 0., z.shape[0] * spacing[1] * 1e-3

    plt.imshow(z, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap='ocean')
    plt.colorbar().ax.set_ylabel('Water Depth (m)')
    plt.xlabel('Along shore (km)')
    plt.ylabel('Cross shore (km)')

plot_coast(spacing, z)

qs = np.zeros_like(z)
qs[0, 100] = 1250

cem.get_var_units('land_surface_water_sediment~bedload__mass_flow_rate')

cem.get_time_step(), cem.get_time_units(), cem.get_current_time()

for time in xrange(3000):
    cem.set_value('land_surface_water_sediment~bedload__mass_flow_rate', qs)
    cem.update_until(time)

cem.get_value('sea_water__depth', out=z)

cem.get_current_time()

plot_coast(spacing, z)

val = np.empty((5, ), dtype=float)
cem.get_value("basin_outlet~coastal_center__x_coordinate", val)
val / 100.

qs[0, 150] = 1500
for time in xrange(3750):
    cem.set_value('land_surface_water_sediment~bedload__mass_flow_rate', qs)
    cem.update_until(time)
    
cem.get_value('sea_water__depth', out=z)

plot_coast(spacing, z)

qs.fill(0.)
for time in xrange(4000):
    cem.set_value('land_surface_water_sediment~bedload__mass_flow_rate', qs)
    cem.update_until(time)
    
cem.get_value('sea_water__depth', out=z)

plot_coast(spacing, z)

