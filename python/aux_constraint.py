import iris
import numpy

import matplotlib.pyplot as plt
import iris.plot as iplt
import iris.quickplot as qplt

get_ipython().magic('matplotlib inline')

infile = '/g/data/ua6/DRSv2/CMIP5/NorESM1-M/historical/mon/ocean/r1i1p1/hfds/latest/hfds_Omon_NorESM1-M_historical_r1i1p1_185001-200512.nc'

cube = iris.load_cube(infile, 'surface_downward_heat_flux_in_sea_water')

print(cube)

#qplt.contourf(cube[0, ::])
#plt.show()

fig, ax = plt.subplots()
im = ax.imshow(cube[0, ::].data)
fig.colorbar(im)

plt.show()

cube.data.mean()

aux_coord_names = [coord.name() for coord in cube.aux_coords]
print(aux_coord_names)

lat_subset = lambda cell: cell >= 0.0    
lat_constraint = iris.Constraint(latitude=lat_subset)

cube.extract(lat_constraint)

def create_mask(latitude_array, target_shape, hemisphere):
    """Create mask from the latitude auxillary coordinate"""

    target_ndim = len(target_shape)

    if hemisphere == 'nh':
        mask_array = numpy.where(latitude_array >= 0, False, True)
    elif hemisphere == 'sh':
        mask_array = numpy.where(latitude_array < 0, False, True)

    mask = broadcast_array(mask_array, [target_ndim - 2, target_ndim - 1], target_shape)
    assert mask.shape == target_shape 

    return mask


def broadcast_array(array, axis_index, shape):
    """Broadcast an array to a target shape.
    
    Args:
      array (numpy.ndarray)
      axis_index (int or tuple): Postion in the target shape that the 
        axis/axes of the array corresponds to
          e.g. if array corresponds to (depth, lat, lon) in (time, depth, lat, lon)
          then axis_index = [1, 3]
          e.g. if array corresponds to (lat) in (time, depth, lat, lon)
          then axis_index = 2
      shape (tuple): shape to broadcast to
      
    For a one dimensional array, make start_axis_index = end_axis_index
    
    """

    if type(axis_index) in [float, int]:
        start_axis_index = end_axis_index = axis_index
    else:
        assert len(axis_index) == 2
        start_axis_index, end_axis_index = axis_index
    
    dim = start_axis_index - 1
    while dim >= 0:
        array = array[numpy.newaxis, ...]
        array = numpy.repeat(array, shape[dim], axis=0)
        dim = dim - 1
    
    dim = end_axis_index + 1
    while dim < len(shape):    
        array = array[..., numpy.newaxis]
        array = numpy.repeat(array, shape[dim], axis=-1)
        dim = dim + 1

    return array

cube.coord('latitude').points

cube.shape

type(cube.data)

nh_mask = create_mask(cube.coord('latitude').points, cube.shape, 'nh')
ocean_mask = cube.data.mask

complete_mask = nh_mask + ocean_mask

cube.data = numpy.ma.asarray(cube.data)
cube.data.mask = complete_mask

cube.data.mean()

fig, ax = plt.subplots()
im = ax.imshow(cube[0, ::].data)
fig.colorbar(im)

#qplt.contourf(cube[0, ::])
#plt.gca().coastlines()
plt.show()



