from netCDF4 import Dataset

#url = ('http://geoport.whoi.edu/thredds/dodsC/usgs/data2/rsignell/gdrive/'
#       'nsf-alpha/Data/MIT_MSEAS/MSEAS_Tides_20160317/mseas_tides_2015071612_2015081612_01h.nc')
url = ('/usgs/data2/rsignell/gdrive/'
       'nsf-alpha/Data/MIT_MSEAS/MSEAS_Tides_20160317/mseas_tides_2015071612_2015081612_01h.nc')
nc = Dataset(url)

vtime = nc['time']
coords = nc['vgrid2']
vbaro = nc['vbaro']

import iris
iris.FUTURE.netcdf_no_unlimited = True

longitude = iris.coords.AuxCoord(coords[:, :, 0],
                                 var_name='vlat',
                                 long_name='lon values',
                                 units='degrees')

latitude = iris.coords.AuxCoord(coords[:, :, 1],
                                var_name='vlon',
                                long_name='lat values',
                                units='degrees')

# Dummy Dimension coordinate to avoid default names.
# (This is either a bug in CF or in iris. We should not need to do this!)
lon = iris.coords.DimCoord(range(866),
                           var_name='x',
                           long_name='lon_range',
                           standard_name='longitude')

lat = iris.coords.DimCoord(range(1032),
                           var_name='y',
                           long_name='lat_range',
                           standard_name='latitude')

vbaro.shape

import numpy as np

u_cubes = iris.cube.CubeList()
v_cubes = iris.cube.CubeList()


for k in range(vbaro.shape[0]):  # vbaro.shape[0]
    time = iris.coords.DimCoord(vtime[k],
                                var_name='time',
                                long_name=vtime.long_name,
                                standard_name='time',
                                units=vtime.units)
    
    u = vbaro[k, :, :, 0]
    u_cubes.append(iris.cube.Cube(np.broadcast_to(u, (1,) + u.shape),
                                  units=vbaro.units,
                                  long_name=vbaro.long_name,
                                  var_name='u',
                                  standard_name='barotropic_eastward_sea_water_velocity',
                                  dim_coords_and_dims=[(time, 0), (lon, 1), (lat, 2)],
                                  aux_coords_and_dims=[(latitude, (1, 2)),
                                                       (longitude, (1, 2))]))

    v = vbaro[k, :, :, 1]
    v_cubes.append(iris.cube.Cube(np.broadcast_to(v, (1,) + v.shape),
                                  units=vbaro.units,
                                  long_name=vbaro.long_name,
                                  var_name='v',
                                  standard_name='barotropic_northward_sea_water_velocity',
                                  dim_coords_and_dims=[(time, 0), (lon, 1), (lat, 2)],
                                  aux_coords_and_dims=[(longitude, (1, 2)),
                                                       (latitude, (1, 2))]))

u_cube = u_cubes.concatenate_cube()
v_cube = v_cubes.concatenate_cube()

cubes = iris.cube.CubeList([u_cube, v_cube])

iris.save(cubes, 'hops.nc')

get_ipython().system('ncdump -h hops.nc')



