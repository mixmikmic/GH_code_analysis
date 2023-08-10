from netCDF4 import Dataset

url = ('http://geoport.whoi.edu/thredds/dodsC/usgs/data2/rsignell/gdrive/'
       'nsf-alpha/Data/MIT_MSEAS/MSEAS_Tides_20160317/mseas_tides_2015071612_2015081612_01h.nc')

nc = Dataset(url)

vtime = nc['time']
coords = nc['vgrid2']
vbaro = nc['vbaro']

itime = -1

import iris
iris.FUTURE.netcdf_no_unlimited = True

time = iris.coords.DimCoord(vtime[itime],
                            var_name='time',
                            long_name=vtime.long_name,
                            standard_name='longitude',
                            units=vtime.units)

longitude = iris.coords.AuxCoord(coords[:, :, 0],
                                 var_name='vlat',
                                 standard_name='longitude',
                                 units='degrees')

latitude = iris.coords.AuxCoord(coords[:, :, 1],
                                var_name='vlon',
                                standard_name='latitude',
                                units='degrees')

import numpy as np

u = vbaro[itime, :, :, 0]
u_cube = iris.cube.Cube(np.broadcast_to(u, (1,) + u.shape),
                        units=vbaro.units,
                        long_name=vbaro.long_name,
                        var_name='u',
                        standard_name='barotropic_eastward_sea_water_velocity',
                        dim_coords_and_dims=[(time, 0)],
                        aux_coords_and_dims=[(latitude, (1, 2)),
                                             (longitude, (1, 2))])

v = vbaro[itime, :, :, 1]
v_cube = iris.cube.Cube(np.broadcast_to(v, (1,) + v.shape),
                        units=vbaro.units,
                        long_name=vbaro.long_name,
                        var_name='v',
                        standard_name='barotropic_northward_sea_water_velocity',
                        dim_coords_and_dims=[(time, 0)],
                        aux_coords_and_dims=[(latitude, (1, 2)),
                                             (longitude, (1, 2))])

cubes = iris.cube.CubeList([u_cube, v_cube])

iris.save(cubes, 'hops.nc')

nc2 = netCDF4.Dataset('hops.nc','r+')
nc2

