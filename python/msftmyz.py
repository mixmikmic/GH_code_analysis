import matplotlib.pyplot as plt
import iris
import iris.plot as iplt
import numpy
import iris.coord_categorisation

get_ipython().magic('matplotlib inline')

infile = '/g/data/ua6/DRSv2/CMIP5/CSIRO-Mk3-6-0/historical/mon/ocean/r1i1p1/msftmyz/latest/msftmyz_Omon_CSIRO-Mk3-6-0_historical_r1i1p1_185001-200512.nc'

cube = iris.load_cube(infile, 'ocean_meridional_overturning_mass_streamfunction')

print(cube)

dim_coord_names = [coord.name() for coord in cube.dim_coords]
print(dim_coord_names)

aux_coord_names = [coord.name() for coord in cube.aux_coords]
print(aux_coord_names)

mf_cube = cube[:, -1, : ,:]

mf_cube

mf_clim_cube = mf_cube.collapsed('time', iris.analysis.MEAN)

mf_clim_cube

iplt.contour(mf_clim_cube, colors='k')
# plt.clabel(contour_plot) fmt='%.1f')
plt.show()

depth_constraint = iris.Constraint(depth= lambda cell: cell <= 250)
lat_constraint = iris.Constraint(latitude=lambda cell: -30.0 <= cell < 30.0)

tropics_cube = mf_clim_cube.extract(depth_constraint & lat_constraint)

iplt.contourf(tropics_cube, cmap='RdBu_r',
              levels=[-5e+10, -4e+10, -3e+10, -2e+10, -1e+10, 0, 1e+10, 2e+10, 3e+10, 4e+10, 5e+10],
              extend='both')
plt.show()

tropics_cube.data.max()

sh_lat_constraint = iris.Constraint(latitude=lambda cell: -30.0 <= cell < 0.0)
nh_lat_constraint = iris.Constraint(latitude=lambda cell: 0.0 < cell <= 30.0)

sh_cube = mf_cube.extract(depth_constraint & sh_lat_constraint)
nh_cube = mf_cube.extract(depth_constraint & nh_lat_constraint)

sh_cube

def convert_to_annual(cube, full_months=False, aggregation='mean'):
    """Convert data to annual timescale.
    Args:
      cube (iris.cube.Cube)
      full_months(bool): only include years with data for all 12 months
    """

    iris.coord_categorisation.add_year(cube, 'time')
    iris.coord_categorisation.add_month(cube, 'time')

    if aggregation == 'mean':
        aggregator = iris.analysis.MEAN
    elif aggregation == 'sum':
        aggregator = iris.analysis.SUM

    cube = cube.aggregated_by(['year'], aggregator)

    if full_months:
        cube = cube.extract(iris.Constraint(month='Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec'))
  
    cube.remove_coord('year')
    cube.remove_coord('month')

    return cube

sh_cube = convert_to_annual(sh_cube)
nh_cube = convert_to_annual(nh_cube)

sh_metric = sh_cube.collapsed(['depth', 'latitude'], iris.analysis.MEAN) # weights=grid_areas)
nh_metric = nh_cube.collapsed(['depth', 'latitude'], iris.analysis.MEAN) # weights=grid_areas)

iplt.plot(sh_metric)
iplt.plot(nh_metric)
plt.show()

iplt.plot(nh_metric - sh_metric)
plt.show()



