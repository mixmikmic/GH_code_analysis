import matplotlib.pyplot as plt
import iris
import iris.plot as iplt
import iris.quickplot as qplt

import glob
from iris.experimental.equalise_cubes import equalise_attributes

import numpy

get_ipython().magic('matplotlib inline')

infile = '/Users/irv033/Desktop/results/energy_budget/energy-budget_yr_CSIRO-Mk3-6-0_historicalMisc_r1i1p4_all.nc'
cubes = iris.load(infile)

print(cubes)

rsdt_nh_cube = iris.load_cube(infile, 'TOA Incident Shortwave Radiation nh sum')
qplt.plot(rsdt_nh_cube)
plt.show()

rsdt_sh_cube = iris.load_cube(infile, 'TOA Incident Shortwave Radiation sh sum')
qplt.plot(rsdt_sh_cube)
plt.show()

rsut_nh_cube = iris.load_cube(infile, 'TOA Outgoing Shortwave Radiation nh sum')
rsut_sh_cube = iris.load_cube(infile, 'TOA Outgoing Shortwave Radiation sh sum')

qplt.plot(rsut_nh_cube, label='NH')
qplt.plot(rsut_sh_cube, label='SH')
plt.legend()
plt.show()

rlut_nh_cube = iris.load_cube(infile, 'TOA Outgoing Longwave Radiation nh sum')
rlut_sh_cube = iris.load_cube(infile, 'TOA Outgoing Longwave Radiation sh sum')

qplt.plot(rlut_nh_cube, label='NH')
qplt.plot(rlut_sh_cube, label='SH')
plt.legend()
plt.show()

rsds_nh_cube = iris.load_cube(infile, 'Surface Downwelling Shortwave Radiation nh ocean sum')
qplt.plot(rsds_nh_cube, label='NH')
plt.legend()
plt.show()

rsds_sh_cube = iris.load_cube(infile, 'Surface Downwelling Shortwave Radiation sh ocean sum')

qplt.plot(rsds_sh_cube, label='SH')
plt.legend()
plt.show()

rsus_nh_cube = iris.load_cube(infile, 'Surface Upwelling Shortwave Radiation nh ocean sum')
qplt.plot(rsus_nh_cube, label='NH')
plt.legend()
plt.show()

ohc_nh_cube = iris.load_cube(infile, 'ocean heat content nh sum')
ohc_sh_cube = iris.load_cube(infile, 'ocean heat content sh sum')

ohc_global_cube = (ohc_nh_cube + ohc_sh_cube) * 60 * 60 * 24 * 365

qplt.plot(ohc_global_cube)
plt.show()

ohc_files = glob.glob('/Users/irv033/Desktop/results/energy_budget/ohc_Oyr_CSIRO-Mk3-6-0_historicalMisc_r1i1p4_*.nc')
ohc_cubes = iris.load(ohc_files)

ohc_cubes

equalise_attributes(ohc_cubes)
iris.util.unify_time_units(ohc_cubes)
ohc_cube = ohc_cubes.concatenate_cube()

def multiply_by_area(cube):
    """Multiply by cell area."""

    if not cube.coord('latitude').has_bounds():
        cube.coord('latitude').guess_bounds()
    if not cube.coord('longitude').has_bounds():
        cube.coord('longitude').guess_bounds()
    area_data = iris.analysis.cartography.area_weights(cube)
    units = str(cube.units)
    cube.units = units.replace('m-2', '')
    cube.data = cube.data * area_data

    return cube

ohc_J_cube = multiply_by_area(ohc_cube)

ohc_J_cube

global_ohc_cube = ohc_J_cube.collapsed(['latitude', 'longitude'], iris.analysis.SUM)

global_ohc_cube.data

dohc = numpy.cumsum(global_ohc_cube.data - global_ohc_cube.data[0])
dt = numpy.arange(0, 365 * 163, 365) * 60 * 60 * 24

dohc_dt_data = numpy.zeros(163)
dohc_dt_data[1:] = dohc[1:] / dt[1:]

hfds_sh_cube = iris.load_cube(infile, 'Downward Heat Flux at Sea Water Surface sh ocean sum')

hfds_sh_cube.data.mean()

dohc_dt_data[1:].mean()

dohc_dt_data[1:].max()

dohc_dt_data[1:].min()

len(numpy.arange(0, 365.25 * 163, 365.25))



