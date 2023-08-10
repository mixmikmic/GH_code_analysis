import warnings
# Warnings make for ugly notebooks, ignore them
warnings.filterwarnings('ignore')

import iris
import iris.plot as iplt
import matplotlib.pyplot as plt
import numpy
import dask

fn = 'data/2017-11-21/work01/output/netcdf/discharge_dailyTot_output.nc'

#dask.set_options(get=dask.threaded.get)

# Run dask in multiprocessing mode
#from dask.multiprocessing import get
#dask.set_options(get=get)

# Run dask in distributed mode
# from dask.distributed import Client
# client = Client()
# dask.set_options(get=client.get)
# client

cube = iris.load_cube(fn)

cube

print(cube)

get_ipython().run_line_magic('matplotlib', 'inline')

clim = cube.collapsed('time', iris.analysis.MEAN)

print(clim)

fig = plt.figure(figsize=[30,15])
get_ipython().run_line_magic('time', "iplt.contourf(clim, levels=numpy.arange(0, 10), extend='max')")

rhine_con = iris.Constraint(time=iris.time.PartialDateTime(day=29), 
                            coord_values={
                                'latitude': lambda d: 45 <d < 55, 
                                'longitude': lambda d: 2 < d < 10
                            })

rhine = cube.extract(rhine_con)
print(rhine)

fig = plt.figure(figsize=[30,15])
iplt.pcolormesh(rhine)



