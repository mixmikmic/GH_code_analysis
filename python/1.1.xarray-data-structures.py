import numpy as np
import xarray as xr

myvar = np.random.random(size=(2, 3, 6))
myvar

my_da = xr.DataArray(myvar)
my_da

# Adding labels/metadata
my_da = xr.DataArray(myvar,
                     dims=('lat', 'lon', 'time'),
                     coords={'lat': [15., 30.], 'lon': [-110., -115., -120.]},
                     attrs={'long_name': 'temperature', 'units': 'C'},
                     name='temp')
my_da

# The underlying data is still there:
my_da.data

# Datasets are dict-like containers of DataArrays

xr.Dataset()

my_ds = xr.Dataset({'temperature': my_da})
# also equivalent to:
# my_da.to_dataset()
my_ds

my_ds['precipitation'] = xr.DataArray(np.random.random(myvar.shape),
                                      dims=('lat', 'lon', 'time'),
                                      coords={'lat': [15., 30.], 'lon': [-110., -115., -120.]},
                                      attrs={'long_name': 'precipitation', 'units': 'mm'},
                                      name='pcp') 

my_ds.attrs['history'] = 'created for the xarray tutorial'

my_ds

