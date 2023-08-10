import xarray as xr

# load a sample dataset
ds = xr.tutorial.load_dataset('air_temperature')
ds

t = ds['air'].data  # numpy array
t

t.shape

# extract a time-series for one spatial location
t[:, 10, 20]

da = ds['air']

# numpy style indexing still works (but preserves the labels/metadata)
da[:, 10, 20]

# Positional indexing using dimension names
da.isel(lat=10, lon=20)

# Label-based indexing
da.sel(lat=50., lon=250.)

# Nearest neighbor lookups
da.sel(lat=52.25, lon=251.8998, method='nearest')

# all of these indexing methods work on the dataset too, e.g.:
ds.sel(lat=52.25, lon=251.8998, method='nearest')

# generate a coordinates for a transect of points
lat_points = xr.DataArray([52, 52.5, 53], dims='points')
lon_points = xr.DataArray([250, 250, 250], dims='points')

# nearest neighbor selection along the transect
da.sel(lat=lat_points, lon=lon_points, method='nearest')

da

arr = da.isel(time=0, lat=slice(5, 10), lon=slice(7, 11))
arr

part = arr[:-1]
part

# default behavior is an "inner join"
(arr + part) / 2

# we can also use an outer join
with xr.set_options(arithmetic_join="outer"):
    print((arr + part) / 2)
    
# notice that missing values (nan) were inserted



