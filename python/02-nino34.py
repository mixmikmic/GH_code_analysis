import xarray
url = 'http://dapds00.nci.org.au/thredds/dodsC/ua6/authoritative/CMIP5/CSIRO-BOM/ACCESS1-0/historical/mon/atmos/Amon/r1i1p1/latest/ts/ts_Amon_ACCESS1-0_historical_r1i1p1_185001-200512.nc'
data = xarray.open_dataset(url)
ts = data['ts']

# Does the data look reasonable?

get_ipython().magic('matplotlib inline')
ts[0,:,:].plot()

# Select the NINO34 area

nino34_area = ts.sel(lat=slice(-5,5), lon=slice(360-170,360-120))
nino34_area[0].plot()

# Get the baseline over 1961 to 1990

baseline = nino34_area.sel(time=slice('1961','1990'))
baseline

# Group the data by month & get time average

baseline_average = baseline.groupby('time.month').mean(dim='time')
baseline_average

# Get the anomaly and area average

anomaly = nino34_area.groupby('time.month') - baseline_average
nino34_index = anomaly.mean(dim=('lat','lon'))
nino34_index.plot()

nino34_index

# To save to disk we need to create a dataset

nino34_dataset = xarray.Dataset({'nino34': nino34_index})
nino34_dataset

# We can copy metadata from the source dataset

nino34_dataset.attrs = data.attrs

from datetime import datetime
notebook = 'https://github.com/ScottWales/swc-climatedata/blob/gh-pages/data/02-xarray-basics.ipynb'
nino34_dataset.attrs['history'] = "{}: nino34 calculated by {}".format(datetime.now(), notebook)

nino34_dataset

# Save with `to_netcdf()`

nino34_dataset.to_netcdf('nino34.nc')

get_ipython().system(' ncdump -h nino34.nc')



