get_ipython().system('cd data && wget ftp://ftp.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.2018030806/gfs.t06z.pgrb2.0p25.f000')

import xarray as xr

ds = xr.open_dataset("data/gfs.t06z.pgrb2.0p25.f000", engine="pynio")

temperature = ds['TMP_P0_L1_GLL0']
temperature

get_ipython().run_line_magic('matplotlib', 'inline')
temperature.plot(figsize=(12,5));

