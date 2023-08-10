import numpy as np
import glob
import xarray as xr
import pandas as pd
dirpet = '/g/data/oe9/user/rg6346/PET_BOM_AWRA/pet_avg_Actual_month.nc'
petnc = xr.open_dataset(dirpet)
petnc = petnc.rename({'e0_avg':'pet'})
petnc = petnc.where(petnc>=0, np.nan)
petnc = petnc.squeeze()
petnc

mask_path = '/g/data/oe9/project/team-drip/MDB_MASK/MASK_ARRAY_AWRA.nc'
petmask = xr.open_dataarray(mask_path)
petnc = petnc.where(petmask, drop=True)
petnc

get_ipython().magic('matplotlib inline')
petnc.pet.isel(time=range(0,4)).plot.imshow(col='time', robust = True, col_wrap=4, cmap = 'RdYlGn')



