import Nio
import Ngl
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# extract needed variables from CESM output (this is a minimal set) while concatenating or make
# a separate file - here is what I did:
#  ncrcat -v U,hyam,hybm,P0,PS 1903*.nc U.nc

ds = xr.open_dataset('U.nc',decode_times=False)
ds

hbcofa = ds.hyam.values
hbcofb = ds.hybm.values
plevo = [1000,900,250]
psfc = ds.PS
intyp = 1
kxtrp = False  # True: extrapolate, # False: don't
p0 = ds.P0.values
np.shape(ds.U.values)

u = Ngl.vinth2p(ds.U, hbcofa, hbcofb, plevo, psfc, intyp, p0, 1, kxtrp)

np.shape(u)

ds_out = xr.Dataset({'lon': ('lon', ds.lon), 'lat': ('lat', ds.lat),                     'plev': ('plev', plevo), 'time': ('time',ds.time)})

ds_out['u'] = (['time','plev','lat','lon'],u)
ds_out.lon.attrs = ds.lon.attrs
ds_out.lat.attrs = ds.lat.attrs
ds_out.time.attrs = ds.time.attrs
ds_out.plev.attrs = [('units','mb')]
ds_out.u.attrs = [('units','m/s')]
ds_out.u.attrs = [('_FillValue',1e30)]

ds2 = ds_out.mean('time').sel(plev = 900)
ds2.where(ds2.u < 10000).u.plot()
plt.savefig('foo.png')

# note that apple formatted my hard drive as case-insensitive!!!

ds_out.to_netcdf('/Users/naomi/projects/OCPhelp/NaomiExamples/U-pld.nc')



