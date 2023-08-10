import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (8,4)
import xgcm
from glob import glob

files = '/net/kage/d5/datasets/ERAInterim/monthly/Surface/*10.nc'
ds1 = xr.open_mfdataset(glob(files),decode_times=False)
ds = ds1.mean('T')
ds

delx = ds.X[1]-ds.X[0]; dely = ds.Y[1]-ds.Y[0]
xh = ds.X + delx/2.0; ds['Xh'] = ('Xh',xh)
yh = ds.Y + dely/2.0; ds['Yh'] = ('Yh',yh)
print('X grid:',ds.X.data[0:10],'\nXh grid:',ds.Xh.data[0:10])
print('Y grid:',ds.Y.data[0:10],'\nYh grid:',ds.Yh.data[0:10])

grid = xgcm.Grid(ds, coords={'X': {'center': 'Xh', 'right': 'X'},
                             'Y': {'center': 'Yh', 'right': 'Y'}}, periodic=['X','Y'])
grid

coslat = np.cos(np.deg2rad(ds.Y))
coslath = np.cos(np.deg2rad(ds.Yh))
meterPdegree = 111000.0

dxm = delx*meterPdegree
dym = dely*meterPdegree

# calculate the divergence of the 10m winds, (u10,v10)
# Method I
dudx = grid.diff(ds.u10,axis='X')/dxm  
dvdy = grid.diff(coslat*ds.v10,axis='Y')/dym
divergence = (grid.interp(dudx,axis='X')+ grid.interp(dvdy,axis='Y'))/coslat
ds['divergence1'] = divergence

# same calculation, 
# Method II (preferred)
uh = grid.interp(ds.u10,axis='X')
vh = grid.interp(ds.v10,axis='Y')
dudx = grid.diff(uh,axis='X')/dxm  
dvdy = grid.diff(coslath*vh,axis='Y')/dym
divergence = (dudx + dvdy)/coslat
ds['divergence'] = divergence

div1 = ds.sel(X=240,Y=slice(30,-30)).divergence1
div2 = ds.sel(X=240,Y=slice(30,-30)).divergence
div_diff = (div2 - div1)*100
div1.plot(marker='o', label="Method I")
div2.plot(label="Method II")
div_diff.plot(label="100*difference")
plt.legend()

del ds['divergence1']

# calculate the gradient of the mean 10m wind speed, si10
ds.si10.plot()

Fhx = grid.interp(ds.si10,axis='X')
Fhy = grid.interp(ds.si10,axis='Y')
ds['wnspgradx'] = grid.diff(Fhx,axis='X')/(dxm*coslat)
ds['wnspgrady'] = grid.diff(Fhy,axis='Y')/dym

#plt.quiver(ds.X, ds.Y[5:-5], gradx[0,5:-5], grady[0,5:-5])
ds.sel(X=240,Y=slice(30,-30)).wnspgradx.plot()
ds.sel(X=240,Y=slice(30,-30)).wnspgrady.plot()
plt.legend()

uh = grid.interp(ds.u10,axis='Y')
vh = grid.interp(ds.v10,axis='X')
dudy = grid.diff(uh*coslath,axis='Y')/dym
dvdx = grid.diff(vh,axis='X')/dxm
ds['curl'] = (dvdx - dudy)/coslat
ds

ds.curl[5:-5].plot(vmin = -1e-5, vmax = 1e-5)

