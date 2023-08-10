#! pip install git+https://github.com/xgcm/xgcm.git

import xarray as xr
import numpy as np
import xgcm
from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10,6)

#!wget http://www.ldeo.columbia.edu/~rpa/mitgcm_example_dataset.nc

dir(xgcm.grid.Grid)

ds = xr.open_dataset('mitgcm_example_dataset.nc')
ds

ds.Eta

ds.Eta[0].plot.contourf(levels=30)

surf_mask_c = ds.hFacC[0] > 0
ds.Eta[0].where(surf_mask_c).plot.contourf(levels=30)

get_ipython().run_line_magic('pinfo', 'xgcm.Grid')

ds.XC.attrs

grid = xgcm.Grid(ds, periodic=['X', 'Y'])
grid

get_ipython().run_line_magic('pinfo', 'grid.interp')

ds.THETA.dims

theta_x_interp = grid.interp(ds.THETA, 'X')
theta_x_interp

ds.THETA

theta_z_interp = grid.interp(ds.THETA, 'Z', boundary='extend')
theta_z_interp

ds.THETA[0,5].plot()

theta_x_interp[0,5,].plot()

ds.THETA.sel(XC=200, YC=45,
             method='nearest').plot.line(y='Z', marker='o',
                                         label='original')
theta_z_interp.sel(XC=200, YC=45,
                   method='nearest').plot.line(y='Zl', marker='o',
                                               label='interpolated')
# plt.ylim([-500, 0])
plt.legend()

# an example of calculating kinetic energy
ke = 0.5*(grid.interp((ds.U*ds.hFacW)**2, 'X') + 
          grid.interp((ds.V*ds.hFacS)**2, 'Y'))
print(ke)
ke[0,0].plot()

zeta = (-grid.diff(ds.U * ds.dxC, 'Y') +
         grid.diff(ds.V * ds.dyC, 'X'))/ds.rAz
zeta

zeta_bt = (zeta * ds.drF).sum(dim='Z')
zeta_bt.plot(vmax=2e-4)

u_bt = (ds.U * ds.hFacW * ds.drF).sum(dim='Z')
v_bt = (ds.V * ds.hFacS * ds.drF).sum(dim='Z')
zeta_bt_alt = (-grid.diff(u_bt * ds.dxC, 'Y') + grid.diff(v_bt * ds.dyC, 'X'))/ds.rAz
zeta_bt_alt.plot(vmax=2e-4)

strain = (grid.diff(ds.U * ds.dyG, 'X') - grid.diff(ds.V * ds.dxG, 'Y')) / ds.rA
strain[0,0].plot()

psi = grid.cumsum(-u_bt * ds.dyG, 'Y', boundary='fill')
psi

(psi[0] / 1e6).plot.contourf(levels=np.arange(-160, 40, 5))

adv_flux_div = (grid.diff(ds.ADVx_TH, 'X') +
                grid.diff(ds.ADVy_TH, 'Y') +
                grid.diff(ds.ADVr_TH, 'Z', boundary='fill'))
adv_flux_div

diff_flux_div = (grid.diff(ds.DFxE_TH, 'X') +
                grid.diff(ds.DFyE_TH, 'Y') +
                grid.diff(ds.DFrE_TH + ds.DFrI_TH, 'Z', boundary='fill'))
diff_flux_div

diff_flux_div.sum(dim=['XC', 'YC']).plot.line(y='Z', marker='.', label='diffusion')
adv_flux_div.sum(dim=['XC', 'YC']).plot.line(y='Z', marker='.', label='advection')
plt.grid()
plt.legend()







