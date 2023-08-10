get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')

dst_gridded_data = "../data/obs.usa_subset.nc"
src_gridded_data = "../data/fgm.REF.1980-2010.ensemble.nc"

get_ipython().system('cdo griddes {dst_gridded_data} > dst.grid')
get_ipython().system('cdo griddes {src_gridded_data} > src.grid')

get_ipython().system('cdo genycon,dst.grid {src_gridded_data} con.weights.nc')
get_ipython().system('cdo genbil,dst.grid {src_gridded_data} bil.weights.nc')

x = src_data.unstack('cell')
x['lat']

weights

import numpy as np
import xarray as xr

field = 'U'
src_data = xr.open_dataset(src_gridded_data)[field]
src_data = src_data.stack(cell=['lat', 'lon'])

# Stacking always produces the trailing dimension in a dataset
src_array = src_data.values
leading_shape, n_cells_src = src_array.shape[:-1], src_array.shape[-1]

weights = xr.open_dataset("con.weights.nc")
n_cells_dst = len(weights.dst_grid_center_lat)
dst_array = np.empty(list(leading_shape) + [n_cells_dst, ])

num_links = len(weights.src_address)
dst_address = weights.dst_address.values - 1
src_address = weights.src_address.values - 1
remap_matrix = weights.remap_matrix.values
# dst_area = weights.dst_grid_area.values
# dst_frac = weights.dst_grid_frac.values

for n in range(num_links):
    dst_addr, src_addr = dst_address[n], src_address[n]
    dst_array[:, dst_addr] += (remap_matrix[n]*src_array[:, src_addr])#/(dst_area[dst_addr]*dst_frac[dst_addr])

from darpy import copy_attrs, append_history
get_ipython().run_line_magic('pinfo', 'append_history')

template = xr.open_dataset(dst_gridded_data)[['lon', 'lat']].copy()
template['time'] = src_data['time']
tgt_stacked = tgt.stack(cell=['lat', 'lon'])
tgt_stacked[field] = (('cell', ), dst_array)
tgt = tgt_stacked.unstack('cell')

from darpy.plot import multipanel_figure, PROJECTION, infer_cmap_params
from stat_pm25.plot import STATE_PROJ, add_usa_states

fig, axs = multipanel_figure(1, 2, aspect=2, projection=STATE_PROJ)
axs = axs.ravel()

cmap_params = infer_cmap_params(tgt[field])
del cmap_params['cnorm']
del cmap_params['levels']
kws = dict(cmap='spectral', transform=PROJECTION, infer_intervals=True)
kws.update(cmap_params)

src_data.isel(time=0).plot.pcolormesh('lon', 'lat', ax=axs[0],**kws)
tgt[field].plot.pcolormesh('lon', 'lat', ax=axs[1], **kws)
for ax in axs:
    _ = add_usa_states(ax, facecolor='None')

isinstance('s', str)

from stat_pm25.regrid import remap_dataset

x = remap_dataset(xr.open_dataset("../data/fgm.all_cases.usa_subset.nc"),
              xr.open_dataset(dst_gridded_data),
              weights)

x['PM25'].isel(time=0, dec=2).plot.pcolormesh('lon', 'lat', col='pol', row='ic', infer_intervals=True)



