get_ipython().system('cat environment.yml')

import numpy as np
import pandas as pd
import xarray as xr
import pyart
from skewt import SkewT
from pyart_radar_tools import *

ID = 'KICX'

t_start = '2015-09-14 18:00:00'
t_end = '2015-09-15 00:00:00'

paths = data_download(ID, t_start, t_end)

sounding = SkewT.Sounding('./tmp/KFGZ_2015-09-15_00_sounding.txt')

fields = ['rain', 'r_kdp', 'r_z']

def process_radar(path, sounding, run_QAQC=True,
                  min_dist_km=5, max_dist_km=250,
                  sw_vel=True, max_time_diff=30):
    
    radar = pyart.io.read(path)
    radar = extract_low_sweeps(radar)

    # run some QAQC:
    if run_QAQC:
        start_gate = get_gate_index(radar, dist_km=min_dist_km)
        end_gate = get_gate_index(radar, dist_km=max_dist_km)
        QAQC_mask = construct_QAQC_mask(radar, start_gate, end_gate,
                                        sw_vel=sw_vel,
                                        max_time_diff=max_time_diff)

    # get out just the sweeps with differential phase values
    radar = extract_field_sweeps(radar, field='differential_phase')
    radar = calculate_hidro_rain(radar, sounding)
    radar = calculate_rain_nexrad(radar)
    radar = calculate_rain_kdp(radar)
    if run_QAQC:
        for field in ['rain', 'r_kdp', 'r_z']:
            radar = interpolate_radially(radar, field, QAQC_mask,
                                         start_gate, end_gate)
    return radar

get_ipython().run_cell_magic('time', '', "nx = 400\nny = 400\nsweep_times = []\ngrid_paths = []\n\nfor path in paths:\n    print('processing', path)\n    radar = process_radar(path, sounding)\n\n    for sweep in range(radar.nsweeps):\n        end_sweep_time = get_end_sweep_time(radar, sweep)\n        sweep_times.append(end_sweep_time)\n        \n        # extract sweep\n        sweepn = radar.extract_sweeps([sweep])\n        \n        # grid sweep\n        m = pyart.map.grid_from_radars([sweepn], grid_shape=(1, ny, nx),\n                                       grid_limits=((0, 10000),\n                                                    (-ny/2.*1000, ny/2.*1000),\n                                                    (-nx/2.*1000, nx/2.*1000)),\n                                       fields=fields)\n        if len(sweep_times) == 1:\n            continue\n        \n        # calculate difference between last sweep time and current sweep time\n        diff_hours = ((sweep_times[-1] - sweep_times[-2]).seconds)/3600.\n        \n\n        for field in fields:\n            # weight the data by that difference - converting from rate to accumulation\n            m.fields[field]['data'] *= diff_hours\n            m.fields[field]['units'] = 'mm'\n\n        t = end_sweep_time.isoformat()[:19]\n        t = t.replace('-', '').replace('T', '_').replace(':', '')\n        grid_path = './tmp/{ID}{t}_grid.nc'.format(ID=ID, t=t)\n        print('writing', grid_path)\n        pyart.io.write_grid(grid_path, m, write_point_lon_lat_alt=True)\n        grid_paths.append(grid_path)")

import os

nx = 400
ny = 400

grid_paths = ['./tmp/{f}'.format(f=f) for f in os.listdir('./tmp') if '.nc' in f]

rain = np.zeros((ny, nx))
r_z = np.zeros((ny, nx))
r_kdp = np.zeros((ny, nx))
rain_mask = np.zeros((ny, nx))
r_z_mask = np.zeros((ny, nx))
r_kdp_mask = np.zeros((ny, nx))

for grid_path in grid_paths:
    m = pyart.io.read_grid(grid_path)
    #rain
    rain += np.ma.filled(m.fields['rain']['data'][0], 0)
    rain_mask += m.fields['rain']['data'][0].mask
    #r_z
    r_z += np.ma.filled(m.fields['r_z']['data'][0], 0)
    r_z_mask += m.fields['r_z']['data'][0].mask
    #r_kdp
    r_kdp += np.ma.filled(m.fields['r_kdp']['data'][0], 0)
    r_kdp_mask += m.fields['r_kdp']['data'][0].mask

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
c = ax.imshow(rain, origin='lower', interpolation='None', vmin=0)
plt.colorbar(c)
plt.show()

lat = m.point_latitude['data'][0]
lon = m.point_longitude['data'][0]

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import GoogleTiles

class ShadedReliefESRI(GoogleTiles):
    # shaded relief
    def _image_url(self, tile):
        x, y, z = tile
        url = ('https://server.arcgisonline.com/ArcGIS/rest/services/'                'World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg').format(
               z=z, y=y, x=x)
        return url

states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
plt.figure(figsize=(8,8))
ax = plt.axes(projection=ccrs.PlateCarree())
cl = ax.contour(lon, lat, rain)
cl.clabel(fmt='%1.1f')
ax.gridlines(draw_labels=True)
ax.add_image(ShadedReliefESRI(), 8)
ax.add_feature(states_provinces, edgecolor='gray')

