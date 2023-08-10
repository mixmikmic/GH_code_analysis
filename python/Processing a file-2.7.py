import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

#needs to be running under pyart35 environment
import pyart
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from scipy import ndimage, signal, integrate
import time
import copy
import netCDF4
import skfuzzy as fuzz
import datetime
import platform
import fnmatch
import os
from csu_radartools import csu_kdp
get_ipython().magic('matplotlib inline')

my_system = platform.system()
if my_system == 'Darwin':
    top = '/data/sample_sapr_data/sgpstage/sur/'
    soundings_dir = '/data/sample_sapr_data/sgpstage/interp_sonde/'
elif my_system == 'Linux':
    top = '/lcrc/group/earthscience/radar/sgpstage/sur/'
    soundings_dir = '/lcrc/group/earthscience/radar/interp_sonde/'

#python 2
import imp
radar_codes = imp.load_source('radar_codes', '/Users/scollis/projects/AGU_2016/scripts/processing_code.py')

radar_codes.hello_world()

radar = pyart.io.read('/data/sample_sapr_data/sgpstage/sur/20110520/110635.mdv')
print(radar.fields.keys())

i_end = 975
radar.range['data']=radar.range['data'][0:i_end]
for key in radar.fields.keys():
    radar.fields[key]['data']= radar.fields[key]['data'][:, 0:i_end]
radar.ngates = i_end


display = pyart.graph.RadarMapDisplay(radar)
fig = plt.figure(figsize = [10,8])
display.plot_ppi_map('reflectivity', sweep = 2, resolution = 'l',
                    vmin = -10, vmax = 64, mask_outside = False,
                    cmap = pyart.graph.cm.NWSRef)


#guess a whole heap of data
radar_start_date = netCDF4.num2date(radar.time['data'][0], radar.time['units'])
print(radar_start_date)
ymd_string = datetime.datetime.strftime(radar_start_date, '%Y%m%d')
hms_string = datetime.datetime.strftime(radar_start_date, '%H%M%S')
print(ymd_string, hms_string)

z_dict, temp_dict, snr = radar_codes.snr_and_sounding(radar, soundings_dir)
texture =  radar_codes.get_texture(radar)

radar.add_field('sounding_temperature', temp_dict, replace_existing = True)
radar.add_field('height', z_dict, replace_existing = True)
radar.add_field('SNR', snr, replace_existing = True)
radar.add_field('velocity_texture', texture, replace_existing = True)

my_fuzz, cats = radar_codes.do_my_fuzz(radar)

radar.add_field('gate_id', my_fuzz, 
                      replace_existing = True)

sw = 0
min_lon=-99.
max_lon=-96.
min_lat=35.7
max_lat=38.
lon_lines=[-95, -96, -97, -98, -99]
lat_lines=[35,36,37,38]

display = pyart.graph.RadarMapDisplay(radar)

f = plt.figure(figsize = [15,10])
plt.subplot(2, 2, 1) 
lab_colors=['gray','yellow', 'red', 'green', 'cyan' ]
cmap = matplotlib.colors.ListedColormap(lab_colors)
display.plot_ppi_map('gate_id', sweep = sw, 
                     min_lon = min_lon, max_lon = max_lon, min_lat = min_lat, max_lat = max_lat,
                     resolution = 'l', cmap = cmap, vmin = 0, vmax = 5)
cbax=plt.gca()
#labels = [item.get_text() for item in cbax.get_xticklabels()]
#my_display.cbs[-1].ax.set_yticklabels(cats)
tick_locs   = np.linspace(0,len(cats) -1 ,len(cats))+0.5
display.cbs[-1].locator     = matplotlib.ticker.FixedLocator(tick_locs)
display.cbs[-1].formatter   = matplotlib.ticker.FixedFormatter(cats)
display.cbs[-1].update_ticks()
plt.subplot(2, 2, 2) 
display.plot_ppi_map('reflectivity', sweep = sw, vmin = -8, vmax = 64,
                      min_lon = min_lon, max_lon = max_lon, min_lat = min_lat, max_lat = max_lat,
                     resolution = 'l', cmap = pyart.graph.cm.NWSRef)

plt.subplot(2, 2, 3) 
display.plot_ppi_map('velocity_texture', sweep = sw, vmin =0, vmax = 14, 
                     min_lon = min_lon, max_lon = max_lon, min_lat = min_lat, max_lat = max_lat,
                     resolution = 'l', cmap = pyart.graph.cm.NWSRef)
plt.subplot(2, 2, 4) 
display.plot_ppi_map('cross_correlation_ratio', sweep = sw, vmin = .5, vmax = 1,
                      min_lon = min_lon, max_lon = max_lon, min_lat = min_lat, max_lat = max_lat,
                     resolution = 'l', cmap = pyart.graph.cm.Carbone42)

print(radar.fields['gate_id']['notes'])
print(cats)

melt_locations = np.where(radar.fields['gate_id']['data'] == 1)
kinda_cold = np.where(radar.fields['sounding_temperature']['data'] < 0)
fzl_sounding = radar.gate_altitude['data'][kinda_cold].min()
if len(melt_locations[0] > 1):
    fzl_pid = radar.gate_altitude['data'][melt_locations].mean()
    fzl = (fzl_pid + fzl_sounding)/2.0
else:
    fzl = fzl_sounding

if fzl > 5000:
    fzl = 3500.0


phidp, kdp = pyart.correct.phase_proc_lp(radar, 0.0, debug=True, fzl=fzl)

radar.add_field('corrected_differential_phase', phidp,replace_existing = True)
radar.add_field('corrected_specific_diff_phase', kdp,replace_existing = True)

print(radar.fields.keys())

csu_kdp_field, csu_filt_dp, csu_kdp_sd = radar_codes.return_csu_kdp(radar)

radar.add_field('bringi_differential_phase', csu_filt_dp, replace_existing = True)
radar.add_field('bringi_specific_diff_phase', csu_kdp_field, replace_existing = True)
radar.add_field('bringi_specific_diff_phase_sd', csu_kdp_sd, replace_existing = True)


print(radar.fields['gate_id']['notes'])
rain_and_snow = pyart.correct.GateFilter(radar)
rain_and_snow.exclude_all()
rain_and_snow.include_equal('gate_id', 1)
rain_and_snow.include_equal('gate_id', 3)
rain_and_snow.include_equal('gate_id', 4)

display = pyart.graph.RadarMapDisplay(radar)
fig = plt.figure(figsize = [10,8])
display.plot_ppi_map('reflectivity', sweep = 2, resolution = 'l',
                    vmin = -10, vmax = 64, mask_outside = False,
                    cmap = pyart.graph.cm.NWSRef,
                    gatefilter = rain_and_snow)

m_kdp, phidp_f, phidp_r = pyart.retrieve.kdp_proc.kdp_maesaka(radar, 
                                                              gatefilter=rain_and_snow)

radar.add_field('maesaka_differential_phase', m_kdp, replace_existing = True)
radar.add_field('maesaka_forward_specific_diff_phase', phidp_f, replace_existing = True)
radar.add_field('maesaka__reverse_specific_diff_phase', phidp_r, replace_existing = True)

height = radar.gate_altitude
lats = radar.gate_latitude
lons = radar.gate_longitude
lowest_lats = lats['data'][radar.sweep_start_ray_index['data'][0]:radar.sweep_end_ray_index['data'][0],:]
lowest_lons = lons['data'][radar.sweep_start_ray_index['data'][0]:radar.sweep_end_ray_index['data'][0],:]
c1_dis_lat = 36.605
c1_dis_lon = -97.485
cost = np.sqrt((lowest_lons - c1_dis_lon)**2 + (lowest_lats - c1_dis_lat)**2)
index = np.where(cost == cost.min())
lon_locn = lowest_lons[index]
lat_locn = lowest_lats[index]
print(lat_locn, lon_locn)

my_system = platform.system()

if my_system == 'Darwin':
    top = '/data/sample_sapr_data/sgpstage/sur/'
    s_dir = '/data/sample_sapr_data/sgpstage/interp_sonde/'
    odir_r = '/data/sample_sapr_data/agu2016/radars/'
    odir_s = '/data/sample_sapr_data/agu2016/stats/'
    odir_i = '/data/sample_sapr_data/agu2016/images/'
elif my_system == 'Linux':
    top = '/lcrc/group/earthscience/radar/sgpstage/sur/'
    s_dir = '/lcrc/group/earthscience/radar/sgpstage/interp_sonde/'
    odir_r = '/lcrc/group/earthscience/radar/agu2016/radars/'
    odir_s = '/lcrc/group/earthscience/radar/agu2016/stats/'
    odir_i = '/lcrc/group/earthscience/radar/agu2016/images/'


dis_output_location = os.path.join(odir_s,ymd_string)
if not os.path.exists(dis_output_location):
    os.makedirs(dis_output_location)

dis_string = ''
time_of_dis = netCDF4.num2date(radar.time['data'], radar.time['units'])[index[0]][0]
tstring = datetime.datetime.strftime(time_of_dis, '%Y%m%d%H%H%S')
dis_string = dis_string + tstring + ' '
for key in radar.fields.keys():
    dis_string = dis_string + key + ' '
    dis_string = dis_string + str(radar.fields[key]['data'][index][0]) + ' '

write_dis_filename = os.path.join(dis_output_location,
                                 'csapr_distro_'+ymd_string+hms_string+'.txt')

dis_fh = open(write_dis_filename, 'w')
dis_fh.write(dis_string)
dis_fh.close()

hts = np.linspace(radar.altitude['data'],15000.0 + radar.altitude['data'],61)
flds =['reflectivity', 
     'bringi_specific_diff_phase',
     'corrected_specific_diff_phase',
     'maesaka_differential_phase',
     'cross_correlation_ratio',
     'velocity_texture']
my_qvp = radar_codes.retrieve_qvp(radar, hts, flds = flds)

hts_string = 'height(m) '
for htss in hts:
    hts_string = hts_string + str(int(htss)) + ' '

write_qvp_filename = os.path.join(dis_output_location,
                                 'csapr_qvp_'+ymd_string+hms_string+'.txt')

dis_fh = open(write_qvp_filename, 'w')
dis_fh.write(hts_string + '\n')
for key in flds:
    print(key)
    this_str = key + ' '
    for i in range(len(hts)):
        this_str = this_str + str(my_qvp[key][i]) + ' '
    this_str = this_str + '\n'
    dis_fh.write(this_str)
dis_fh.close()

plt.plot(my_qvp['height'], my_qvp['cross_correlation_ratio'])

im_output_location = os.path.join(odir_i,ymd_string)
if not os.path.exists(im_output_location):
    os.makedirs(im_output_location)

display = pyart.graph.RadarMapDisplay(radar)
fig = plt.figure(figsize = [20,6])
plt.subplot(1,3,1)
display.plot_ppi_map('bringi_specific_diff_phase', sweep = 0, resolution = 'l',
                    mask_outside = False,
                    cmap = pyart.graph.cm.NWSRef,
                    vmin = 0, vmax = 6, title='Bringi/CSU')
plt.subplot(1,3,2)
display.plot_ppi_map('corrected_specific_diff_phase', sweep = 0, resolution = 'l',
                    mask_outside = False,
                    cmap = pyart.graph.cm.NWSRef,
                    vmin = 0, vmax = 6, title='Giangrande/LP')

plt.subplot(1,3,3)
display.plot_ppi_map('maesaka_differential_phase', sweep = 0, resolution = 'l',
                    mask_outside = False,
                    cmap = pyart.graph.cm.NWSRef,
                    vmin = 0, vmax = 6, title='North/Maesaka')

plt.savefig(os.path.join(im_output_location, 'csapr_kdp_comp_'+ymd_string+hms_string+'.png'))

r_output_location = os.path.join(odir_r,ymd_string)
if not os.path.exists(r_output_location):
    os.makedirs(r_output_location)
rfilename = os.path.join(r_output_location, 'csaprsur_' + ymd_string + '.' + 'hms_string.nc')
pyart.io.write_cfradial(rfilename, radar)



