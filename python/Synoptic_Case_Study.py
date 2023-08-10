from siphon.ncss import NCSS

ncss = NCSS('https://nomads.ncdc.noaa.gov/thredds/ncss/grid/gfs-004-anl/201102/20110202/gfsanl_4_20110202_0000_000.grb2')

ncss.variables

from datetime import datetime

query = ncss.query()

query.variables('Temperature',
                'Geopotential_height',
                'U-component_of_wind',
                'V-component_of_wind',
                'Pressure_reduced_to_MSL')
query.time(datetime(2011,2,2,0))
query.add_lonlat()
query.lonlat_box(north=60,south=15,west=-140,east=-50)

data = ncss.get_data(query)

print(data)

temp_var = data.variables['Temperature']
print('Temp units = ',temp_var.units)

hght_var = data.variables['Geopotential_height']
print('Height units = ',hght_var.units)

uwnd_var = data.variables['U-component_of_wind']
print('Uwind units = ',uwnd_var.units)

vwnd_var = data.variables['V-component_of_wind']
print('Vwind units = ',vwnd_var.units)

mslp_var = data.variables['Pressure_reduced_to_MSL']
print('MSLP units = ',mslp_var.units)

dtime = temp_var.dimensions[0]
dlev  = temp_var.dimensions[1]

print(dlev, 'level units = ',data.variables[dlev].units)

from netCDF4 import num2date
import numpy as np

# Set lat/lon values from data
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]

# Make lat and lon 2D, if necessary
if (lat.ndim < 2):
    lon, lat = np.meshgrid(lon, lat)

# Find index values for pressure levels
levs = data.variables[dlev][:]
lev_1000 = np.where(levs == 1000*100)[0][0]
lev_850 = np.where(levs == 850*100)[0][0]
lev_500 = np.where(levs == 500*100)[0][0]
lev_300 = np.where(levs == 300*100)[0][0]

# Get time into a better format
times = data.variables[dtime]
vtimes = num2date(times[:], units=times.units)

from metpy.units import units

hght_1000 = hght_var[0,lev_1000] * units.meter
mslp = mslp_var[0] * units(mslp_var.units)
mslp_hPa = mslp.to('hPa')

hght_850 = hght_var[0,lev_850] * units.meter
tmpk_850 = temp_var[0,lev_850] * units.K
tmpc_850 = tmpk_850.to('degC')
uwnd_850 = uwnd_var[0,lev_850] * units('m/s')
vwnd_850 = vwnd_var[0,lev_850] * units('m/s')

hght_500 = hght_var[0,lev_500] * units.meter
uwnd_500 = uwnd_var[0,lev_500] * units('m/s')
vwnd_500 = vwnd_var[0,lev_500] * units('m/s')

hght_300 = hght_var[0,lev_300] * units.meter
uwnd_300 = uwnd_var[0,lev_300] * units('m/s')
vwnd_300 = vwnd_var[0,lev_300] * units('m/s')

# Helper function to calculate distance between lat/lon points
# to be used in differencing calculations
def calc_dx_dy(longitude,latitude,shape='sphere',radius=6370997.):
    ''' This definition calculates the distance between grid points that are in
        a latitude/longitude format.
        
        Using pyproj GEOD; different Earth Shapes
        https://jswhit.github.io/pyproj/pyproj.Geod-class.html
        
        Common shapes: 'sphere', 'WGS84', 'GRS80'
        
        Accepts, 1D or 2D arrays for latitude and longitude
        
        Assumes [Y, X] for 2D arrays
        
        Returns: dx, dy; 2D arrays of distances between grid points 
                 in the x and y direction with units of meters 
    '''
    import numpy as np
    from metpy.units import units
    from pyproj import Geod
    
    if (radius != 6370997.):
        g = Geod(ellps=shape, a=radius, b=radius)
    else:
        g = Geod(ellps=shape)
    
    if (latitude.ndim == 1):
        longitude, latitude = np.meshgrid(longitude,latitude)
    
    dy = np.zeros(latitude.shape)
    dx = np.zeros(longitude.shape)
        
    _, _, dy[:-1,:] = g.inv(longitude[:-1,:],latitude[:-1,:],longitude[1:,:],latitude[1:,:])
    dy[-1,:] = dy[-2,:]
    
    _, _, dx[:,:-1] = g.inv(longitude[:,:-1],latitude[:,:-1],longitude[:,1:],latitude[:,1:])
    dx[:,-1] = dx[:,-2]
    
    xdiff_sign = np.sign(longitude[0,1]-longitude[0,0])
    ydiff_sign = np.sign(latitude[1,0]-latitude[0,0])
    return xdiff_sign*dx*units.meter, ydiff_sign*dy*units.meter

from metpy.calc import v_vorticity, get_wind_speed, coriolis_parameter

dx, dy = calc_dx_dy(lon, lat)

f = coriolis_parameter(np.deg2rad(lat)).to('1/s')

vor_500 = v_vorticity(uwnd_500, vwnd_500, dx, dy, dim_order='yx')

avor_500 = vor_500 + f

wspd_300 = get_wind_speed(uwnd_300, vwnd_300).to('knots')

import matplotlib.pylab as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat

# Data projection
# GFS data is lat/lon so need PlateCarree
dataproj = ccrs.PlateCarree()

# Plot projection
# The look you want for the view, LambertConformal for mid-latitude view
plotproj = ccrs.LambertConformal(central_longitude=-100, central_latitude=45,
                                 standard_parallels=[30,60])

# Use shapefiles from http://www.naturalearthdata.com/downloads/
# to plot state and country borders
states_provinces = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='50m',
                                             facecolor='none')

country_borders = cfeat.NaturalEarthFeature(category='cultural',
                                            name='admin_0_countries', 
                                            scale='50m',
                                            facecolor='none')

# Simple definition to create axes and map background features
def map_background(num):
    ax = plt.subplot(num, projection=plotproj)
    ax.set_extent([-124,-72,20,54], ccrs.Geodetic())
    ax.add_feature(states_provinces, edgecolor='black', linewidth=1)
    ax.add_feature(country_borders, edgecolor='black', linewidth=1)
    return ax

from scipy.ndimage import gaussian_filter

fig = plt.figure(figsize=(18,14.2))


# Upper-Left Panel 300 hPa
ax1 = map_background(221)
ax1.set_title('300-hPa Geo. Height (m), Wind Speed (kt)', loc='left')
ax1.set_title('Valid: {}'.format(vtimes[0]), loc='right')
cs1 = ax1.contour(lon, lat, gaussian_filter(hght_300, sigma=5),
                  np.arange(0,11000,120), colors='black', transform=dataproj)
plt.clabel(cs1, fontsize=10, inline=1, inline_spacing=5,
           fmt='%i', rightside_up=True, use_clabeltext=True)
cf1 = ax1.contourf(lon, lat, wspd_300.to('knots'), np.arange(50,200,20), cmap=plt.cm.BuPu, transform=dataproj)
plt.colorbar(cf1, orientation='horizontal', pad=0, aspect=50, use_gridspec=True)

# Upper-right Panel 500 hPa
ax2 = map_background(222)
clevavor500 = [-4,-3,-2,-1,0,7,10,13,16,19,22,25,28,31,34,37,40,43,46]
colorsavor500 = ('#660066', '#660099', '#6600CC', '#6600FF', 'w', '#ffE800', '#ffD800',
                 '#ffC800', '#ffB800', '#ffA800', '#ff9800', '#ff8800', '#ff7800',
                 '#ff6800', '#ff5800', '#ff5000', '#ff4000', '#ff3000')
ax2.set_title(r'500-hPa Geo. Height (m), AVOR ($10^5$ $s^{-1}$)', loc='left')
ax2.set_title('Valid: {}'.format(vtimes[0]), loc='right')
cs2 = ax2.contour(lon, lat, gaussian_filter(hght_500, sigma=5),
                  np.arange(0,6500,60), colors='black', transform=dataproj)
plt.clabel(cs2, fontsize=10, inline=1, inline_spacing=5,
           fmt='%i', rightside_up=True, use_clabeltext=True)
cf2 = ax2.contourf(lon, lat, avor_500*1e5, clevavor500, colors=colorsavor500, extend='both', transform=dataproj)
plt.colorbar(cf2, orientation='horizontal', pad=0, aspect=50, extendrect=True, use_gridspec=True)

# Lower-left Panel 850 hPa
ax3 = map_background(223)
ax3.set_title('850-hPa Geo. Height (m), Temp. (C)', loc='left')
ax3.set_title('Valid: {}'.format(vtimes[0]), loc='right')
cs3 = ax3.contour(lon, lat, gaussian_filter(hght_850, sigma=5),
                  np.arange(0,6500,30), colors='black', transform=dataproj)
plt.clabel(cs3, fontsize=10, inline=1, inline_spacing=10,
           fmt='%i', rightside_up=True, use_clabeltext=True)
cf3 = ax3.contourf(lon, lat, tmpc_850, np.arange(-40,40,2), extend='both', transform=dataproj)
plt.colorbar(cf3, orientation='horizontal', pad=0, aspect=50, extendrect=True, use_gridspec=True)
cs6 = ax3.contour(lon, lat, tmpc_850, np.arange(-40,40,2), colors='black', linestyles='dotted', transform=dataproj)
plt.clabel(cs6, fontsize=10, inline=1, inline_spacing=5,
           fmt='%i', rightside_up=True, use_clabeltext=True)

# Lower-right Panel surface
ax4 = map_background(224)
clevprecip = [0,0.01,0.03,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,
              0.60,0.70,0.80,0.90,1.00,1.25,1.50,1.75,2.00,2.50]
ax4.set_title('MSLP (hPa), 1000-500 hPa Thickness, Precip (in)', loc='left')
ax4.set_title('Valid: {}'.format(vtimes[0]), loc='right')
cs4 = ax4.contour(lon, lat, gaussian_filter(mslp_hPa, sigma=5),
                  np.arange(0,1100,4), colors='black', transform=dataproj)
plt.clabel(cs4, fontsize=10, inline=1, inline_spacing=10,
           fmt='%i', rightside_up=True, use_clabeltext=True)
cf4 = ax4.contourf(lon, lat, np.zeros_like(hght_500), clevprecip, cmap=plt.cm.BuPu, transform=dataproj)
plt.colorbar(cf4, orientation='horizontal', pad=0, aspect=50, use_gridspec=True)
cs5 = ax4.contour(lon, lat, gaussian_filter(hght_500-hght_1000, sigma=5),
                  np.arange(0,6000,60), colors='tab:red', linestyles='dashed', transform=dataproj)
plt.clabel(cs5, fontsize=10, inline=1, inline_spacing=5,
           fmt='%i', rightside_up=True, use_clabeltext=True)



plt.tight_layout()
plt.show()

from netCDF4 import Dataset

# Data is located in local working directory
ds = Dataset('goes13.2011.032.233148.BAND_01.nc')

# print file metadata
print(ds)

# Visible data
vis_data = ds.variables['data']

# Get time and convert to datetime object
sattime = ds.variables['time']
vsattime = num2date(sattime[:], units=sattime.units)

# Set up satellite projection
proj = ccrs.Geostationary(central_longitude=-75)

# Get lat/lons and mask values in space
lon = np.ma.masked_values(ds.variables['lon'][:],2.1432893e+09)
lat = np.ma.masked_values(ds.variables['lat'][:],2.1432893e+09)

# Find exent of image in lat/lon
ilon_min = np.unravel_index(np.argmin(lon),lon.shape)
ilon_max = np.unravel_index(np.argmax(lon),lon.shape)
ilat_min = np.unravel_index(np.argmin(lat),lat.shape)
ilat_max = np.unravel_index(np.argmax(lat),lat.shape)

# Convert extent to projection coords
LONpt0 = proj.transform_point(lon[ilon_min],lat[ilon_min],ccrs.Geodetic())
LONpt1 = proj.transform_point(lon[ilon_max],lat[ilon_max],ccrs.Geodetic())
LATpt0 = proj.transform_point(lon[ilat_min],lat[ilat_min],ccrs.Geodetic())
LATpt1 = proj.transform_point(lon[ilat_max],lat[ilat_max],ccrs.Geodetic())

# Start Visible Satellite Image
fig = plt.figure(2, figsize=(12,12))
ax = fig.add_subplot(1, 1, 1, projection=proj)

# Set titles
ax.set_title('GOES-13 Visible Imagery', loc='left')
ax.set_title(vsattime[0], loc='right')

# Acutal satellite plot
im = ax.imshow(vis_data[0,:,:]/100, origin='upper', extent=(LONpt0[0],LONpt1[0],LATpt0[1],LATpt1[1]),
               cmap='Greys_r', norm = plt.Normalize(0,255))

# Add state borders and coastlines
ax.coastlines(resolution='50m', color='white', linewidth=0.75)
ax.add_feature(states_provinces,edgecolor='white', linewidth=0.75)

plt.show()

# 6.5um Water Vapor is BAND 3
ds = Dataset('goes13.2011.032.233148.BAND_03.nc')

# Grab Water Vapor Data
WV_data = ds.variables['data']

# Get time and convert to datetime object
sattime = ds.variables['time']
vsattime = num2date(sattime[:], units=sattime.units)

# Get lat/lons and mask values in space
# Need to do this again, because resolution change, resulting
# in a different number of points
lon = np.ma.masked_values(ds.variables['lon'][:],2.1432893e+09)
lat = np.ma.masked_values(ds.variables['lat'][:],2.1432893e+09)

# Find exent of image in lat/lon
ilon_min = np.unravel_index(np.argmin(lon),lon.shape)
ilon_max = np.unravel_index(np.argmax(lon),lon.shape)
ilat_min = np.unravel_index(np.argmin(lat),lat.shape)
ilat_max = np.unravel_index(np.argmax(lat),lat.shape)

# Convert extent to projection coords
LONpt0 = proj.transform_point(lon[ilon_min],lat[ilon_min],ccrs.Geodetic())
LONpt1 = proj.transform_point(lon[ilon_max],lat[ilon_max],ccrs.Geodetic())
LATpt0 = proj.transform_point(lon[ilat_min],lat[ilat_min],ccrs.Geodetic())
LATpt1 = proj.transform_point(lon[ilat_max],lat[ilat_max],ccrs.Geodetic())

# Start Water Vapor Satellite Image
fig = plt.figure(3, figsize=(12,12))
ax = fig.add_subplot(1, 1, 1, projection=proj)

# Set titles
ax.set_title('GOES-13 Water Vapor 6.5um', loc='left')
ax.set_title(vsattime[0], loc='right')

# Plot water vapor data
im = ax.imshow(WV_data[0,:,:]/100, origin='upper', extent=(LONpt0[0],LONpt1[0],LATpt0[1],LATpt1[1]),
               cmap='Greys')

# Plot state borders and coastlines
ax.coastlines(resolution='50m', color='white', linewidth=0.75)
ax.add_feature(states_provinces,edgecolor='white', linewidth=0.75)

plt.show()

# 10.7um Water Vapor is BAND 4
ds = Dataset('goes13.2011.032.233148.BAND_04.nc')

# Grab Infrared Data
IR_data = ds.variables['data']

# Get time and convert to datetime object
sattime = ds.variables['time']
vsattime = num2date(sattime[:], units=sattime.units)

# Get lat/lons and mask values in space
# Need to do this again, because resolution change, resulting
# in a different number of points
lon = np.ma.masked_values(ds.variables['lon'][:],2.1432893e+09)
lat = np.ma.masked_values(ds.variables['lat'][:],2.1432893e+09)

# Find exent of image in lat/lon
ilon_min = np.unravel_index(np.argmin(lon),lon.shape)
ilon_max = np.unravel_index(np.argmax(lon),lon.shape)
ilat_min = np.unravel_index(np.argmin(lat),lat.shape)
ilat_max = np.unravel_index(np.argmax(lat),lat.shape)

# Convert extent to projection coords
LONpt0 = proj.transform_point(lon[ilon_min],lat[ilon_min],ccrs.Geodetic())
LONpt1 = proj.transform_point(lon[ilon_max],lat[ilon_max],ccrs.Geodetic())
LATpt0 = proj.transform_point(lon[ilat_min],lat[ilat_min],ccrs.Geodetic())
LATpt1 = proj.transform_point(lon[ilat_max],lat[ilat_max],ccrs.Geodetic())

# Start Water Vapor Satellite Image
fig = plt.figure(3, figsize=(12,12))
ax = fig.add_subplot(1, 1, 1, projection=proj)

# Set titles
ax.set_title('GOES-13 Infrared 10.7um', loc='left')
ax.set_title(vsattime[0], loc='right')

# Plot infrared data
im = ax.imshow(IR_data[0,:,:]/100, origin='upper', extent=(LONpt0[0],LONpt1[0],LATpt0[1],LATpt1[1]),
               cmap='Greys')

# Plot state borders and coastlines
ax.coastlines(resolution='50m', color='white', linewidth=0.75)
ax.add_feature(states_provinces,edgecolor='white', linewidth=0.75)

plt.show()

from metpy.io import Level2File

# Use metpy reader for level 2 data
raddata = Level2File('KLOT20110202_000026_V03')

# Valid Time
radtime = raddata.dt
print(radtime)

# Pull data out of the file
sweep = 0

# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in raddata.sweeps[sweep]])

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = raddata.sweeps[sweep][0][4][b'REF'][0]
ref_range = np.arange(ref_hdr.num_gates) * ref_hdr.gate_width + ref_hdr.first_gate
ref = np.array([ray[4][b'REF'][1] for ray in raddata.sweeps[sweep]])

# Get radar site lat/lon
lat_0 = raddata.sweeps[0][0][1].lat
lon_0 = raddata.sweeps[0][0][1].lon

def get_lonlat(x, y, nex_lon, nex_lat):
    from pyproj import Proj

    p = Proj(proj='aeqd', ellps='sphere',
             lon_0=nex_lon,
             lat_0=nex_lat)

    return p(x,y,inverse=True)

from metpy.plots import ctables

# Turn into an array, then mask
data = np.ma.array(ref)
data[np.isnan(data)] = np.ma.masked

# Convert az,range to x,y
xlocs = ref_range * np.sin(np.deg2rad(az[:, np.newaxis])) * 1000
ylocs = ref_range * np.cos(np.deg2rad(az[:, np.newaxis])) * 1000

# Get lat/lon values from x,y and center lon/lat
lon, lat = get_lonlat(xlocs, ylocs, lon_0, lat_0)

# Plot the data
fig = plt.figure(5, figsize=(10,12))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Set titles
ax.set_title('KLOT Reflectivity', loc='left')
ax.set_title(radtime, loc='right')

# Plot radar data with a colormap from metpy
cmap = ctables.registry.get_colortable('viridis')
ax.pcolormesh(lon, lat, data, cmap=cmap)

# Limit graphics area extent
ax.set_extent([-92.5,-84.5,38.5,45.5],ccrs.PlateCarree())

# Plot state borders
hiresstates = cfeat.NaturalEarthFeature(category='cultural',
                                        name='admin_1_states_provinces_lakes',
                                        scale='10m',
                                        facecolor='none')

ax.add_feature(hiresstates,edgecolor='black', linewidth=0.75)

plt.tight_layout()
plt.show()

from metpy.io import get_upper_air_data
from metpy.plots import SkewT
from metpy.calc import parcel_profile

# Set values for desire date/time
year = 2011
month = 2
day = 2
hour = 0

# Get data and make plots for two stations with data coming from Univ. of Wyoming
for sound_stn in ['ILX', 'GRB']:
    # Get data with metpy function 'get_upper_air_data'
    data = get_upper_air_data(datetime(year,month,day,hour),sound_stn,'wyoming')
    
    # Parse out Temperature (T), Dewpoint (Td), Pressure (p),
    # U-component of wind (u), V-component of wind (v)
    T = data.variables['temperature'][:]
    Td = data.variables['dewpoint'][:]
    p = data.variables['pressure'][:]
    u = data.variables['u_wind'][:]
    v = data.variables['v_wind'][:]

    # Change default to be better for skew-T
    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig)

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot
    skew.plot(p, T, 'r')
    skew.plot(p, Td, 'g')
    skew.plot_barbs(p[::4], u[::4], v[::4])

    # Add the relevant special lines
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    skew.ax.set_ylim(1000, 100)

    # Calculate full parcel profile and add to plot as black line
    # Requires that the variables have associated units
    prof = parcel_profile(p, T[0], Td[0]).to('degC')
    skew.plot(p, prof, 'k', linewidth=2)

    # Make some titles
    plt.title('K'+sound_stn+' RAOB Obs',loc='left')
    plt.title(datetime(year,month,day,hour), loc='right')

    # Show the plot
    plt.show()

import pandas as pd
from metpy.calc import get_wind_components
from metpy.plots import StationPlot
from metpy.plots.wx_symbols import current_weather, sky_cover

# Read surface data from file, skipping first five lines and using the fifth as the header
# Remove any spaces after commas and make missing values ('M') into np.nan
sfcdata = pd.read_csv('sfc_obs_201102020000_201102020000.txt', header=5,
                      skipinitialspace=True, na_values='M')

# Convert string date field ('valid') to datetime objects with Pandas method to_datetime
sfcdata['valid'] = pd.to_datetime(sfcdata['valid'])

# Masking our data for only times between 0000 and 0030 UTC
LLlon = -95
LLlat = 38
URlon = -82
URlat = 45.5
mask_time = (sfcdata['valid'] >= datetime(2011,2,2,0,0)) & (sfcdata['valid'] < datetime(2011,2,2,0,30))
mask_lon = ((sfcdata['lon'] > LLlon) & (sfcdata['lon'] < URlon))
mask_lat = ((sfcdata['lat'] < URlat) & (sfcdata['lat'] > LLlat+.25))
mask = mask_time&mask_lon&mask_lat

# Only keep sfc data where mask is True
sfcdata = sfcdata.loc[mask]

# Group by station and select first one (which should be closest to 0000 UTC)
gb = sfcdata.groupby('station')
sfc_stn_00 = gb.head(1)

# Pull out variables
stid = sfc_stn_00['station'].values
st_lon = sfc_stn_00['lon'].values
st_lat = sfc_stn_00['lat'].values
tmpf = sfc_stn_00['tmpf'].values * units.degF
dwpf = sfc_stn_00['dwpf'].values * units.degF
alti = (sfc_stn_00['alti'].values * units('in Hg')).to('hPa')
u, v = get_wind_components(sfc_stn_00['sknt'].values * units('knots'),
                           sfc_stn_00['drct'].values * units.degree)

# Decode present weather
present_wx = sfc_stn_00['presentwx']
wx_text = present_wx.fillna(value='M')
wx_codes = {'': 0, 'HZ': 5, 'BR': 10, '-DZ': 51, 'DZ': 53, '+DZ': 55, 'M': 0, 'UP': 0, '-SNPL': 79,
            '-FZRA': 66, 'FZRA': 67, '-FZDZPL': 56, '+PL': 79, 'PL': 79, '-RA': 61, 'RA': 63,
            '+RA': 65, '-SN': 71, 'SN': 73, '+SN': 75, 'FZFG': 5, 'BLSN': 0, 'np.nan': 0}
wx = [wx_codes[s.split()[0] if ' ' in s else s] for s in wx_text]

# Decode skycover
skycover = [[sfc_stn_00['skyc1'].fillna(value='M').values],
            [sfc_stn_00['skyc2'].fillna(value='M').values],
            [sfc_stn_00['skyc3'].fillna(value='M').values],
            [sfc_stn_00['skyc4'].fillna(value='M').values]]
skyc_codes = {'SKC': 0, 'CLR': 0, 'FEW': 2, 'SCT': 4, 'BKN': 6, 'OVC': 8, 'VV ': 9, 'M': 0}
skyc = np.array([skyc_codes[s] for j in range(4) for s in skycover[j][0]])
skyc = skyc.reshape((4, len(sfc_stn_00['skyc1'])))
skyc = np.max(skyc,axis=0)

# Set up projection for surface data plot
sfcproj = ccrs.LambertConformal(central_longitude=-90, central_latitude=45,
                                standard_parallels=[30,60])

# Create the figure and an axes set to the projection
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(1, 1, 1, projection=sfcproj)

# Set plot bounds
ax.set_extent([LLlon,URlon,LLlat,URlat], ccrs.PlateCarree())

# Set titles
ax.set_title('Surface Observations', loc='left')
ax.set_title(datetime(2011,2,2,0), loc='right')

# Add some various map elements to the plot to make it recognizable
ax.add_feature(states_provinces, edgecolor='black')

# Here's the actual station plot

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 12 pt.
stationplot = StationPlot(ax, st_lon, st_lat, transform=ccrs.PlateCarree(),
                          fontsize=12)

# Plot the temperature and dew point to the upper and lower left, respectively, of
# the center point. Each one uses a different color.
stationplot.plot_parameter('NW', tmpf, color='red')
stationplot.plot_parameter('SW', dwpf, color='darkgreen')

# A more complex example uses a custom formatter to control how the sea-level pressure
# values are plotted. This uses the standard trailing 3-digits of the pressure value
# in tenths of millibars.
stationplot.plot_parameter('NE', alti.m,
                           formatter=lambda v: format(10 * v, '.0f')[-3:])

# Plot the cloud cover symbols in the center location. This uses the codes made above and
# uses the `sky_cover` mapper to convert these values to font codes for the
# weather symbol font.
stationplot.plot_symbol('C', skyc, sky_cover)

# Same this time, but plot current weather to the left of center, using the
# `current_weather` mapper to convert symbols to the right glyphs.
stationplot.plot_symbol('W', wx, current_weather)

# Add wind barbs
stationplot.plot_barb(u, v, transform=ccrs.PlateCarree())

# Also plot the actual text of the station id. Instead of cardinal directions,
# plot further out by specifying a location of 2 increments in x and 0 in y.
stationplot.plot_text((1.5, -1), stid)

plt.tight_layout()
plt.show()

