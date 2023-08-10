from mpl_toolkits.basemap import Basemap, cm
import os.path
import sys
from matplotlib import rcParams
from matplotlib.animation import ArtistAnimation
import matplotlib
import matplotlib.pyplot as plt
import pyart
from siphon.radarserver import RadarServer
from datetime import datetime, timedelta
from siphon.cdmr import Dataset
import pyart
import numpy as np
import numpy.ma as ma
import netCDF4
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy.ma as ma
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS
from metpy.plots import StationPlot
from metpy.plots.wx_symbols import sky_cover
from metpy.calc import get_wind_components, get_layer
from metpy.units import units
from metpy.calc import lcl, log_interp
import scipy.ndimage as ndimage

rs = RadarServer('http://thredds-aws.unidata.ucar.edu/thredds/radarServer/nexrad/level2/S3/')
#rs = RadarServer('http://thredds.ucar.edu/thredds/radarServer/nexrad/level2/IDD/')

query = rs.query()
dt = datetime(2017, 6, 12, 22, 0) # Our specified time
query.stations('KCYS').time(dt)
cat = rs.get_catalog(query)
for item in sorted(cat.datasets.items()):
    # Pull the actual Dataset object out
    # of our list of items and access over OPENDAP
    ds = item[1]
radar = pyart.io.nexrad_cdm.read_nexrad_cdm(ds.access_urls['OPENDAP'])
#Pull out only the lowest tilt
radar = radar.extract_sweeps([0])
#Pull the timestamp from the radar file
time_start = netCDF4.num2date(radar.time['data'][0], radar.time['units'])

rlons = radar.gate_longitude['data']
rlats = radar.gate_latitude['data']
cenlat = radar.latitude['data'][0]
cenlon = radar.longitude['data'][0]
#Pull reflectivity from the radar object
refl = radar.fields['reflectivity']['data']
#Mask noisy stuff below 20 dbZ
refl = ma.masked_less(refl, 20.)

# Set up our projection
crs = ccrs.LambertConformal(central_longitude=-100.0, central_latitude=45.0)
# Set up our array of latitude and longitude values and transform to 
# the desired projection.
tlatlons = crs.transform_points(ccrs.LambertConformal(central_longitude=265, central_latitude=25, standard_parallels=(25.,25.)),rlons,rlats)
tlons = tlatlons[:,:,0]
tlats = tlatlons[:,:,1]
# Limit the extent of the map area to around the radar, must convert to proper coords.
LL = (cenlon-1.,cenlat-1.,ccrs.PlateCarree())
UR = (cenlon+1.,cenlat+1.0,ccrs.PlateCarree())
# Get data to plot state and province boundaries
#states_provinces = cfeature.NaturalEarthFeature(
#        category='cultural',
#        name='admin_1_states_provinces_lakes',
#        scale='50m',
#        facecolor='none')
#Read in state and county boundary shapefiles
fname = 'cb_2016_us_county_20m/cb_2016_us_county_20m.shp'
fname2 = 'cb_2016_us_state_20m/cb_2016_us_state_20m.shp'
counties = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black')
states = ShapelyFeature(Reader(fname2).geometries(),ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black')
#Create the figure
get_ipython().magic('matplotlib inline')
fig=plt.figure(1,figsize=(30.,25.))
ax = plt.subplot(111,projection=ccrs.PlateCarree())
ax.coastlines('50m',edgecolor='black',linewidth=0.75)
#ax.add_feature(states_provinces,edgecolor='black',linewidth=0.5)
#Add states/counties and set extent
ax.add_feature(counties, edgecolor = 'black', linewidth = 0.5)
ax.add_feature(states, edgecolor = 'black', linewidth = 1.5)
ax.set_extent([LL[0],UR[0],LL[1],UR[1]])
#Plot our radar data
refp = ax.pcolormesh(rlons, rlats, refl, cmap=plt.cm.gist_ncar, vmin = 10, vmax = 70)

# Set up our projection
crs = ccrs.LambertConformal(central_longitude=-100.0, central_latitude=45.0)
# Set up our array of latitude and longitude values and transform to 
# the desired projection.
tlatlons = crs.transform_points(ccrs.LambertConformal(central_longitude=265, central_latitude=25, standard_parallels=(25.,25.)),rlons,rlats)
tlons = tlatlons[:,:,0]
tlats = tlatlons[:,:,1]
# Limit the extent of the map area to around the radar, must convert to proper coords.
LL = (cenlon-1.,cenlat-1.,ccrs.PlateCarree())
UR = (cenlon+1.,cenlat+1.0,ccrs.PlateCarree())
# Get data to plot state and province boundaries
#states_provinces = cfeature.NaturalEarthFeature(
#        category='cultural',
#        name='admin_1_states_provinces_lakes',
#        scale='50m',
#        facecolor='none')
#Read in state and county boundary shapefiles
fname = 'cb_2016_us_county_20m/cb_2016_us_county_20m.shp'
fname2 = 'cb_2016_us_state_20m/cb_2016_us_state_20m.shp'
counties = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black')
states = ShapelyFeature(Reader(fname2).geometries(),ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black')
#Create the figure
get_ipython().magic('matplotlib inline')
fig=plt.figure(1,figsize=(30.,25.))
ax = plt.subplot(111,projection=ccrs.PlateCarree())
ax.coastlines('50m',edgecolor='black',linewidth=0.75)
#ax.add_feature(states_provinces,edgecolor='black',linewidth=0.5)
#Add states/counties and set extent
ax.add_feature(counties, edgecolor = 'black', linewidth = 0.5)
ax.add_feature(states, edgecolor = 'black', linewidth = 1.5)
ax.set_extent([LL[0],UR[0],LL[1],UR[1]])
#Plot our radar data
rlevs = np.arange(30,70,10)
#refp = ax.contour(rlons, rlats, refl, cmap=plt.cm.gist_ncar)
refl_sm = ndimage.gaussian_filter(refl, sigma=1, order=0)
refp = ax.contour(rlons, rlats, refl_sm, rlevs, cmap=plt.cm.gist_ncar)

#Point to the URL for the station data on the Unidata thredds server
metar_cat_url = 'http://thredds.ucar.edu/thredds/catalog/nws/metar/ncdecoded/catalog.xml?dataset=nws/metar/ncdecoded/Metar_Station_Data_fc.cdmr'
# Parse the xml
catalog = TDSCatalog(metar_cat_url)
metar_dataset = catalog.datasets['Feature Collection']
ncss_url = metar_dataset.access_urls['NetcdfSubset']
# Import ncss client
ncss = NCSS(ncss_url)
#Set date/time to be the time from our radar scan
start_time = datetime(time_start.year, time_start.month, time_start.day, time_start.hour, time_start.minute)
#Create a query to access the data
query = ncss.query()
query.lonlat_box(north=cenlat+1., south=cenlat-1., east=cenlon+1., west=cenlon-1)
query.time(start_time)
query.variables('air_temperature', 'dew_point_temperature', 'inches_ALTIM',
                'wind_speed', 'wind_from_direction', 'cloud_area_fraction')
query.accept('csv')
#Get the data
data = ncss.get_data(query)

# Access is just like netcdf4-python
lats = data['latitude'][:]
lons = data['longitude'][:]
tair = data['air_temperature'][:] * units('degC')
dewp = data['dew_point_temperature'][:] * units('degC')
tair = tair.to('degF').magnitude
dewp = dewp.to('degF').magnitude
slp = (data['inches_ALTIM'][:] * units('inHg')).to('mbar')

# Convert wind to components
u, v = get_wind_components(data['wind_speed'], data['wind_from_direction'] * units.degree)

# Need to handle missing (NaN) and convert to proper code
cloud_cover = 8 * data['cloud_area_fraction']
cloud_cover[np.isnan(cloud_cover)] = 10
cloud_cover = cloud_cover.astype(np.int)

#Create the figure
fig=plt.figure(2,figsize=(30.,25.))
ax = plt.subplot(111,projection=ccrs.PlateCarree())
ax.coastlines('50m',edgecolor='black',linewidth=0.75)
#ax.add_feature(states_provinces,edgecolor='black',linewidth=0.5)
#Add states/counties and set extent
ax.add_feature(counties, edgecolor = 'black', linewidth = 0.5)
ax.add_feature(states, edgecolor = 'black', linewidth = 1.5)
ax.set_extent([LL[0],UR[0],LL[1],UR[1]])
#Plot our radar data
refp = ax.pcolormesh(rlons, rlats, refl, cmap=plt.cm.gist_ncar, vmin = 10, vmax = 70)
# For some reason these come back as bytes instead of strings
stid = np.array([s.decode() for s in data['station']])
# Create a station plot pointing to an Axes to draw on as well as the location of points
stnp = stationplot = StationPlot(ax, lons, lats, transform=ccrs.PlateCarree(),
                          fontsize=20)
stnt = stationplot.plot_parameter('NW', tair, color='red')
stnd = stationplot.plot_parameter('SW', dewp, color='darkgreen')
stnpr = stationplot.plot_parameter('NE', slp)

# Add wind barbs
stnpb = stationplot.plot_barb(u, v)

# Plot the sky cover symbols in the center. We give it the integer code values that
# should be plotted, as well as a mapping class that can convert the integer values
# to the appropriate font glyph.
stns = stationplot.plot_symbol('C', cloud_cover, sky_cover)

# Plot station id -- using an offset pair instead of a string location
stntx = stationplot.plot_text((2, 0), stid)
cs = plt.colorbar(refp, ticks = [20,25,30,35,40,45,50,55,60,65,70],norm=matplotlib.colors.Normalize(vmin=20, vmax=70))

cs.ax.tick_params(labelsize=25)
plt.title('Radar Reflectivity and Surface Obs '+str(time_start.year)+'-'+str(time_start.month)+'-'+str(time_start.day)+
          ' '+str(time_start.hour)+':'+str(time_start.minute)+' UTC', size = 25)
plt.show()

cat = TDSCatalog('http://nomads.ncdc.noaa.gov/thredds/catalog/rap130/'+str(time_start.year)+'0'+str(time_start.month)+'/'+str(time_start.year)+'0'+str(time_start.month)+str(time_start.day)+'/catalog.html?dataset=rap130/'+str(time_start.year)+'0'+str(time_start.month)+'/'+str(time_start.year)+'0'+str(time_start.month)+str(time_start.day)+'/rap_130_'+str(time_start.year)+'0'+str(time_start.month)+str(time_start.day)+'_'+str(time_start.hour)+'00_001.grb2')
#cat = TDSCatalog('http://nomads.ncdc.noaa.gov/thredds/catalog/rap130/'+str(year)+str(month)+'/'+str(year)+str(month)+str(day)+'/catalog.html?dataset=rap130/'+str(year)+str(month)+'/'+str(year)+str(month)+str(day)+'/rap_130_'+str(year)+str(month)+str(day)+'_'+str(UTC)+'_000.grb2')
latest_ds = list(cat.datasets.values())[0]
print(latest_ds.access_urls)
ncss = NCSS(latest_ds.access_urls['NetcdfServer'])

query = ncss.query()
query.variables('Convective_available_potential_energy_surface').variables('U-component_of_wind').variables('V-component_of_wind').variables('Storm_relative_helicity').variables('Pressure_surface').variables('Dew_point_temperature').variables('Temperature_height_above_ground').variables('Vertical_u-component_shear').variables('Vertical_v-component_shear').variables('Geopotential_height').variables('Geopotential_height_surface')
query.add_lonlat().lonlat_box(cenlon-1.1, cenlon +1.1, cenlat-1.1, cenlat+1.1)
data1 = ncss.get_data(query)
dtime = data1.variables['Convective_available_potential_energy_surface'].dimensions[0]
dlat = data1.variables['Convective_available_potential_energy_surface'].dimensions[1]
dlev = data1.variables['Geopotential_height'].dimensions[1]
dlon = data1.variables['Convective_available_potential_energy_surface'].dimensions[2]
CAPE = data1.variables['Convective_available_potential_energy_surface'][:] * units('J/kg')
SRH = data1.variables['Storm_relative_helicity'][:] * units('m/s')
SFCP = (data1.variables['Pressure_surface'][:]/100.) * units('hPa')
Td = data1.variables['Dew_point_temperature'][:] * units('kelvin')
T = data1.variables['Temperature_height_above_ground'][:] * units('kelvin')
ushr = data1.variables['Vertical_u-component_shear'][:] * units('m/s')
vshr = data1.variables['Vertical_v-component_shear'][:] * units('m/s')
hgt = data1.variables['Geopotential_height'][:] * units('meter')
sfc_hgt = data1.variables['Geopotential_height_surface'][:] * units('meter')
uwnd = data1.variables['U-component_of_wind'][:] * units('m/s')
vwnd = data1.variables['V-component_of_wind'][:] * units('m/s')

# Get the dimension data
lats_r = data1.variables[dlat][:]
lons_r= data1.variables[dlon][:]
lev = (data1.variables[dlev][:]/100.) * units('hPa')

# Set up our array of latitude and longitude values and transform to 
# the desired projection.
crs = ccrs.PlateCarree()
crlons, crlats = np.meshgrid(lons_r[:]*1000, lats_r[:]*1000)
trlatlons = crs.transform_points(ccrs.LambertConformal(central_longitude=265, central_latitude=25, standard_parallels=(25.,25.)),crlons,crlats)
trlons = trlatlons[:,:,0]
trlats = trlatlons[:,:,1]

#Create the figure
fig=plt.figure(2,figsize=(30.,25.))
ax = plt.subplot(111,projection=ccrs.PlateCarree())
ax.coastlines('50m',edgecolor='black',linewidth=0.75)
#ax.add_feature(states_provinces,edgecolor='black',linewidth=0.5)
#Add states/counties and set extent
ax.add_feature(counties, edgecolor = 'black', linewidth = 0.5)
ax.add_feature(states, edgecolor = 'black', linewidth = 1.5)
ax.set_extent([LL[0],UR[0],LL[1],UR[1]])
srhlev = np.arange(50,700,50)
ch = ax.contourf(trlons, trlats, SRH[0,1,:,:], srhlev, cmap=plt.cm.viridis, alpha = .65)
#Plot our radar data
refp = ax.pcolormesh(rlons, rlats, refl, cmap=plt.cm.gist_ncar, vmin = 10, vmax = 70)
# For some reason these come back as bytes instead of strings
stid = np.array([s.decode() for s in data['station']])
# Create a station plot pointing to an Axes to draw on as well as the location of points
stnp = stationplot = StationPlot(ax, lons, lats, transform=ccrs.PlateCarree(),
                          fontsize=18)
stnt = stationplot.plot_parameter('NW', tair, color='red')
stnd = stationplot.plot_parameter('SW', dewp, color='darkgreen')
stnpr = stationplot.plot_parameter('NE', slp)

# Add wind barbs
stnpb = stationplot.plot_barb(u, v)

# Plot the sky cover symbols in the center. We give it the integer code values that
# should be plotted, as well as a mapping class that can convert the integer values
# to the appropriate font glyph.
stns = stationplot.plot_symbol('C', cloud_cover, sky_cover)

# Plot station id -- using an offset pair instead of a string location
stntx = stationplot.plot_text((2, 0), stid)

#Plot CAPE because why not
cplev = np.arange(500,4000,500)
cf = ax.contour(trlons, trlats, CAPE[0,:,:], cplev, cmap=plt.cm.autumn_r, linewidths = 4, alpha = .65)
#ch = ax.contourf(trlons, trlats, SRH[0,1,:,:], srhlev, cmap=plt.cm.BuPu, alpha = .65)

plt.clabel(cf, fontsize=18, inline=1, inline_spacing=10, fmt='%i', rightside_up=True, use_clabeltext=True)

cs = plt.colorbar(refp, ticks = [20,25,30,35,40,45,50,55,60,65,70],norm=matplotlib.colors.Normalize(vmin=20, vmax=70),
                  shrink = .75, pad = 0)
cg = plt.colorbar(ch, shrink = .75, pad = .01)
cg.ax.tick_params(labelsize=25)
cg.set_label("Storm-Relative Helicity",size = 30)
cs.ax.tick_params(labelsize=25)
cs.set_label("Reflectivity",size = 30)
plt.title('Radar Reflectivity, RAP Analysis, and Surface Obs '+str(time_start.year)+'-'+str(time_start.month)+'-'+str(time_start.day)+
          ' '+str(time_start.hour)+':'+str(time_start.minute)+' UTC', size = 35)
plt.tight_layout()
plt.savefig("ReallyCoolMap.png")
plt.show()

from metpy.calc import lcl, log_interp



levs = np.zeros((hgt.shape[1], hgt.shape[2], hgt.shape[3]))
for i in range(hgt.shape[2]):
    for j in range(hgt.shape[3]):
        levs[:,i,j] = lev

SHR6 = np.sqrt(ushr ** 2 + vshr ** 2)

lcl_rap = lcl(SFCP[0,:,:], T[0,0,:,:], Td[0,0,:,:])

lcl_h = log_interp(lcl_rap[0], levs * units('hPa'), hgt[0,:,:,:], axis = 0)

print(lcl_h.shape)
print(lcl_rap[0].shape)

lcl_h = np.zeros((lcl_rap[0].shape))
for i in range(lcl_rap[0].shape[0]):
    for j in range(lcl_rap[0].shape[1]):
        lcl_h[i,j] = (log_interp(lcl_rap[0][i,j], levs[:,i,j] * units('hPa'), hgt[0,:,i,j])).magnitude


lcl_agl = lcl_h * units('meter')-sfc_hgt[0,:,:]

def sigtor(sbcape, sblcl, srh1, shr6):
    r"""Calculate the significant tornado parameter (fixed layer).
     
    The significant tornado parameter is designed to identify
    environments favorable for the production of significant
    tornadoes contingent upon the development of supercells.
    It's calculated according to the formula used on the SPC
    mesoanalysis page, updated in [Thompson, Edwards, and Mead, 2004]:
    
    sigtor = (sbcape / 1500 J/kg) * ((2000 m - sblcl) / 1000 m) * (srh1 / 150 m^s/s^2) * (shr6 / 20 m/s)
    
    The sblcl term is set to zero when the lcl is above 2000m and 
    capped at 1 when below 1000m, and the shr6 term is set to 0 
    when shr6 is below 12.5 m/s and maxed out at 1.5 when shr6
    exceeds 30 m/s.
    
    Parameters
    ----------
    sbcape : array-like
        Surface-based CAPE
    sblcl : array-like
        Surface-based lifted condensation level
    srh1 : array-like
        Surface-1km storm-relative helicity
    shr6 : array-like
        Surface-6km bulk shear
        
    Returns
    -------
    number
        significant tornado parameter
    
    Citation:
    Thompson, R.L., R. Edwards, and C. M. Mead, 2004b:
        An update to the supercell composite
        and significant tornado parameters.
        Preprints, 22nd Conf. on Severe Local
        Storms, Hyannis, MA, Amer.
        Meteor. Soc.
        
    """
    
    shr6 = shr6.to('m/s')
    shr6 = shr6.magnitude
    sblcl = sblcl.magnitude
    ind = np.where((sblcl <= 2000.) & (sblcl >= 1000.))
    ind1 = np.where(sblcl < 1000.)
    ind2 = np.where(sblcl > 2000.)
    sind = np.where((shr6 <= 30.) & (shr6 >= 12.5))
    sind1 = np.where(shr6 < 12.5)
    sind2 = np.where(shr6 > 30.)
    sblcl[ind] = (2000. - sblcl[ind]) / 1000.
    sblcl[ind1] = 1.
    sblcl[ind2] = 0.
    shr6[sind] = shr6[sind] / 20.
    shr6[sind1] = 0.
    shr6[sind2] = 1.5
    sigtor = (sbcape.magnitude / 1500.) * sblcl * (srh1.magnitude / 150.) * shr6
     
    return sigtor

sigtor_p = sigtor(CAPE[0,:,:], lcl_agl, SRH[0,0,:,:], SHR6[0,0,:,:])

#Create the figure
fig=plt.figure(2,figsize=(30.,25.))
ax = plt.subplot(111,projection=ccrs.PlateCarree())
ax.coastlines('50m',edgecolor='black',linewidth=0.75)
#ax.add_feature(states_provinces,edgecolor='black',linewidth=0.5)
#Add states/counties and set extent
ax.add_feature(counties, edgecolor = 'black', linewidth = 0.5)
ax.add_feature(states, edgecolor = 'black', linewidth = 1.5)
ax.set_extent([LL[0],UR[0],LL[1],UR[1]])
stlev = [.25,.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7]
ch = ax.contourf(trlons, trlats, sigtor_p, stlev, cmap=plt.cm.magma, alpha = .65)
#Plot our radar data
refp = ax.pcolormesh(rlons, rlats, refl, cmap=plt.cm.gist_ncar, vmin = 10, vmax = 70)
# For some reason these come back as bytes instead of strings
stid = np.array([s.decode() for s in data['station']])
# Create a station plot pointing to an Axes to draw on as well as the location of points
stnp = stationplot = StationPlot(ax, lons, lats, transform=ccrs.PlateCarree(),
                          fontsize=20)
stnt = stationplot.plot_parameter('NW', tair, color='red')
stnd = stationplot.plot_parameter('SW', dewp, color='darkgreen')
stnpr = stationplot.plot_parameter('NE', slp)

# Add wind barbs
stnpb = stationplot.plot_barb(u, v)

# Plot the sky cover symbols in the center. We give it the integer code values that
# should be plotted, as well as a mapping class that can convert the integer values
# to the appropriate font glyph.
stns = stationplot.plot_symbol('C', cloud_cover, sky_cover)

# Plot station id -- using an offset pair instead of a string location
stntx = stationplot.plot_text((2, 0), stid)

#Plot CAPE because why not
cplev = np.arange(500,4000,500)
#cf = ax.contour(trlons, trlats, CAPE[0,:,:], cplev, cmap=plt.cm.autumn_r, linewidths = 4, alpha = .65)
#ch = ax.contourf(trlons, trlats, SRH[0,1,:,:], srhlev, cmap=plt.cm.BuPu, alpha = .65)

#plt.clabel(cf, fontsize=16, inline=1, inline_spacing=10, fmt='%i', rightside_up=True, use_clabeltext=True)

cs = plt.colorbar(refp, ticks = [20,25,30,35,40,45,50,55,60,65,70],norm=matplotlib.colors.Normalize(vmin=20, vmax=70),
                  shrink = .75, pad = 0)
cg = plt.colorbar(ch, shrink = .75, pad = .01)
cg.ax.tick_params(labelsize=25)
cg.set_label("Significant Tornado",size = 30)
cs.ax.tick_params(labelsize=25)
cs.set_label("Reflectivity",size = 30)
plt.title('Radar Reflectivity, RAP Analysis, and Surface Obs '+str(time_start.year)+'-'+str(time_start.month)+'-'+str(time_start.day)+
          ' '+str(time_start.hour)+':'+str(time_start.minute)+' UTC', size = 35)
plt.tight_layout()
plt.savefig("SigTorMap.png")
plt.show()

def bulk_shear(u, v, pressure, height, top, bottom = None):
    r"""Calculate bulk shear through a layer. 
    
    Layer top and bottom specified in meters AGL:
    
    Parameters
    ----------
    u : array-like
        U-component of wind.
    v : array-like
        V-component of wind.
    p : array-like
        Atmospheric pressure profile
    hgt : array-like
        Heights from sounding
    top: `pint.Quantity`
        The top of the layer in meters AGL
    bottom: `pint.Quantity`
        The bottom of the layer in meters AGL.
        Default is the surface.
        
    Returns
    -------
    `pint.Quantity'
        u_shr: u-component of layer bulk shear, in m/s
    `pint.Quantity'
        v_shr: v-component of layer bulk shear, in m/s
    `pint.Quantity'
        shr_mag: magnitude of layer bulk shear, in m/s
        
    """   
    
    
    u = u.to('meters/second')
    v = v.to('meters/second')
    
    sort_inds = np.argsort(pressure[::-1])
    pressure = pressure[sort_inds]
    height = height[sort_inds]
    u = u[sort_inds]
    v = v[sort_inds]
    
    if bottom:
        depth_s = top - bottom
        bottom = bottom + height[0]
    else:
        depth_s = top
        
    w_int = get_layer(pressure, u, v, heights=height, bottom=bottom, depth=depth_s)
    
    u_shr = w_int[1][-1] - w_int[1][0]
    v_shr = w_int[2][-1] - w_int[2][0]
    
    shr_mag = np.sqrt((u_shr ** 2) + (v_shr ** 2))
    
    return u_shr, v_shr, shr_mag

height_agl = hgt - sfc_hgt

print(np.max(sfc_hgt))
print(hgt.shape)

print(uwnd)



sfc1_u = np.zeros((lcl_rap[0].shape))
sfc1_v = np.zeros((lcl_rap[0].shape))
for i in range(lcl_rap[0].shape[0]):
    for j in range(lcl_rap[0].shape[1]):
        shr1 = bulk_shear(uwnd[0,:,i,j], vwnd[0,:,i,j], levs[:,i,j] * units('hPa'), height_agl[0,:,i,j], top = 1000 * units('meter'), bottom = 0 * units('meter'))
        sfc1_u[i,j] = shr1[0].magnitude
        sfc1_v[i,j] = shr1[1].magnitude

sfc1_u = (sfc1_u * units('m/s')).to('knot')
sfc1_v = (sfc1_v * units('m/s')).to('knot')

#Create the figure
fig=plt.figure(2,figsize=(30.,25.))
ax = plt.subplot(111,projection=ccrs.PlateCarree())
ax.coastlines('50m',edgecolor='black',linewidth=0.75)
#ax.add_feature(states_provinces,edgecolor='black',linewidth=0.5)
#Add states/counties and set extent
ax.add_feature(counties, edgecolor = 'black', linewidth = 0.5)
ax.add_feature(states, edgecolor = 'black', linewidth = 1.5)
ax.set_extent([LL[0],UR[0],LL[1],UR[1]])
stlev = [.25,.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7]
ch = ax.contourf(trlons, trlats, sigtor_p, stlev, cmap=plt.cm.magma, alpha = .65)
#Plot our radar data
refp = ax.pcolormesh(rlons, rlats, refl, cmap=plt.cm.gist_ncar, vmin = 10, vmax = 70)
# For some reason these come back as bytes instead of strings
stid = np.array([s.decode() for s in data['station']])
# Create a station plot pointing to an Axes to draw on as well as the location of points
stnp = stationplot = StationPlot(ax, lons, lats, transform=ccrs.PlateCarree(),
                          fontsize=20)
stnt = stationplot.plot_parameter('NW', tair, color='red')
stnd = stationplot.plot_parameter('SW', dewp, color='darkgreen')
stnpr = stationplot.plot_parameter('NE', slp)

# Add wind barbs
stnpb = stationplot.plot_barb(u, v)

# Plot the sky cover symbols in the center. We give it the integer code values that
# should be plotted, as well as a mapping class that can convert the integer values
# to the appropriate font glyph.
stns = stationplot.plot_symbol('C', cloud_cover, sky_cover)

# Plot station id -- using an offset pair instead of a string location
stntx = stationplot.plot_text((2, 0), stid)

#Plot CAPE because why not
cplev = np.arange(500,4000,500)
#cf = ax.contour(trlons, trlats, CAPE[0,:,:], cplev, cmap=plt.cm.autumn_r, linewidths = 4, alpha = .65)
#ch = ax.contourf(trlons, trlats, SRH[0,1,:,:], srhlev, cmap=plt.cm.BuPu, alpha = .65)

#plt.clabel(cf, fontsize=16, inline=1, inline_spacing=10, fmt='%i', rightside_up=True, use_clabeltext=True)

cs = plt.colorbar(refp, ticks = [20,25,30,35,40,45,50,55,60,65,70],norm=matplotlib.colors.Normalize(vmin=20, vmax=70),
                  shrink = .75, pad = 0)
cg = plt.colorbar(ch, shrink = .75, pad = .01)
cg.ax.tick_params(labelsize=25)
cg.set_label("Significant Tornado",size = 30)
cs.ax.tick_params(labelsize=25)
cs.set_label("Reflectivity",size = 30)
ax.barbs(trlons,trlats,sfc1_u,sfc1_v,length=10,regrid_shape=12, color = 'k')
plt.title('Radar Reflectivity, RAP Analysis, and Surface Obs '+str(time_start.year)+'-'+str(time_start.month)+'-'+str(time_start.day)+
          ' '+str(time_start.hour)+':'+str(time_start.minute)+' UTC', size = 35)
plt.tight_layout()
plt.savefig("SigTorShearMap.png")
plt.show()

tors = np.genfromtxt('June12Tornadoes.csv', usecols =(0,1), delimiter = ',')
torlats = tors[:,0]/100.
torlons = tors[:,1]/-100.

print(torlats)
print(torlons)

#Create the figure
fig=plt.figure(2,figsize=(30.,25.))
ax = plt.subplot(111,projection=ccrs.PlateCarree())
ax.coastlines('50m',edgecolor='black',linewidth=0.75)
#ax.add_feature(states_provinces,edgecolor='black',linewidth=0.5)
#Add states/counties and set extent
ax.add_feature(counties, edgecolor = 'black', linewidth = 0.5)
ax.add_feature(states, edgecolor = 'black', linewidth = 1.5)
ax.set_extent([LL[0],UR[0],LL[1],UR[1]])
stlev = [.25,.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7]
ch = ax.contourf(trlons, trlats, sigtor_p, stlev, cmap=plt.cm.magma, alpha = .65)
#Plot our radar data
refp = ax.pcolormesh(rlons, rlats, refl, cmap=plt.cm.gist_ncar, vmin = 10, vmax = 70)
# For some reason these come back as bytes instead of strings
stid = np.array([s.decode() for s in data['station']])
# Create a station plot pointing to an Axes to draw on as well as the location of points
stnp = stationplot = StationPlot(ax, lons, lats, transform=ccrs.PlateCarree(),
                          fontsize=20)
stnt = stationplot.plot_parameter('NW', tair, color='red')
stnd = stationplot.plot_parameter('SW', dewp, color='darkgreen')
stnpr = stationplot.plot_parameter('NE', slp)

# Add wind barbs
stnpb = stationplot.plot_barb(u, v)

# Plot the sky cover symbols in the center. We give it the integer code values that
# should be plotted, as well as a mapping class that can convert the integer values
# to the appropriate font glyph.
stns = stationplot.plot_symbol('C', cloud_cover, sky_cover)

# Plot station id -- using an offset pair instead of a string location
stntx = stationplot.plot_text((2, 0), stid)

#Plot CAPE because why not
cplev = np.arange(500,4000,500)
#cf = ax.contour(trlons, trlats, CAPE[0,:,:], cplev, cmap=plt.cm.autumn_r, linewidths = 4, alpha = .65)
#ch = ax.contourf(trlons, trlats, SRH[0,1,:,:], srhlev, cmap=plt.cm.BuPu, alpha = .65)

#plt.clabel(cf, fontsize=16, inline=1, inline_spacing=10, fmt='%i', rightside_up=True, use_clabeltext=True)

cs = plt.colorbar(refp, ticks = [20,25,30,35,40,45,50,55,60,65,70],norm=matplotlib.colors.Normalize(vmin=20, vmax=70),
                  shrink = .75, pad = 0)
cg = plt.colorbar(ch, shrink = .75, pad = .01)
cg.ax.tick_params(labelsize=25)
cg.set_label("Significant Tornado",size = 30)
cs.ax.tick_params(labelsize=25)
cs.set_label("Reflectivity",size = 30)
ax.barbs(trlons,trlats,sfc1_u,sfc1_v,length=10,regrid_shape=12, color = 'k')
ax.scatter(torlons, torlats, s = 250, color = 'r', marker = 'v')
plt.title('Radar Reflectivity, SFC-1 Shear/SigTor Parameter, Tornado Reports, \n and Surface Obs '+str(time_start.year)+'-'+str(time_start.month)+'-'+str(time_start.day)+
          ' '+str(time_start.hour)+':'+str(time_start.minute)+' UTC', size = 35)
plt.tight_layout()
plt.savefig("SigTorwReportsMap.png")
plt.show()

#Create the figure
fig=plt.figure(2,figsize=(30.,25.))
ax = plt.subplot(111,projection=ccrs.PlateCarree())
ax.coastlines('50m',edgecolor='black',linewidth=0.75)
#ax.add_feature(states_provinces,edgecolor='black',linewidth=0.5)
#Add states/counties and set extent
ax.add_feature(counties, edgecolor = 'black', linewidth = 0.5)
ax.add_feature(states, edgecolor = 'black', linewidth = 1.5)
ax.set_extent([LL[0],UR[0],LL[1],UR[1]])
srhlev = np.arange(50,700,50)
ch = ax.contourf(trlons, trlats, SRH[0,1,:,:], srhlev, cmap=plt.cm.viridis, alpha = .65)
#Plot our radar data
refp = ax.pcolormesh(rlons, rlats, refl, cmap=plt.cm.gist_ncar, vmin = 10, vmax = 70)
# For some reason these come back as bytes instead of strings
stid = np.array([s.decode() for s in data['station']])
# Create a station plot pointing to an Axes to draw on as well as the location of points
stnp = stationplot = StationPlot(ax, lons, lats, transform=ccrs.PlateCarree(),
                          fontsize=18)
stnt = stationplot.plot_parameter('NW', tair, color='red')
stnd = stationplot.plot_parameter('SW', dewp, color='darkgreen')
stnpr = stationplot.plot_parameter('NE', slp)

# Add wind barbs
stnpb = stationplot.plot_barb(u, v)

# Plot the sky cover symbols in the center. We give it the integer code values that
# should be plotted, as well as a mapping class that can convert the integer values
# to the appropriate font glyph.
stns = stationplot.plot_symbol('C', cloud_cover, sky_cover)

# Plot station id -- using an offset pair instead of a string location
stntx = stationplot.plot_text((2, 0), stid)

#Plot CAPE because why not
cplev = np.arange(500,4000,500)
cf = ax.contour(trlons, trlats, CAPE[0,:,:], cplev, cmap=plt.cm.autumn_r, linewidths = 4, alpha = .65)
#ch = ax.contourf(trlons, trlats, SRH[0,1,:,:], srhlev, cmap=plt.cm.BuPu, alpha = .65)

plt.clabel(cf, fontsize=18, inline=1, inline_spacing=10, fmt='%i', rightside_up=True, use_clabeltext=True)

cs = plt.colorbar(refp, ticks = [20,25,30,35,40,45,50,55,60,65,70],norm=matplotlib.colors.Normalize(vmin=20, vmax=70),
                  shrink = .75, pad = 0)
cg = plt.colorbar(ch, shrink = .75, pad = .01)
cg.ax.tick_params(labelsize=25)
cg.set_label("Storm-Relative Helicity",size = 30)
cs.ax.tick_params(labelsize=25)
cs.set_label("Reflectivity",size = 30)
plt.title('Radar Reflectivity, RAP Analysis, and Surface Obs '+str(time_start.year)+'-'+str(time_start.month)+'-'+str(time_start.day)+
          ' '+str(time_start.hour)+':'+str(time_start.minute)+' UTC', size = 35)
plt.tight_layout()
ax.scatter(torlons, torlats, s = 250, color = 'r', marker = 'v')
plt.savefig("ReallyCoolMapwTors.png")
plt.show()



