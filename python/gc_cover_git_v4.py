get_ipython().run_line_magic('matplotlib', 'inline')
from netCDF4 import Dataset
from gc_set_cover import *
import shapely.geometry as sgeom
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from shapely.ops import unary_union
from descartes.patch import PolygonPatch
from sklearn.utils.extmath import cartesian
from functools import reduce
import pickle

##loading nc4 file
nc_f = 'small_scan_blocks.nc4'  # filename
nc_fid = Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file
                             # and create an instance of the netCDF4 class

##blocks are groups into 5 min blocks
## naming format is 'degrees lat - scan time from E-W'
centroid_lat_lon = [group for group in nc_fid.groups] 
#print (centroid_lat_lon)

#for converting lat/lon to Geo
geo = ccrs.Geostationary(central_longitude=-85.0, satellite_height=42.336e6)
geo_proj4 = geo.proj4_init
latlon = {'init' :'epsg:4326'}

#creating geometries from scan block info using the corners
pgon = list()
blockname = []
for i in range(len(centroid_lat_lon)):
    temp_lon_arr = (nc_fid.groups[centroid_lat_lon[i]].variables['longitude_centre'][:].T - 360.)
    temp_lon_crns = [temp_lon_arr[0,-1],  #upper left corner
                     temp_lon_arr[0,0],   #upper right corner
                     temp_lon_arr[-1,0],  #lower right corner
                     temp_lon_arr[-1,-1]] #lower left corner
    temp_lat_arr = (nc_fid.groups[centroid_lat_lon[i]].variables['latitude_centre'][:].T)
    temp_lat_crns = [temp_lat_arr[0,-1],  #upper left corner
                     temp_lat_arr[0,0],   #upper right corner
                     temp_lat_arr[-1,0],  #lower right corner
                     temp_lat_arr[-1,-1]] #lower left corner
    temp_scan = zip(temp_lon_crns, temp_lat_crns) #(lon, lat) to conform to Cartesian (x, y)
    pgon.append(sgeom.Polygon(temp_scan))

#reframe as GeoSeries
blockset = gpd.GeoDataFrame( {'centroid_lat_lon' : centroid_lat_lon, 'geometry':pgon} )
blockset.crs = latlon #initialize the CRS

#convert lat/lon to geostationary coordinates using built-in Cartopy tools for analysis and plotting
blockset = blockset.to_crs(geo_proj4)
blockset = blockset[blockset['geometry'].is_valid].reset_index(drop=True) #takes out the blocks that scan into space
blockset = blockset.drop(where(blockset['geometry'].area > average(blockset.area))[0]).reset_index(drop=True) #drop blocks were generated with errors(size too large)

#load AF mesh scores
af_scores, timewindow = calc_afmesh_window(2007, 3, 20)
min_mesh = reduce(minimum, af_scores)

#creating a bounding envelope for Universal set
#We are generally concerned with the land masses lying between 50N and 50S and have minimum AF < 5.

#creates a grid for calculating airmass
# x = linspace(-130, -30, 201)
# y = linspace( 50, -50, 201)
# xv, yv = meshgrid(x,y)
#eliminate areas where minimum Airmass Factor > 5.
xind5 = where(min_mesh<=5.0)[0]
yind5 = where(min_mesh<=5.0)[1]
zone5 = sgeom.MultiPoint(list(zip(xv[xind5, yind5], yv[xind5, yind5]))).buffer(.51)
#read in Geometries
df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
df.loc[df['name'].isin(['Panama', 'Trinidad and Tobago']), 'continent'] = 'South America' #technically caribbean
n_am = df.query('continent == "North America"')
caribbean = ['Bahamas', 'Cuba', 'Dominican Rep.', 'Haiti', 'Jamaica', 'Puerto Rico', 'Trinidad and Tobago']
n_am = n_am[logical_not(n_am['name'].isin(caribbean))][['continent', 'geometry']] #remove caribbean islands
s_am = df.query('continent == "South America"')[['continent', 'geometry']]
frguiana = gpd.GeoDataFrame({'continent':'South America', 'geometry' : df[df['name'] == 'France'].intersection(zone5)}) #french guiana is listed under France < Europe
s_am = s_am.append(frguiana)
westhem = n_am.append(s_am)
continents = westhem.dissolve(by='continent')

extras = sgeom.box(-79.5, 23, -75, 27.5) #Northern Bermuda islands
n_am = n_am.difference(extras)

#eliminate extreme latitudes
U = continents.intersection(zone5)
U.crs = {'init' :'epsg:4326'}
envelope = sgeom.box(-130, -50, -30, 50)
ocean = envelope.difference(U.unary_union)
U = U.to_crs(geo_proj4).buffer(0)

#fixing precision issues by adding a buffer
U_set = U
blockset = blockset[blockset.intersects(U_set.unary_union)].reset_index(drop=True) #drop blocks that don't cover any land
blockset['geometry'] = blockset.buffer(5000)
coverage = blockset.unary_union.buffer(0)
print('Containment:', coverage.contains(U_set.unary_union))

#scan blocks aren't covering the Eastern portion of Nova Scotia
#so I'm subtracting the difference for the universe_set used in the algorithm
diff = U_set.difference(coverage)
universe_set = U_set.difference(diff.buffer(1000))
print('Difference area =\n', diff.area)
print('New difference area =\n',universe_set.difference(coverage).area) 

ax1= subplots(figsize=(5,5), dpi=200)
ax1 = axes(projection=geo)
ax1.add_feature(cfeat.NaturalEarthFeature('cultural', 'admin_0_countries', '110m'), facecolor='white', edgecolor='black', linewidth=0.5)
ax1.gridlines()
ax1.axis('scaled')
#built-in plotting function for geopandas objects
blockset.plot(ax=ax1, zorder=5, alpha=0.1, color='red', edgecolor='black')
#title('Candidate Scan Blocks')
#plt.savefig('candidates2')
show()

#plotting multipolygon geometry
#min_mesh = load('min_mesh_jan.npy')
BLUE = '#6699cc'
fig = plt.subplots(figsize=(5,5),dpi=200)
ax1 = plt.axes(projection=geo)
ax1.gridlines(zorder=0)
ax1.axis('scaled')
ax1.add_feature(cfeat.NaturalEarthFeature('cultural', 'admin_0_countries', '110m'),
                facecolor='white', edgecolor='black', linewidth=0.5, zorder=1)
ax1.add_patch( PolygonPatch(polygon=universe_set.unary_union, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2 ))
#ax1.add_patch( PolygonPatch(polygon=af_mesh.buffer(5000), color='red', alpha=0.5, zorder=3 ))
#blockset[blockset.intersects(universe_set)==True].plot(ax=ax1, alpha=0.1, facecolor='red',edgecolor='black', zorder=5)
blockset.plot(ax=ax1, alpha=0.1, facecolor='red',edgecolor='black', zorder=4)
#mesh = ax1.contourf(xv, yv, min_mesh, arange(2,8), zorder=2, transform=ccrs.PlateCarree(), cmap='autumn', alpha=0.5)
#colorbar(mesh, shrink=0.5 )
#title('Universe Set')
#savefig('universe_set_wcands')
show()

latlon_mesh = sgeom.MultiPoint(cartesian([x,y]))
geo_mesh = latlon_to_geo(latlon_mesh)
af_mesh = geo_mesh.intersection(universe_set.unary_union)

#plotting multipolygon geometry
#min_mesh = load('min_mesh_jan.npy')
BLUE = '#6699cc'
fig = plt.subplots(figsize=(3,3),dpi=200)
ax1 = plt.axes(projection=geo)
ax1.gridlines(zorder=0)
ax1.axis('scaled')
ax1.add_feature(cfeat.NaturalEarthFeature('cultural', 'admin_0_countries', '110m'),
                facecolor='white', edgecolor='black', linewidth=0.5, zorder=1)
ax1.add_patch( PolygonPatch(polygon=universe_set.unary_union, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2 ))
ax1.add_patch( PolygonPatch(polygon=af_mesh.buffer(2000), color='green', alpha=1, zorder=3 ))
#title('Universe Set with AF Mesh')
#savefig('Usetwmesh')
show()

coverset = gpd.GeoDataFrame()

print('Covering South America')
cover_sam = greedy_gc_cost(blockset = blockset,
                           universe_set = universe_set['South America'],
                           ocean = ocean,
                           mesh_pts = af_mesh,
                           mesh_airmass = af_scores,
                           tol = .001,
                           setmax = len(timewindow))
coverset = coverset.append(cover_sam, ignore_index = True)
print('Covering North America')
nam_new = universe_set['North America'].difference(cover_sam.unary_union)
cover_nam = greedy_gc_cost(blockset = blockset,
                           universe_set = nam_new,
                           ocean = ocean,
                           mesh_pts = af_mesh,
                           mesh_airmass = af_scores,
                           t = len(cover_sam),
                           tol = .001,
                           setmax = len(timewindow)-len(coverset))
coverset = coverset.append(cover_nam, ignore_index = True)
print('Covering extra time')
cover_extra = greedy_gc_cost(blockset = blockset,
                                universe_set = universe_set.unary_union,
                                ocean = ocean,
                                mesh_pts = af_mesh,
                                mesh_airmass = af_scores,
                                t = len(coverset),
                                tol = .001,
                                setmax = len(timewindow)-len(coverset))

beep()

#mpl plot object
ax1 = subplots(figsize=(5,5), dpi=200)
ax1 = axes(projection=geo)
ax1.add_feature(cfeat.NaturalEarthFeature('cultural', 'admin_0_countries', '110m'),
                linewidth=0.5,facecolor='white', edgecolor='black')
ax1.gridlines()
ax1.axis('scaled')
#U.plot(ax=ax1, zorder=4)
#universe_set.plot(ax=ax1, color = 'lightblue', alpha = 0.5, zorder=4)
#ax1.add_patch(PolygonPatch(af_mesh.buffer(2000), color='green', alpha=0.3, zorder=5 ))
#ax1.add_patch(PolygonPatch(dec.geometry[29], color='green', alpha=0.3, zorder=5 ))
cover_sam.plot(ax=ax1, zorder=6, alpha=0.2, edgecolor='black', facecolor='red')
cover_nam.plot(ax=ax1, zorder=7, alpha=0.2, edgecolor='black', facecolor='red')
cover_extra.plot(ax=ax1, zorder=8, alpha=0.2, edgecolor='black', facecolor='purple')
#mesh = ax1.contourf(xv, yv, min_mesh, arange(2,8), zorder=2, transform=ccrs.PlateCarree(), cmap='autumn', alpha=0.4)
#colorbar(mesh, shrink=0.5 )
#title('Cover Set, cost function')
#savefig('cover_2018dec21_bookended')
show()

coverset['time'] = timewindow[:len(coverset)]
coverset = coverset[['time', 'centroid_lat_lon', 'geometry']]

cover_extra['time'] = timewindow[len(coverset):]
cover_extra = cover_extra[['time', 'centroid_lat_lon', 'geometry']]

#save outputs
coverset.to_pickle('coverset_'+str(timewindow[0].isoformat())+'_pandas.pickle')
#save blocklist to txt
namelist=coverset['centroid_lat_lon']
with open('blocklist_'+str(timewindow[0].date())+'.txt', 'w') as output:
    output.write(namelist.to_string())

#save outputs
cover_extra.to_pickle('cover_extra_'+str(timewindow[0].isoformat())+'_pandas.pickle')
#save blocklist to txt
namelist=cover_extra['centroid_lat_lon']
with open('blocklist_extra_'+str(timewindow[0].isoformat())+'.txt', 'w') as output:
    output.write(namelist.to_string())

get_ipython().run_line_magic('run', "-i 'plot_coverset_mov.py'")

nc_f = 'monthly_modis_wsa_0.5deg_200703.nc'  # filename
modis_march = Dataset(nc_f, 'r') 

alb_map = modis_march.variables['wsa_band6'][19] #the map is oriented (x,y) = (0,0) at bottom left corner

error = calc_set_err(coverset=coverset, albedo = alb_map, af_mesh = af_mesh)

import seaborn as sns
style.use('ggplot')
fig, ax1 = subplots(1,1,figsize=(8,8))
sns.distplot(error[:,2][error[:,2]<14.], rug=True, kde_kws={'lw': 1})
plt.show()



