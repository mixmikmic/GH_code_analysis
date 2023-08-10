get_ipython().magic('matplotlib inline')
import datacube
import pandas
pandas.set_option('display.max_colwidth', 200)
pandas.set_option('display.max_rows', None)
from datacube.storage.masking import mask_invalid_data

#app is a note to let GA know what we are doing with it, user-defined
#when loading data, #make sure data is on same coordinate scale or convert 
dc=datacube.Datacube(app='learn-data-access')
dc

products = dc.list_products()
products.columns.tolist()

display_columns = ['name', 'description', 'platform', 'product_type', 'instrument', 'crs', 'resolution']
# #list only nbar products
nbar_list = products[products['product_type'] == 'nbar'][display_columns].dropna()
# productlist = products[display_columns].dropna()
nbar_list

measurements = dc.list_measurements()
measurements.columns.tolist()

display_columns = ['units', 'nodata', 'aliases']
# display meausrements for one product
measurements[display_columns].loc['ls8_nbar_albers']

#display measurements in all nbar products
m_list = nbar_list.name.tolist() #list of nbar measurements
measurements[display_columns].loc[m_list]

query = {
    'time': ('2013-01-01', '2014-12-31'),
    'lat': (-35.2, -35.3),
    'lon': (149.0, 149.1),
}
#for landsat, even 1x1 degree for 4 yrs is too much data

# attempt landsat
#2 stars unpack the limits of our query, we load specific measurements from a product
# data = dc.load(product='ls8_nbar_albers', measurements=['red','nir'], **query)
#2 stars unpack the limits of our query, we load specific measurements from a product
data = dc.load(product='ls5_nbar_albers', measurements=['red', 'green', 'blue'], **query)

#mask invalid data removes clouds/shadows
data = mask_invalid_data(data)
data

data.time

#imshow needs the thing plotted to be the last dimension, so reshape.  
#imshow wants data from 0 to 1; 4000 is a scaling factor for landsat data

rgb=img.transpose('y','x','band') / 4000
plt.imshow(rgb)

#take mean along time dimension, then stick them together along new dimension band
# so mean creates three matrices (R,G,B), to_array puts them together in a 3D array where the third dimension is band
img = data.mean(dim='time').to_array(dim='band')
img

ndvi = (data.nir-data.red)/(data.nir+data.red)
ndvi

query2 = {
    'lat': (-35.2, -35.4),
    'lon': (149.0, 149.2),
}
#make sure data is on same coordinate sclae
tp = dc.load(product='srtm_dem1sv1_0',measurements=['dem'],output_crs='EPSG:3577',resolution=(-25,25),resampling='cubic',**query2)
tp

tp.dem.squeeze().plot.contourf(levels=10,cmap='cubehelix')

#plot 2 data sets on same axis
fig,ax=plt.subplots(1,figsize=(10,10))
ax.imshow(rgb)
ax.contour(dem)
fig

data.green.isel(time=6).plot.imshow(robust=True) 

# ## attempt MODIS
# #2 stars unpack the limits of our query, we load specific measurements from a product
# data = dc.load(product='modis_mcd43a3_tile', measurements=['BRDF_Albedo_Band_Mandatory_Quality_Band1', 'Nadir_reflectance_Band2', 'Nadir_reflectance_Band3'], **query)
# #mask invalid data removes clouds/shadows
# data = mask_invalid_data(data)
# data



