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

display_columns = ['name','discription','platform','product_type','instrument','crs','resolution']
display_columns

products = dc.list_products()
products.columns.tolist()

display_columns = ['name', 'description', 'platform', 'product_type', 'instrument', 'crs', 'resolution']
# #list only nbar products
nbar_list = products[products['product_type'] == 'nbar'][display_columns].dropna()
# productlist = products[display_columns].dropna()
productlist

query = {
    'time': ('2000-01-01', '2017-12-31'),
    'lat': (-35, -36),
    'lon': (146.0, 147.0),
}

#2 stars unpack the limits of our query, we load specific measurements from a product
data = dc.load(product='modis_mcd43a3_tile', measurements=['BRDF_Albedo_Band_Mandatory_Quality_Band1', 'Nadir_reflectance_Band2', 'Nadir_reflectance_Band3'], **query)
#mask invalid data removes clouds/shadows
data = mask_invalid_data(data)
data



