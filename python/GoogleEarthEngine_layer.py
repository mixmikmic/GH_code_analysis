import ee
import folium

ee.Initialize()
print(folium.__version__)

def folium_gee_map(image,vis_params=None,folium_kwargs={}):
    """
    Function to view Google Earth Engine tile layer as a Folium map.
    
    Parameters
    ----------
    image : Google Earth Engine Image.
    vis_params : Dict with visualization parameters.
    folium_kwargs : Keyword args for Folium Map.
    """
    
    # Get the MapID and Token after applying parameters
    image_info = image.getMapId(vis_params)
    mapid = image_info['mapid']
    token = image_info['token']
    folium_kwargs['attr'] = ('Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a> ')
    folium_kwargs['tiles'] = "https://earthengine.googleapis.com/map/%s/{z}/{x}/{y}?token=%s"%(mapid,token)
    
    return folium.Map(**folium_kwargs)

def folium_gee_layer(folium_map,image,vis_params=None,folium_kwargs={}):
    """
    Function to add Google Earch Engine tile layer as a Folium layer.
    
    Parameters
    ----------
    folium_map : Folium map to add tile to.
    image : Google Earth Engine Image.
    vis_params : Dict with visualization parameters.
    folium_kwargs : Keyword args for Folium Map.
    """
    
    # Get the MapID and Token after applying parameters
    image_info = image.getMapId(vis_params)
    mapid = image_info['mapid']
    token = image_info['token']
    folium_kwargs['attr'] = ('Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a> ')
    folium_kwargs['tiles'] = "https://earthengine.googleapis.com/map/%s/{z}/{x}/{y}?token=%s"%(mapid,token)
    
    layer = folium.TileLayer(**folium_kwargs)
    layer.add_to(folium_map)

# Get an area to look at
lat = 39.495159
lon = -107.3689237
zoom_start=10

# Open Street Map Base
m = folium.Map(location=[lat, lon], tiles="OpenStreetMap", zoom_start=zoom_start)

# Add GEE Terrain Layer
image = ee.Image('srtm90_v4')
vis_params = {'min':0.0, 'max':3000, 'palette':'00FFFF,0000FF'}
folium_gee_layer(m,image,vis_params=vis_params,folium_kwargs={'overlay':True,'name':'SRTM'})

# Create a reference to the image collection
l8 = ee.ImageCollection('LANDSAT/LC8_L1T_TOA')
# Filter the collection down to a two week period
filtered = l8.filterDate('2013-05-01', '2013-05-15');
# Use the mosaic reducer, to select the most recent pixel in areas of overlap
l8_image = filtered.median()
l8_vis_params = {'min': 0, 'max':0.3, 'bands':'B4,B3,B2'}
folium_gee_layer(m,l8_image,l8_vis_params,folium_kwargs={'overlay':True,'name':'Visual'})
m.add_child(folium.LayerControl())
m



