# See requirements.txt to set up your dev environment.
import os
import sys
import utm
import json
import scipy
import overpy
import urllib
import datetime 
import urllib3
import rasterio
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
from osgeo import gdal
from planet import api
from planet.api import filters
from traitlets import link
import rasterio.tools.mask as rio_mask
from shapely.geometry import mapping, shape
from IPython.display import display, Image, HTML
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from scipy import ndimage
import warnings
from osgeo import gdal


from osmapi import OsmApi
from geopy.geocoders import Nominatim

urllib3.disable_warnings()
from ipyleaflet import (
    Map,
    Marker,
    TileLayer, ImageOverlay,
    Polyline, Polygon, Rectangle, Circle, CircleMarker,
    GeoJSON,
    DrawControl
)

get_ipython().magic('matplotlib inline')
# will pick up api_key via environment variable PL_API_KEY
# but can be specified using `api_key` named argument
api_keys = json.load(open("apikeys.json",'r'))
client = api.ClientV1(api_key=api_keys["PLANET_API_KEY"])
gdal.UseExceptions()
api = overpy.Overpass()

# Basemap Mosaic (v1 API)
mosaicsSeries = 'global_quarterly_2017q1_mosaic'
# Planet tile server base URL (Planet Explorer Mosaics Tiles)
mosaicsTilesURL_base = 'https://tiles0.planet.com/experimental/mosaics/planet-tiles/' + mosaicsSeries + '/gmap/{z}/{x}/{y}.png'
# Planet tile server url
mosaicsTilesURL = mosaicsTilesURL_base + '?api_key=' + api_keys["PLANET_API_KEY"]
# Map Settings 
# Define colors
colors = {'blue': "#009da5"}
# Define initial map center lat/long
center = [45.5231, -122.6765]
# Define initial map zoom level
zoom = 13
# Set Map Tiles URL
planetMapTiles = TileLayer(url= mosaicsTilesURL)
# Create the map
m = Map(
    center=center, 
    zoom=zoom,
    default_tiles = planetMapTiles # Uncomment to use Planet.com basemap
)
# Define the draw tool type options
polygon = {'shapeOptions': {'color': colors['blue']}}
rectangle = {'shapeOptions': {'color': colors['blue']}} 

# Create the draw controls
# @see https://github.com/ellisonbg/ipyleaflet/blob/master/ipyleaflet/leaflet.py#L293
dc = DrawControl(
    polygon = polygon,
    rectangle = rectangle
)
# Initialize an action counter variable
actionCount = 0
AOIs = {}

# Register the draw controls handler
def handle_draw(self, action, geo_json):
    # Increment the action counter
    global actionCount
    actionCount += 1
    # Remove the `style` property from the GeoJSON
    geo_json['properties'] = {}
    # Convert geo_json output to a string and prettify (indent & replace ' with ")
    geojsonStr = json.dumps(geo_json, indent=2).replace("'", '"')
    AOIs[actionCount] = json.loads(geojsonStr)
    
# Attach the draw handler to the draw controls `on_draw` event
dc.on_draw(handle_draw)
m.add_control(dc)
m

print AOIs[1]
myAOI = AOIs[1]["geometry"]

# build a query using the AOI and
# a cloud_cover filter that excludes 'cloud free' scenes

old = datetime.datetime(year=2017,month=1,day=1)

query = filters.and_filter(
    filters.geom_filter(myAOI),
    filters.range_filter('cloud_cover', lt=5),
    filters.date_range('acquired', gt=old)
)

# build a request for only PlanetScope imagery
request = filters.build_search_request(
    query, item_types=['PSScene3Band']
)

# if you don't have an API key configured, this will raise an exception
result = client.quick_search(request)
scenes = []
planet_map = {}
for item in result.items_iter(limit=500):
    planet_map[item['id']]=item
    props = item['properties']
    props["id"] = item['id']
    props["geometry"] = item["geometry"]
    props["thumbnail"] = item["_links"]["thumbnail"]
    scenes.append(props)
scenes = pd.DataFrame(data=scenes)
# now let's clean up the datetime stuff
# make a shapely shape from our aoi
portland = shape(myAOI)
footprints = []
overlaps = []
# go through the geometry from our api call, convert to a shape and calculate overlap area.
# also save the shape for safe keeping
for footprint in scenes["geometry"].tolist():
    s = shape(footprint)
    footprints.append(s)
    overlap = 100.0*(portland.intersection(s).area / portland.area)
    overlaps.append(overlap)
# take our lists and add them back to our dataframe
scenes['overlap'] = pd.Series(overlaps, index=scenes.index)
scenes['footprint'] = pd.Series(footprints, index=scenes.index)
# now make sure pandas knows about our date/time columns.
scenes["acquired"] = pd.to_datetime(scenes["acquired"])
scenes["published"] = pd.to_datetime(scenes["published"])
scenes["updated"] = pd.to_datetime(scenes["updated"])

scenes = scenes[scenes['overlap']>0.9]


print len(scenes)
# now let's clean up the datetime stuff
# make a shapely shape from our aoi
portland = shape(myAOI)
footprints = []
overlaps = []
# go through the geometry from our api call, convert to a shape and calculate overlap area.
# also save the shape for safe keeping
for footprint in scenes["geometry"].tolist():
    s = shape(footprint)
    footprints.append(s)
    overlap = 100.0*(portland.intersection(s).area / portland.area)
    overlaps.append(overlap)
# take our lists and add them back to our dataframe
scenes['overlap'] = pd.Series(overlaps, index=scenes.index)
scenes['footprint'] = pd.Series(footprints, index=scenes.index)
# now make sure pandas knows about our date/time columns.
scenes["acquired"] = pd.to_datetime(scenes["acquired"])
scenes["published"] = pd.to_datetime(scenes["published"])
scenes["updated"] = pd.to_datetime(scenes["updated"])

# first create a list of colors
colors = ["#ff0000","#00ff00","#0000ff","#ffff00","#ff00ff","#00ffff"]
# grab our scenes from the geometry/footprint geojson
footprints = scenes["geometry"].tolist()
# for each footprint/color combo

for footprint,color in zip(footprints,colors):
    # create the leaflet object
    feat = {'geometry':footprint,"properties":{
            'style':{'color': color,'fillColor': color,'fillOpacity': 0.1,'weight': 1}},
            'type':u"Feature"}
    # convert to geojson
    gjson = GeoJSON(data=feat)
    # add it our map
    m.add_layer(gjson)
# now we will draw our original AOI on top 
feat = {'geometry':myAOI,"properties":{
            'style':{'color': "#FFFFFF",'fillColor': "#FFFFFF",'fillOpacity': 0.1,'weight': 2}},
            'type':u"Feature"}
gjson = GeoJSON(data=feat)
m.add_layer(gjson)   
m 

def get_products(client, scene_id, asset_type='PSScene3Band'):    
    """
    Ask the client to return the available products for a 
    given scene and asset type. Returns a list of product 
    strings
    """
    out = client.get_assets_by_id(asset_type,scene_id)
    temp = out.get()
    return temp.keys()

def activate_product(client, scene_id, asset_type="PSScene3Band",product="analytic"):
    """
    Activate a product given a scene, an asset type, and a product.
    
    On success return the return value of the API call and an activation object
    """
    temp = client.get_assets_by_id(asset_type,scene_id)  
    products = temp.get()
    if( product in products.keys() ):
        return client.activate(products[product]),products[product]
    else:
        return None 

def download_and_save(client,product):
    """
    Given a client and a product activation object download the asset. 
    This will save the tiff file in the local directory and return its 
    file name. 
    """
    out = client.download(product)
    fp = out.get_body()
    fp.write()
    return fp.name

def scenes_are_active(scene_list):
    """
    Check if all of the resources in a given list of
    scene activation objects is read for downloading.
    """
    retVal = True
    for scene in scene_list:
        if scene["status"] != "active":
            print "{} is not ready.".format(scene)
            return False
    return True

to_get = scenes["id"][0:7].tolist()
activated = []
# for each scene to get
for scene in to_get:
    # get the product 
    product_types = get_products(client,scene)
    for p in product_types:
        # if there is a visual product
        if p == "visual": # p == "basic_analytic_dn"
            print "Activating {0} for scene {1}".format(p,scene)
            # activate the product
            _,product = activate_product(client,scene,product=p)
            activated.append(product)

tiff_files = []
asset_type = "_3B_Visual"
# check if our scenes have been activated
if True:#scenes_are_active(activated):
    for to_download,name in zip(activated,to_get):
        # create the product name
        name = name + asset_type + ".tif"
        # if the product exists locally
        if( os.path.isfile(name) ):
            # do nothing 
            print "We have scene {0} already, skipping...".format(name)
            tiff_files.append(name)
        elif to_download["status"] == "active":
            # otherwise download the product
            print "Downloading {0}....".format(name)
            fname = download_and_save(client,to_download)
            tiff_files.append(fname)
            print "Download done."
        else:
            print "Could not download, still activating"
else:
    print "Scenes aren't ready yet"

sorted(tiff_files)
print tiff_files 

infile = tiff_files[0]
# Open the file
gtif = gdal.Open(infile)
# Get the project reference object this knows the UTM zone
reff = gtif.GetProjectionRef()
# arr is the actual image data.
arr = gtif.ReadAsArray()
# Trans is our geo transfrom array. 
trans = gtif.GetGeoTransform()
# print the ref object
print reff
# find our UTM zone
i = reff.find("UTM")
print reff[i:i+12]

def pixel2utm(ds, x, y):
    """
    Returns utm coordinates from pixel x, y coords
    """
    xoff, a, b, yoff, d, e = ds.GetGeoTransform()
    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff
    return(xp, yp)

def draw_point(x,y,img):
    t = 20
    # a cloud_cover filter that ex
    img[y-t:y+t,x-t:x+t,:] = [255,0,0]

pos = [3000,1400] # this is the pixel we want info abou
ds = gdal.Open(infile)
# take the GDAL info and make it into UTM
my_utm = pixel2utm(ds,pos[0],pos[1])
# convert UTM into Lat Long
# need to figure out how to get zone info
my_lla = utm.to_latlon(my_utm[0],my_utm[1],10,"N")
# do the lat long look up from OSM
geolocator = Nominatim()
# reverse look up the are based on lat lon
location = geolocator.reverse("{0},{1}".format(my_lla[0],my_lla[1]))
# print location info
print location.address
print location.raw
# get the OSM ID info
osm_id = int(location.raw["place_id"])
print osm_id
# create an interface to the OSM API
MyApi = OsmApi()
# Look up our position 
print MyApi.NodeGet(osm_id)

from matplotlib.patches import Circle
fig,ax = plt.subplots(1)

# create our plot
plt.imshow(arr[:3,:,:].transpose((1, 2, 0)))#, extent=extent)
fig = plt.gcf()
# add our annotation
plt.annotate(location.address, xy=pos, xycoords='data',
             xytext=(0.25, 0.5), textcoords='figure fraction',color="red",
             arrowprops=dict(arrowstyle="->"))
ax.set_aspect('equal')
# Set a point
circ = Circle((pos[0],pos[1]),60,color="red")
ax.add_patch(circ)
fig.set_size_inches(18.5, 10.5)
plt.show()

import geopandas as gpd
fname = "./portland_parks_small.geojson"
park_df = gpd.read_file(fname)
portland_parks = json.load(open(fname,'r'))
# raw geojson works better with GDAL
geojson = [p for p in portland_parks["features"]]
# no area out of the box
p = [p.area for p in park_df["geometry"].tolist()]
park_df["area"] = pd.Series(p)
park_df["geojson"] = pd.Series(geojson)
park_df.sort_values(['area',], ascending=[1])
park_df.head()
#len(park_df)
#print park_df["wikipedia"].dropna()

for p in portland_parks["features"]:
    feat = {'geometry':p["geometry"],"properties":{
            'style':{'color': "#00FF00",'fillColor': "#00FF00",'fillOpacity': 0.0,'weight': 1}},
            'type':u"Feature"}
    # convert to geojson
    gjson = GeoJSON(data=feat)
    # add it our map
    m.add_layer(gjson)
m

park_sz = park_df.groupby("name").sum() 
park_sz = park_sz.sort_values(by='area',ascending=[0])
display(park_sz)

def write_geojson_by_name(df,name,outfile):
    """
    Take in a dataframe, a park name, and an output file name 
    Save the park's geojson to the specified file.
    """
    temp = df[df["name"]==name]
    to_write = {"type": "FeatureCollection",
                "features": temp["geojson"].tolist()}
    with open(outfile,'w') as fp:
        fp.write(json.dumps(to_write))
def crop_scenes_to_geojson(geojson,scenes,out_name):
    """
    Take in a geojson file, a list of scenes, and an output name
    Call gdal and warp the scenes to match the geojson file and save the results to outname.
    """
    commands = ["gdalwarp", # t
           "-t_srs","EPSG:3857",
           "-cutline",geojson,
           "-crop_to_cutline",
           "-tap",
            "-tr", "3", "3"
           "-overwrite"]
    for tiff in scenes:
        commands.append(tiff)
    commands.append(out_name)
    print " ".join(commands)
    subprocess.call(commands)

geo_json_files = []
tif_file_names = []
unique_park_names = list(set(park_df["name"].tolist()))
for name in list(unique_park_names):
    # Generate our file names 
    geojson_name = "./parks/"+name.replace(" ","_")+".geojson"
    tif_name = "./parks/"+name.replace(" ","_")+".tif"
    # write geojson
    write_geojson_by_name(park_df,name,geojson_name)
    # write to park file
    crop_scenes_to_geojson(geojson_name,tiff_files,tif_name)
    # Save the results to lists
    geo_json_files.append(geojson_name)
    tif_file_names.append(tif_name)

magic = ["mogrify","-format", "jpg", "./parks/*.tif"]
subprocess.call(magic)
for p in tif_file_names[0:20]:
    print p
    display(Image(p.replace('tif','jpg')))

def load_image3(filename):
    """Return a 3D (r, g, b) numpy array with the data in the specified TIFF filename."""
    path = os.path.abspath(os.path.join('./', filename))
    if os.path.exists(path):
        with rasterio.open(path) as src:
            b,g,r,mask = src.read()
            return np.dstack([b, g, r])
        

def get_avg_greeness(filename):
    retVal = -1.0
    try:
        # load the image
        img = load_image3(filename)
        if img is not None:
            # add all the channels together, black pixels will still be zero
            # this isn't a perfect method but there are very few truly black spots 
            # on eart
            black_like_my_soul = np.add(np.add(img[:,:,0],img[:,:,1]),img[:,:,2])
            # sum up the not black pixels
            not_black = np.count_nonzero(black_like_my_soul)
            # sum up all the green
            img = np.array(img,dtype='int16')
            total_green = np.sum(img[:,:,1]-((np.add(img[:,:,0],img[:,:,2])/2)))
            # calculate our metric
            if total_green != 0 and not_black > 0:
                retVal = total_green / float(not_black)
        return retVal
    except Exception as e:
        print e
        return -1.0

greens = [get_avg_greeness(f) for f in tif_file_names]
print greens

paired = zip(tif_file_names,greens)
paired.sort(key=(lambda tup: tup[1]))
paired.reverse()
labels = [p[0][8:-4].replace("_"," ") for p in paired]
data = [p[1] for p in paired]
plt.figure(figsize=(20,6))
xlocations = np.array(range(len(paired)))+0.5
width = 1
plt.bar(xlocations, data, width=width)
plt.yticks(range(-1,25,1))
plt.xticks(xlocations+ width/2, labels)
plt.xlim(0, xlocations[-1]+width*2)
plt.ylim(-2,np.max(data)+1)
plt.title("Greeness over Average Red and Blue Per Park")
plt.gca().get_xaxis().tick_bottom()
plt.gca().get_yaxis().tick_left()
xa = plt.gca()
xa.set_xticklabels(xa.xaxis.get_majorticklabels(), rotation=90)
plt.show()

imgs = [p[0] for p in paired]
for p in imgs[0:35]:
    print p[8:-4].replace("_"," ")
    display(Image(p.replace('tif','jpg')))

opacity_map = {}
gmax = np.max(greens)
gmin = np.min(greens)
# this is a nonlinear mapping
opacity = [np.clip((float(g**2)-gmin)/float(gmax-gmin),0,1) for g in greens]
for op,name in zip(opacity,imgs):
    opacity_map[name]=op

m = Map(
    center=center, 
    zoom=zoom,
    default_tiles = planetMapTiles # Uncomment to use Planet.com basemap
)
dc = DrawControl(
    polygon = polygon,
    rectangle = rectangle
)
# Initialize an action counter variable
actionCount = 0
AOIs = {}

# Register the draw controls handler
def handle_draw(self, action, geo_json):
    # Increment the action counter
    global actionCount
    actionCount += 1
    # Remove the `style` property from the GeoJSON
    geo_json['properties'] = {}
    # Convert geo_json output to a string and prettify (indent & replace ' with ")
    geojsonStr = json.dumps(geo_json, indent=2).replace("'", '"')
    AOIs[actionCount] = json.loads(geojsonStr)
    
# Attach the draw handler to the draw controls `on_draw` event
dc.on_draw(handle_draw)
m.add_control(dc)
m

for p in portland_parks["features"]:
    t = "./parks/"+p["properties"]["name"].replace(" ","_") + ".tif"
    feat = {'geometry':p["geometry"],"properties":{
            'style':{'color': "#00FF00",'fillColor': "#00FF00",'fillOpacity': opacity_map[t],'weight': 1}},
            'type':u"Feature"}
    # convert to geojson
    gjson = GeoJSON(data=feat)
    # add it our map
    m.add_layer(gjson)
m



