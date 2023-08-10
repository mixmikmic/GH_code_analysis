# Set height (y-axis length) and width (x-axis length) which will be used to train model on later
img_height, img_width = (256,256)  #Default to (256,256)

# Import all the necessary libraries
import os
import glob
import io
import sys

import matplotlib.pyplot as plt
import skimage.io                                     #Used for imshow function
import tqdm
from PIL import Image, ImageDraw

import numpy as np

import geopandas as gpd
import ogr
from owslib.wms import WebMapService
import rasterio
import rasterio.features
import shapely.geometry

print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
print('Geopandas    :', gpd.__version__)
print('Skimage      :', skimage.__version__)

# Have a look at our data folder
print(os.listdir('data'))  #see what's in the input folder (where data is in)

def get_arrays(gdf:gpd.GeoDataFrame, output_shape:tuple=(256,256)):
    
    ## Setup Sentinel Hub WMS Service to access Sentinel 2 RGB imagery
    api_key = '5b63030a-a576-4cf4-af58-b2f28ca6b987'  #get this from the sentinel hub website
    wms = WebMapService(url=f'https://services.sentinel-hub.com/ogc/wms/{api_key}', version='1.1.1')
    
    ## Loop through our digitized vector polygons used as training labels
    rasterList = []
    maskList = []
    
    for i in tqdm.tqdm(range(len(gdf))):
        geofeature = gdf['geometry'][i]  #get a single geometry feature first
        
        midx, midy = geofeature.centroid.x, geofeature.centroid.y  #get centroid of geometry obj
        tight_bbox = geofeature.bounds  #get tight boundaries of the geometry object (xmin, ymin, xmax, ymax)
        
        # Retrieve bounding box
        maxlength = max([tight_bbox[2]-tight_bbox[0], tight_bbox[3]-tight_bbox[1]])  #get max length, either x width or y height
        maxlength = max(maxlength, 10000)  #set minimum maxlength to 10000 so we don't get crops too close to earth
        xmin, xmax = midx-maxlength/2, midx+maxlength/2
        ymin, ymax = midy-maxlength/2, midy+maxlength/2
        
        bbox = (xmin, ymin, xmax, ymax)  #tuple format bounding box
        boundingbox = shapely.geometry.box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)  #shapely format bounding box
                
        # Use Sentinel Hub WMS service to get an image
        img = wms.getmap(layers=['TRUE_COLOR'],
                 styles='',
                 format='image/jpeg',
                 transparent=False,
                 maxcc=50,
                 gain=0.3,
                 gamma=0.8,
                 time='2018-02-10%2F2018-02-14',
                 size=output_shape,
                 srs='EPSG:3857',
                 bbox=bbox
                )
        im = Image.open(fp=io.BytesIO(img.read()))
        raster = np.asarray(a=im, dtype=np.uint8)
        rasterList.append(raster)
        
        ## MASK - We burn the vectors inside the bounding box into a raster mask
        intermediate_vector = gdf.cx[xmin:xmax, ymin:ymax]   #get all vectors that intersects in the bbox
        vector = intermediate_vector.intersection(boundingbox)  #do full intersection crop!
        
        transform = rasterio.transform.from_bounds(west=xmin, south=ymin, east=xmax, north=ymax, width=output_shape[1], height=output_shape[0])
        burned_vector = rasterio.features.rasterize(shapes=vector.geometry, out_shape=output_shape, transform=transform,
                                                    fill=0, all_touched=True, default_value=1, dtype=np.uint8)
        mask = np.expand_dims(a=burned_vector.astype(np.bool), axis=-1)  #convert to shape like (256,256,1) with dtype=boolean
        maskList.append(mask)
    
    #return vector, raster, boundingbox, mask
    return np.stack(arrays=rasterList, axis=0), np.stack(arrays=maskList, axis=0)  #output raster (RGB image) and mask (Boolean mask)

#Read in list of geojson files as WGS84, convert to 3857, concatenate into a single geopandas geodataframe
geojsons = [gpd.read_file(filename=geojson).to_crs(epsg=3857) for geojson in glob.glob(pathname='data/sentinel2-20180212/crevasses_*.geojson')]
gdf = gpd.concat(objs=geojsons, ignore_index=True)
print(f'{len(gdf)} polygons loaded from the geojson file(s)')

if not os.path.exists('model/train/X_data.npy') or not os.path.exists('model/train/Y_data.npy'):
    X_data, Y_data = get_arrays(gdf=gdf)
    # Save array to disk
    os.makedirs(name='model/train', exist_ok=True)
    np.save('model/train/X_data.npy', X_data.data)
    np.save('model/train/Y_data.npy', Y_data.data)
elif os.path.exists('model/train/X_data.npy') and os.path.exists('model/train/Y_data.npy'):
    X_data = np.load('model/train/X_data.npy')
    Y_data = np.load('model/train/Y_data.npy')

print(X_data.shape, X_data.dtype)
print(Y_data.shape, Y_data.dtype)

id = 128
fig, axarr = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(15,15))
axarr[0, 0].imshow(X_data[id])
axarr[0, 1].imshow(Y_data[id][:,:,0])
plt.show()

import quilt

if os.path.exists('model/build.yml'):
    os.remove('model/build.yml')
quilt.generate(directory="model")

quilt.login()

quilt.build(package='weiji14/nz_space_challenge', path='model/build.yml')

quilt.push(package='weiji14/nz_space_challenge', is_public=True)



