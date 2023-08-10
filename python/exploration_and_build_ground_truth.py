import matplotlib.pyplot as plt
import numpy as np

get_ipython().magic('matplotlib inline')


import rasterio

src = rasterio.open('../AOI_4_Shanghai_Train/MUL-PanSharpen/MUL-PanSharpen_AOI_4_Shanghai_img1911.tif')
img = src.read([5, 3, 2]).transpose([1,2,0])

def scale_bands(img, lower_pct = 1, upper_pct = 99):
    """
    Rescale the bands of a multichannel image for display
    """
    # Loop through the image bands, rescaling each one
    img_scaled = np.zeros(img.shape, np.uint8)
    
    for i in range(img.shape[2]):
        
        band = img[:, :, i]
        
        # Pick out the lower and upper percentiles
        lower, upper = np.percentile(band, [lower_pct, upper_pct])
        
        # Normalize the band
        band = (band - lower) / (upper - lower) * 255
        
        # Clip the high and low values, and cast to uint8
        img_scaled[:, :, i] = np.clip(band, 0, 255).astype(np.uint8)
        
    return img_scaled


# Plot the rescaled image
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(scale_bands(img))

src.bounds

src2 = rasterio.open('../AOI_4_Shanghai_Train/MUL-PanSharpen/MUL-PanSharpen_AOI_4_Shanghai_img1910.tif')
src2.bounds

import os

import geopandas as gpd
import shapely.geometry

image_dir_path = '../AOI_4_Shanghai_Train/RGB-PanSharpen'

# Collect a list of all the images with the .tif extension
tiff_images = [os.path.join(image_dir_path, image_name) 
               for image_name in os.listdir(image_dir_path) 
               if image_name.lower().endswith('tif')]

res = []

for image_file_name in tiff_images:
    
    with rasterio.open(image_file_name) as src:
        
        # Convert the image bounding box into a shapely polygon
        bbox = shapely.geometry.box(*src.bounds)
        
    res.append((image_file_name, bbox))

# Convert the results into a GeoDataFrame
image_summary = gpd.GeoDataFrame(res, columns=['image_name', 'geometry'], crs={'init': 'epsg:4326'})

image_summary.head()

image_summary.plot()

output_file_name = 'vectors/shanghai_RGB_image_summary.geojson'

# Make sure the file doesnt exist (the program will crash if it does)
if os.path.exists(output_file_name):
    os.remove(output_file_name)

# Save the file as a geojson
image_summary.to_file(output_file_name, driver='GeoJSON')

meta_df = gpd.read_file('vectors/shanghai_RGB_image_summary.geojson')
poly = shapely.geometry.box(*meta_df.unary_union.bounds)

import geopandas_osm.osm
landuse_df = geopandas_osm.osm.query_osm('way', poly, recurse='down', tags='landuse')

landuse = landuse_df[~landuse_df.landuse.isnull()][['landuse', 'name', 'geometry']]

landuse.to_file('vectors/shangai_landuse_RGB.geojson', 'GeoJSON');

from collections import Counter
Counter(landuse.landuse.values)

fig, ax = plt.subplots()
landuse.set_geometry(landuse.geometry.apply(shapely.geometry.Polygon), inplace=True)
landuse[landuse.landuse == 'farmland'].plot(color='g', ax=ax)

farmland = landuse[landuse.landuse == 'farmland'].unary_union

images_containing_farmland = image_summary[image_summary.intersects(farmland)]
images_containing_farmland.head()

images_containing_farmland.size

from descartes import PolygonPatch

# # Calculate the proportion of each image taken up by forest
# proportion_forest = images_containing_forest.intersection(all_forest).area / images_containing_forest.area

# # Find the images with between 20% and 80% forest
# has_some_forest = np.logical_and(0.2 < proportion_forest, proportion_forest < 0.8)
# images_containing_some_forest = images_containing_forest[has_some_forest]

# # Pick the first image on the list
# file_name = images_containing_some_forest.image_name.values[0]

# # Load the image
# with rasterio.open(file_name) as src:
#         img = scale_bands(src.read([5, 3, 2]).transpose([1,2,0]))
        #print(src.read()[[5, 3, 2, 0, 1, 4, 6, 7], :, :])
img_bounds = shapely.geometry.box(*src.bounds)
img_transform = list(np.array(~src.affine)[[0, 1, 3, 4, 2, 5]])
        
# Get the intersection between the forest and the image bounds
image_farmland_area = farmland.intersection(img_bounds)

# Transform it into pixel coordinates
image_farmland_area_pxcoords = shapely.affinity.affine_transform(image_farmland_area, img_transform)

fig, ax = plt.subplots(figsize=(12,12))

# Plot the image
ax.imshow(scale_bands(img))

# Plot the forest on top of the image
ax.add_patch(PolygonPatch(image_farmland_area_pxcoords, fc='g', alpha=0.4, hatch='//'))

farmyard = landuse[landuse.landuse == 'farmyard'].unary_union
image_farmyard_area = farmyard.intersection(img_bounds)
image_farmyard_area_pxcoords = shapely.affinity.affine_transform(image_farmyard_area, img_transform)

# Subtract out the farmyards from the farmland coordinates
image_farmland_area_pxcoords = image_farmland_area_pxcoords - image_farmyard_area_pxcoords

fig, ax = plt.subplots(figsize=(12,12))

# Plot the image
ax.imshow(scale_bands(img))

# Plot the forest on top of the image
ax.add_patch(PolygonPatch(image_farmland_area_pxcoords, fc='g', alpha=0.4, hatch='//'))

import cv2

def polycoords(poly):
    """Convert a polygon into the format expected by OpenCV
    """
    if poly.type in ['MultiPolygon', 'GeometryCollection']:
        return [np.array(p.exterior.coords) for p in poly if p.type == 'Polygon']
    elif poly.type == 'Polygon':
        return [np.array(poly.exterior.coords)]
    else:
        print('Encountered unrecognized geometry type {}. Ignoring.'.format(poly.type))
        return []

def make_mask(img_shape, poly):
    """Make a mask from a polygon"""
    poly_pts = polycoords(poly)
    polys = [x.astype(int) for x in poly_pts]
    # Create an empty mask and then fill in the polygons
    mask = np.zeros(img_shape[:2])
    cv2.fillPoly(mask, polys, 255)
    return mask.astype('uint8')

# Convert the forest polygon into a mask
farmland_mask = make_mask(img.shape, image_farmland_area_pxcoords)

# Plot it
plt.imshow(farmland_mask, cmap=plt.cm.gray)



