import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import random

import os
import rasterio
import shapely.geometry

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

#!ls /home/ubuntu/data/TX_post/*.tif

# Tell the script where to look for images
path_to_harvey_images = '/home/ubuntu/data/TX_post/'

# Load the shapefile with the flooding
file = '/home/ubuntu/data/vector_data/Flood_Analysis_shapefile/20170831_130614_Flood_Analysis.shp'
#file = 'digitalglobe_crowdsourcing_hurricane_harvey_raw_results_20170905.geojson'
flooding = gpd.read_file(file)


# Convert the CRS of the flooding file
flooding.to_crs({'init': 'epsg:4326'}, inplace=True)

flooding.head()

#flooding.plot(figsize=(15,15))

#flooding.plot()

# Get a list of all of the tif files
tif_files = [os.path.join(path_to_harvey_images, fn) for fn in os.listdir(path_to_harvey_images) if fn.endswith('tif')]
tif_files[:3]

# Get the shape of each image
geometry = []
for tif_file in tif_files:
    with rasterio.open(tif_file) as src:
        geometry.append(shapely.geometry.box(*src.bounds))

# Create a GeoDataFrame with the image footprints
footprints = gpd.GeoDataFrame({'file_name': tif_files}, geometry=geometry, crs={'init': 'epsg:4326'})

# Filter for just the images that have flooding in them (I know I sqeezed a lot into one line)
footprints_with_flooding = footprints[footprints.index.isin(gpd.sjoin(flooding, footprints, how='inner', op='intersects').index_right.unique())]

#footprints_with_flooding.plot()

def tile_image(file_name, tile_size):
    src = rasterio.open(file_name)
    
    width = src.width
    height = src.height
    
    for i in range(0, width - tile_size, tile_size):
        for j in range(0, height - tile_size, tile_size):
            window = ((j, j + tile_size), (i, i + tile_size))
            
            # Load the tile
            img = src.read(window=window).transpose([1,2,0])
            
            # Get metadata about the tile
            transform = list(np.array(~src.window_transform(window))[[0, 1, 3, 4, 2, 5]])
            box = shapely.geometry.box(*src.window_bounds(window))
            
            # Skip any image with more than 10% missing pixels
            #print((img == 0).all(axis=2).sum() / (1.0*img.size))
            if (img == 0).all(axis=2).sum() / (1.0*img.size) > 0.1:
                continue
                
            yield img, transform, box

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

def get_mask(mask_poly, img_shape, transform):
    # Transform the poly into image coords
    mask_poly_pxcoords = shapely.affinity.affine_transform(mask_poly, transform)
    
    # Add a buffer to pad it out a little
    mask_poly_pxcoords = mask_poly_pxcoords.buffer(2)
    
    return make_mask(img_shape, mask_poly_pxcoords)

print(len(footprints_with_flooding),"files with flooding found")

# Start Iterating through tiles

#os.makedirs('/home/ubuntu/data/TX_post/training_tiles')



print("everything prepped, starting sub-image extraction")
tiles = []
tile_no = 0
control = 0
image_width = 300
for i, row in footprints_with_flooding.iterrows():
    print("file ",i,row.file_name,"Tile count starts at",tile_no)
    for tile_img, tile_transform, tile_box in tile_image(row.file_name, image_width):
        
        # Skip tiles with no flooding (you can try keeping them)
        #keep a SMALL percentage of the unflooded tiles for training (do rest of loop), but reject the rest if not flooded
        mask_poly = flooding[flooding.intersects(tile_box)].intersection(tile_box).unary_union
            
        if mask_poly.area < 1e-12:
            #print("no flooding on tile ",tile_no)
            p_keep = 0.01    #dataset could have almost 200000 512x512 images, P==0.01 should give ~2000 back
            dice_roll = np.random.random()
            if dice_roll < p_keep:
                
                mask = np.zeros(tile_img.shape[:2])
                print(str(control)+"th unflooded tile added, tile no:",tile_no)
                np.save("/home/ubuntu/data/TX_post/training_tiles/%d_mask"%tile_no, mask)
                np.save("/home/ubuntu/data/TX_post/training_tiles/%d_dry_img"%tile_no, tile_img)
                control += 1
                tile_no += 1
        
            continue
        
        mask = get_mask(mask_poly, tile_img.shape[:2], tile_transform)
        
        # skip tiles between 0 and 10% flooding (too many false-positives)
        
        if (mask.sum()/255.)/(mask.shape[0]*mask.shape[1]) < 0.10:
            print("flooding found, but less than 10%, rejecting image")
            continue
        
        print("flooding found, tile no:",tile_no)
        np.save("/home/ubuntu/data/TX_post/training_tiles/%d_mask"%tile_no, mask)
        np.save("/home/ubuntu/data/TX_post/training_tiles/%d_img"%tile_no, tile_img)
        tile_no += 1
        #if tile_no > 20: break

tile_no = 935
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

tile_pick = random.randint(0,tile_no-1)  #chooses a random tiletile_no = tile_no + 73
print(tile_pick)
img = np.load('/home/ubuntu/data/TX_post/training_tiles/%d_img.npy'%tile_pick)
mask = np.load('/home/ubuntu/data/TX_post/training_tiles/%d_mask.npy'%tile_pick)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,14))
ax1.imshow(img)
ax2.imshow(mask);









