import rasterio
src = rasterio.open('assets/MUL-PanSharpen_AOI_3_Paris_img100.tif')

import matplotlib.pyplot as plt
import numpy as np

get_ipython().magic('matplotlib inline')

# Load the RGB bands and transpose the image shape
img = src.read([5, 3, 2]).transpose([1,2,0])

# Plot it
plt.imshow(img)

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

print(src.width, src.height, src.count)

print(src.crs)
print(src.transform)

src.bounds

import os

import geopandas as gpd
import shapely.geometry

image_dir_path = '/mnt/OSN_Data/spacenet/AOI_3_Paris_Train/MUL-PanSharpen'

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

output_file_name = 'vectors/image_summary.geojson'

# Make sure the file doesnt exist (the program will crash if it does)
if os.path.exists(output_file_name):
    os.remove(output_file_name)

# Save the file as a geojson
image_summary.to_file(output_file_name, driver='GeoJSON')



