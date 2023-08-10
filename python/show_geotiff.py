import gdal, osr
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')

# Open raster data
dome_sub = gdal.Open("../data/dome_sub_sub_utm.tif")

arr_sub = dome_sub.GetRasterBand(1).ReadAsArray()

plt.style.use("ggplot")
plt.figure(figsize=(5,3))
# plt.imshow(arr[450:500,450:500], origin='lower left', cmap='viridis', interpolation='nearest')
plt.imshow(arr_sub, # origin='lower left', 
           cmap='viridis', 
           interpolation='nearest')



