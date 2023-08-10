from snappy import ProductIO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
get_ipython().magic('matplotlib inline')
from skimage import exposure
import os

print('Modules properly imported')

file_path = os.getcwd()
file_name = 'MER_FR__1PNUPA20030723_105132_000000982018_00223_07291_0388.N1'
p = ProductIO.readProduct(os.getcwd() + "\\" + file_name)
print list(p.getBandNames())

rad7 = p.getBand('radiance_15')
width = rad7.getRasterWidth()
height = rad7.getRasterHeight()
print('Width = ' + str(width))
print('Height = ' + str(height))

rad_data_7 = np.zeros(width*height, dtype = np.float32)
print('The array has been initialized')

rad7.readPixels( 0, 0, height, width, rad_data_7)
print('Reading data...')

rad_data_7.shape = width, height
print('raster image ready to be displayed')

plt.figure(figsize=(8, 8))
fig = plt.imshow(rad_data_7, cmap = cm.gray)  #matplotlib settings for the current image
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()

plt.figure(figsize=(8, 8))  
fig = plt.imshow(exposure.equalize_hist(rad_data_7),cmap = cm.gray)
fig.axes.get_xaxis().set_visible(False)  #matplotlib settings for the current figure
fig.axes.get_yaxis().set_visible(False)
plt.show()

B_rad = p.getBand('radiance_2')
B_rad_data = np.zeros(width*height, dtype = np.float32)
B_rad.readPixels( 0, 0, height, width, B_rad_data)
B_rad_data.shape = width, height
print('Blue channel ready')

G_rad = p.getBand('radiance_3')
G_rad_data = np.zeros(width*height, dtype = np.float32)
G_rad.readPixels( 0, 0, height, width, G_rad_data)
G_rad_data.shape = width, height
print('Green channel ready')

R_rad = p.getBand('radiance_4')
R_rad_data = np.zeros(width*height, dtype = np.float32)
R_rad.readPixels( 0, 0, height, width, R_rad_data)
R_rad_data.shape = width, height
print('Red channel ready')

cube = np.zeros((width,height,3), dtype = np.float32)

val1,val2 = np.percentile(B_rad_data, (4,95))
sat_B_rad_data = exposure.rescale_intensity(B_rad_data, in_range=(val1,val2))

val1,val2 = np.percentile(G_rad_data, (4,95))
sat_G_rad_data = exposure.rescale_intensity(G_rad_data, in_range=(val1,val2))

val1,val2 = np.percentile(R_rad_data, (4,95))
sat_R_rad_data = exposure.rescale_intensity(R_rad_data, in_range=(val1,val2))

cube[:,:,0] =sat_R_rad_data
cube[:,:,1] =sat_G_rad_data
cube[:,:,2] =sat_B_rad_data

plt.figure(figsize=(8, 8))
fig = plt.imshow(cube)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()




