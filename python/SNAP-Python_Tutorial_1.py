from snappy import ProductIO

file_path = 'C:\Program Files\snap\S2A_MSIL1C_20170202T090201_N0204_R007_T35SNA_20170202T090155.SAFE\MTD_MSIL1C.xml'
product = ProductIO.readProduct(file_path)

list(product.getBandNames())

B5 = product.getBand('B5')

Width = B5.getRasterWidth()
Height = B5.getRasterHeight()
print(Width,Height)

import numpy as np
B5_data = np.zeros(Width*Height, dtype = np.float32)
B5.readPixels(0,0,Width,Height,B5_data)
B5_data.shape = (Height,Width)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
get_ipython().magic('matplotlib inline')
plt.figure(figsize=(8, 8))                 # adjusting the figure window size
fig = plt.imshow(B5_data, cmap = cm.gray)  #matplotlib settings for the current image
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()

import skimage.exposure as exposure
val1,val2 = np.percentile(B5_data, (2.5,97.5))
B5_data_new = exposure.rescale_intensity(B5_data, in_range=(val1,val2))

plt.figure(figsize=(8, 8))                     
fig = plt.imshow(B5_data_new, cmap = cm.gray)  
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()



