import cv2
import matplotlib.pyplot as plt
import rasterio

import numpy as np
from rasterio.plot import show_hist

import plotly.plotly as py



img_path = ("/home/ubuntu/data/sar/experiment_crops_20170815/train/50x50/oil_and_gas_infrastructure/"                     "S1B_IW_GRDH_1SDV_20170709T061319_20170709T061344_006407_00B43B_EFB1_terrain_correction_69.png")
print img_path
png_img = cv2.imread(img_path)
plt.figure(figsize=(10,12))
plt.imshow(png_img, cmap='pink')
plt.show()

img_path = ("/home/ubuntu/data/sar/experiment_crops_20170815/train/50x50/oil_and_gas_infrastructure/"                     "S1B_IW_GRDH_1SDV_20170709T061319_20170709T061344_006407_00B43B_EFB1_terrain_correction_69.tif")
print img_path
src = rasterio.open(img_path)
plt.imshow(src.read(1), cmap='pink')
plt.show()

print "tif array"
np.set_printoptions(threshold=np.nan)
rasterio_array = src.read(1)
print(rasterio_array[25,:])

print "png array"
print(png_img[25,:,1])

print type(png_img), type(png_img[0]), png_img[0].shape

print type(rasterio_array), type(rasterio_array[0]), rasterio_array[0].shape

plt.imshow(png_img, cmap='hot')
plt.show()

test_png_img = png_img[:,:,0]
plt.imshow(test_png_img, cmap='pink')
plt.show()

plt.imshow(rasterio_array, cmap='pink')
plt.show()

print len(test_png_img)
for indx, element in enumerate(test_png_img):
    print indx, 'PNG', min(element), max(element), 'TIF', min(rasterio_array[indx]), max(rasterio_array[indx])

no_bins = 256
plt.hist(rasterio_array.flatten(), no_bins, range=(0.0, 1.0), fc='k', ec='k')
plt.title("tif Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

no_bins = 256
plt.hist(test_png_img.flatten(), no_bins, range=(0.0, 255.0), fc='k', ec='k')
plt.title("png Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

plt.hist(rasterio_array.flatten(), range=(0.5, 1.0), fc='k', ec='k')
plt.title("tif Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

plt.hist(test_png_img.flatten(), range=(127, 255), fc='k', ec='k')
plt.title("png Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

plt.hist(rasterio_array.flatten(), range=(0.0, 0.1), fc='k', ec='k')
plt.title("tif Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

plt.hist(test_png_img.flatten(), range=(0.0, 25), fc='k', ec='k')
plt.title("png Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

esaimagefilehdr = "/home/ubuntu/data/sar/Sigma0_VH.hdr"
esaimagefile = "/home/ubuntu/data/sar/Sigma0_VH.img"
headerfile = open(esaimagefilehdr, "r")
print headerfile.readlines()

img_path = ("/home/ubuntu/data/sar/experiment_crops_20170815/train/50x50/oil_and_gas_infrastructure/"                     "S1B_IW_GRDH_1SDV_20170709T061319_20170709T061344_006407_00B43B_EFB1_terrain_correction_69.tif")
print esaimagefile
imgfile = rasterio.open(esaimagefile)
array = imgfile.read(1)

print type(array), type(array[0]), array[0].shape

crop = array[0:2500]

plt.imshow(crop)
plt.show()

print len(crop)
for indx, element in enumerate(crop):
    print indx, 'img', min(element), max(element)

print array.min(), array.max()


plt.hist(array.flatten(), range=(0.0, 614.0), fc='k', ec='k')
plt.title("envi Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

plt.hist(array.flatten(), range=(0.0,1), fc='k', ec='k')
plt.title("envi Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

plt.hist(array.flatten(), range=(1,50), fc='k', ec='k')
plt.title("envi Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

plt.hist(array.flatten(), range=(500, 614), fc='k', ec='k')
plt.title("envi Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

plt.hist(np.percentile(array, 50, axis=0))
plt.show()

print array.size
print array.shape
print len(array)
print 21355 * 54682

# print len(crop)
# for indx, element in enumerate(array):
#     print indx, 'img', min(element), max(element)

unique, counts = np.unique(array, return_counts=True)
dict(zip(unique, counts))

print len(unique)

print np.count_nonzero(array <= 1)

a = 1167734110/1167720158 * 100.0
print a
b = 1167734110 - 1167720158
print b



