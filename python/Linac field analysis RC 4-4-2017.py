import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from skimage import io, filters, feature, measure
from skimage.morphology import erosion, dilation, square
from skimage.feature import blob_log

#from pylinac.core.image import Image 
#from pylinac.core.profile import SingleProfile 
get_ipython().magic('matplotlib inline')

tiff_file = 'Image.tif'
image = io.imread(tiff_file)
print(image.shape)     # 3 layers are RGB
image = image[:,:,0]   # keep only the red channel
print(image.shape)     

thresh = filters.threshold_otsu(image)
print('The otsu threshold is ' + str(thresh))     # determines the image threshold

fig_width = 12
fig_height = 4

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(fig_width, fig_height))

ax1.imshow(image, cmap='gray')  
ax2.imshow(image <= thresh);  
ax3.hist(image.flatten() , bins = 160); 

main_perimeter = int(measure.perimeter(image <= thresh)/4)   # get the length of the sides of the binary object
main_perimeter

main_ROI = image[main_perimeter:, 0:main_perimeter]
thresh2 = filters.threshold_otsu(main_ROI)
print('The otsu threshold is ' + str(thresh2)) 

fig_width = 12
fig_height = 4

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(fig_width, fig_height))

ax1.imshow(main_ROI, cmap='gray');    # plot the main ROI
ax2.hist(main_ROI.flatten() , bins = 160); 
ax3.imshow(main_ROI <= thresh2);

radiation_perimeter = int(measure.perimeter(main_ROI <= thresh2)/4)   # get the length of the sides of the binary object
radiation_perimeter

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(fig_width, fig_height))

err_dil_factor = 10
eroded_ROI = erosion(main_ROI <= thresh2, square(err_dil_factor))   # apply erosion to remove writing
dilated_ROI = dilation(eroded_ROI, square(err_dil_factor))        # apply dilation to restor radiaiton ROI

ax1.imshow(main_ROI <= thresh2);
ax2.imshow(dilated_ROI);
ax3.imshow(dilated_ROI - (main_ROI <= thresh2));    # what is removed by the operation

radiation_perimeter = int(measure.perimeter(eroded_ROI)/4)   # get the length of the sides of the eroded ROI
radiation_perimeter

main_ROI_dim = int(np.mean(main_ROI.shape)) # get approx main_ROI length/width dimension
main_ROI_dim

ROI_difference = int((main_ROI_dim - radiation_perimeter)/2)   # will subtract difference from each edge
ROI_difference

scale_factor = 1.8
ROI_difference = ROI_difference*scale_factor    # scale the ROI to capture more or less of the radiation

radiation_ROI = main_ROI[ROI_difference:main_ROI_dim-ROI_difference, ROI_difference:main_ROI_dim-ROI_difference]

plt.imshow(radiation_ROI, cmap='gray');   # Now plot the ratiaion ROI

thresh3 = filters.threshold_otsu(radiation_ROI)
print('The otsu threshold is ' + str(thresh3)) 

fig_width = 12
fig_height = 4

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(fig_width, fig_height))

ax1.imshow(radiation_ROI, cmap='gray');    # plot the main ROI
ax2.hist(radiation_ROI.flatten() , bins = 160); 
ax3.imshow(radiation_ROI <= thresh3);

blobs_log = blob_log(radiation_ROI <= thresh3, max_sigma=30, num_sigma=10, threshold=.1)

blobs_log

blobs_log[0]

f, (ax1) = plt.subplots(1, 1)

ax1.imshow(radiation_ROI <= thresh3)
for blob in blobs_log:
    print(blob)
    y, x, r = blob
    circ = Circle((x, y), 10, color='r', linewidth=2, fill=False)
    ax1.add_patch(circ)



