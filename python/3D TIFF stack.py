import os
import numpy as np
from PIL import Image
import nibabel as nib
import scipy.misc

## This .nii was generated from the compute cloud

TokenName = 'Fear197ds10.nii'
img = nib.load(TokenName)

## Sanity check for shape
img.shape

## Convert into np array (or memmap in this case)
data = img.get_data()
print data.shape
print type(data)


plane = 0;

##Iterate through all planes to get slices
for plane in range(data.shape[0]):
    ## Convert memmap array into ndarray for toimage process
    output = np.asarray(data[plane])
    ## Save as TIFF for Ilastik
    scipy.misc.toimage(output).save('outfile' + TokenName + str(plane) + '.tiff')



