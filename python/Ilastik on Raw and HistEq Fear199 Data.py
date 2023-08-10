## Script used to download nii run on Docker
from ndreg import *
import matplotlib
import ndio.remote.neurodata as neurodata
import nibabel as nb
inToken = "Fear199"
nd = neurodata()
print(nd.get_metadata(inToken)['dataset']['voxelres'].keys())
inImg = imgDownload(inToken, resolution=5)
imgWrite(inImg, "./Fear199.nii")

## Method 1:
import os
import numpy as np
from PIL import Image
import nibabel as nib
import scipy.misc
TokenName = 'Fear199.nii'
img = nib.load(TokenName)

## Convert into np array (or memmap in this case)
data = img.get_data()
print data.shape
print type(data)

## Method 2:
rawData = sitk.GetArrayFromImage(inImg)  ## convert to simpleITK image to normal numpy ndarray
print type(rawData)

## if we have (i, j, k), we want (k, j, i)  (converts nibabel format to sitk format)
new_im = newer_img.swapaxes(0,2) # just swap i and k

plane = 0;
for plane in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 100, 101, 102, 103, 104):
    output = np.asarray(rawData[plane])
    ## Save as TIFF for Ilastik
    scipy.misc.toimage(output).save('RAWoutfile' + TokenName + 'ITK' + str(plane) + '.tiff')



