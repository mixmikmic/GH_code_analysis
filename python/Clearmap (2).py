get_ipython().system('git clone https://github.com/ChristophKirst/ClearMap.git')

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

import os
import numpy as np
from PIL import Image
import nibabel as nib
import scipy.misc

rawData = sitk.GetArrayFromImage(inImg)  ## convert to simpleITK image to normal numpy ndarray
print type(rawData)

plane = 0;
for plane in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
    output = np.asarray(rawData[plane])
    ## Save as TIFF for Ilastik
    scipy.misc.toimage(output).save('clarity'+str(plane).zfill(4)+'.tif')

