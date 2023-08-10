get_ipython().magic('matplotlib inline')

import os
os.chdir('/Users/albert/ndreg')

from ndreg import *
import matplotlib
import ndio.remote.neurodata as neurodata
import nibabel as nb

inImg = imgRead("../atlasfull.nii")
imgShow(inImg, vmax=500)

print(inImg.GetSpacing())

inImg = imgResample(inImg, spacing=(1.8719999119639397, .04999999888241291, 1.8719999119639397))
imgShow(inImg, vmax=500)

imgWrite(inImg, "../seelviz/miniatlas.nii")

inImg = imgRead("../seelviz/miniatlas.nii")
imgShow(inImg, vmax=500)

inImg = imgResample(inImg, spacing=(3.6719999119639397, .16999999888241291, 3.6719999119639397))
imgShow(inImg, vmax=500)

imgWrite(inImg, "../seelviz/miniatlas.nii")

inImg = imgRead("../seelviz/miniatlas.nii")
imgShow(inImg, vmax=500)

inImg = imgResample(inImg, spacing=(1.8719999119639397, .04999999888241291, 1.8719999119639397))
imgShow(inImg, vmax=500)

