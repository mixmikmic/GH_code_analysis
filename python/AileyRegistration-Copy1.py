get_ipython().magic('matplotlib inline')

import os
os.chdir('/Users/albert/ndreg')

from ndreg import *
import matplotlib
import ndio.remote.neurodata as neurodata
import nibabel as nb

refToken = "ara_ccf2"
refImg = imgDownload(refToken)

imgShow(refImg)

imgShow(refImg, vmax=500)

refAnnoImg = imgDownload(refToken, channel="annotation")
imgShow(refAnnoImg, vmax=1000)

randValues = np.random.rand(1000,3)

randValues = np.concatenate(([[0,0,0]],randValues))

randCmap = matplotlib.colors.ListedColormap (randValues)
imgShow(refAnnoImg, vmax=1000, cmap=randCmap)

imgShow(refImg, vmax=500, newFig=False)
imgShow(refAnnoImg, vmax=1000, cmap=randCmap, alpha=0.2, newFig=False)
plt.show()

inToken = "152DLS_tail/"
nd = neurodata()
print(nd.get_metadata(inToken)['dataset']['voxelres'].keys())

inImg = imgDownload(inToken, resolution=0)
imgShow(inImg, vmax=500)

inImg = imgRead("/Users/albert/ndreg/Cocaine174ARACoronal.nii")
#inImg(load_data)
imgShow(inImg, vmax=500)

inImg.SetSpacing([0.01872, 0.01872, 0.005]) # Setting manually due to bug https://github.com/neurodata/ndstore/issues/326

print(inImg.GetSpacing())

print(refImg.GetSpacing())

inImg = imgResample(inImg, spacing=refImg.GetSpacing())
imgShow(inImg, vmax=500)

imgShow(refImg, vmax=500)

imgShow(inImg, vmax=500)

inImg = imgReorient(inImg, "LAI", "RSA")
imgShow(inImg, vmax=500)

(values, bins) = np.histogram(sitk.GetArrayFromImage(inImg), bins=100, range=(0,500))
plt.plot(bins[:-1], values)

lowerThreshold = 100
upperThreshold = sitk.GetArrayFromImage(inImg).max()+1

inImg = sitk.Threshold(inImg,lowerThreshold,upperThreshold,lowerThreshold) - lowerThreshold
imgShow(inImg, vmax = 500)

(values, bins) = np.histogram(sitk.GetArrayFromImage(inImg), bins=100, range=(0,500))
plt.plot(bins[:-1], values)

imgShow(inImg, vmax = 500)

(values, bins) = np.histogram(sitk.GetArrayFromImage(inImg), bins=1000)
cumValues = np.cumsum(values).astype(float)
cumValues = (cumValues - cumValues.min()) / cumValues.ptp()

maxIndex = np.argmax(cumValues>0.95)-1
threshold = bins[maxIndex]
print(threshold)

inMask = sitk.BinaryThreshold(inImg, 0, threshold, 1, 0)
imgShow(inMask)

imgShow(imgMask(inImg,inMask))

spacing=[2.0,2.0,2.0]
refImg_ds = imgResample(refImg, spacing=spacing)
print(inImg)
imgShow(refImg_ds, vmax=500)

inImg_ds = imgResample(inImg, spacing=spacing)
imgShow(inImg_ds, vmax=500)

inMask_ds = imgResample(inMask, spacing=spacing, useNearest=True)
imgShow(inMask_ds)

affine = imgAffineComposite(inImg_ds, refImg_ds, inMask=inMask_ds, iterations=100, useMI=True, verbose=True)

inImg_affine = imgApplyAffine(inImg, affine, size=refImg.GetSize())
imgShow(inImg_affine, vmax=500)

inMask_affine = imgApplyAffine(inMask, affine, size=refImg.GetSize(), useNearest=True)
imgShow(inMask_affine)

inImg_ds = imgResample(inImg_affine, spacing=spacing)

inMask_ds = imgResample(inMask_affine, spacing=spacing, useNearest=True)

(field, invField) = imgMetamorphosisComposite(inImg_ds, refImg_ds, inMask=inMask_ds, alphaList=[0.05, 0.02, 0.01], useMI=True, iterations=100, verbose=True)

inImg_lddmm = imgApplyField(inImg_affine, field, size=refImg.GetSize())

inMask_lddmm = imgApplyField(inMask_affine, field, size=refImg.GetSize(), useNearest=True)

imgShow(inImg_lddmm, vmax = 500)

imgShow(inMask_lddmm)

imgShow(inImg_lddmm, vmax=500, newFig=False, numSlices=1)
imgShow(refAnnoImg, vmax=1000, cmap=randCmap, alpha=0.2, newFig=False, numSlices=1)

imgShow(imgChecker(inImg_lddmm, refImg, useHM=False), vmax=500, numSlices=1)

logJDet = -sitk.Log(sitk.DisplacementFieldJacobianDeterminant(field))

logJDet = imgResample(logJDet, spacing=inImg_lddmm.GetSpacing(), size=inImg_lddmm.GetSize())

imgShow(inImg_lddmm, vmax=500, newFig=False)
imgShow(logJDet, newFig=False, alpha=0.5, cmap=plt.cm.jet, vmin=-2, vmax=2)
plt.show()

imgShow(inImg_lddmm, vmax=500, newFig=False)
imgShow(logJDet, newFig=False, alpha=0.5, cmap=plt.cm.jet, vmin=-2, vmax=2)
fig = plt.gcf()

fig.axes

ax = fig.axes[0]

img_ax = ax.get_images()[1]

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])
cbar = fig.colorbar(img_ax, cax=cbar_ax) 

imgShow(inImg_lddmm, vmax=500, newFig=False)
imgShow(logJDet, newFig=False, alpha=0.5, cmap=plt.cm.jet, vmin=-2, vmax=2)
fig = plt.gcf()
ax = fig.axes[0]
img_ax = ax.get_images()[1]
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])
cbar = fig.colorbar(img_ax, cax=cbar_ax) 
plt.show()



