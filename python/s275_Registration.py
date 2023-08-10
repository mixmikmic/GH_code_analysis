get_ipython().magic('matplotlib inline')

from ndreg import *
import matplotlib
import ndio.remote.neurodata as neurodata

server = "dev.neurodata.io"
userToken = txtRead("userToken.pem").strip()

refToken = "ara3"
refImg = imgDownload(refToken, channel="average", server=server, userToken=userToken)

refThreshold = imgPercentile(refImg, 0.99)
print(refThreshold)

imgShow(refImg, vmax=refThreshold)

refAnnoImg = imgDownload(refToken, channel="annotation", server=server, userToken=userToken)
refAnnoImgOrig = refAnnoImg[:,:,:]
imgShow(refAnnoImg, vmax=1000)

randValues = np.random.rand(1000,3)
randValues = np.concatenate(([[0,0,0]],randValues))

randCmap = matplotlib.colors.ListedColormap (randValues)
imgShow(refAnnoImg, vmax=1000, cmap=randCmap)

imgShow(refImg, vmax=refThreshold, newFig=False)
imgShow(refAnnoImg, vmax=1000, cmap=randCmap, alpha=0.3, newFig=False)
plt.show()

inToken = "s275"
nd = neurodata(hostname="dev.neurodata.io", user_token=userToken)
print(nd.get_metadata(inToken)['dataset']['imagesize']) 

inImg = imgDownload(inToken, resolution=1, userToken=userToken, server=server)
inImg.SetSpacing(np.array(inImg.GetSpacing())*1000) ###
inImgOrig = inImg[:,:,:]

inImg = inImgOrig[:,:,:]
inThreshold = imgPercentile(inImg, 0.95)
imgShow(inImg, vmax=inThreshold)

imgShow(refImg, vmax=refThreshold)

imgShow(inImg, vmax=inThreshold)

inOrient = "IAL"
refOrient = "RSA"
inImg = imgReorient(inImg, inOrient, refOrient)
imgShow(inImg, vmax=inThreshold)

inImgSize_reorient = inImg.GetSize()
inImgSpacing_reorient= inImg.GetSpacing()

spacing = [0.1,0.1, 0.1]
inImg_ds = imgResample(inImg, spacing)
imgShow(inImg_ds, vmax=inThreshold)

refImg_ds = imgResample(refImg, spacing)
imgShow(refImg_ds, vmax=refThreshold)

imgWrite(inImg_ds,"/cis/project/clarity/data/ailey/s275_ch0_rsa_100um.img")

roiStart = [5.4, 1.2, 2.1]
roiSize = [4.5,6.5,7.5]

roiStartVoxel = (roiStart / np.array(spacing)).astype('uint16').tolist()
print(roiStartVoxel)
roiSizeVoxel = (roiSize / np.array(spacing)).astype('uint16').tolist()
print(roiSizeVoxel)

roiImg = sitk.Image(roiSizeVoxel,sitk.sitkUInt8)
roiImg += 255

emptyImg = sitk.Image(refImg_ds.GetSize(),sitk.sitkUInt8) # Create an empty image
emptyImg.CopyInformation(refImg_ds) # Copy spacing, origin and direction from reference image
refMask = sitk.Paste(emptyImg, roiImg, roiSizeVoxel, [0,0,0], roiStartVoxel)
imgShow(refMask, vmax=255)

refImg_ds = sitk.Mask(refImg_ds, refMask)
imgShow(refImg_ds, vmax=refThreshold)

imgShow(inImg_ds, vmax=inThreshold)

threshold = imgPercentile(inImg_ds,0.95)
inMask_ds = sitk.BinaryThreshold(inImg_ds, 0, threshold, 255, 0)
imgShow(inMask_ds, vmax=255)

translation = -np.array(roiStart)
inAffine = [1.2,0,0,0,1.2,0,0,0,1]+translation.tolist()
print(inAffine)
imgShow(imgApplyAffine(inImg_ds, inAffine, size=refImg_ds.GetSize()),vmax = inThreshold)

affine = imgAffineComposite(inImg_ds, refImg_ds, inMask=inMask_ds, iterations=100, useMI=True, verbose=True, inAffine=inAffine)

inImg_affine = imgApplyAffine(inImg, affine, size=refImg.GetSize(), spacing=refImg.GetSpacing())
imgShow(inImg_affine, vmax=inThreshold)

inImg_ds = imgResample(inImg_affine, spacing=spacing, size=refImg_ds.GetSize())
imgShow(imgChecker(inImg_ds, refImg_ds), vmax=refThreshold)

inMask_ds = imgApplyAffine(inMask_ds, affine, useNearest=True, size=refImg_ds.GetSize())
imgShow(inMask_ds, vmax=255)
imgShow(inImg_ds, vmax=inThreshold)

inImg_ds = imgResample(inImg_affine, spacing=spacing, size=refImg_ds.GetSize())
(field, invField) = imgMetamorphosisComposite(inImg_ds, refImg_ds, inMask=inMask_ds, alphaList=[0.1, 0.05,0.02],
                                              scaleList = [1.0, 1.0,1.0], useMI=True, iterations=100, verbose=True)

inImg_lddmm = imgApplyField(inImg_affine, field, size=refImg.GetSize())
imgShow(inImg_lddmm, vmax=inThreshold)

inImg_ds = imgResample(inImg_lddmm, spacing=spacing, size=refImg_ds.GetSize())
imgShow(imgChecker(inImg_ds, refImg_ds), vmax=refThreshold)

imgShow(inImg_lddmm, vmax=inThreshold, newFig=False)
imgShow(refAnnoImg, vmax=1000, cmap=randCmap, alpha=0.2, newFig=False)
plt.show()

outToken = inToken + "_to_" + refToken
imgUpload(inImg_lddmm, outToken, server=server, userToken=userToken)
### imgWrite(inImg_lddmm, "/cis/project/clarity/data/ailey/"+outToken+"_new.img")

spacing_ds = invField.GetSpacing()
size_ds = np.ceil(np.array(refAnnoImg.GetSize())*np.array(refAnnoImg.GetSpacing())/np.array(spacing_ds))
size_ds = list(size_ds.astype(int))

invAffine = affineInverse(affine)
invAffineField = affineToField(invAffine, size_ds, spacing_ds)
invField2 = fieldApplyField(invAffineField, invField)
inAnnoImg = imgApplyField(refAnnoImg, invField2,useNearest=True, size=inImgSize_reorient, spacing=inImgSpacing_reorient)
inAnnoThreshold = imgPercentile(inAnnoImg,0.99)
imgShow(inAnnoImg, vmax=inAnnoThreshold)

inAnnoImg = imgReorient(inAnnoImg, refOrient, inOrient)
imgShow(inAnnoImg, vmax=inAnnoThreshold)

outToken = "ara3_to_AutA"
outChannel = "annotation_draft"
imgUpload(inAnnoImg, outToken, outChannel, resolution=5)

