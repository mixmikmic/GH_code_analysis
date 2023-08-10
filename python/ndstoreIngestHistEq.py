get_ipython().magic('matplotlib inline')

from ndreg import *
import matplotlib
import ndio.remote.neurodata as neurodata

## Script used to download nii run on Docker
import nibabel as nb
inToken = "Aut1367"
nd = neurodata()
print(nd.get_metadata(inToken)['dataset']['voxelres'].keys())
inImg = imgDownload(inToken, resolution=5)

imgWrite(inImg, "./Aut1367.nii")

import nibabel as nib
import cv2

## Script from clviz_web_viz imgGet.py
(values, bins) = np.histogram(sitk.GetArrayFromImage(inImg), bins=100, range=(0,500))
print "level 5 brain obtained"
counts = np.bincount(values)
maximum = np.argmax(counts)

lowerThreshold = maximum
upperThreshold = sitk.GetArrayFromImage(inImg).max()+1

inImg = sitk.Threshold(inImg,lowerThreshold,upperThreshold,lowerThreshold) - lowerThreshold
print "applied filtering"

(values, bins) = np.histogram(sitk.GetArrayFromImage(inImg), bins=100, range=(0,500))
plt.plot(bins[:-1], values)

## Kwame's script
lowerThreshold = 100
upperThreshold = sitk.GetArrayFromImage(inImg).max()+1

inImg = sitk.Threshold(inImg,lowerThreshold,upperThreshold,lowerThreshold) - lowerThreshold
imgShow(inImg, vmax = 500)

(values, bins) = np.histogram(sitk.GetArrayFromImage(inImg), bins=100, range=(0,500))
plt.plot(bins[:-1], values)

imgWrite(inImg, "./Aut1367Threshold.nii")

print type(inImg)

"""Applies local equilization to the img's histogram and outputs a .nii file"""
print('Generating Histogram...')
path = os.getcwd()
im = nib.load(path + "/" + "Aut1367Threshold.nii")

im = im.get_data()
img = im[:,:,:]

shape = im.shape
#affine = im.get_affine()

x_value = shape[0]
y_value = shape[1]
z_value = shape[2]

#####################################################

imgflat = img.reshape(-1)

#img_grey = np.array(imgflat * 255, dtype = np.uint8)

#img_eq = exposure.equalize_hist(img_grey)

#new_img = img_eq.reshape(x_value, y_value, z_value)
#globaleq = nib.Nifti1Image(new_img, np.eye(4))

#nb.save(globaleq, '/home/albert/Thumbo/AutAglobaleq.nii')

######################################################

#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

img_grey = np.array(imgflat * 255, dtype = np.uint8)
#threshed = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)

cl1 = clahe.apply(img_grey)

#cv2.imwrite('clahe_2.jpg',cl1)
#cv2.startWindowThread()
#cv2.namedWindow("adaptive")
#cv2.imshow("adaptive", cl1)
#cv2.imshow("adaptive", threshed)
#plt.imshow(threshed)

localimgflat = cl1 #cl1.reshape(-1)

newer_img = localimgflat.reshape(x_value, y_value, z_value)
localeq = nib.Nifti1Image(newer_img, np.eye(4))

print type(newer_img)

print type(localeq)
print newer_img.shape
print localeq.shape

## Convert back into simpleITK image
Aut1367_histeq_ITK = sitk.GetImageFromArray(newer_img, isVector=False)

imgWrite(Aut1367_histeq_ITK, "./Aut1367LocalHistEq.nii")

imgShow(inImg, vmax = 500)

## post thresholding + local histogram equalization
imgShow(Aut1367_histeq_ITK)

print Aut1367_histeq_ITK.GetSize()

## Z, Y, X:

## I made an Open Connectome account
## I then made a dataset/project.
## The dataset is called "Aut1367_histeq", the token is "Aut1367_tony", the channel is "histogram_equalization"
## I set the X image size, Y image size, Z image size on my ndstore dataset to the values from above.

token = "Aut1367_tony"
channel = "histogram_equalization"
imgUpload(Aut1367_histeq_ITK, token, channel, resolution=0)



