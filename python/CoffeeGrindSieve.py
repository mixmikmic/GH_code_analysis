get_ipython().system('pip install --user opencv-python')
#!pip install --user imageio
#!pip list --format=columns

# imports for image processing
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import io as skyio
from mpl_toolkits.mplot3d import Axes3D

cgsCreds = {
    'IBM_API_KEY_ID': '...',
    'IAM_SERVICE_ID': '...',
    'ENDPOINT': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
    'IBM_AUTH_ENDPOINT': 'https://iam.ng.bluemix.net/oidc/token',
    'BUCKET': '...',
    'FILE': '...'
}

# The code was removed by DSX for sharing.

import sys
import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
cgsClient = ibm_boto3.client(service_name='s3',
    ibm_api_key_id = cgsCreds['IBM_API_KEY_ID'],
    ibm_auth_endpoint="https://iam.ng.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3-api.us-geo.objectstorage.service.networklayer.com')
type(cgsClient)


# Extract slots from credentials
# Bucket
cgsBucket = cgsCreds['BUCKET']
# Filename
cgsImage = cgsCreds['FILE']
# Verify current values
print(cgsBucket, cgsImage)

# Your data file was loaded into a botocore.response.StreamingBody object.
# Please read the documentation of ibm_boto3 and pandas to learn more about your possibilities to load the data.
# ibm_boto3 documentation: https://ibm.github.io/ibm-cos-sdk-python/
# pandas documentation: http://pandas.pydata.org/

# Method to read image from cos into numpy.ndarray
def cgsReadImage(client, bucket, file):
    # Download file from COS into a 'ibm_botocore.response.StreamingBody' object
    isr = client.get_object(Bucket=bucket, Key=file)
    # Extract the jpeg image data from the 'Body' slot into a byte array
    jpg = isr['Body']
    print(type(jpg))
    # needed by skyio.imread
    if not hasattr(jpg, "__iter__"): jpg.__iter__ = types.MethodType( __iter__, jpg )
    # Convert the jpeg image data into a numpy.ndarray of size rRows*cCols*nColorChannels
    img = skyio.imread(jpg)
    # Print some stats
    print("cgsReadImage: \n\tBucket=%s \n\tFile=%s \n\tArraySize=%d %s Type=%s\n" % (bucket, file, img.size, img.shape, img.dtype))
    return(img)

# Read image from COS
jpg = cgsReadImage(cgsClient, cgsBucket, cgsImage)

# now that the images is in a numpy.ndarray it needs to somehow be written to an object that represents a jpeg image
# the memory structure to hold that representation of the jpeg is a io.BytesIO object, suiteable for the Body arg of client.put_object
import io as libio
from PIL import Image

def cgsWriteImage(client, bucket, file, image):
    # Convert numpy.ndarray into PIL.Image.Image object. This features a 'save' method that will be used below
    # Determine number of dimensions
    n = image.ndim
    # RGB image
    if (n==3):
            img = Image.fromarray(image,'RGB')
    # Binary or graylevel image
    else:
        # Binary
        if (image.max()==1):
            img = Image.fromarray(image,'1').convert('RGB')  
        # Graylevel
        else:
            img = Image.fromarray(image,'L').convert('RGB')            
        
    # Create buffer to hold jpeg representation of image in 'io.BytesIO' object
    bufImage = libio.BytesIO()
    # Store jpeg representation of image in buffer
    img.save(bufImage,"JPEG") 
    # Rewind the buffer to beginning
    bufImage.seek(0)
    # Provide the jpeg object to the Body parameter of put_object to write image to COS
    isr = client.put_object(Bucket=bucket, 
                            Body = bufImage,
                            Key = file, 
                            ContentType = 'image/jpeg')
    print("cgsWriteImage: \n\tBucket=%s \n\tFile=%s \n\tArraySize=%d %s RawSize=%d\n" % (bucket, file, image.size, image.shape, bufImage.getbuffer().nbytes))

# Write image in numpy.ndarray object into jpep file on COS
imgFile = 'CoffeeGrind-Copy.jpg'
cgsWriteImage(cgsClient, cgsBucket, imgFile, jpg)

# Read jpeg image from COS into numpy.ndarray
jpg = cgsReadImage(cgsClient, cgsBucket, imgFile)

cgsDefPlotSize=[12,12]
mpl.rcParams['figure.figsize'] = cgsDefPlotSize

# plot image
plt.imshow(jpg)
plt.show()

def imgStats(img):
    print("Image stats: ", "min:", img.min(), " max:", img.max(), " mean:", img.mean(), " median:", np.median(img), " Type:", img.dtype)

# Select a small part of the sample image and convert it to binary. 
orig = jpg[1340:1440,1285:1385,2]
orig = np.uint8(orig>96)
# Insert a little whole
orig[55:60,70:75] = 0

# Create a structuring element of size 7 shaped like a circle
t=7
strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(t,t))
# The structural element is a binary t*t matrix
print(strel)

# Calculate basic MM operations
erode = cv2.morphologyEx(orig, cv2.MORPH_ERODE, strel)
dilate = cv2.morphologyEx(orig, cv2.MORPH_DILATE, strel)
opening = cv2.morphologyEx(orig, cv2.MORPH_OPEN, strel)
closing = cv2.morphologyEx(orig, cv2.MORPH_CLOSE, strel)

# Create plot of results of above MM operations
mpl.rcParams['figure.figsize'] = [15,3]
plt.subplot(1,5,1); plt.imshow(orig,cmap= 'gray')
plt.subplot(1,5,2); plt.imshow(erode,cmap = 'gray')
plt.subplot(1,5,3); plt.imshow(dilate,cmap = 'gray')
plt.subplot(1,5,4); plt.imshow(opening,cmap = 'gray')
plt.subplot(1,5,5); plt.imshow(closing,cmap = 'gray')
plt.tight_layout()
plt.show()
mpl.rcParams['figure.figsize'] = cgsDefPlotSize

# use openCV's split method
b,g,r = cv2.split(jpg)

# use subplots to show the three color channels
mpl.rcParams['figure.figsize'] = [15,10]
plt.subplot(1,3,1); plt.imshow(b,cmap= 'gray', vmin=0, vmax=255)
plt.subplot(1,3,2); plt.imshow(g,cmap = 'gray', vmin=0, vmax=255)
plt.subplot(1,3,3); plt.imshow(r,cmap = 'gray', vmin=0, vmax=255)
plt.tight_layout()
plt.show()
mpl.rcParams['figure.figsize'] = cgsDefPlotSize

imgStats(r)

# select the red channel for further processing
gray = r
# invert
gray = 255-gray
# adjust graylevels such that the darkest pixel becomes black
gray = gray-gray.min()

plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
plt.show()
imgStats(gray)

# downscale image by a factor downScale
downScale = 4
gray = cv2.resize(gray, None, fx=1/downScale, fy=1/downScale, interpolation = cv2.INTER_LINEAR)

print(gray.shape, gray.size, gray.dtype)
imgStats(gray)

# Write image to COS
cgsWriteImage(cgsClient, cgsBucket, 'CoffeeGrind-Gray.jpg', gray)

def cgsPlot3d(img, plotScale=1, plotElev=75):   
    # set figure size
    fig = plt.figure(figsize = (20,20))
    # downscale image to speed up plotting
    ims = cv2.resize(img, None, fx=1/plotScale, fy=1/plotScale, interpolation = cv2.INTER_LINEAR)
    # initialize 3d plot
    ax = fig.add_subplot(111, projection='3d')
    # set viewing angle
    ax.view_init(elev=plotElev)
    # create a xy grid along the size of the image
    xx, yy = np.mgrid[0:ims.shape[0], 0:ims.shape[1]]
    # construct the 3d plot on the 2D grid with z coords the gray levels
    ax.plot_surface(xx, yy, ims, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
    plt.show()

# plot inverted sample image (background black, coffee grind white)
cgsPlot3d(gray, plotScale=2, plotElev=75)

# Edge preserving smoothing
h = 3
smooth = cv2.medianBlur(gray, h)
smooth = np.uint8(smooth)
imgStats(smooth)
plt.imshow(smooth, cmap='gray', vmin=0, vmax=255)
plt.show()

# strel size experimental --> will be 512 for original size
t=np.int(512/downScale)
strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(t,t))
tophat = cv2.morphologyEx(smooth, cv2.MORPH_TOPHAT, strel)
plt.imshow(tophat, cmap='gray', vmin=0, vmax=255)
plt.show()

imgStats(tophat)

# Write image to COS
cgsWriteImage(cgsClient, cgsBucket, 'CoffeeGrind-TopHat.jpg', tophat)

# plot inverted sample image (background white, coffee grind black)
cgsPlot3d(255-tophat, plotScale=2, plotElev=45)

# Calc intensity histogram of tophat for threshold estimation
hist = cv2.calcHist([tophat],[0],None,[256],[0,256])
mpl.rcParams['figure.figsize']=[8,4]
plt.plot(hist); plt.grid()
plt.xticks(np.arange(0, 100, step=10))
plt.xlim(0,100); plt.ylim(0,5000)
plt.show()
mpl.rcParams['figure.figsize']=cgsDefPlotSize

# convert to BW. Threshold experimental
bw1 = cv2.threshold(tophat,28, 255, cv2.THRESH_BINARY)[1]
bw2 = cv2.threshold(tophat,32, 255, cv2.THRESH_BINARY)[1]
bw3 = cv2.threshold(tophat,48, 255, cv2.THRESH_BINARY)[1]

# use subplots to show three threshold values
mpl.rcParams['figure.figsize'] = [15,10]
plt.subplot(1,3,1); plt.imshow(bw1[0:200,500:800],cmap = 'gray')
plt.subplot(1,3,2); plt.imshow(bw2[0:200,500:800],cmap = 'gray')
plt.subplot(1,3,3); plt.imshow(bw3[0:200,500:800],cmap = 'gray')
plt.tight_layout()
plt.show()
mpl.rcParams['figure.figsize'] = cgsDefPlotSize

# Classify into background and coffee grind pixels
b = 32
bw = cv2.threshold(tophat,b, 255, cv2.THRESH_BINARY)[1]

# fill small holes
c=5
strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(c,c))
bwc = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, strel)

# remove small particles
o=5
strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(o,o))
bwco = cv2.morphologyEx(bwc, cv2.MORPH_OPEN, strel)

# Show effects of MM operations 'opening' and subsequent 'closing'
mpl.rcParams['figure.figsize'] = [15,10]
plt.subplot(1,3,1); plt.imshow(bw[400:600,400:700],cmap = 'gray')
plt.subplot(1,3,2); plt.imshow(bwc[400:600,400:700],cmap = 'gray')
plt.subplot(1,3,3); plt.imshow(bwco[400:600,400:700],cmap = 'gray')
plt.tight_layout()
plt.show()
mpl.rcParams['figure.figsize'] = cgsDefPlotSize

# Classification mask 
cgsMask = bwco>0
cgsImg = tophat * cgsMask
plt.imshow(cgsImg, cmap='gray')
plt.show()
imgStats(cgsImg)

# Write image to COS
cgsWriteImage(cgsClient, cgsBucket, 'CoffeeGrind-Mask.jpg', cgsImg)

# Geodesic distance transform
msk = np.uint8(cgsMask)
dt = cv2.distanceTransform(msk, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
plt.imshow(np.sqrt(dt/dt.max())*255, cmap='gray', vmin=0, vmax=255)
plt.show()
imgStats(dt/dt.max()*255)

# Write image to COS
cgsWriteImage(cgsClient, cgsBucket, 'CoffeeGrind-Dist.jpg', dt)

t=2
strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(t,t))
print(strel)
th2 = cv2.morphologyEx(dt, cv2.MORPH_BLACKHAT, strel)
plt.imshow(th2, cmap='gray')
plt.show()
imgStats(th2)

# Find local maxima in dt DEM
from skimage import morphology as sm

t=3
strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(t,t))
c=[]
for h in range(1,32):
    mxa = sm.h_maxima(dt,h,strel)
    n,cc = cv2.connectedComponents(np.uint8(mxa>0))  
    c.append(n)
      
mpl.rcParams['figure.figsize']=[8,4]
plt.plot(range(1,32),c)
plt.grid()
plt.show()
mpl.rcParams['figure.figsize']=cgsDefPlotSize

# Show local maxima found in distance transform of classification mask
t=3
strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(t,t))

# calculate local maxima
h=2
seeds = sm.h_maxima(dt,h,strel)
seeds = np.uint8(seeds)

# count and identify local maxima as connected components
n,cc = cv2.connectedComponents(seeds)  
print("Number of coffee grind particles:", n)

# grow markers for visual clarity
t=5
strel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(t,t))
markers = cv2.morphologyEx(seeds, cv2.MORPH_DILATE, strel5)

# plot markers as black dots on top of the coffee grind
plt.imshow(cgsImg * (1-markers), cmap='gray')
plt.show()
imgStats(markers)

# Write image to COS
cgsWriteImage(cgsClient, cgsBucket, 'CoffeeGrind-Gray.jpg', cgsImg * (1-markers))

# must convert to rgb image for watershed
rgb = cv2.cvtColor(cgsImg, cv2.COLOR_GRAY2BGR)
# Run watershed. Walls are identifed as '-1'. 
ws = cv2.watershed(rgb,cc)+1
ws = np.uint8(ws)

# set image background to 0. don't know why background is just another segment
hist = cv2.calcHist([ws],[0],None,[256],[0,256])
idx = np.argmax(hist)
ws[ws==idx] = 0

# increase size of segmentation walls for visual clarity
t=3
strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(t,t))
wsd = cv2.morphologyEx(ws, cv2.MORPH_ERODE, strel)

# plot black segmentation walls on top of the coffee grind
cgs = (wsd>1)*cgsImg
plt.imshow(cgs, cmap='gray')
plt.show()

# Write image to COS
cgsWriteImage(cgsClient, cgsBucket, 'CoffeeGrind-WaterShed.jpg', cgs)

# Final connected components
cc_n, cc_lbl, cc_stats, cc_cntr = cv2.connectedComponentsWithStats(ws, connectivity=4)
print("Number of coffee grind particles:", cc_n)

def ccHist(cc):
    m=126; k=5; n = np.uint8(m/k)
    x = np.double(range(0,m,k))
    h = np.histogram(cc, bins=x**2)
    fig = plt.figure()
    plt.bar(x[0:n]+k/2,h[0][0:n],k-1)
    plt.xticks(np.uint8(x),np.uint8(x))
    plt.grid(); plt.xlim(0,m-1)
    plt.xlabel("Mean diameter of particles [~pixels]")
    plt.ylabel("Number of particles")
    plt.show()
    

# Area of cc's
mpl.rcParams['figure.figsize']=[10,5]

# Extract size of connected components
cc_area = cc_stats[:,cv2.CC_STAT_AREA]

# Call histogram method defined above
ccHist(cc_area)
mpl.rcParams['figure.figsize']=cgsDefPlotSize

cc_diameter = np.sqrt(cc_area)
print("MeanDiameter=%d+-%d    MedianDiameter=%d IQR=%d" % 
      (np.mean(cc_diameter), np.std(cc_diameter), np.median(cc_diameter), np.percentile(cc_diameter,75)-np.percentile(cc_diameter,25)))



