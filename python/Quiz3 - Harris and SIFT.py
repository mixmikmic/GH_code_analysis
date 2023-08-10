get_ipython().magic('matplotlib inline')
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
import random as rnd
from scipy.ndimage import filters
from PIL import Image
from numpy import *
from pylab import *
from pandas import *

np.seterr(divide='ignore', invalid='ignore')

#Compute the Algorithm Harris corner detecion for implementation in grayscale image
def compute_harris_points(img, sigma=3):
    #compute derivates in the image 
    imx = np.zeros(img.size)
    imy = np.zeros(img.size)
    
    imx = filters.gaussian_filter(img, (sigma,sigma), (0,1))
    imy = filters.gaussian_filter(img, (sigma,sigma), (1,0))
    
    # compute the products of derivatives at every pixel
    Sxx = filters.gaussian_filter(imx*imx,sigma)
    Sxy = filters.gaussian_filter(imx*imy,sigma)
    Syy = filters.gaussian_filter(imy*imy,sigma)
    
    # determinant and trace
    Mdet = Sxx*Syy - Sxy**2
    Mtr = Sxx + Syy
    harris = np.divide(Mdet, Mtr)
    harris[np.isposinf(harris)] = 0
    harris[np.isnan(harris)] = 0
    return harris

def doHarrisNonMaxSupression(harrisim,min_dist=10,threshold=0.1):
    #Return corners from a Harris response image
    #min_dist is the minimum number of pixels separating
    #corners and image boundary. 
    global t 
    global dist
    dist=min_dist
    t=threshold
    #print(t)
    # find top corner candidates above a threshold
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1
    # get coordinates of candidates
    coords = array(harrisim_t.nonzero()).T
    # ...and their values
    candidate_values = [harrisim[c[0],c[1]] for c in coords]
    # sort candidates
    index = argsort(candidate_values)
    # store allowed point locations in array
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
            (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
    return filtered_coords

def plot_harris_points(image,filtered_coords):
    #""" Plots corners found in image. """
    plt.figure(figsize=(20,12))
    gray()
    plt.imshow(image)
    plt.title('Harris corner detection, dist=%s and threshold=%s'%(dist,t))
    plt.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'*',color = 'r')
    plt.axis('off')
    plt.show()

im = Image.open('boat_images/img1.pgm')
plt.figure(figsize=(20,12))
gray()
plt.imshow(im, cmap = 'gray')

harrisim = compute_harris_points(im)


for i in range(1, 11,1):
        xx=i* 0.01
        j=10
        filtered_coords = doHarrisNonMaxSupression(harrisim,j,xx)
        plot_harris_points(im, filtered_coords)

def getGaussianKernel(sigma, kernelHeight=51, kernelWidth=51):
    assert(kernelHeight % 2 == 1 and kernelWidth % 2 == 1)

    yOffset = (kernelHeight - 1) / 2
    xOffset = (kernelWidth - 1) / 2

    kernel = np.ndarray((kernelHeight, kernelWidth), np.float64)

    for y in range(-yOffset, yOffset+1, 1):
        for x in range(-xOffset, xOffset+1, 1):
            kernel[y+yOffset][x+xOffset] = (1. / (2.*np.pi*sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def calcGaussianPyramid(org_img):
    img = org_img.copy()
    bluredImg = img.copy()

    sigma = 1.6

    octaveCount = 7
    sigmaCount = 4

    gp = np.ndarray(shape=(octaveCount,), dtype=np.ndarray)

    for o in range(0, octaveCount):
        gp[o] = np.ndarray(shape=(sigmaCount+1, img.shape[0], img.shape[1]), dtype=np.float64)
        gp[o][0] = bluredImg.copy()
        for s in range(1, sigmaCount + 1):
            #k = 2**(float(s)/float(sigmaCount))
            k = np.sqrt(2.0)**s
            kernel = getGaussianKernel(k*sigma)
            bluredImg = cv2.filter2D(img, -1, kernel)
            gp[o][s] = bluredImg.copy()

        if (o < octaveCount-1):
            img = downscale(img)
            bluredImg = downscale(bluredImg)

    return gp

def calcDifference(img0, img1, threshold = 0):
    assert(img0.shape == img1.shape)

    diffImg = np.ndarray(img0.shape, np.float64)

    for y in range(diffImg.shape[0]):
        for x in range(diffImg.shape[1]):
            difference = abs(img1[y][x] - img0[y][x])
            if difference > threshold:
                diffImg[y][x] = difference
            else:
                diffImg[y][x] = 0

    return diffImg


def calcDoG(gp):
 
    DoG = np.ndarray(shape=gp.shape, dtype=np.ndarray)

    for o in range(DoG.shape[0]):
        DoG[o] = np.ndarray(shape=(gp[o].shape[0]-1, gp[o].shape[1], gp[o].shape[2]), dtype=np.float64)
        for s in range(DoG[o].shape[0]):
            DoG[o][s] = calcDifference(gp[o][s], gp[o][s+1])

    return DoG

def getNeighbourhood(octave, s, y, x, radius=1):
    neighbourhood = octave[s-radius:s+radius+1, y-radius:y+radius+1, x-radius:x+radius+1]
    return neighbourhood

def calcExtrema(DoG, threshold=0.3, radius=1):
    keypoints = np.ndarray(shape=DoG.shape, dtype=np.ndarray)

    sigma = 1.6
    sigmaCount = DoG[0].shape[0]

    for o in range(DoG.shape[0]):
        keypoints[o] = np.ndarray(shape=(DoG[o].shape[0]-(2*radius),), dtype=list)
        for s in range(radius, DoG[o].shape[0]-radius):
            keypoints[o][s-radius] = []
            k = 2**(float(s)/float(sigmaCount))
            for y in range(radius, DoG[o].shape[1]-radius):
                for x in range(radius, DoG[o].shape[2]-radius):
                    value = DoG[o][s, y, x]
                    neighbourhood = getNeighbourhood(DoG[o], s, y, x, radius=radius).flatten()
                    neighbourhood.sort()
                    min2 = neighbourhood[1]
                    max2 = neighbourhood[-2]
                    if value < min2 or (value > threshold and value > max2):
                        scale = 2**o
                        keypoints[o][s-radius].append((scale * y + scale/2, scale * x + scale/2, scale * k*sigma))

    return keypoints

def normalize(img):
    normImg = np.ndarray(shape=img.shape, dtype=np.float64)

    max_val = img.max()
    if max > 0:
        normImg = img/float(max_val)
        normImg *= 255.
    else:
        return img.copy()

    return normImg.astype(np.uint8)

def scale(img, factor=2):
    assert(len(img.shape) == 2)
    rows, cols = img.shape
    scaledImg = np.ndarray((rows*factor, cols*factor), np.float64)

    for y in range(0, scaledImg.shape[0]):
        for x in range(0, scaledImg.shape[1]):
            scaledImg[y][x] = img[y/factor][x/factor]

    return scaledImg

def downscale(img):
    assert(len(img.shape) == 2)
    rows, cols = img.shape
    scaledImg = np.ndarray((rows/2, cols/2), np.float64)

    for y in range(0, scaledImg.shape[0]):
        for x in range(0, scaledImg.shape[1]):
            scaledImg[y][x] = img[2*y][2*x]

    return scaledImg


def drawKeypoints(img, kp):
    if (len(img.shape) < 3 or img.shape[2] == 1):
        kpImg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        kpImg = img.copy()
    for y, x, scale in kp:
        r = rnd.randrange(0,255)
        g = rnd.randrange(0,255)
        b = rnd.randrange(0,255)
        cv2.circle(kpImg, (int(x), int(y)), int(scale), (r, g, b))
    
    return kpImg

def plotImage(title,image):
    plt.figure(figsize=(20,12))
    gray()
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

images = glob.glob('boat_images/img_*.png')
print('Images Loaded!')    
for filename in images:
    img = cv2.imread(filename, 0)
    gp = calcGaussianPyramid(img)
    DoG = calcDoG(gp)

    radius = 1
    keypoints = calcExtrema(DoG, radius=radius)
    kpImg = img.copy()
    for o in range(keypoints.shape[0]):
        for s in range(radius, DoG[o].shape[0]-radius):
            kp = keypoints[o][s-radius]
            kpImg = drawKeypoints(kpImg, kp)

    plotImage("Custom SIFT "+ filename, kpImg)    
    


        

#img = (Image.open('boat_images/img1.pgm').convert('L'))
#img.save('boat_images/img_1.png')
#img = (Image.open('boat_images/img2.pgm').convert('L'))
#img.save('boat_images/img_2.png')
#img = (Image.open('boat_images/img3.pgm').convert('L'))
#img.save('boat_images/img_3.png')
#img = (Image.open('boat_images/img4.pgm').convert('L'))
#img.save('boat_images/img_4.png')
#img = (Image.open('boat_images/img5.pgm').convert('L'))
#img.save('boat_images/img_5.png')
#img = (Image.open('boat_images/img6.pgm').convert('L'))
#img.save('boat_images/img_6.png')

images = glob.glob('boat_images//img*.pgm')
for filename in images:
    img = cv2.imread(filename, 0)
    sift = cv2.SIFT()
    kp, desc = sift.detectAndCompute(img, None)
    imgfinal=cv2.drawKeypoints(img,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plotImage("SIFT Opencv  "+filename, imgfinal)   


## matching
sift = cv2.SIFT()
imgA = cv2.imread('boat_images//img1.pgm', 0)
imgB = cv2.imread('boat_images//img2.pgm', 0)
kpA, desA = sift.detectAndCompute(imgA,None)
kpB, desB = sift.detectAndCompute(imgB,None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(desA,desB, k=2)
#img3 = cv2.drawMatchesKnn(imgA,kpA,imgB,kpB,good,flags=2)
#plt.imshow(img3)
#plt.show()



