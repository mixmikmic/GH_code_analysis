# -*- coding: utf-8 -*-
from __future__ import print_function
import os
from os import listdir
from os.path import isfile, join
import numpy  
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import string
from numpy.linalg import inv  #matrix calc
import math #log
import scipy
import scipy.ndimage
import scipy.signal #median filter for zero bad pixels
from astropy.io import fits #fits file read and writeTo
np.set_printoptions(suppress=True) #no scientific notations
import re #sort fits files
from IPython.display import clear_output #clear output
from pylab import *
from matplotlib import colors

def readFile(path, string):   #search string "white" and "raw"
    allFiles = [f for f in listdir(path) if isfile(join(path,f))]  
    for n in range(0, len(allFiles)):
        allFiles[n] = join(path,allFiles[n])
    allFiles.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)]) 
    #use "regular expression" to sort filenames
    listfile = [];
    for seq in range (0, len(allFiles)):
        if string in allFiles[seq]:
            listfile.append(allFiles[seq])
    print ('There are '+ str(len(listfile))+ ' '+ string + ' files')        
    return(listfile)

def funcReplacePixelsLessThanOrEqualZeros(data):
    listCoordinates = np.transpose(np.where(data <= 0))
    #if (len(listCoordinates == 0)):
    #    return(data)
    dataFiltered = scipy.ndimage.filters.median_filter(data, 3)
    dataOutput = data
    for i in range(len(listCoordinates)):
        [r, c] = listCoordinates[i]
        dataOutput[r, c] = dataFiltered[r, c]
    return(dataOutput)

def funcReplacePixelsComplex(data):
    listCoordinates = np.transpose(np.where(data == complex))
    if (len(listCoordinates == 0)):
        return(data)
    #dataFiltered = scipy.signal.medfilt(data, 1)
    dataFiltered = scipy.ndimage.filters.median_filter(data, 3)
    dataOutput = data
    for i in range(len(listCoordinates)):
        [r, c] = listCoordinates[i]
        dataOutput[r, c] = dataFiltered[r, c]
    return(dataOutput)

def funcCalculateAbsorption(closed, Open, raw):
    numerator = Open - closed
    denominator = raw - closed
    numerator = funcReplacePixelsLessThanOrEqualZeros(numerator)
    denominator = funcReplacePixelsLessThanOrEqualZeros(denominator)
    dataAbsorption = np.log(numerator / denominator)
    dataAbsorption = funcReplacePixelsComplex(dataAbsorption)
    return(dataAbsorption)

pathRoot = '/Users/jumaoyuan/Dropbox/VisTrails_1/workflow_mathematica_bullet'
pathRaw = os.path.join(pathRoot, 'TXT_raw_images')
pathAbsorption = os.path.join(pathRoot, 'FITS_abs_Py')
pathSinograms = os.path.join(pathRoot, 'FITS_sinograms')
pathFigures = os.path.join(pathRoot, 'PNG_figures')
try:
    os.stat(pathAbsorption)
except:
    os.mkdir(pathAbsorption)       
try:
    os.stat(pathFigures)
except:
    os.mkdir(pathFigures)       

filename = readFile(pathRaw, "Open")
dataOpen = np.loadtxt(filename[0]) #read text image file
[rows, columns] = dataOpen.shape
print (rows, columns)
plt.imshow(dataOpen, cmap = 'gray')
plt.show()
plt.plot(dataOpen.flatten())
plt.show()

pixelSizeMM = 0.055

filename = readFile(pathRaw, 'Closed')
dataClosed = np.loadtxt(filename[0])
plt.imshow(dataClosed, cmap = 'gray')
plt.show()
plt.plot(dataOpen.flatten())
plt.show()

listFilenames = readFile(pathRaw, 'PSI_')
listFilenames[0:4]
numRaw = len(listFilenames)
numRaw

print (listFilenames[4].split('_'))
print ()
print (np.int(listFilenames[4].split('_')[7]))
print ()
print (listFilenames[4].split('_')[8].split('.')[0].split('p'))
firstAngle = listFilenames[4].split('_')[8].split('.')[0].split('p')[0]
secondAngle = np.str(np.int(listFilenames[4].split('_')[8].split('.')[0].split('p')[1]))
np.float(".".join([firstAngle, secondAngle]))

listSequenceNumbers = []
listAngles = []
for i in range(numRaw):
    firstAngle = listFilenames[i].split('_')[8].split('.')[0].split('p')[0]
    secondAngle = np.str(np.int(listFilenames[i].split('_')[8].split('.')[0].split('p')[1]))
    listAngles.append(np.float(".".join([firstAngle, secondAngle])))
    listSequenceNumbers.append(np.int(listFilenames[i].split('_')[7]))

print (listAngles)
plt.plot(listAngles)
plt.show()

dataRawStack = np.zeros((numRaw, rows, columns))
for i in range(numRaw):
    dataRawStack[i,:,:] = np.loadtxt(listFilenames[i])

print (dataRawStack.shape)

index = 2
plt.imshow(dataRawStack[index,:,:], cmap = 'gray')
annotate('$1.8^o$',xy=(0,0), xytext=(10,30), fontsize=15, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.3))
plt.show()
listAngles[index]

index = 21
plt.imshow(dataRawStack[index-1, :, :], cmap = 'gray')
annotate('#21',xy=(0,0), xytext=(10,30), fontsize=15, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.3))
plt.show()
plt.imshow(dataRawStack[index, :, :], cmap = 'gray')
annotate('#22',xy=(0,0), xytext=(10,30), fontsize=15, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.3))
plt.show()
plt.imshow(dataRawStack[index+1, :, :], cmap = 'gray')
annotate('#23',xy=(0,0), xytext=(10,30), fontsize=15, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.3))
plt.show()

index = 45
plt.imshow(dataRawStack[index-1, :, :], cmap = 'gray')
annotate('#45',xy=(0,0), xytext=(10,30), fontsize=15, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.3))
plt.show()
plt.imshow(dataRawStack[index, :, :], cmap = 'gray')
annotate('#46',xy=(0,0), xytext=(10,30), fontsize=15, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.3))
plt.show()
plt.imshow(dataRawStack[index+1, :, :], cmap = 'gray')
annotate('#47',xy=(0,0), xytext=(10,30), fontsize=15, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.3))
plt.show()

index = 21
dataRawStack[index, :, :] = (dataRawStack[index-1, :, :] + dataRawStack[index+1, :, :])/2

plt.imshow(dataRawStack[index-1, :, :], cmap = 'gray')
annotate('#21',xy=(0,0), xytext=(10,30), fontsize=15, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.3))
plt.show()
plt.imshow(dataRawStack[index, :, :], cmap = 'gray')
annotate('#22',xy=(0,0), xytext=(10,30), fontsize=15, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.3))
plt.show()
plt.imshow(dataRawStack[index+1, :, :], cmap = 'gray')
annotate('#23',xy=(0,0), xytext=(10,30), fontsize=15, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.3))
plt.show()

index = 45
dataRawStack[index, :, :] = (dataRawStack[index-1, :, :] + dataRawStack[index+1, :, :])/2

plt.imshow(dataRawStack[index-1, :, :], cmap = 'gray')
annotate('#45',xy=(0,0), xytext=(10,30), fontsize=15, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.3))
plt.show()
plt.imshow(dataRawStack[index, :, :], cmap = 'gray')
annotate('#46',xy=(0,0), xytext=(10,30), fontsize=15, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.3))
plt.show()
plt.imshow(dataRawStack[index+1, :, :], cmap = 'gray')
annotate('#47',xy=(0,0), xytext=(10,30), fontsize=15, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.3))
plt.show()

for indexImage in range(numRaw):
    dataRaw = dataRawStack[indexImage, :, :]
    dataAbsorption = funcCalculateAbsorption(dataClosed, dataOpen, dataRaw)
    sequenceNumber = listSequenceNumbers[indexImage]
    angle = listAngles[indexImage]
    print(indexImage, sequenceNumber, angle)
    newFilename = 'absorption_' + np.str(sequenceNumber).zfill(4) + '_' + np.str(angle) + '.fits'
    absFilename = os.path.join(pathAbsorption, newFilename)
    try: #remove file if exits
        os.remove(absFilename)
    except OSError:
        pass
    absFits = fits.HDUList([fits.PrimaryHDU(dataAbsorption)])
    absFits.writeto(absFilename)
    absFits.close()
    clear_output()

index = 2
listFilenames = readFile(pathAbsorption, 'abs')
angle = listAngles[index]
oneImage = fits.open(listFilenames[index])
oneImage.info()
#oneImage.close()
oneImageArray = oneImage[0].data
oneImageArray.shape
plt.imshow(oneImageArray, cmap = 'gray')
annotate('$1.8^o$',xy=(0,0), xytext=(10,30), fontsize=15, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.3))
plt.show()
angle

