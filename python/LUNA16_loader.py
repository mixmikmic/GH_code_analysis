import SimpleITK as sitk
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import ntpath
get_ipython().magic('matplotlib inline')

# To get the data:
# wget https://www.dropbox.com/sh/mtip9dx6zt9nb3z/AAAs2wbJxbNM44-uafZyoMVca/subset5.zip
# The files are 7-zipped. Regular linux unzip won't work to uncompress them. Use 7za instead.
# 7za e subset5.zip

DATA_DIR = "/Volumes/data/tonyr/dicom/LUNA16/"
# img_file = 'subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.213140617640021803112060161074.mhd'
# img_file = 'subset1/1.3.6.1.4.1.14519.5.2.1.6279.6001.768276876111112560631432843476.mhd'
# img_file = 'subset1/1.3.6.1.4.1.14519.5.2.1.6279.6001.282512043257574309474415322775.mhd'  # Transformation matrix is flipped. This is where the np.absolute comes in
# img_file = 'subset1/1.3.6.1.4.1.14519.5.2.1.6279.6001.134370886216012873213579659366.mhd'
img_file = 'subset2/1.3.6.1.4.1.14519.5.2.1.6279.6001.922852847124879997825997808179.mhd'
img_file= 'subset1/1.3.6.1.4.1.14519.5.2.1.6279.6001.336894364358709782463716339027.mhd'
img_file = 'subset1/1.3.6.1.4.1.14519.5.2.1.6279.6001.231834776365874788440767645596.mhd'
img_file = 'subset1/1.3.6.1.4.1.14519.5.2.1.6279.6001.208737629504245244513001631764.mhd'
img_file = 'subset8/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.mhd'
# img_file = 'subset5/1.3.6.1.4.1.14519.5.2.1.6279.6001.110678335949765929063942738609.mhd'
# img_file = 'subset9/1.3.6.1.4.1.14519.5.2.1.6279.6001.141345499716190654505508410197.mhd'
# img_file = 'subset8/1.3.6.1.4.1.14519.5.2.1.6279.6001.724562063158320418413995627171.mhd'
#img_file = 'subset5/1.3.6.1.4.1.14519.5.2.1.6279.6001.112740418331256326754121315800.mhd'  # 10 nodules here.
#img_file='subset5/1.3.6.1.4.1.14519.5.2.1.6279.6001.275755514659958628040305922764.mhd'  # 5 nodules here. Order varies between candidates and annotations file
cand_path = 'CSVFILES/candidates_with_annotations.csv'

df = pd.read_csv(DATA_DIR+cand_path)

df['diameter_mm'].hist();
plt.title('Histogram of nodules');
plt.xlabel('Size [mm]');
plt.ylabel('Count');

df['diameter_mm'].describe()



"""
Normalize pixel depth into Hounsfield units (HU)

This tries to get all pixels between -1000 and 400 HU.
All other HU will be masked.

"""
def normalizePlanes(npzarray):
     
    maxHU = 400.
    minHU = -1000.
 
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

def extractCandidates(img_file):
    
    # Get the name of the file
    subjectName = ntpath.splitext(ntpath.basename(img_file))[0]  # Strip off the .mhd extension
    
    # Read the list of candidate ROI
    dfCandidates = pd.read_csv(DATA_DIR+cand_path)
    
    
    numCandidates = dfCandidates[dfCandidates['seriesuid']==subjectName].shape[0]
    print('There are {} candidate nodules in this file.'.format(numCandidates))
    
    numNonNodules = sum(dfCandidates[dfCandidates['seriesuid']==subjectName]['class'] == 0)
    numNodules = sum(dfCandidates[dfCandidates['seriesuid']==subjectName]['class'] == 1)
    print('{} are true nodules (class 1) and {} are non-nodules (class 0)'.format(numNodules, numNonNodules))
    
    # Read if the candidate ROI is a nodule (1) or non-nodule (0)
    candidateValues = dfCandidates[dfCandidates['seriesuid']==subjectName]['class'].values
    
    # Get the world coordinates (mm) of the candidate ROI center
    worldCoords = dfCandidates[dfCandidates['seriesuid']==subjectName][['coordX', 'coordY', 'coordZ']].values
    
    # Use SimpleITK to read the mhd image
    itkimage = sitk.ReadImage(DATA_DIR+img_file)
    
    # Get the real world origin (mm) for this image
    originMatrix = np.tile(itkimage.GetOrigin(), (numCandidates,1))  # Real world origin for this image (0,0)
    
    # Subtract the real world origin and scale by the real world (mm per pixel)
    # This should give us the X,Y,Z coordinates for the candidates
    candidatesPixels = (np.round(np.absolute(worldCoords - originMatrix) / itkimage.GetSpacing())).astype(int)
    
    # Replace the missing diameters with the 50th percentile diameter 
    
    
    candidateDiameter = dfCandidates['diameter_mm'].fillna(dfCandidates['diameter_mm'].quantile(0.5)).values / itkimage.GetSpacing()[1]
     
    candidatePatches = []
    
    imgAll = sitk.GetArrayFromImage(itkimage) # Read the image volume
    
    for candNum in range(numCandidates):
        
        #print('Extracting candidate patch #{}'.format(candNum))
        candidateVoxel = candidatesPixels[candNum,:]
        xpos = int(candidateVoxel[0])
        ypos = int(candidateVoxel[1])
        zpos = int(candidateVoxel[2])
        
        # Need to handle the candidates where the window would extend beyond the image boundaries
        windowSize = 32
        x_lower = np.max([0, xpos - windowSize])  # Return 0 if position off image
        x_upper = np.min([xpos + windowSize, itkimage.GetWidth()]) # Return  maxWidth if position off image
        
        y_lower = np.max([0, ypos - windowSize])  # Return 0 if position off image
        y_upper = np.min([ypos + windowSize, itkimage.GetHeight()]) # Return  maxHeight if position off image
         
        # SimpleITK is x,y,z. Numpy is z, y, x.
        imgPatch = imgAll[zpos, y_lower:y_upper, x_lower:x_upper]
        
        # Normalize to the Hounsfield units
        # TODO: I don't think we should normalize into Housefield units
        imgPatchNorm = imgPatch #normalizePlanes(imgPatch)
        
        candidatePatches.append(imgPatchNorm)  # Append the candidate image patches to a python list

    return candidatePatches, candidateValues, candidateDiameter

patchesArray, valuesArray, noduleDiameter = extractCandidates(img_file)

# Display the positive candidates (nodules)

numPositives = np.where(valuesArray==1)[0]

if (len(numPositives) > 1):
    plt.figure(figsize=(10,10))

for i, candidateNum in enumerate(numPositives):

    imgPatch = patchesArray[candidateNum]
    plt.subplot(len(numPositives),1,i+1)
    plt.imshow(imgPatch, cmap='bone');
    plt.title('Value = {}, Diameter = {:.1f}'.format(valuesArray[candidateNum], noduleDiameter[candidateNum]));
    circle = plt.Circle((imgPatch.shape[0]/2, 
                         imgPatch.shape[1]/2), 
                        radius=noduleDiameter[candidateNum], fill=False, color='r')
    plt.gca().add_patch(circle)
    

# Display three candidate non-nodules

numNegatives = np.where(valuesArray==0)[0]
plt.figure(figsize=(15,15))
for i, candidateNum in enumerate(numNegatives[:3]):

    imgPatch = patchesArray[candidateNum]
    plt.subplot(3,1,i+1)
    plt.imshow(imgPatch, cmap='bone');
    plt.title('Value = {}, Diameter = {:.1f}'.format(valuesArray[candidateNum], noduleDiameter[candidateNum]));
    circle = plt.Circle((imgPatch.shape[0]/2, 
                         imgPatch.shape[1]/2), 
                        radius=noduleDiameter[candidateNum], fill=False, color='y')
    plt.gca().add_patch(circle)
    



