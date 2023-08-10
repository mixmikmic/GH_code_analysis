import numpy as np
import cv2
import glob
import os
import codecs
import matplotlib.pyplot as plt

#This fetches all the folders in the lekha-ocr-database/train_images folder 
def returnFolders(path):
    
    folders = [ f for f in os.listdir(path) ]
        
    print "Found {} folders".format(len(folders))
    
    return folders

def preprocessImage(image):
    #Does Adaptive Gaussian Thresholding (See sudoku image in documentation for understanding)
    adaptiveGThreshold = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    retVal,thresholded_img = cv2.threshold(adaptiveGThreshold,127,255,cv2.THRESH_BINARY)
    img = cv2.resize(thresholded_img,(32,32),interpolation = cv2.INTER_CUBIC)
    return img

def getImagesInFolder(imageFolder):
    
    image_paths = glob.glob(imageFolder+'*.png')
    image_label = imageFolder.split('/')[-2]
    len_images = len(image_paths)
    print "Found {} Images of Label {}".format(len_images,image_label)
    
    input_images = [ cv2.imread(img,0) for img in image_paths ]
    input_images = [ preprocessImage(img) for img in input_images ]
    
    labels =  [ image_label for img in input_images ]
    
    images,labels =  np.array(input_images),np.array(labels)
    
    return images,labels

def prepareDataset(path):
    trainX = None
    trainy = None
    
    folders =  returnFolders(path)
    for f in folders:
        imageFolder = path+str(f)+'/'
        images,labels = getImagesInFolder(imageFolder)
        
        if trainX is None:
            trainX = images
        else:
            trainX = np.append(trainX,images,axis = 0)
        
        if trainy is None:
            trainy = labels
        else:
            trainy = np.append(trainy,labels,axis=0)
            
    print "TrainX Shape: ",trainX.shape
    print "Trainy Shape: ",trainy.shape
    return trainX,trainy

def saveNumpyArrays(trainX,trainy):
    np.save('NP-Dataset/X.npy',trainX)
    np.save('NP-Dataset/y.npy',trainy)
    print "Saved Numpy Arrays to NP-Dataset/"
    return

trainX,trainy = prepareDataset("/home/amrith/Machine-Learning/MalayalamOCR/IN/")

saveNumpyArrays(trainX,trainy)

def getShape():
    return trainX.shape,trainy.shape

