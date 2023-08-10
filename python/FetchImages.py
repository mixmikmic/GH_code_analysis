# Import Dependencies
import os
import cv2
import numpy as np
from urllib import request

# Function to Fetch Rose Images
def fetchRoseImages():
    # Check if output directory exists or not
    # if it dosen't exists, create it
    # I am using the name 'rose' for the rose images directory. We'll use the directory name later on as the image label.
    if not os.path.exists('rose'):
        os.makedirs('rose/')
    
    # Copy Paste here the URL of the Imagenet webpage containing all the image URL's 
    rosesURL = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04971313'
    # Read the URL's and decode them
    readURL = request.urlopen(rosesURL).read().decode()
    
    # This is just to keep different names of images
    count = 1
    
    print('\n\n Fetching Images of Roses . . . \n\n')
    
    # Since, all URL's are in separate line, split it by '\n'
    for img in readURL.split('\n'):
        # Put code in a try except as some URL's might not have an image or might be broken.
        try:
            # Print all URL's
            print('Fetched: ',img)
            # Retrieve image from URL and save in 'rose/' directory
            request.urlretrieve(img,'rose/'+str(count)+'.jpg')
            count += 1
        except Exception as e:
            print('Exception Occured: ',str(e))           

# Function to Fetch Tulip Images
def fetchTulipImages():
    # Check if output directory exists or not
    # if it dosen't exists, create it
    # I am using the name 'tulip' for the tulip images directory. We'll use the directory name later on as the image label.
    if not os.path.exists('tulip'):
        os.makedirs('tulip/')
    
    # Copy Paste here the URL of the Imagenet webpage containing all the image URL's 
    tulipsURL = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n12454159'
    # Read the URL's and decode them
    readURL = request.urlopen(tulipsURL).read().decode()
    
    # This is just to keep different names of images
    count = 1
    
    print('\n\n Fetching Images of Tulips . . . \n\n')
    
    # Since, all URL's are in separate line, split it by '\n'
    for img in readURL.split('\n'):
        # Put code in a try except as some URL's might not have an image or might be broken.
        try:
            # Print all URL's
            print('Fetched: ',img)
            # Retrieve image from URL and save in 'rose/' directory
            request.urlretrieve(img,'tulip/'+str(count)+'.jpg')
            count += 1
        except Exception as e:
            print('Exception Occured: ',str(e))           

from IPython.display import Image
Image('InvalidImages/ugly.jpg')

Image('InvalidImages/ugly1.jpg')

# Function to Remove Invalid Images
# Compare all files in the rose and tulip folders and comapre all images with the invalid images
# If there is an invalid image, remove it from folder
def removeInvalidImages():
    for files in ['rose']:
        for img in os.listdir(files):
            for invalidImage in os.listdir('InvalidImages'):
                try:
                    imagePath = str(files)+'/'+str(img)
                    invalidImage = cv2.imread('InvalidImages/'+str(invalidImage))
                    currentImage = cv2.imread(imagePath)

                    if invalidImage.shape == currentImage.shape and not(np.bitwise_xor(invalidImage,currentImage).any()):
                        print('Removing Image: ',imagePath)
                        os.remove(imagePath)
                except Exception as e:
                    print('Exception Occured: ',str(e))
                    
    for files in ['tulip']:
        for img in os.listdir(files):
            for invalidImage in os.listdir('InvalidImages'):
                try:
                    imagePath = str(files)+'/'+str(img)
                    invalidImage = cv2.imread('InvalidImages/'+str(invalidImage))
                    currentImage = cv2.imread(imagePath)

                    if invalidImage.shape == currentImage.shape and not(np.bitwise_xor(invalidImage,currentImage).any()):
                        print('Removing Image: ',imagePath)
                        os.remove(imagePath)
                except Exception as e:
                    print('Exception Occured: ',str(e))

# Run both Functions
if __name__ == '__main__':
    fetchRoseImages()
    fetchTulipImages()
    removeInvalidImages()

