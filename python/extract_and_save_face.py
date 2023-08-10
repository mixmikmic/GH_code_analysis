get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import concurrent.futures
import os

# Direction is the target directory to save images
def detectFace(filename, direction):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.05, 3, minSize = (100,100))

    if len(faces):
        filename = filename.split('/')[1]
        cv2.imwrite(direction + filename, image)

# Open the source directory of images
photoDir = 'female/'
photoList = os.listdir(photoDir)
# The target directory to save images
direction = 'female_detect/'

# Multi-threading to speed up
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    for photo in photoList:
        try:
            filename = photoDir+photo
            executor.submit(detectFace, filename, direction)
        except Exception as exc:
            print(exc)

# Open the source directory of images
photoDir = 'woman_without_hair/'
photoList = os.listdir(photoDir)
# The target directory to save faces
direction = 'woman_without_hair/'

# Direction is the target directory to save images
def extractFace(filename, direction):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.05, 3, minSize = (100,100))
    
    filealt = filename.split('/')[1]

    im = Image.open(filename)
    i = 0
    for (x, y, w, h) in faces:
        filename = str(i) + filealt
        center_x = x+w/2
        center_y = y+h/2
        b_dim = min(max(w,h)*1.2,im.width, im.height) 
        box = (center_x-b_dim/2, center_y-b_dim/2, center_x+b_dim/2, center_y+b_dim/2)
        # Crop Image
        crpim = im.crop(box).resize((224,224))
        # Save Image
        crpim.save(direction + filename)
        i += 1

# Multi-threading to speed up
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    for photo in photoList:
        try:
            filename = photoDir+photo
            executor.submit(extractFace, filename, direction)
        except Exception as exc:
            print(exc)

