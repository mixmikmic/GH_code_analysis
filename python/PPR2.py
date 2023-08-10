from __future__ import print_function
import numpy as np
from IPython.display import display, Image
import os
import PIL.Image
from sklearn.cross_validation import train_test_split
import pandas as pd

image_size = 350
"""
filelist   = os.listdir("data/images")
x = np.array([np.array(PIL.Image.open("data/images/"+fname)) for fname in filelist])
print(x.shape)
x.dump('file.npy')"""
filelist   = os.listdir("data/images")
filelist[0:10]

x = np.load('file.npy')
print(x[0].shape)

data = pd.read_csv("data/legend.csv", usecols = [1,2])
data.head(10)

data.head(3)
data['emotion']=data['emotion'].str.lower()
data['emotion']

dictn = data.set_index('image').to_dict()
dictn = dictn['emotion']
dictn

rawlabel = list(data['emotion'])
label = []
for i in rawlabel:
    var = 0
    if i=="anger":
        var = 1
    elif i=="surprise":
        var = 2
    elif i=="happiness":
        var = 3
    elif i =="neutral":
        var = 0
    elif i=="disgust":
        var = 4
    elif i=="fear":
        var = 5
    elif i=="contempt":
        var = 6
    else: #sadness
        var = 7 
    label.append(var)
Y = np.array(label)

import pickle
pickle_out = open("dict.pickle","wb")
pickle.dump(Y, pickle_out)
pickle_out.close()
Y

import cv2

filelist = os.listdir("data/images")
image = cv2.imread("data/images/"+filelist[0])
image = cv2.resize(image,(64,64), interpolation = cv2.INTER_CUBIC)
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image.shape

image

import cv2
import os
X = []
Y = []
filelist = os.listdir("data/images")
for i in filelist:
    try:
        filelist = os.listdir("data/images")
        image = cv2.imread("data/images/"+i)
        
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if(image.shape==(350,350)):
            image = cv2.resize(image,(64,64), interpolation = cv2.INTER_CUBIC)
            #image = np.array(image)
            X.append(image)
            Y.append(dictn[i])
    except KeyError:
        X.pop()
Y

X

len(X)

len(Y)

cleandata = {'x':X,'y':Y}
pickle_out = open("cleandata.pickle","wb")
pickle.dump(cleandata, pickle_out)
pickle_out.close()



