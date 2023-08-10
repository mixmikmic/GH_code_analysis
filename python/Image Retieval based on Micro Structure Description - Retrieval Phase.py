#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import operator
from IPython.display import Image

from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017')
collection = client.test_database.coral2

db = []
num = input('Enter the input Image: ')
for x in collection.find():
    db = np.array(x['distances'])
Image(str(num)+'.jpg')

inputImage = db[num]

distance = np.zeros(1000*72).reshape(1000,72)

distance = abs(db - inputImage)

distanceSum = np.sum(distance,axis=1)

keys = np.arange(len(distanceSum),dtype=int)

Imagedictionary = dict(zip(keys, distanceSum))
sorted_images = sorted(Imagedictionary.items(), key=operator.itemgetter(1))

i = 0;
Resultimages = []
ResultHists = np.zeros(21*72).reshape(21,72)
for key in sorted_images:
    if(i<=20):
        ResultHists[i]=db[key[0]]
        i = i +1
        imageName = str(key[0])+'.jpg'
        Resultimages.append(imageName)
        print imageName
        Image(imageName)
    else:
        break;

for ima in Resultimages:
    imageD = Image(ima)
    display(imageD)

i=0
for hist in ResultHists:
    i=i+1
    plt.figure(i)
    plt.axis([0, 72, 0, 1])
    plt.bar(np.arange(72),hist)
    plt.xlabel('Bin size')
    plt.ylabel('Frequency')
    plt.title('Histogram of MSD of image')
    plt.grid(True)
plt.show()



