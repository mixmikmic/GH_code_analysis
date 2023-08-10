get_ipython().magic('matplotlib inline')
import cv2
import numpy as np

def invertChannels(img):
    b,g,r = cv2.split(img) # split channels
    return cv2.merge([r,g,b]) # merge in rgb order to display with matplotlib

get_ipython().magic('matplotlib inline')
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('..\colorImages\iris.jpg')
img1 = cv2.imread('..\colorImages\iris_1.jpg')
img2 = cv2.imread('..\colorImages\iris_2.jpg')
img3 = cv2.imread('..\colorImages\iris_3.jpg')
img4 = cv2.imread('..\colorImages\iris_4.jpg')

plt.figure(figsize=(10,5)) 

plt.subplot(121)
plt.imshow(invertChannels(img))

plt.subplot(122)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])    

plt.figure(figsize=(10,20)) 
plt.subplot(421);plt.imshow(invertChannels(img1))
plt.subplot(422)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img1],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])  

plt.subplot(423);plt.imshow(invertChannels(img2))
plt.subplot(424)
for i,col in enumerate(color):
    histr = cv2.calcHist([img2],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])  
    
plt.subplot(425);plt.imshow(invertChannels(img3))
plt.subplot(426)
for i,col in enumerate(color):
    histr = cv2.calcHist([img3],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])  
    
plt.subplot(427);plt.imshow(invertChannels(img4))
plt.subplot(428)
for i,col in enumerate(color):
    histr = cv2.calcHist([img4],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])  

img1[:,:,1].shape

# increase brightness

plt.figure(figsize=(10,5)) 
img1_r1 = (img1 + 100).astype('uint8')
plt.subplot(121)
plt.imshow(invertChannels(img1_r1))

plt.subplot(122)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img1_r1],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256]) 

# build look up table
high = np.max(img1_r1)
low = np.min(img1_r1)

x = np.linspace(0,255,256);

declive = 255./(high - low);
ordenada = - declive * low;
print high
print low
print ordenada
print declive
table = declive * x + ordenada;
table[0:low] = 0;
table[high:256] = 255;

table = np.array([table.astype('uint8')])

img1_r1[:,:,1]

table

img1_r2 = cv2.LUT(img1_r1,table)

plt.imshow(invertChannels(img1_r2))

for i,col in enumerate(color):
    histr = cv2.calcHist([img1_r2],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])  



