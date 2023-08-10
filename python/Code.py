Path = './yalefaces/'

Path2 = './yaleface_jpg/'

files = os.listdir(Path)
files2 = os.listdir(Path2)
images = []

for name in files:
    temp  = Image.open(Path+name).save('./yaleface_jpg/'+name+'.jpg')


for name in files2:
    temp = cv2.imread(Path2+name)
    images.append(temp)

#importing required library

import os
import cv2
import numpy as np
import math
from PIL import Image

#initializing global parameters

# folder_tr = "yaleface_jpg/tr"
# folder_te = "yaleface_jpg/te"
folder_tr = "tr"
folder_te = "te"
total_tr = len(next(os.walk(os.getcwd()+"/"+folder_tr))[2])
total_te = len(next(os.walk(os.getcwd()+"/"+folder_te))[2])
dim = 200

# input image maytix -> column
# matrix L: all images column vise

f = np.empty(shape=(dim*dim,total_tr), dtype='float64')
i = 0
for filename in os.listdir(os.getcwd()+"/"+folder_tr):
    #Read a image 
    image = cv2.imread(folder_tr+"/"+filename)
    #resize image
    resized = cv2.resize(image, (dim,dim), interpolation = cv2.INTER_AREA)
    #Convert it into grayscale image
    gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #flattening an image
    gray_vector = np.array(gray_image, dtype='float64').flatten()
    f[:, i] = gray_vector[:] 
    i = i + 1

#Mean face
mean_face = np.sum(f, axis=1) /total_tr

#subtract mean face
for i in range(total_tr):
    f[:,i] -= mean_face[:]

# SVD
u, s, vh = np.linalg.svd(f, full_matrices=False)

# Creating emplty array to store f*u
fu = np.empty(shape = (u.shape[0]*u.shape[1], u.shape[1]),  dtype=np.int8)

# Creating temp array
temp = np.empty(shape = (u.shape[0], u.shape[1]),  dtype=np.int8)

# loop over number of faces f, for f*u, store the answer of array by flattning it in fu array

for i in range(f.shape[1]):
    
    for c in range(u.shape[1]):    
        temp[:,c] = f[:,i] * u[:,c]
    
    tempF = np.array(temp, dtype='int8').flatten()
    
    fu[:, i] = tempF[:]

tt = np.empty(shape = (u.shape[0], u.shape[1]),  dtype=np.int8)

i = 0
wrong = 0
for filename in os.listdir(os.getcwd()+"/"+folder_te):
    print("input face: ",filename)
    # read image
    test = cv2.imread(folder_te+"/"+filename)   
    # resize
    test = cv2.resize(test, (dim,dim), interpolation = cv2.INTER_AREA)
    # convert to gray scale
    test= cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    # flatten image
    test = np.array(test, dtype='float64').flatten()  
    # subtract mean
    test -= mean_face     

    # matrix: tt (test image as column vector)
    tt[:, i] = test[:]
    i=i+1

# test image * u (u of SVD)

# Creating emplty array to store f*u
t = np.empty(shape = (u.shape[0], u.shape[1]),  dtype=np.int8)
    
for col in range(u.shape[1]):    
    t[:,col] = tt[:,0] * u[:,col]
    
# flatten whole tt*u matrix
tF = np.array(t, dtype='int8').flatten()

# fu - tf

fu_tu = np.empty(shape = (u.shape[0]*u.shape[1], u.shape[1]))

for col in range(u.shape[1]):
    fu_tu[:,col] = fu[:,col] - tF[:]

# find norm

from numpy import linalg as LA

ans = np.empty(shape=(u.shape[1],))

for c in range(fu_tu.shape[1]):    
    ans[c] = np.linalg.norm(fu_tu[:,c])

# shorting norm values

temp_ans = np.empty(shape=(u.shape[1],))
temp=np.copy(ans)

temp.sort()
check = temp[0]
print("norm: ",check)

print(ans)
print(temp)

# find image index at the minimum norm
index=0

for i in range(ans.shape[0]):
    if check == ans[i]:
        print("image index:",i)
        index=i
        break

# print output image name
i = 0
index
for filename in os.listdir(os.getcwd()+"/"+folder_tr):
    
    if index == i:
        print("recognized face: ",filename)
        break
        
    else:
        i=i+1



