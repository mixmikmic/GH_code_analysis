import math
import cv2
import numpy as np

import torch
from torch.utils.serialization import load_lua
from torch.legacy import nn

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')
plt.ion()

def getNameList(filePath):
    names = []
    with open(filePath) as f:
        names = [ line.strip() for line in f ]
    return names

idNames = getNameList("../../data/lab3/names.txt")
print "Number of identities = ", len(idNames)
print "List of the first 5 identities = ", idNames[:5]

dataset = load_lua("../../data/lab3/Experiment_1/facerec-image-dataset.t7")
print "No. of images = ", len(dataset['images'])
for imgName in dataset['images']:
    print imgName

dispImg = mpimg.imread("../../data/lab3/Experiment_1/" + dataset['images'][0])
imgplot = plt.imshow(dispImg)

# load the image using OpenCV
inputImg = cv2.imread("../../data/lab3/Experiment_1/" + dataset['images'][0])

# re-scale the smaller dim (among width, height) to refSize
refSize, targetSize = 256, 224
imgRows, imgCols = inputImg.shape[0], inputImg.shape[1]
if imgCols < imgRows: resizedImg = cv2.resize(inputImg, (refSize, refSize * imgRows / imgCols))
else: resizedImg = cv2.resize(inputImg, (refSize * imgCols / imgRows, refSize))
    
# center-crop
oH, oW = targetSize, targetSize
iH, iW = resizedImg.shape[0], resizedImg.shape[1]
anchorH, anchorW = int(math.ceil((iH - oH)/2)), int(math.ceil((iW - oW) / 2))
croppedImg = resizedImg[anchorH:anchorH+oH, anchorW:anchorW+oW]

print croppedImg.shape

# convert shape from (height, width, 3) to (3, width, height)
channel_1, channel_2, channel_3 = croppedImg[:, :, 0], croppedImg[:, :, 1], croppedImg[:, :, 2]
croppedImg = np.empty([3, croppedImg.shape[0], croppedImg.shape[1]])
croppedImg[0], croppedImg[1], croppedImg[2] = channel_1, channel_2, channel_3

print croppedImg.shape

# subtract training mean
inputImg = inputImg.astype(float)
trainingMean = [129.1863, 104.7624, 93.5940]
for i in range(3): croppedImg[i] = croppedImg[i] - trainingMean[i]

# load pre-trained VGG-Face network
vggFace = load_lua("../../data/lab3/VGG_FACE_pyTorch_small.t7")
vggFace.modules[31] = nn.View(1, 25088)
vggFace = vggFace.cuda()
print vggFace

# forward pass
input = torch.Tensor(croppedImg).unsqueeze(0)
output = vggFace.forward(input.cuda())
output = output.cpu().numpy()

print output.shape

print "ID = ", idNames[np.argmax(output)]

# top-k predictions
ind = output[0].argsort()[-5:][::-1]
for idx in ind:
    print idNames[idx]

vggFeatures = vggFace.modules[35].output.cpu().numpy()
print vggFeatures.shape
print vggFeatures

print np.linalg.norm(vggFeatures)

