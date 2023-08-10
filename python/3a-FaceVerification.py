import cv2, math
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.serialization import load_lua
from torch.legacy import nn

from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')
plt.ion()

def loadImage(imgPath):
    inputImg = cv2.imread(imgPath)

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

    # convert shape from (height, width, 3) to (3, width, height)
    channel_1, channel_2, channel_3 = croppedImg[:, :, 0], croppedImg[:, :, 1], croppedImg[:, :, 2]
    croppedImg = np.empty([3, croppedImg.shape[0], croppedImg.shape[1]])
    croppedImg[0], croppedImg[1], croppedImg[2] = channel_1, channel_2, channel_3

    # subtract training mean
    inputImg = inputImg.astype(float)
    trainingMean = [129.1863, 104.7624, 93.5940]
    for i in range(3): croppedImg[i] = croppedImg[i] - trainingMean[i]
    return croppedImg

def getVggFeatures(imgPaths, preTrainedNet):
    nImgs = len(imgPaths)
    preTrainedNet.modules[31] = nn.View(nImgs, 25088)
    preTrainedNet = preTrainedNet.cuda()
    
    batchInput = torch.Tensor(nImgs, 3, 224, 224)
    for i in range(nImgs): batchInput[i] = torch.from_numpy(loadImage(imgPaths[i]))
    
    batchOutput = preTrainedNet.forward(batchInput.cuda())
    return preTrainedNet.modules[35].output.cpu()

dataset = load_lua("../../data/lab3/Experiment_3/cfpw-facePairs-dataset.t7")
print "# image pairs in the dataset = ", len(dataset['pairs'])

imgPath1 = "../../data/lab3/Experiment_3/" + dataset['pairs'][1].img1
imgPath2 = "../../data/lab3/Experiment_3/" + dataset['pairs'][1].img2
dispImg1, dispImg2 = mpimg.imread(imgPath1), mpimg.imread(imgPath2)

f, axarr = plt.subplots(1, 2)
axarr[0].imshow(dispImg1)
axarr[1].imshow(dispImg2)

print "label = ", dataset['pairs'][1].label

imgPath1 = "../../data/lab3/Experiment_3/" + dataset['pairs'][99].img1
imgPath2 = "../../data/lab3/Experiment_3/" + dataset['pairs'][99].img2
dispImg1, dispImg2 = mpimg.imread(imgPath1), mpimg.imread(imgPath2)

f, axarr = plt.subplots(1, 2)
axarr[0].imshow(dispImg1)
axarr[1].imshow(dispImg2)

print "label = ", dataset['pairs'][99].label

vggFace = load_lua("../../data/lab3/VGG_FACE_pyTorch_small.t7")

nPairs, batchSize = len(dataset['pairs']), 10
classifierScores, labels = [], []

for startIdx in range(0, nPairs, batchSize):
    endIdx = min(startIdx+batchSize-1, nPairs-1)
    size = (endIdx - startIdx + 1)

    imgPaths1, imgPaths2, batchLabels = [], [], []
    for offset in range(size):
        pair = dataset['pairs'][startIdx+offset]
        imgPaths1.append("../../data/lab3/Experiment_3/" + pair.img1)
        imgPaths2.append("../../data/lab3/Experiment_3/" + pair.img2)
        batchLabels.append(int(pair.label) * -1)
    
    descrs1 = getVggFeatures(imgPaths1, vggFace).clone()
    descrs2 = getVggFeatures(imgPaths2, vggFace).clone()
    for i in range(size):
        descr1, descr2 = descrs1[i].numpy(), descrs2[i].numpy()
        normDescr1, normDescr2 = descr1 / np.linalg.norm(descr1), descr2 / np.linalg.norm(descr2)
        classifierScores.append( np.linalg.norm(normDescr1 - normDescr2) )
        labels.append(batchLabels[i])


fpr, tpr, thresholds = metrics.roc_curve(labels, classifierScores)
auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate (fpr)")
plt.ylabel("True Positive Rate (tpr)")
plt.show()

print "AUC = ", auc

