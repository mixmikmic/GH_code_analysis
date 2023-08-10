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

class Siamese(torch.nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.fc1 = torch.nn.Linear(4096, 64)

    def forward(self, x1, x2):
        o1 = self.fc1(x1)
        o2 = self.fc1(x2)
        o = torch.sqrt(torch.sum(torch.mul(o1-o2, o1-o2), 1))
        return o

def evaluate(net, dataset):
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
        batchOutput = net(Variable(descrs1).cuda(), Variable(descrs2).cuda())
        
        classifierScores += batchOutput.data.cpu().numpy().T[0].tolist()
        labels += batchLabels
    
    return classifierScores, labels

class Metrics():
    def __init__(self, classifierScores, labels):
        self.scores = classifierScores
        self.labels = labels

    def getAvgDist(self):
        nSim, nDiss = 0, 0
        avgDistSim, avgDistDiss = 0.0, 0.0
        for i in range(len(self.scores)):
            if self.labels[i] == 1: 
                avgDistDiss += self.scores[i] 
                nDiss += 1
            else: 
                avgDistSim += self.scores[i]
                nSim += 1
        return avgDistSim/nSim, avgDistDiss/nDiss
    
    def getROC(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.labels, self.scores)
        auc = metrics.auc(fpr, tpr)
        eer, r = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1., full_output=True)
        return eer, auc, fpr, tpr

vggFace = load_lua("../../data/lab3/VGG_FACE_pyTorch_small.t7")
dataset = load_lua("../../data/lab3/Experiment_3/cfpw-facePairs-dataset.t7")

torch.manual_seed(0)
np.random.seed(0)

net = Siamese()
criterion = torch.nn.HingeEmbeddingLoss()
optimizer = optim.SGD(net.parameters(), lr=0.00005, weight_decay=0.0005)

net = net.cuda()
criterion = criterion.cuda()

# Before training

scores, labels = evaluate(net, dataset)
verifMetric = Metrics(scores, labels)
avgDistSim, avgDistDiss = verifMetric.getAvgDist()
print "avgDistSim = ", avgDistSim, ", avgDistDiss = ", avgDistDiss

eer, auc, fpr, tpr = verifMetric.getROC()
print "EER = ", eer, ", AUC = ", auc

nEpochs, nPairs, batchSize = 2, len(dataset['pairs']), 10
for epochCtr in range(nEpochs):
    
    shuffle = np.random.permutation(nPairs)
    runningLoss, iterCnt = 0.0, 0
    for startIdx in range(0, nPairs, batchSize):
        endIdx = min(startIdx + batchSize - 1, nPairs - 1)
        size = endIdx - startIdx + 1
    
        imgPaths1, imgPaths2, labels = [], [], []
        for offset in range(size):
            pair = dataset['pairs'][shuffle[startIdx+offset]]
            imgPaths1.append("../../data/lab3/Experiment_3/" + pair.img1)
            imgPaths2.append("../../data/lab3/Experiment_3/" + pair.img2)
            labels.append(int(pair.label))
        
        descrs1 = getVggFeatures(imgPaths1, vggFace).clone()
        descrs2 = getVggFeatures(imgPaths2, vggFace).clone()
        
        batchOutput = net(Variable(descrs1).cuda(), Variable(descrs2).cuda())
        loss = criterion(batchOutput, Variable(torch.Tensor(labels)).cuda())
        loss.backward()
        optimizer.step()
        
        runningLoss += loss.data[0]
        iterCnt += 1
    
    print "epoch ", epochCtr, "/", nEpochs, ": loss = ", runningLoss/iterCnt 

# After training

scores, labels = evaluate(net, dataset)
verifMetric = Metrics(scores, labels)
avgDistSim, avgDistDiss = verifMetric.getAvgDist()
print "avgDistSim = ", avgDistSim, ", avgDistDiss = ", avgDistDiss

eer, auc, fpr, tpr = verifMetric.getROC()
print "EER = ", eer, ", AUC = ", auc

