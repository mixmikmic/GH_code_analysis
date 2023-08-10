import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.autograd import Variable

import numpy as np
import random
from PIL import Image
from ipywidgets import FloatProgress
from IPython.display import display
from __future__ import print_function

from model import ModelDefinition
from dataset import ReadImages, collection
import os
import os.path as path
import glob

import cv2

def readFrameAnnotation(annotationFile):
    """
        read annotation file
        return the list of annotation ([start, end], gesture)
    """
    anno = []
    for l in open(rootDir+'annotation/'+fName).read().splitlines():
        s = l.split(' ')
        anno += [ ([int(s[1]), int(s[2])], int(s[0]))]
    return anno

def findGestureFrame(frameNumber, annotationFile):
    """
        from Frame Number and the list of annotation
        return the Gesture or None if not in annation
    """
    for seq, gest in annotationFile:
        if frameNumber >= seq[0] and frameNumber <= seq[1]:
            return gest
    return None

def copyParameters(net, netBase):
    for i, f in enumerate(net.features):
        if type(f) is torch.nn.modules.conv.Conv2d:
            f.weight.data = netBase.features[i].weight.data
            f.bias.data = netBase.features[i].bias.data
    for i, c in enumerate(net.classifier):
        if type(c) is torch.nn.modules.linear.Linear:
            if c.weight.size() == netBase.classifier[i].weight.size():
                c.weight.data = netBase.classifier[i].weight.data
                c.bias.data = netBase.classifier[i].bias.data

def fillInput(nframe, video):
    t = transforms.Compose(
                (transforms.ToPILImage(),
                transforms.Scale(225),
                transforms.RandomCrop(225),
                transforms.ToTensor())
                )
    inputs = torch.Tensor(nframe,3,225,225)
    for j in range(nframe):
        ret, frame = video.read()
        inputs[j] = t(frame)
    return inputs

#TODO : test if difference between learning only gesture per batch

def learnSequence(sequence, gesture, video, model, criterion, optimize, batchSize=32):
    numberFrame = seq[1] - seq[0]
    running_loss = 0
    while numberFrame > 0:
        if numberFrame >= batchSize:
            inputs = fillInput(batchSize, video)
            numberFrame -= batchSize
            
            labels = torch.LongTensor([gesture]*batchSize)
        else:
            inputs = fillInput(numberFrame, video)
            labels = torch.LongTensor([gesture]*numberFrame)
            numberFrame = 0
        #inputs.cuda()
        #labels.cuda()
        inputs = Variable(inputs)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, Variable(labels))
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
    return running_loss

rootDir = '/video/Gesture/'
model = models.AlexNet(num_classes=7)
copyParameters(model, models.alexnet(pretrained=True))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=0.01, momentum=0.9)

for video in glob.glob(rootDir+'*.mp4'):
    print("Video ", video)
    
    fName = path.splitext(path.basename(video))[0] #basename
    annotation = readFrameAnnotation(rootDir+'annotation/'+fName) #read annotation
    
    videoCap = cv2.VideoCapture(video)
    
    for seq, gesture in annotation:
        print("Sequence ", seq, " Gesture : ", gesture)
        rl = learnSequence(seq, gesture, videoCap, model, criterion, optimizer)
        print(rl)
videoCap.release()



