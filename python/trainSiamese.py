from __future__ import print_function

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

from model import siamese
from dataset import ReadImages, collection

def selectPairs(net, imageList, criterion, transform, nbSelect=10, batchSize=32):
    """
        Produce nbSelect pairs of not corresponding images that maximize the loss function
    """
    CDict = collection.createConceptDict(imageList)
    net.eval().cuda() #set the network to eval mode and on the GPU
    for im in imageList: #for each image
        candidate = [random.choice(CDict[k]) for k in CDict.keys()] #select one random image per concept
        for i in range(len(candidate)/batchSize):
            t = torch.Tensor(batchSize, 3, 225, 225).cuda()
            for j in range(batchSize):
                t[j] = transform(Image.open(candidate[i*batchSize+j]))
            out = net(t)
            #TODO compute the good labels
            loss = criterion(out, lab)
        
        rest = len(candidate)%batchSize
        #TODO handle the rest of the image
        
    

def run1():
    dataset = readImageswithPattern('/video/CLICIDE', lambda x:x.split('/')[-1].split('-')[0]) #read Clicide dataset
    
    model = siamese.siamese().train().cuda() #load the model
    
    
    

imageList = ReadImages.readImageswithPattern('/video/CLICIDE/', lambda x:x.split("/")[-1].split("-")[0])

s = selectPairs(None, imageList, None)

s

