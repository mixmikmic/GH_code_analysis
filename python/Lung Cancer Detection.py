import dicom
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pandas as pd
from tqdm import tqdm
import timeit
import time
from skimage import transform, io
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import os
from __future__ import division, print_function
import shutil
get_ipython().run_line_magic('matplotlib', 'inline')
import glob
import torchvision.transforms as T
from torchvision import utils
import scipy.misc as m

# Defining the data directories
root_dir = '/Users/navneetmkumar/Documents/Paper Implementations'
sample_images_dir = root_dir+'/sample_images'
patients = os.listdir(sample_images_dir)
labels = pd.read_csv('/Volumes/Nav/Datasets/stage1_labels.csv', index_col=0)

print(labels.head())

for patient in patients[:1]:
    label = labels.get_value(patient, 'cancer')
    path = os.path.join(sample_images_dir, patient)
    slices = [dicom.read_file(path+'/'+s) for s in os.listdir(path)] # Get all the dicom files for the particular patient
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    print(len(slices), label)
    print(slices[0])

len(patients)

import cv2
import math


IMG_SIZE=50
NUM_SLICES = 20

# Create chunks of the data
def chunks(l, n):
    """ Yields successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]
        
def mean(l):
    return sum(l)/len(l)

def process_data(patient, labels, img_px=50, num_slices=20, visualize=False):
    label = labels.get_value(patient, 'cancer')
    path = os.path.join(sample_images_dir, patient)
    slices = [dicom.read_file(path+'/'+s) for s in os.listdir(path)] # Get all the dicom files for the particular patient
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    
    new_slices = []
    slices = [cv2.resize(np.array(each_slice.pixel_array), (IMG_SIZE, IMG_SIZE)) for each_slice in slices]
    chunk_size = math.ceil(len(slices)/NUM_SLICES)
    
    for slice_chunk in chunks(slices, chunk_size):
        slice_chunk =  list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)
    
    if len(new_slices) == NUM_SLICES-1:
        new_slices.append(new_slices[-1])
        
    if len(new_slices) == NUM_SLICES-2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])
        
    if len(new_slices) == NUM_SLICES+2:
        new_val = list(map(mean, zip(*[new_slices[NUM_SLICES-1], new_slices[NUM_SLICES]])))
        del new_slices[NUM_SLICES]
        new_slices[NUM_SLICES-1]= new_val
        
    if len(new_slices) == NUM_SLICES+1:
        new_val = list(map(mean, zip(*[new_slices[NUM_SLICES-1], new_slices[NUM_SLICES]])))
        del new_slices[NUM_SLICES]
        new_slices[NUM_SLICES-1]= new_val
        
    if visualize:
        fig = plt.figure()
        for num, each_slice in enumerate(new_slices):
            y = fig.add_subplot(4,5, num+1)
            y.imshow(each_slice)
        plt.show()
        
    if label==1:
        label = np.array([0,1])
    elif label==0:
        label = np.array([1,0])
        
    return np.array(new_slices), label
    

new_data = []
for num, patient in tqdm(enumerate(patients)):
    try:
        img_data, label = process_data(patient, labels)
        new_data.append([img_data, label])
    except KeyError as e:
        print('This is unlabeled data')
        
np.save('train_data.npy', new_data)

data = np.load('train_data.npy')
print(len(data))

# Developing the dataloader
class LungCancerDataset(Dataset):
    
    def __init__(self, train_data, n_classes=2, is_transform=False):
        self.train_data = train_data
        self.n_classes = n_classes
        self.is_transform = is_transform
        
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, i):
        img = self.train_data[i][0]
        lbl = self.train_data[i][1]
        sample  = {'image': img, 'target': lbl}
        if self.is_transform:
            sample = self.transform(sample)
        return sample
    
    def transform(self, sample):
        img = sample['image']
        lbl = sample['target']
        
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        
        sample  = {'image': img, 'target': lbl}
        return sample

d = LungCancerDataset(data, is_transform=True)
# Instead of using a simple for loop we will be using the torch.utils.DataLoader
# It provides the following features:
# 1.Batching the data
# 2.Shuffling the data
# 3.Load data in parallel using multiprocessing workers

dataloader = DataLoader(d, batch_size=10, shuffle=True, num_workers=4)

for i_batch, sample_batched in enumerate(dataloader):
    imgs, labels = sample_batched['image'], sample_batched['target']
    if i_batch == 0:
        d = imgs[0].numpy()
        print(d.shape)

# Defining the model
IMG_SIZE = 50
NUM_SLICES = 20

n_classes = 2

class Convolutional3DNetwork(nn.Module):
    
    def __init__(self, n_classes=2):
        super(Convolutional3DNetwork, self).__init__()
        self.n_classes =  n_classes
        
        self.conv1block = nn.Sequential(
                            nn.Conv3d(20, 32, 3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(32, 32, 3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.MaxPool3d(2, ceil_mode=True),
                            )
        self.conv2block = nn.Sequential(
                            nn.Conv3d(32, 64, 3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(64, 64, 3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.MaxPool3d(2, stride=2, ceil_mode=True),
                            )
        
        self.conv3block = nn.Sequential(
                            nn.Conv3d(64, 128, 3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(128, 128, 3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.MaxPool3d(2, stride=2, ceil_mode=True),
                            )
        self.classifer = nn.Linear(6272, 2)
        
    # Define the forward pass for the network
    def forward(self, x):
        x = x.view([-1, NUM_SLICES, 1, IMG_SIZE, IMG_SIZE])
        print(x.size())
        conv1 = self.conv1block(x)
        print(conv1.size())
        conv2 = self.conv2block(conv1)
        print(conv2.size())
        conv3 = self.conv3block(conv2)
        print(conv3.size())
        conv3 = conv3.view(-1, 6272)
        output = self.classifer(conv3)
        return output

# Define the training method
def train_model(model, optimizer, criterion, scheduler, num_epochs = 25):
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
                
        running_loss = 0.0
        running_corrects = 0
        
        for data in dataloader:
            inputs, labels = data['image'], data['target']
            inputs, labels = Variable(inputs), Variable(labels)
            labels=labels[:,0]
            
            # Zero the parameter gradients
            optimizer.zero_grad()
                
            #Forward Pass
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.data[0]
        
        epoch_loss = running_loss / len(data)
        
        print('Loss: {:.4f}'.format(epoch_loss))

    return model

model = Convolutional3DNetwork()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=0.3, gamma=0.1)
model  = train_model(model, optimizer, criterion, lr_scheduler)

model(Variable(torch.from_numpy(data[10][0]).float()))



