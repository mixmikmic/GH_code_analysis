from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import sys
sys.path.insert(0, '../')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from cpe775.dataset import FaceLandmarksDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Compose
from cpe775.transforms import ToTensor, CropFace, ToGray

# load the dataset
dataset = FaceLandmarksDataset(csv_file='../data/train.csv',
                               root_dir='../data/',
                               transform=CropFace())

crop_face = CropFace()

images = []
landmarks = []
for idx in range(len(dataset)):
    sample = dataset[idx]
    
    images.append(sample['image'])
    landmarks.append(sample['landmarks'])

np_images = np.stack([np.array(img) for img in images], axis=0)
np_landmarks = np.stack([np.array(land) for land in landmarks], axis=0)

np_images.shape

np_landmarks.shape

np.savez('../data/train.npz', images=np_images, landmarks=np_landmarks)

import h5py

f = h5py.File('../data/train.h5', 'w')
f['images'] = np_images
f['landmarks'] = np_landmarks
f.close()

from cpe775.dataset import CroppedFaceLandmarksDataset
from cpe775.transforms import ToPILImage

cropped_dataset = CroppedFaceLandmarksDataset('../data/train.npz',
                                              transform=ToPILImage())
hdf5_cropped_dataset = CroppedFaceLandmarksDataset('../data/train.h5',
                                                   transform=ToPILImage())

get_ipython().run_line_magic('timeit', 'cropped_dataset[np.random.randint(len(cropped_dataset))]')

get_ipython().run_line_magic('timeit', 'dataset[np.random.randint(len(dataset))]')

get_ipython().run_line_magic('timeit', 'hdf5_cropped_dataset[np.random.randint(len(hdf5_cropped_dataset))]')

