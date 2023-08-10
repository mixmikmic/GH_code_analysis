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
                               root_dir='../data/')

from torchvision import transforms
import torchvision.transforms.functional as F
import random

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    """Horizontally flip the given PIL Image randomly with a probability of 0.5."""
    
    _horizontal_flip_indexes = np.array([
        [1,2,3,4,5,6,7,8,18,19,20,21,22,37,38,39,40,41,42,32,33,49,50,51,61,62,68,60,59],
        [17,16,15,14,13,12,11,10,27,26,25,24,23,46,45,44,43,48,47,36,35,55,54,53,65,64,66,56,57]
    ]) - 1

    def __call__(self, sample):
        """
        Args:
            sample: a dict containing the `image` (PIL image) and the landmarks.
        Returns:
            dict: Randomly flipped image and landmarks
        """
        image, landmarks = sample['image'], sample['landmarks']
        
        if random.random() < 1.0:
            
            w, h = image.size
            
            # Horizontal flip of all x coordinates:
            landmarks = landmarks.copy()
            landmarks[..., 0] = (landmarks[..., 0] - 0.5)* -1 + .5
            
            flipped_landmarks = landmarks.copy()
            
            flipped_landmarks[..., self._horizontal_flip_indexes[0], :] = landmarks[..., self._horizontal_flip_indexes[1], :]
            flipped_landmarks[..., self._horizontal_flip_indexes[1], :] = landmarks[..., self._horizontal_flip_indexes[0], :]
            return {'image': F.hflip(image), 'landmarks': flipped_landmarks}
        return sample

from cpe775.utils.img_utils import show_landmarks

flip = RandomHorizontalFlip()

len(flip._horizontal_flip_indexes[0])

len(flip._horizontal_flip_indexes[1])

idx = 49

image = dataset[idx]['image']
landmarks = dataset[idx]['landmarks']

w,h = image.size

landmarks /= np.array([w,h])

sample = flip({'image': image, 'landmarks': landmarks})

show_landmarks(sample['image'], sample['landmarks'], normalized=True)

show_landmarks(image, landmarks, normalized=True)

from cpe775.transforms import RandomHorizontalFlip

flip = RandomHorizontalFlip()

image = dataset[idx]['image']
landmarks = dataset[idx]['landmarks']

w,h = image.size

landmarks /= np.array([w,h])

sample = flip({'image': image, 'landmarks': landmarks})

show_landmarks(sample['image'], sample['landmarks'], normalized=True)

from cpe775.transforms import CropFace, RandomCrop
import torchvision.transforms.functional as F

output_size = (224, 224)

crop_face = CropFace()
random_crop = RandomCrop((224, 224))
face_sample = crop_face(dataset[idx])

show_landmarks(face_sample['image'], face_sample['landmarks'], normalized=True)

sample = random_crop(face_sample)
image = sample['image']
landmarks = sample['landmarks']
show_landmarks(image, landmarks, normalized=True)

from cpe775.transforms import CenterCrop

crop_face = CropFace()
center_crop = CenterCrop((224, 224))
face_sample = crop_face(dataset[idx])

sample = center_crop(face_sample)
image = sample['image']
landmarks = sample['landmarks']
show_landmarks(image, landmarks, normalized=True)



