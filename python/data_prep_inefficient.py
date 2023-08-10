import os
import random
import itertools
import copy
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from tqdm import trange
from PIL import Image
from collections import Counter

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

data_dir = './data/processed/'
back_dir = data_dir + 'background/'
eval_dir = data_dir + 'evaluation/'

# ensuring reproducibility
np.random.seed(0)
random.seed(0)

# params
percentage_split = False
augment = False

# get list of all alphabets
background_alphabets = [os.path.join(back_dir, x) for x in next(os.walk(back_dir))[1]]
background_alphabets.sort()

# list of all drawers (1 to 20)
background_drawers = list(np.arange(1, 21))

print("There are {} alphabets.".format(len(background_alphabets)))

# 80-20 train-valid split
if percentage_split:
    valid_size = 0.2
    num_alphabets = len(background_alphabets)

    indices = list(range(num_alphabets))
    split = int(np.floor(valid_size * num_alphabets))
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_alphabets = [background_alphabets[idx] for idx in train_idx]
    valid_alphabets = [background_alphabets[idx] for idx in valid_idx]

    train_alphabets.sort()
    valid_alphabets.sort()

    # from 20 drawers, randomly select 12
    train_drawers = list(np.random.choice(background_drawers, size=12, replace=False))
    remaining_drawers = [x for x in background_drawers if x not in train_drawers]

    print("There are {} train alphabets".format(len(train_alphabets)))
    print("There are {} valid alphabets".format(len(valid_alphabets)))
# 30 train, 10 valid
else:
    # from 40 alphabets, randomly select 30
    train_alphabets = list(np.random.choice(background_alphabets, size=30, replace=False))
    valid_alphabets = [x for x in background_alphabets if x not in train_alphabets]
    
    train_alphabets.sort()
    valid_alphabets.sort()

    # from 20 drawers, randomly select 12
    train_drawers = list(np.random.choice(np.arange(20), size=12, replace=False))
    remaining_drawers = [x for x in background_drawers if x not in train_drawers]

def check_same_alph_character(filenames):
    # same alphabet
    if filenames[0].split('/')[4] == filenames[1].split('/')[4]:
        # same character
        if filenames[0].split('/')[5] == filenames[1].split('/')[5]:
            return True
        return False
    return False

num_iters = int(100e3 / 2)

same_drawer = 0
selected_alphabets = []
redoing = 0
img_pairs = []
label_pairs = []
for i in trange(num_iters):
    # sample a like pair
    if i % 2 == 0:
        # uniformly select 1 alphabet
        alph = np.random.choice(train_alphabets)
        selected_alphabets.append(alph)
                
        # uniformly sample 1 character
        chars = [os.path.join(alph, x) for x in next(os.walk(alph))[1]]
        char = np.random.choice(chars)
                
        # uniformly sample 2 drawers
        ds = np.random.choice(train_drawers, size=2, replace=True)
                
        # get list of filenames to read in char dir
        filenames = [
            os.path.join(char, x) for x in next(os.walk(char))[-1] if int(
                x.split("_")[1][0:2].lstrip("0")
            ) in ds
        ]
        
        # in case I get the same drawer
        if len(filenames) == 1:
            same_drawer += 1
            filenames = filenames * 2
        
        # load pair as numpy array and store
        pair = []
        for name in filenames:
            img_arr = img2array(name, gray=True, expand=True)
            img_arr = np.transpose(img_arr, (0, 3, 1, 2))
            pair.append(img_arr)        
        img_pairs.append(np.concatenate(pair, axis=0))
        
        # store ground truth lbl
        gd_truth = np.array([1], dtype=np.int64)
        label_pairs.append(gd_truth)
        
    # sample a dissimilar pair
    else:
        redo = True
        while redo:
            # uniformly select 2 alphabets
            alph = np.random.choice(train_alphabets, size=2, replace=True)
            selected_alphabets.extend(alph)

            # uniformly sample 2 drawers
            ds = np.random.choice(train_drawers, size=2, replace=True)

            filenames = []
            for i, a in enumerate(alph):
                # uniformly sample 1 character
                chars = [os.path.join(a, x) for x in next(os.walk(a))[1]]
                char = np.random.choice(chars)

                # get list of filenames to read in char dir
                name = [
                    os.path.join(char, x) for x in next(os.walk(char))[-1] if int(
                        x.split("_")[1][0:2].lstrip("0")
                    ) == ds[i]
                ]
                filenames.append(*name)
            
            # reject (same alph, same char, same drawer) and (same alph, same char, diff drawer)
            # i.e. just check if same alph and same char
            redo = check_same_alph_character(filenames)
            if redo:
                redoing += 1

        # load pair as numpy array and store
        pair = []
        for name in filenames:
            img_arr = img2array(name, gray=True, expand=True)
            img_arr = np.transpose(img_arr, (0, 3, 1, 2))
            pair.append(img_arr)        
        img_pairs.append(np.concatenate(pair, axis=0))

         # store ground truth lbl
        gd_truth = np.array([0], dtype=np.int64)
        label_pairs.append(gd_truth)

print("Redid {} false pairs...".format(redoing))
print("Selected the same drawer {} times...".format(same_drawer))

# confirm that each alphabet gets approximately equal representation
Counter(selected_alphabets)

# shuffle img and labels to prevent monotone (same, different) sequence
indices = list(range(len(img_pairs)))
np.random.shuffle(indices)

img_pairs = [img_pairs[idx] for idx in indices]
label_pairs = [label_pairs[idx] for idx in indices]

pickle_dump(img_pairs, data_dir + 'X_train.p')
pickle_dump(label_pairs, data_dir + 'y_train.p')

arr2pil = transforms.ToPILImage()

def pil2array(im):
    x = np.asarray(im, dtype=np.float32)
    x = np.expand_dims(x, axis=-1)
    x = np.expand_dims(x, axis=0)
    x = x / 255
    return x

augmented_img_pairs = []
augmented_label_pairs = []
for idx in trange(len(img_pairs)):
    # get gd truth label
    label = label_pairs[idx]
    
    # grab img pair
    pair = img_pairs[idx]
    pair = np.transpose(pair, (0, 2, 3, 1))
    im1, im2 = np.array(pair)
    
    # convert back to [0, 255] range
    im1 *= 255
    im2 *= 255
    
    # transform to PIL image
    im1, im2 = arr2pil(im1), arr2pil(im2)
    
    # compose 8 transforms
    for i in range(8):
        # randomly select transform with proba 0.5
        rot = random.choice([0, [-10, 10]])
        shear = random.choice([None, [-0.3, 0.3]])
        scale = random.choice([None, [0.8, 1.2]])
        trans = random.choice([None, [2/150, 2/150]]) # absolute value
        
        # apply affine transformation
        aff = transforms.RandomAffine(rot, trans, scale, shear)
        aug_im1, aug_im2 = aff(im1), aff(im2)
        
        # convert to numpy array
        aug_im1 = pil2array(aug_im1)
        aug_im2 = pil2array(aug_im2)
        
        # transpose to C,H,W
        aug_im1 = np.transpose(aug_im1, (0, 3, 1, 2))
        aug_im2 = np.transpose(aug_im2, (0, 3, 1, 2))
        
        # add to list
        aug_pairs = np.concatenate([aug_im1, aug_im2], axis=0)
        augmented_img_pairs.append(aug_pairs)
        augmented_label_pairs.append(label)

# shuffle img and labels
indices = list(range(len(augmented_img_pairs)))
np.random.shuffle(indices)

augmented_img_pairs = [augmented_img_pairs[idx] for idx in indices]
augmented_label_pairs = [augmented_label_pairs[idx] for idx in indices]

train_img_pairs = img_pairs + augmented_img_pairs
train_label_pairs = label_pairs + augmented_label_pairs

print("Effective Train Size: {}".format(2 * len(train_img_pairs)))

pickle_dump(train_img_pairs, data_dir + 'X_train_aug.p')
pickle_dump(train_label_pairs, data_dir + 'y_train_aug.p')

# from remaining 8 drawers, select 4
valid_drawers = list(np.random.choice(remaining_drawers, size=4, replace=False))
remaining_drawers = [x for x in remaining_drawers if x not in valid_drawers]

num_iters = len(valid_alphabets)

# number of characters to sample in each alphabet
pop = 14

valid_img_pairs = []
valid_label_pairs = []
for iter, alph in enumerate(valid_alphabets):
    print("{}/{}".format(iter+1, num_iters))
    for j in range(2):
        # grab drawers
        ds = [valid_drawers[2*j], valid_drawers[2*j + 1]]
        
        # sample pop characters uniformly
        chars = [os.path.join(alph, x) for x in next(os.walk(alph))[1]]
        chars = np.random.choice(chars, size=pop, replace=False)
        
        # grab filenames for both drawers
        filenames = []
        for d in ds:
            for char in chars:
                names = [
                    os.path.join(char, x) for x in next(os.walk(char))[-1] if int(
                        x.split("_")[1][0:2].lstrip("0")
                    ) == d
                ]
                filenames.append(*names)
        d1 = filenames[:pop]
        d2 = filenames[pop:]
        
        
        for i, left in enumerate(d1):
            way_pairs = []
            way_labels = []
            for right in d2:
                img_names = [left, right]
                pair = []
                for name in img_names:
                    img_arr = img2array(name, gray=True, expand=True)
                    img_arr = np.transpose(img_arr, (0, 3, 1, 2))
                    pair.append(img_arr) 
                # create img and store
                pair = np.concatenate(pair, axis=0)
                way_pairs.append(pair)
            
            # create pop-way task
            way_pairs = [np.expand_dims(x, axis=0) for x in way_pairs]
            way_pairs = np.concatenate(way_pairs, axis=0)
            valid_img_pairs.append(way_pairs)
            label = np.array([i], dtype=np.int64)
            valid_label_pairs.append(label)

pickle_dump(valid_img_pairs, data_dir + 'X_valid.p')
pickle_dump(valid_label_pairs, data_dir + 'y_valid.p')

# get list of alphabets
test_alphabets = [os.path.join(eval_dir, x) for x in next(os.walk(eval_dir))[1]]
test_alphabets.sort()

# there are 20 drawers
test_drawers = remaining_drawers

num_iters = len(test_alphabets)

# number of characters to sample in each alphabet
pop = 20

test_img_pairs = []
test_label_pairs = []
for iter, alph in enumerate(test_alphabets):
    print("{}/{}".format(iter+1, num_iters))
    for j in range(2):
        # sample a pair of drawers
        ds = np.random.choice(test_drawers, size=2, replace=False)
        
        # sample 20 characters uniformly
        chars = [os.path.join(alph, x) for x in next(os.walk(alph))[1]]
        chars = np.random.choice(chars, size=pop, replace=False)
        
        # grab filenames for both drawers
        filenames = []
        for d in ds:
            for char in chars:
                names = [
                    os.path.join(char, x) for x in next(os.walk(char))[-1] if int(
                        x.split("_")[1][0:2].lstrip("0")
                    ) == d
                ]
                filenames.append(*names)
        d1 = filenames[:pop]
        d2 = filenames[pop:]

        for i, left in enumerate(d1):
            way_pairs = []
            way_labels = []
            for right in d2:
                img_names = [left, right]
                pair = []
                for name in img_names:
                    img_arr = img2array(name, gray=True, expand=True)
                    img_arr = np.transpose(img_arr, (0, 3, 1, 2))
                    pair.append(img_arr) 
               # create img and store
                pair = np.concatenate(pair, axis=0)
                way_pairs.append(pair)
                
            # create pop-way task
            way_pairs = [np.expand_dims(x, axis=0) for x in way_pairs]
            way_pairs = np.concatenate(way_pairs, axis=0)
            test_img_pairs.append(way_pairs)
            label = np.array([i], dtype=np.int64)
            test_label_pairs.append(label)       

pickle_dump(test_img_pairs, data_dir + 'X_test.p')
pickle_dump(test_label_pairs, data_dir + 'y_test.p')

