import numpy as np
from skimage import io, transform
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

TARGET_SIZE = (299, 299, 3)

# * Train *
# Filenames
f_root = '../datasets/Food-5K/training/'
f_train = [os.path.join(f_root, f) for f in os.listdir(f_root) if f.endswith('.jpg')]

# Read the images
X_train, y_train = [], []
for f_im in tqdm(f_train):
    im = io.imread(f_im)
    if len(im.shape) == 3:
        im = transform.resize(im, output_shape=TARGET_SIZE)
        im = (im * 255.0).astype(np.uint8)
        X_train.append(im)
        y_train.append(int(os.path.basename(f_im)[0]))
X_train = np.array(X_train)
y_train = np.array(y_train)
print 'X_train.shape:', X_train.shape
print 'y_train.shape:', y_train.shape
np.save('./data/X_train.npy', X_train)
np.save('./data/y_train.npy', y_train)

# * Validation *
# Filenames
f_root = '../datasets/Food-5K/validation/'
f_val = [os.path.join(f_root, f) for f in os.listdir(f_root) if f.endswith('.jpg')]

# Read the images
X_val, y_val = [], []
for f_im in tqdm(f_val):
    im = io.imread(f_im)
    if len(im.shape) == 3:
        im = transform.resize(im, output_shape=TARGET_SIZE)
        im = (im * 255.0).astype(np.uint8)
        X_val.append(im)
        y_val.append(int(os.path.basename(f_im)[0]))
X_val = np.array(X_val)
y_val = np.array(y_val)
print 'X_val.shape:', X_val.shape
print 'y_val.shape:', y_val.shape
np.save('./data/X_val.npy', X_val)
np.save('./data/y_val.npy', y_val)

# * Validation *
# Filenames
f_root = '../datasets/Food-5K/evaluation/'
f_test = [os.path.join(f_root, f) for f in os.listdir(f_root) if f.endswith('.jpg')]

# Read the images
X_test, y_test = [], []
for f_im in tqdm(f_test):
    im = io.imread(f_im)
    if len(im.shape) == 3:
        im = transform.resize(im, output_shape=TARGET_SIZE)
        im = (im * 255.0).astype(np.uint8)
        X_test.append(im)
        y_test.append(int(os.path.basename(f_im)[0]))
X_test = np.array(X_test)
y_test = np.array(y_test)
print 'X_test.shape:', X_test.shape
print 'y_test.shape:', y_test.shape
np.save('./data/X_test.npy', X_test)
np.save('./data/y_test.npy', y_test)





