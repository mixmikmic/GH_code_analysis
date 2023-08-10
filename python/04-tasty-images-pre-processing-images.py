reset -fs

import numpy as np
import pandas as pd
import os
import glob
import pickle
import gzip
import dl_functions
from IPython.display import display
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from skimage import io, color, exposure, transform
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

IMG_SIZE = 50

ok_images='data/downloads/ok'

nok_images='data/downloads/nok'

X = np.vstack((dl_functions.normalize_images_array(ok_images, IMG_SIZE), dl_functions.normalize_images_array(nok_images, IMG_SIZE)))

y = np.vstack((np.array([1]*(len(X)/2)).reshape((len(X)/2), 1), np.array([0]*(len(X)/2)).reshape((len(X)/2), 1)))

with gzip.open('pre_processed_images/image_data_' + str(len(X)) + '_' + str(IMG_SIZE) + '.txt.gz', 'wb') as fp:
    pickle_file = pickle.dump((X, y), fp)

with gzip.open('pre_processed_images/image_data_' + str(len(X)) + '_' + str(IMG_SIZE) + '.pklz', 'wb') as fp:
    pickle_file = pickle.dump((X, y), fp)

with open('pre_processed_images/image_data_' + str(len(X)) + '_' + str(IMG_SIZE) + '.txt', 'wb') as fp:
    pickle_file = pickle.dump((X, y), fp)

with open('pre_processed_images/image_data_' + str(len(X)) + '_' + str(IMG_SIZE) + '.pkl', 'wb') as fp:
    pickle_file = pickle.dump((X, y), fp)

get_ipython().system("gsutil cp -r 'pre_processed_images' 'gs://wellio-kadaif-tasty-images-project-pre-processed-images'")



