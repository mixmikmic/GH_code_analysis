reset -fs

import os, sys
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
from scipy.misc import imread
from keras.preprocessing.image import array_to_img, img_to_array, load_img
get_ipython().magic('matplotlib inline')

path_ok = 'data/downloads/ok'

path_nok = 'data/downloads/nok'

number_nok_files = sum(os.path.isfile(os.path.join(path_nok, f)) for f in os.listdir(path_nok))

training_sample_nok_images = int(number_nok_files * 0.7)

testing_sample_nok_images = int(number_nok_files * 0.3)

number_ok_files = sum(os.path.isfile(os.path.join(path_ok, f)) for f in os.listdir(path_ok))

training_sample_ok_images = int(number_ok_files * 0.7)

testing_sample_ok_images = int(number_ok_files * 0.3)

# http://stackoverflow.com/questions/35975799/loop-through-sub-directories-to-sample-files
def create_train_test_samples(source_files, train_destination_files, test_destination_files):

    number_files = sum(os.path.isfile(os.path.join(source_files, f)) for f in os.listdir(source_files))
    samples_train = int(number_files * 0.7)
    samples_test = int(number_files * 0.3)
    #take random samples.
    train_filenames = random.sample(os.listdir(source_files), samples_train)
    test_filenames = random.sample(os.listdir(source_files), samples_test)

    #copy the files to the new destination
    for i in train_filenames:
        shutil.copy2(source_files + '/'  + i, train_destination_files)
    
    for i in test_filenames:
        shutil.copy2(source_files + '/'  + i, test_destination_files)
        
    return 'Done.'

create_train_test_samples('data/downloads/ok', 'data/train/ok', 'data/test/ok')

create_train_test_samples('data/downloads/nok', 'data/train/nok', 'data/test/nok')

