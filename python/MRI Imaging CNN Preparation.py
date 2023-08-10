from parsing import *
import numpy as np
import os

''' Let's start by loading DICOM-contour map into memory.
Will build a map from patiend_id (DICOM subdir) to original_id (contour subdir).
'''
def get_DICOM_map():
    link_map = {}
    with open('final_data/link.csv', 'r') as link:
        header = link.readline()
        for line in link:
            line = line.strip().split(',')
            link_map[line[0]] = line[1]
    print("Finished patient_id - original_id (DICOM - contours) link.")
    return link_map


''' Now lets assemble our data (DICOM images)
To do this I will use a map from patiend_id (DICOM subdir) to a map of DICOM name to DICOM data
i.e. {p_id -> {DICOM_id -> DICOM_data}}
'''
def get_DICOM_data():
    # Load DICOM data from disk
    DICOM_map = {}
    DICOM_dir = 'final_data/dicoms/'
    for subdir in os.listdir(DICOM_dir):
        path = os.path.join(DICOM_dir, subdir)
        if not os.path.isdir(path): continue
        DICOM_map[subdir] = {}
        for filename in os.listdir(path):
            # here we make use of 'parse_dicom_file' method from starter code in parsing.py
            dcm_dict = parse_dicom_file(os.path.join(path, filename))
            dcm_id = int(filename.split('.')[0])
            DICOM_map[subdir][dcm_id] = dcm_dict['pixel_data']
        print("Have %d dicoms for patient %s." % (len(DICOM_map[subdir]), subdir))
    print("Finished loading DICOM data.")
    return DICOM_map
    
    
''' Finally lets follow suit on contour data, much as above
{o_id -> {contour_id -> contour_data*}}

* note that we will transform contour data to its boolean mask 
in place and store that.
'''
def get_contour_data():
    # Load contour data from disk
    contour_map = {}
    contour_dir = 'final_data/contourfiles/'
    for subdir in os.listdir(contour_dir):
        path = os.path.join(contour_dir, subdir)
        if not os.path.isdir(path): continue
        contour_map[subdir] = {}
        # ignore o-contours, hard-code in contour type of interest
        path = os.path.join(path, 'i-contours')
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            if not os.path.isfile(filepath) or not filename.endswith('.txt'): continue
            # make use of 'parse_contour_file' and 'poly_to_mask' methods from starter code
            coords_list = parse_contour_file(filepath)
            # transformation to boolean mask
            mask = poly_to_mask(coords_list, 256, 256)
            contour_id = int(filename.split('-')[2])
            contour_map[subdir][contour_id] = mask
        print("Have %d contours for original id %s." % (len(contour_map[subdir]), subdir))
    print("Finished loading contour data.")
    return contour_map   
    
links = get_DICOM_map()
dicoms = get_DICOM_data()
contours = get_contour_data()

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Let's visualize some of these DICOM images
def preview():
    for i in range(1,6):
        if i == 3: continue
        subdir = 'SCD0000%d01' % i
        plt.figure()
        plt.imshow(dicoms[subdir][1])

preview()

import filecmp, random

''' Check parsing contour correctly by comparing derived file 
from coordinates list against original file using filecmp 
'''
def compare_contour(original_filepath):
    #print(original_filepath)
    coords_list = parse_contour_file(original_filepath)
    with open("test_file", 'w') as testfile:
        for coords in coords_list:
            line = '%.2f %.2f\n' % (coords[0], coords[1])
            testfile.write(line)
    if filecmp.cmp(original_filepath, "test_file"): print("Comparison passed.")
    else: print("Error: comparison failed on %s." % original_filepath)

# testing random contour files
def test(n=5):
    contour_dir = 'final_data/contourfiles'
    contour_type = 'i-contours'
    for i in range(n):
        subdir = random.choice(os.listdir(contour_dir))
        while subdir[0] == '.': subdir = random.choice(os.listdir(contour_dir))
        path = os.path.join(contour_dir, subdir, contour_type)
        filename = random.choice(os.listdir(path))
        while filename[0] == '.': filename = random.choice(os.listdir(path))
        filepath = os.path.join(path, filename)
        compare_contour(filepath)


test(10)

# Now let us build a data set for our CNN of associated DICOM - contour pairs.

def build_dataset():
    X, y = [], []
    for patient_id, original_id in links.items():
        count = 0
        for dcm_id in dicoms[patient_id]:
            if dcm_id in contours[original_id]:
                X.append(dicoms[patient_id][dcm_id])
                y.append(contours[original_id][dcm_id])
                count += 1
        print("Found %d matches out of %d DICOMS and %d contours." % (count, len(dicoms[patient_id]), len(contours[original_id])))
    return np.array(X), np.array(y)
    
train_x, train_y = build_dataset()
print("\nTrain DICOM data shape: ", train_x.shape)
print("Train contour target shape: ", train_y.shape)

import math

# Now we develop a method to load batches of our data so we can later run our model

def run_model(nepochs=1, batch_size=8):       
    iters = 0  
    for epoch in range(nepochs):
        # shuffle indices each epoch so as to have random order every time
        train_indices = np.arange(train_x.shape[0])
        np.random.shuffle(train_indices)

        # make sure we cycle over the dataset once, loading single batch at each step
        for i in range(int(math.ceil(train_x.shape[0] / batch_size))):
            # generate indices for the batch
            start_idx = i*batch_size
            idx = train_indices[start_idx:start_idx+batch_size]
            print("On iteration %d have batch following indices: " % iters, idx)
            # generate batches which can be fed into model
            X, y = train_x[idx], train_y[idx]     
            iters += 1

run_model()



