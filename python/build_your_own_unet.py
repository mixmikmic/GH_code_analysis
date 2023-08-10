import nibabel as nib  # By convention, nibabel is abbreviated to nib, much as numpy is abbreviated to np

input_file = 'BRATS_10_Updated/Brats17_2013_18_1/Brats17_2013_18_1_flair.nii.gz'
loaded_nifti = nib.load(input_file)

print(loaded_nifti)

scan_array = loaded_nifti.get_data()
print(type(scan_array))
print(scan_array.shape)

# Just for Jupyter Notebooks
get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt

im = plt.imshow(scan_array, interpolation='none', origin='lower', cmap='gray')
plt.show()

print(scan_array.shape)
total_axial_slices = scan_array.shape[2]

for axial_slice_num in range(total_axial_slices):
    
    print(axial_slice_num)
    im = plt.imshow(scan_array[:, :, axial_slice_num], interpolation='none', origin='lower', cmap='gray')
    plt.show()

import numpy as np

total_axial_slices = scan_array.shape[2]

for axial_slice_num in range(total_axial_slices):
    
    # Using numpy, we rotate 90 degrees clockwise 3 times, making the nose point "up"
    im = plt.imshow(np.rot90(scan_array[:, :, axial_slice_num], 1), interpolation='none', origin='lower', cmap='gray')
    plt.show()

input_label = 'BRATS_10_Updated/Brats17_2013_18_1/Brats17_2013_18_1_seg.nii.gz'

loaded_label = nib.load(input_label)
label_array = loaded_label.get_data()

for axial_slice_num in range(total_axial_slices):

    plt.subplot(1, 2, 1)
    im = plt.imshow(np.rot90(scan_array[:, :, axial_slice_num], 3), interpolation='none', origin='lower', cmap='gray')

    plt.subplot(1, 2, 2)
    im = plt.imshow(np.rot90(label_array[:, :, axial_slice_num], 3), interpolation='none', origin='lower', cmap='Dark2')
    
    plt.show()

import os

data_directory = 'BRATS_10_Updated/Brats17_2013_18_1'

def load_all_sequences_from_patients(input_directory, patient_id):

    output_arrays = []
    sequences = ['t1ce', 't1', 'flair', 't2', 'seg']
    
    for sequence in sequences:
        target_file = os.path.join(input_directory, patient_id + '_' + sequence + '.nii.gz')
        print(target_file)
        data_array = nib.load(target_file).get_data()
        output_arrays.append(data_array)
        
        # Alternate Ways to write append
        # output_arrays += [data_array]
        # output_arrays = output_arrays + [data_array]
        
    return output_arrays

# Alternate
# outputs = load_all_sequences_from_patients(data_directory, 'Brats...')
# T1POST = outputs[0]
# T1PRE = outputs[1]
# FLAIR = outputs[2]
# etc.

T1POST, T1PRE, FLAIR, T2, GROUND_TRUTH = load_all_sequences_from_patients(data_directory, 'Brats17_2013_18_1')

input_label = ''

print(total_axial_slices)

for axial_slice_num in range(60, total_axial_slices):

    #scan_number = 0
    
    plt.figure(figsize = (20, 5))
    
    for scan_number, scan in enumerate([T1POST, T1PRE, FLAIR, T2, GROUND_TRUTH]):
        
        # Alternate
        # scan_number = scan_number + 1
    
        plt.subplot(1, 5, scan_number + 1)
        im = plt.imshow(np.rot90(scan[:, :, axial_slice_num], 1), interpolation='none', origin='lower', cmap='gray')
        plt.axis('off')
        
    plt.show()

import numpy as np

def load_all_sequences_from_patients(input_directory, patient_id):

    output_arrays = []
    sequences = ['t1ce', 't1', 'flair', 't2', 'seg']
    
    for sequence in sequences:
        target_file = os.path.join(input_directory, patient_id, patient_id + '_' + sequence + '.nii.gz')
        data_array = nib.load(target_file).get_data()
        output_arrays.append(data_array)
    
    stacked_output_array = np.stack(output_arrays[0:4], axis=-1)
    ground_truth_array = np.expand_dims(output_arrays[-1], axis=-1)    
    return stacked_output_array, ground_truth_array


stacked_sequences, ground_truth = load_all_sequences_from_patients('BRATS_10_Updated', 'Brats17_2013_18_1')

def split_3d_array_into_2d_slices(input_array, skip=20):
    
    total_axial_slices = input_array.shape[2]
    output_slices = list()
#     print('empty list', output_slices)
    
    for current_axial_num in range(skip, total_axial_slices-skip):
        
        extracted_slice = input_array[:, :, current_axial_num, :]
        output_slices.append(extracted_slice)
        
    return output_slices
        
patient_slices_2d = split_3d_array_into_2d_slices(stacked_sequences)
ground_truth_slices_2d = split_3d_array_into_2d_slices(ground_truth)

# for patient_slice in patient_slices_2d:
#     print(patient_slice.shape, 'input_data')
# for patient_slice in ground_truth_slices_2d:
#     print(patient_slice.shape, 'ground_truth')

def assign_ground_truth_from_slices(input_ground_truth_slices):
    
    ground_truth_labels = list()
    
    for ground_truth_slice in input_ground_truth_slices:
        
        if np.sum(ground_truth_slice) > 0:
            ground_truth_labels.append(1)
        else:
            ground_truth_labels.append(0)
        
    return ground_truth_labels

patient_ground_truth_labels = assign_ground_truth_from_slices(ground_truth_slices_2d)
print(patient_ground_truth_labels)

def normalize_images(input_3d_data):
        
    number_of_channels = input_3d_data.shape[-1]
    normalized = []
    
    for channel in range(number_of_channels):
        
        extracted_channel = input_3d_data[:, :, :, channel].copy()

        masked_channel = np.copy(extracted_channel)
        masked_channel[masked_channel == 0] = -100
        masked_channel = np.ma.masked_where(masked_channel == -100, masked_channel)

        masked_channel = masked_channel - np.mean(masked_channel)
        masked_channel = masked_channel / np.std(masked_channel)
        
        normalized.append(masked_channel.astype('float16'))
        
    return np.stack(normalized, axis=3)

normalized_stacked_sequences = normalize_images(stacked_sequences)

# for patient_slice in patient_slices_2d:
#     for channel in range(patient_slice.shape[-1]):
#         print(channel, 'channel_number')
#         print(np.mean(patient_slice[:, :, channel]), np.std(patient_slice[:, :, channel]))

import h5py  # python package 
import numpy as np
# samples, rows, columns, channels
# n, 240, 240, 4

# samples, 1
# 0, 0, 0, 1, 1, ..., 0

def save_hdf5_file(train_data, ground_truth, output_filename):
    
    with h5py.File(output_filename, 'w') as file_handle:

        file_handle.create_dataset('train', data=train_data, dtype=train_data.dtype)
        file_handle.create_dataset('labels', data=ground_truth, dtype=ground_truth.dtype)

# Generate fake data
X = np.zeros(shape=[10, 240, 240, 4])
y = np.zeros(shape=(10, 1))
save_hdf5_file(X, y, 'fake_data.h5')

# This code was updated outside of the tutorial, so that we can process subsets of the data more easily
import os

def generate_dataset(patient_data_list):

    X, y = [], [] 

    for patient_directory in patient_data_list:

        # Load nifti files for MR sequences and tumor segmentation
        patient_sequences, ground_truth = load_all_sequences_from_patients(data_directory, patient_directory)

        # Normalize input volumes
        patient_norm = normalize_images(patient_sequences)  # this is a new addition too

        # 4D volumes to slices
        sequence_slices = split_3d_array_into_2d_slices(patient_norm)
        ground_truth_slices = split_3d_array_into_2d_slices(ground_truth)

        # Get ground truth vector
        ground_truth_vector = assign_ground_truth_from_slices(ground_truth_slices)

        # Append this patient to our lists
        X.append(sequence_slices)
        y.append(ground_truth_vector)

    X = np.asarray(X)
    y = np.hstack(y)

    # Grab the dimensions of the 5D array
    patients, slices, rows, cols, ch = X.shape

    # Combine the first two dimension (patients, slices) into one
    X = X.reshape(patients*slices, rows, cols, ch)
    return X, y

data_directory = 'BRATS_10_Updated'
all_patients = os.listdir(data_directory)

X_train, y_train = generate_dataset(all_patients[:8])
X_val, y_val = generate_dataset(all_patients[8:])

save_hdf5_file(X_train, y_train, 'training.h5')
save_hdf5_file(X_val, y_val, 'validation.h5')

# Let's make a U-net
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras

from keras.layers import Input, Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model

max_channels = 128  # change this if you run out of GPU memory!

# First block
input_layer = Input(shape=(240, 240, 4))
conv1 = Conv2D(max_channels // 16, (3, 3), padding='same', activation='relu')(input_layer)
conv2 = Conv2D(max_channels // 16, (3, 3), padding='same', activation='relu')(conv1)
conv2 = BatchNormalization()(conv2)
pool1 = MaxPool2D((2, 2))(conv2)

# Second block
conv3 = Conv2D(max_channels // 8, (3, 3), padding='same', activation='relu')(pool1)
conv4 = Conv2D(max_channels // 8, (3, 3), padding='same', activation='relu')(conv3)
conv4 = BatchNormalization()(conv4)
pool2 = MaxPool2D((2, 2))(conv4)

# Third block
conv5 = Conv2D(max_channels // 4, (3, 3), padding='same', activation='relu')(pool2)
conv6 = Conv2D(max_channels // 4, (3, 3), padding='same', activation='relu')(conv5)
conv6 = BatchNormalization()(conv6)
pool3 = MaxPool2D((2, 2))(conv6)

# Fourth block
conv7 = Conv2D(max_channels // 2, (3, 3), padding='same', activation='relu')(pool3)
conv8 = Conv2D(max_channels // 2, (3, 3), padding='same', activation='relu')(conv7)
conv8 = BatchNormalization()(conv8)
pool4 = MaxPool2D((2, 2))(conv8)

# Fifth block
conv9 = Conv2D(max_channels, (3, 3), padding='same', activation='relu')(pool4)
conv10 = Conv2D(max_channels, (3, 3), padding='same', activation='relu')(conv9)
conv10 = BatchNormalization()(conv10)
pool5 = GlobalAveragePooling2D()(conv10)

# Fully-connected
dense1 = Dense(128, activation='relu')(pool5)
drop1 = Dropout(0.5)(dense1)
output = Dense(1, activation='sigmoid')(drop1)

# Create model object
model = Model(inputs=input_layer, outputs=output)
print(model.summary())

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.io_utils import HDF5Matrix
seed = 0

# At train time, we can diversify our dataset by applying random transformations
data_gen_args = dict( 
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.2,
    channel_shift_range=0.005,
    horizontal_flip=True,
    vertical_flip=True
)

# Generator for the training data
train_datagen = ImageDataGenerator(**data_gen_args)
X_train = HDF5Matrix('training.h5', 'train')
y_train = HDF5Matrix('training.h5', 'labels')
train_generator = train_datagen.flow(X_train, y_train, seed=0, batch_size=16)

# Generator for the validation data
val_datagen = ImageDataGenerator()  # no augmentation! why? we want a consist measure, not one that changes during training
X_val = HDF5Matrix('validation.h5', 'train')
y_val = HDF5Matrix('validation.h5', 'labels')
val_generator = val_datagen.flow(X_val, y_val, seed=0, batch_size=1)  # gives us accuracy over whole validation set

from keras.callbacks import ModelCheckpoint, EarlyStopping

mc_cb = ModelCheckpoint('best_model.h5')
el_cb = EarlyStopping(patience=5)

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_generator, epochs=50, shuffle='batch',
                    validation_data=val_generator, callbacks=[mc_cb, el_cb])
model.save('final_model.h5')

