import numpy as np
import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
import sys
import glob
import argparse
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pickle as pkl
import tensorflow as tf
from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.models import Model, Sequential
from keras.layers import Dropout, Flatten,Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.preprocessing import image 
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from PIL import Image
import seaborn as sns

print('TensorFlow Version:', tf.__version__)
print('Keras Version:', __version__)

top_level_train = os.listdir('data/train')
top_level_test = os.listdir('data/test')
top_level_valid = os.listdir('data/valid')
print(top_level_train, top_level_test, top_level_train)

def file_parser(high_level, dirs):
    files = []
    for directory in dirs:
        for file in os.listdir(high_level + directory):
            files.append(directory + '/' + file)
    return files
train_files = file_parser('data/train/', top_level_train)
test_files = file_parser('data/test/', top_level_test)
valid_files = file_parser('data/valid/', top_level_valid)
print(len(train_files), len(test_files), len(valid_files))

#Image Hyperparameters
IM_WIDTH = 299
IM_HEIGHT = 299

def img_prep(img_path):
    '''
    Function to fetch the image from the image path and return a numpy array of the image.
    The size of the returned image will be defined by IM_WIDTH and IM_HEIGHT (defined above)
    '''
    img = image.load_img(img_path, target_size=(IM_WIDTH, IM_HEIGHT))
    return image.img_to_array(img)

trn_files = []
trn_labels = []
for file in train_files:
    trn_files.append(img_prep('data/train/' + file))
    dir_file = file.split('/')
    trn_labels.append(dir_file[0])

tst_files = []
tst_labels = []
for file in test_files:
    tst_files.append(img_prep('data/test/' + file))
    dir_file = file.split('/')
    tst_labels.append(dir_file[0])

vld_files = []
vld_labels = []
for file in valid_files:
    vld_files.append(img_prep('data/valid/' + file))
    dir_file = file.split('/')
    vld_labels.append(dir_file[0])

#all_files = [trn_files, trn_labels, tst_files, tst_labels, vld_files, vld_labels]

def convert_to_array(lst):
    return np.asarray(lst)

trn_files = convert_to_array(trn_files)
trn_labels = convert_to_array(trn_labels)
tst_files = convert_to_array(tst_files)
tst_labels = convert_to_array(tst_labels)
vld_files = convert_to_array(vld_files)
vld_labels = convert_to_array(vld_labels)

print(type(trn_files))

print(len(trn_files), len(trn_labels))
with open('pickled/trn_files.pkl', 'wb') as f:
    pkl.dump(trn_files, f)
with open('pickled/trn_labels.pkl', 'wb') as f:
    pkl.dump(trn_labels, f)

print(len(tst_files), len(tst_labels))
with open('pickled/tst_files.pkl', 'wb') as f:
    pkl.dump(tst_files, f)
with open('pickled/tst_labels.pkl', 'wb') as f:
    pkl.dump(tst_labels, f)

print(len(vld_files), len(vld_labels))
with open('pickled/vld_files.pkl', 'wb') as f:
    pkl.dump(vld_files, f)
with open('pickled/vld_labels.pkl', 'wb') as f:
    pkl.dump(vld_labels, f)

#Load Pickled Datasets
with open('pickled/trn_files.pkl', 'rb') as f:
    trn_files = pkl.load(f)
with open('pickled/trn_labels.pkl', 'rb') as f:
    trn_labels = pkl.load(f)
with open('pickled/tst_files.pkl', 'rb') as f:
    tst_files = pkl.load(f)
with open('pickled/tst_labels.pkl', 'rb') as f:
    tst_labels = pkl.load(f)
with open('pickled/vld_files.pkl', 'rb') as f:
    vld_files = pkl.load(f)
with open('pickled/vld_labels.pkl', 'rb') as f:
    vld_labels = pkl.load(f)

lb = LabelBinarizer()
lb.fit(trn_labels)
trn_label_vecs = lb.transform(trn_labels)
test_lb = LabelBinarizer()
test_lb.fit(tst_labels)
tst_label_vecs = test_lb.transform(tst_labels)
valid_lb = LabelBinarizer()
valid_lb.fit(vld_labels)
vld_label_vecs = valid_lb.transform(vld_labels)

#Validate One-Hot Vectorization is correct
trn_label_vecs[20]

#Shuffle the Data
def randomize_data(files, labels):
    randomize = np.arange(len(files))
    np.random.shuffle(randomize)
    return trn_files[randomize], trn_label_vecs[randomize]
trn_files, trn_label_vecs = randomize_data(trn_files, trn_label_vecs)
tst_files, tst_label_vecs = randomize_data(tst_files, tst_label_vecs)
vld_files, vld_label_vecs = randomize_data(vld_files, vld_label_vecs)

print(len(trn_label_vecs), len(trn_files))
print(type(trn_label_vecs), type(trn_files))

Melanoma = 0
Nevus = 0 
Seborrheic_Keratosis = 0
for label in trn_labels:
    if label == 'melanoma':
        Melanoma += 1
    if label == 'nevus':
        Nevus += 1
    if label == 'seborrheic_keratosis':
        Seborrheic_Keratosis += 1
print(Melanoma, Nevus, Seborrheic_Keratosis)

def add_new_last_layer(base_model, nb_classes, fc_size, fc_size2, drop):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(fc_size, activation='relu')(x) #new FC layer, random init
    x = Dropout(drop)(x)
    x = Dense(fc_size2, activation='relu')(x) #new FC layer, random init
    x = Dropout(drop)(x)
    predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
    model = Model(input=base_model.input, output=predictions)
    return model

#Model Hyperparamaters
nb_classes = 3
fc_size = 512
fc_size2 = 256
batch_size = 100
dropout = 0.5

base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
model = add_new_last_layer(base_model, nb_classes, fc_size, fc_size2, dropout)

#Print a summary of the model architecture.
model.summary()

model.save_weights('saved_models/inception_baseline.h5')

i = 0
for layer in model.layers:
    i += 1
print(i)

def setup_to_transfer_learn(model):
    """Freeze all pretrained layers and compile the model"""
    for layer in model.layers[:312]:
        layer.trainable = False
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
setup_to_transfer_learn(model)
#Other Optimizers 'adam'. Not much success with SGD(lr=0.0001, momentum=0.9), 'rmsprop'

epochs = 20
batch_size = batch_size

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4, verbose=1, mode='auto')

hist = model.fit(np.array(trn_files), np.array(trn_label_vecs), 
          validation_data=(np.array(vld_files), np.array(vld_label_vecs)),
          epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, early_stopping], verbose=1)

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

def predict_image(model, img_matrix): 
    '''
    This function accepts the final model from above and returns a classification of all all three types. 
    ***Args***
    model: CNN Trained above
    img_matrix: image data in a numpy array
    '''
    img = np.expand_dims(img_matrix, axis=0)
    img = preprocess_input(img)
    return model.predict(img)

def convert_prediction(values):
    '''
    This function accepts a numpy array of predicted values and returns a dictionary of predictions of each label.
    ***Args***
    values: Numpy Array of predicted values.
    '''
    labels = ['Melanoma', 'Nevus', 'Seborrheic Keratosis']
    value_hash = {}
    for i in range(len(labels)):
        value_hash[labels[i]] = values[0][i]
    return value_hash

values = predict_image(model, trn_files[0])
predictions = convert_prediction(values)

max(predictions)

trn_label_vecs[0]



