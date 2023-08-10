reset -fs

import numpy as np
import pandas as pd
import os
import glob
import pickle
import gzip
import h5py
import dl_functions
from IPython.display import display
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

with gzip.open('pre_processed_images/image_data_20000_50.txt.gz', 'rb') as f:
    X, y = pickle.load(f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

y_train_sparse = np_utils.to_categorical(y_train, 2)

y_test_sparse = np_utils.to_categorical(y_test, 2)

IMG_SIZE = 50

model_1 = dl_functions.cnn_model_v_2(IMG_SIZE)

model_1.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model_1.summary()

model_1.fit(X_train, y_train_sparse, batch_size=128, epochs=20, verbose=1, validation_split=0.3)

score = model_1.evaluate(X_test, y_test_sparse, verbose=1)

print('Test loss: {:0,.4f}'.format(score[0]))
print('Test accuracy: {:.2%}'.format(score[1]))

model_1.save('models/model_v2_20000_50.h5')

model_1.save_weights('models/model_v2_20000_50_weights.h5')

predicted_images = []
for i in model_1.predict(X_test):
    predicted_images.append(np.where(np.max(i) == i)[0])

dl_functions.show_confusion_matrix(confusion_matrix(y_test, predicted_images), ['Class 0', 'Class 1'])

pd.DataFrame(confusion_matrix(y_test, predicted_images),columns = ['nok_image','ok_image'])

print("AUC: {:.2%}\n".format(roc_auc_score(y_test, predicted_images)))

predictions_probability = model_1.predict_proba(X_test)

plt.figure(figsize=(7, 7))
dl_functions.plot_roc(y_test, predictions_probability[:,1], "CNN - " + str(len(model_1.layers)) + " layers | # images: " + str(len(X)) + " | image size: " + str(IMG_SIZE), "Tasty Food Images")

trump = dl_functions.normalize_images_array('images', IMG_SIZE)

trump_prediction = model_1.predict_classes(trump)

trump_prediction[0]

trump.shape

trump = trump.reshape(trump.shape[1], trump.shape[2], trump.shape[3])

img = array_to_img(trump)
display(img)

model_2 = dl_functions.cnn_model_v_1(IMG_SIZE)

model_2.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model_2.summary()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

y_train_sparse = np_utils.to_categorical(y_train, 2)

y_test_sparse = np_utils.to_categorical(y_test, 2)

model_2.fit(X_train, y_train_sparse, batch_size=128, epochs=50, verbose=1, validation_split=0.3)

score = model_2.evaluate(X_test, y_test_sparse, verbose=1)

print('Test loss: {:0,.4f}'.format(score[0]))
print('Test accuracy: {:.2%}'.format(score[1]))

model_2.save('models/model_v1_20000_50.h5')

model_2.save_weights('models/model_v1_20000_50_weights.h5')

predicted_images = []
for i in model_2.predict(X_test):
    predicted_images.append(np.where(np.max(i) == i)[0])

dl_functions.show_confusion_matrix(confusion_matrix(y_test, predicted_images), ['Class 0', 'Class 1'])

pd.DataFrame(confusion_matrix(y_test, predicted_images),columns = ['nok_image','ok_image'])

print("AUC: {:.2%}\n".format(roc_auc_score(y_test, predicted_images)))

predictions_probability = model_2.predict_proba(X_test)

plt.figure(figsize=(7, 7))
dl_functions.plot_roc(y_test, predictions_probability[:,1], "CNN - " + str(len(model_2.layers)) + " layers | # images: " + str(len(X)) + " | image size: " + str(IMG_SIZE), "Tasty Food Images")

trump = dl_functions.normalize_images_array('images', IMG_SIZE)

trump_prediction = model_2.predict_classes(trump)

trump_prediction[0]

trump.shape

trump = trump.reshape(trump.shape[1], trump.shape[2], trump.shape[3])

img = array_to_img(trump)
display(img)

get_ipython().system("gsutil cp -r 'models' 'gs://wellio-kadaif-tasty-images-project-pre-processed-images'")



