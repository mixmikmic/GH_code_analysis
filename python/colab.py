get_ipython().system('apt-get install -y -qq software-properties-common python-software-properties module-init-tools')
get_ipython().system('add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null')
get_ipython().system('apt-get update -qq 2>&1 > /dev/null')
get_ipython().system('apt-get -y install -qq google-drive-ocamlfuse fuse')
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
get_ipython().system('google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL')
vcode = getpass.getpass()
get_ipython().system('echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}')

get_ipython().system('mkdir -p drive')
get_ipython().system('google-drive-ocamlfuse drive')

import os
os.chdir('drive/MGH/Teaching/qtim_Tutorials/tutorial_5/')
get_ipython().system('ls')

from keras.layers import Input, Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model

max_channels = 1024

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
val_datagen = ImageDataGenerator()  # no augmentation! why?
X_val = HDF5Matrix('validation.h5', 'train')
y_val = HDF5Matrix('validation.h5', 'labels')
val_generator = val_datagen.flow(X_val, y_val, seed=0, batch_size=1)

from keras.callbacks import ModelCheckpoint, EarlyStopping

mc_cb = ModelCheckpoint('best_model.h5')
el_cb = EarlyStopping(patience=5)

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_generator, epochs=50, shuffle='batch',
                    validation_data=val_generator, callbacks=[mc_cb, el_cb])
model.save('final_model.h5')

from keras.models import load_model
import numpy as np
import h5py

model = load_model('best_model.h5')

# We will use testing data in future... this is somewhat biased!
val_data = h5py.File('validation.h5', 'r')
X_val, y_val = val_data['train'], val_data['labels']

y_pred = model.predict(X_val)  # get network predictions over entire dataset
y_true = np.asarray(y_val)  # using np.asarray explicitly loads the HDF5 data

import pandas as pd
pd.DataFrame([y_pred.squeeze(), y_true]).T

from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

# Confusion matrix, optionally normalized
normalize = False
cm = confusion_matrix(y_true, np.round(y_pred).astype('bool'))
fmt = 'd'  # for displaying the values

if normalize:
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # optional!
  fmt = '.2%'

# Use some fancy plotting
labels = ['No tumor', 'Tumor']
ax = sns.heatmap(cm, annot=True, fmt=fmt, xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
plt.savefig('confusion.png', dpi=300)

fpr, tpr, _ = roc_curve(y_true, y_pred)
plt.plot(fpr, tpr, label='AUC: {:.2f}'.format(auc(fpr, tpr)))
plt.title('ROC analysis of my first tumor detector')
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.legend()
plt.savefig('roc.png', dpi=300)

