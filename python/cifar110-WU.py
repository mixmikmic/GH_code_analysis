from __future__ import (print_function, absolute_import)
import os
import numpy as np
import pandas as pd
import keras
import keras.backend as K
from keras import datasets
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from models import CNN, VGG8
from wide_resnet import WideResidualNetwork
from pulearn.utils import synthesize_pu_labels

from keras.callbacks import (
    ReduceLROnPlateau,
    LearningRateScheduler,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint)

from keras_tqdm import TQDMNotebookCallback

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="0"

global _SESSION
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
_SESSION = tf.Session(config=config)
K.set_session(_SESSION)

num_classes = 11
batch_size = 128
epochs = 200
data_augmentation = True
checkpoint = None
# checkpoint = 'model_checkpoint_cifar10_wide_resnet.h5'
title = 'cifar110_vgg8'
# title = 'cifar110_wide_resnet'

(x_train_10, y_train_10), (x_test_10, y_test_10) = datasets.cifar10.load_data()
(x_train_100, y_train_100), (x_test_100, y_test_100) = datasets.cifar100.load_data()

y_train_10 = y_train_10 + 1
y_test_10 = y_test_10 + 1
y_train_100[...] = 0
y_test_100[...] = 0

x_train = np.concatenate((x_train_10, x_train_100), axis=0).astype('float32')
y_train = np.concatenate((y_train_10, y_train_100), axis=0).astype('int8')
x_test = np.concatenate((x_test_10, x_test_100), axis=0).astype('float32')
y_test = np.concatenate((y_test_10, y_test_100), axis=0).astype('int8')

x_train = x_train.astype(K.floatx())
x_test = x_test.astype(K.floatx())

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def normalize(x):
    """Substract mean and Divide by std."""
    x -= np.array([125.3, 123.0, 113.9], dtype=K.floatx())
    x /= np.array([63.0, 62.1, 66.7], dtype=K.floatx())
    return x


x_train = normalize(x_train)
x_test = normalize(x_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

pct_missing = 0.
y_train_pu = synthesize_pu_labels(y_train, random_state=None, verbose=True)

y_train_enc = y_train_pu[pct_missing]
from sklearn.utils import class_weight
y_train_dec = 1 - y_train_enc[:, 0]
class_weights = class_weight.compute_class_weight('balanced',
                                                 [0, 1],
                                                 y_train_dec)
class_weight = {}
class_weight[0] = class_weights[0]
for k in range(1, num_classes):
    class_weight[k] = class_weights[1]

# pu_ratio = np.sum(y_train_enc[:, 1:]) / np.sum(y_train_enc[:, 0])

# class_weight = {0: 1}
# for k in range(1, num_classes):
#     class_weight[k] = 1 / pu_ratio

class_weight[0] = class_weight[0] * 0.5
print('class_weight', class_weight)

if title == 'cifar110_vgg8':
    model = VGG8(input_shape=x_train.shape[1:], num_classes=num_classes)
elif title == 'cifar110_wide_resnet':
    model = WideResidualNetwork(depth=28, width=8, dropout_rate=0.5,
                                classes=num_classes, include_top=True,
                                weights=None)
if checkpoint is not None:
    model = load_model(checkpoint)
    
optimizer = keras.optimizers.Adam(lr=1e-4)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()

# Checkpoint
checkpointer = ModelCheckpoint(
    monitor='val_acc',
    filepath="model_checkpoint_{}_wu_{}.h5".format(title, pct_missing),
    verbose=1,
    save_best_only=True)

# csvlogger
csv_logger = CSVLogger(
    'csv_logger_{}_wu_{}.csv'.format(title, pct_missing))

# EarlyStopping
early_stopper = EarlyStopping(monitor='val_acc',
                              min_delta=0.001,
                              patience=50)
# Reduce lr with schedule
# def schedule(epoch):
#     lr = K.get_value(model.optimizer.lr)
#     if epoch in [60, 120, 160]:
#         lr = lr * np.sqrt(0.1)
#     return lr
# lr_scheduler = LearningRateScheduler(schedule)

# Reduce lr on plateau
lr_reducer = ReduceLROnPlateau(monitor='val_acc',
                               factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=20,
                               min_lr=1e-6)

if not data_augmentation:
    print('No data augmentation applied.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              class_weight=class_weight,
              shuffle=True,
              callbacks=[csv_logger, checkpointer, early_stopper, lr_reducer]) 
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        # randomly rotate images in the range (degrees, 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train_pu[pct_missing],
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        class_weight=class_weight,
                        callbacks=[checkpointer, early_stopper,
                                   lr_reducer, csv_logger,
                                   TQDMNotebookCallback()])
    model.save('{}_wu_{}.h5'.format(title, pct_missing))
    
    # Predict
    y_train_pred_enc = model.predict(x_train)
    y_test_pred_enc = model.predict(x_test)
    
    lst = []
    y_train_pred = np.argmax(y_train_pred_enc, axis=-1)
    y_train_true = np.argmax(y_train, axis=-1)
    y_train_label = np.argmax(y_train_pu[pct_missing], axis=-1)
    for y_pred, y_true, y_label in zip(y_train_pred, y_train_true, y_train_label):
        lst.append(dict(y_pred=y_pred, y_true=y_true, y_label=y_label, split='train'))

    y_test_pred = np.argmax(y_test_pred_enc, axis=-1)
    y_test_true = np.argmax(y_test, axis=-1)
    for y_pred, y_true in zip(y_test_pred, y_test_true):
        lst.append(dict(y_pred=y_pred, y_true=y_true, y_label=y_true, split='test'))
    
    df = pd.DataFrame(lst)
    df.to_csv('prediction_{}_wu_{}.csv'.format(title, pct_missing), index=False, mode='a')



