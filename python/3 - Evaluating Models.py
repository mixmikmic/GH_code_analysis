import numpy as np
np.warnings.filterwarnings('ignore')  # Hide np.floating warning

import keras

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# Prevent TensorFlow from grabbing all the GPU memory
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

import holoviews as hv
hv.extension('bokeh')

from keras.datasets import cifar10
import keras.utils

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Save an unmodified copy of y_test for later, flattened to one column
y_test_true = y_test[:,0].copy()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# The data only has numeric categories so we also have the string labels below 
cifar10_labels = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck'])

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=x_train.shape[1:]))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=128,
          epochs=8,
          verbose=1,
          validation_data=(x_test, y_test))

get_ipython().run_cell_magic('opts', 'Curve [width=400 height=300]', "%%opts Curve (line_width=3)\n%%opts Overlay [legend_position='top_left']\n\ntrain_acc = hv.Curve((history.epoch, history.history['acc']), 'epoch', 'accuracy', label='training')\nval_acc = hv.Curve((history.epoch, history.history['val_acc']), 'epoch', 'accuracy', label='validation')\n\n(train_acc * val_acc).redim(accuracy=dict(range=(0.4, 1.1)))")

from sklearn.metrics import confusion_matrix

y_pred = model.predict_classes(x_test)
confuse = confusion_matrix(y_test_true, y_pred)

# Holoviews hack to tilt labels by 45 degrees
from math import pi
def angle_label(plot, element):
    plot.state.xaxis.major_label_orientation = pi / 4

get_ipython().run_cell_magic('opts', "HeatMap [width=500 height=400 tools=['hover'] finalize_hooks=[angle_label]]", "hv.HeatMap((cifar10_labels, cifar10_labels, confuse)).redim.label(x='true', y='predict')")

model2 = Sequential()
model2.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=x_train.shape[1:]))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Dropout(0.25))

model2.add(Flatten())
model2.add(Dense(128, activation='relu'))

model2.add(Dropout(0.5))

model2.add(Dense(num_classes, activation='softmax'))

model2.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history2 = model2.fit(x_train, y_train,
          batch_size=128,
          epochs=11,
          verbose=1,
          validation_data=(x_test, y_test))

get_ipython().run_cell_magic('opts', 'Curve [width=600 height=450]', "%%opts Curve (line_width=3)\n%%opts Overlay [legend_position='top_left']\n\ntrain_acc = hv.Curve((history.epoch, history.history['acc']), 'epoch', 'accuracy', label='training without dropout')\nval_acc = hv.Curve((history.epoch, history.history['val_acc']), 'epoch', 'accuracy', label='validation without dropout')\ntrain_acc2 = hv.Curve((history2.epoch, history2.history['acc']), 'epoch', 'accuracy', label='training with dropout')\nval_acc2 = hv.Curve((history2.epoch, history2.history['val_acc']), 'epoch', 'accuracy', label='validation with dropout')\n\n(train_acc * val_acc * train_acc2 * val_acc2).redim(accuracy=dict(range=(0.4, 1.1)))")

model3 = Sequential()
model3.add(Conv2D(32, kernel_size=(3, 3), padding='same',
                 activation='relu',
                 input_shape=x_train.shape[1:]))
model3.add(Conv2D(32, (3, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.25))

# Second layer of convolutions
model3.add(Conv2D(64, kernel_size=(3, 3), padding='same',
                 activation='relu'))
model3.add(Conv2D(64, (3, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.25))

model3.add(Flatten())
model3.add(Dense(512, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(num_classes, activation='softmax'))

model3.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history3 = model3.fit(x_train, y_train,
          batch_size=128,
          epochs=15,
          verbose=1,
          validation_data=(x_test, y_test))

get_ipython().run_cell_magic('opts', 'Curve [width=600 height=500]', "%%opts Curve (line_width=3)\n%%opts Overlay [legend_position='top_left']\n\ntrain_acc = hv.Curve((history2.epoch, history2.history['val_acc']), 'epoch', 'accuracy', label='validation (simple model)')\ntrain_acc2 = hv.Curve((history3.epoch, history3.history['acc']), 'epoch', 'accuracy', label='training (complex model)')\nval_acc = hv.Curve((history3.epoch, history3.history['val_acc']), 'epoch', 'accuracy', label='validation (complex model)')\n\n(train_acc * val_acc * train_acc2).redim(accuracy=dict(range=(0.4, 1.1)))")

