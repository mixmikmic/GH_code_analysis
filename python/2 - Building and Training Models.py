import numpy as np
np.warnings.filterwarnings('ignore')  # Hide np.floating warning
import holoviews as hv
hv.extension('bokeh')

# Prevent TensorFlow from grabbing all the GPU memory
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

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

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

### Convolution and max pool layers

# Group 1: Convolution
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Group 2: Convolution
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=x_train.shape[1:]))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Group 3: Dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summary()

get_ipython().run_cell_magic('time', '', 'history = model.fit(x_train, y_train,\n          batch_size=256,\n          epochs=5,\n          verbose=1,\n          validation_data=(x_test, y_test))')

model.fit(x_train, y_train,
          batch_size=256,
          epochs=2,
          verbose=1,
          validation_data=(x_test, y_test))

early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.05, patience=2, verbose=1)
model.fit(x_train, y_train,
          batch_size=256,
          epochs=10,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[early_stop])

print(history.epoch)
history.history

get_ipython().run_cell_magic('opts', 'Curve [width=400 height=300]', "%%opts Curve (line_width=3)\n%%opts Overlay [legend_position='top_left']\n\ntrain_acc = hv.Curve((history.epoch, history.history['acc']), 'epoch', 'accuracy', label='training')\nval_acc = hv.Curve((history.epoch, history.history['val_acc']), 'epoch', 'accuracy', label='validation')\n\ntrain_acc * val_acc")

y_predict = model.predict(x_test)
y_predict[:5]

y_predict = model.predict_classes(x_test)
y_predict[:5]

y_predict_labels = cifar10_labels[y_predict]
y_true_labels = cifar10_labels[y_test_true]
print(y_predict_labels[:5])
print(y_true_labels[:5])

get_ipython().run_cell_magic('output', 'size=64', "%%opts RGB [xaxis=None yaxis=None]\n\nimages = [hv.RGB(x_test[i], label='%s(%s)' % (y_true_labels[i], y_predict_labels[i]) ) for i in range(12)]\nhv.Layout(images).cols(4)")

failed = y_predict != y_test_true
print('Number failed:', np.count_nonzero(failed))

get_ipython().run_cell_magic('output', 'size=64', "%%opts RGB [xaxis=None yaxis=None]\n\nimages = [hv.RGB(x_test[failed][i], label='%s(%s)' % \n                 (y_true_labels[failed][i],\n                  y_predict_labels[failed][i]) ) for i in range(12)]\nhv.Layout(images).cols(4)")



