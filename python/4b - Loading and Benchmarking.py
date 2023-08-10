import numpy as np
np.warnings.filterwarnings('ignore')  # Hide np.floating warning

import keras

from keras.datasets import cifar10

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

from keras.models import load_model

get_ipython().run_cell_magic('time', '', "gpu_model = load_model('cifar10_model.hdf5')")

gpu_model.predict_classes(x_test)

with tf.device("/device:CPU:0"):
    cpu_model = load_model('cifar10_model.hdf5')
    print(cpu_model.predict_classes(x_test))

print('GPU performance: %d images' % x_test.shape[0])
get_ipython().run_line_magic('timeit', 'gpu_model.predict_classes(x_test)')
print('CPU performance: %d images' % x_test.shape[0])
get_ipython().run_line_magic('timeit', 'cpu_model.predict_classes(x_test)')

print('GPU performance: 1 image')
get_ipython().run_line_magic('timeit', 'gpu_model.predict_classes(x_test[:1])')
print('CPU performance: 1 image')
get_ipython().run_line_magic('timeit', 'cpu_model.predict_classes(x_test[:1])')



