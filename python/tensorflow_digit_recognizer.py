get_ipython().magic("config InlineBackend.figure_format = 'retina'")
get_ipython().magic('matplotlib inline')

import os

import tensorflow as tf

from   keras.backend.tensorflow_backend import set_session
from   keras.datasets import mnist
from   keras          import backend as K
from   keras.utils    import np_utils

img_rows, img_cols = 28, 28 # input image dimensions
nb_classes         = 10

# Limit GPU memory consumption to 30%

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

os.environ['KERAS_BACKEND'] = 'tensorflow'

dummy, (X_test, y_test) = mnist.load_data()

print(X_test.shape[0], 'test samples')

if K.image_dim_ordering() == 'th':
    X_test      = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
else:
    X_test      = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_test  = X_test.astype('float32')
X_test  /= 255
# convert class vectors to binary class matrices
Y_test  = np_utils.to_categorical(y_test, nb_classes)

get_ipython().run_cell_magic('time', '', "from keras.models import load_model\n\nmodel = load_model('tf_digit_model_10epoch_10class_128batch.h5')")

get_ipython().run_cell_magic('time', '', "score   = model.evaluate(X_test, Y_test, verbose=0)\nprint('Test score:',    score[0])\nprint('Test accuracy:', score[1])")

y_predict = model.predict_classes(X_test)
fails     = y_predict != y_test

dummy, (X_test, y_test) = mnist.load_data()

X_test_fails    = X_test[fails]
y_test_fails    = y_test[fails]
y_predict_fails = y_predict[fails]

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 10, 5

for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.imshow(X_test_fails[i], cmap='gray_r')
    plt.title('Predict: %d, Actual: %d' % (y_predict_fails[i], y_test_fails[i]))
    plt.xticks([])
    plt.yticks([])
#plt.tight_layout()



