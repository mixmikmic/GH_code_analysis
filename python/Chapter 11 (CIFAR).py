import os
data_folder = os.path.join(os.path.expanduser("~"), "Data", "cifar-10-batches-py")
batch1_filename = os.path.join(data_folder, "data_batch_1")

import pickle
# Bigfix thanks to: http://stackoverflow.com/questions/11305790/pickle-incompatability-of-numpy-arrays-between-python-2-and-3
def unpickle(filename):
    with open(filename, 'rb') as fo:
        return pickle.load(fo, encoding='latin1')

batch1 = unpickle(batch1_filename)

image_index = 100
image = batch1['data'][image_index]

image = image.reshape((32,32, 3), order='F')
import numpy as np
image = np.rot90(image, -1)

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

plt.imshow(image)

import numpy as np
batches = []
for i in range(1, 6):
    batch_filename = os.path.join(data_folder, "data_batch_{}".format(i))
    batches.append(unpickle(batch1_filename))
    break    #IMPORTANT -- see chapter for explanation of this line

X = np.vstack([batch['data'] for batch in batches])

X = np.array(X) / X.max()
X = X.astype(np.float32)

from sklearn.preprocessing import OneHotEncoder
y = np.hstack(batch['labels'] for batch in batches).flatten()
y = OneHotEncoder().fit_transform(y.reshape(y.shape[0],1)).todense()
y = y.astype(np.float32)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

X_train = X_train.reshape(-1, 3, 32, 32)
X_test = X_test.reshape(-1, 3, 32, 32)

from lasagne import layers
layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ]

from nolearn.lasagne import NeuralNet
from lasagne.nonlinearities import sigmoid, softmax
nnet = NeuralNet(layers=layers,
                 input_shape=(None, 3, 32, 32),
                 conv1_num_filters=32,
                 conv1_filter_size=(3, 3),
                 conv2_num_filters=64,
                 conv2_filter_size=(2, 2),
                 conv3_num_filters=128,
                 conv3_filter_size=(2, 2),
                 pool1_ds=(2,2),
                 pool2_ds=(2,2),
                 pool3_ds=(2,2),
                 hidden4_num_units=500,
                 hidden5_num_units=500,
                 output_num_units=10,
                 output_nonlinearity=softmax,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 regression=True,
                 max_epochs=3,
                 verbose=1)

nnet.fit(X_train, y_train)

from sklearn.metrics import f1_score
y_pred = nnet.predict(X_test)
print(f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1)))



