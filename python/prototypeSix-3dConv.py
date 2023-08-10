import tensorflow as tf
#import tensorflow.contrib.learn.python.learn as learn
import tflearn 
import scipy as sp
import numpy as np 
import matplotlib.pyplot as plt
import six
from sklearn.metrics import roc_curve
import datetime
get_ipython().magic('matplotlib inline')

ecgData = np.load("/media/john/SAMSUNG/CNN/3D-conv/ecgData.npy")
ecgClass = np.load("/media/john/SAMSUNG/CNN/3D-conv/ecgClass.npy")
unseenData = np.load("/media/john/SAMSUNG/CNN/3D-conv/unseenData.npy")
unseenClass = np.load("/media/john/SAMSUNG/CNN/3D-conv/unseenClass.npy")

def processClassData(classData):
    """
    Process classData.
    
    Returns a one-hot array of shape [len(classData), 2].
    """
    # Convert label data to one-hot array
          
    classDataOH = np.zeros((len(classData),2))
    classDataOH[np.arange(len(classData)), classData] = 1
    
    return classDataOH

ecgData = ecgData[..., np.newaxis]
unseenData = unseenData[..., np.newaxis]
ecgClassOH = processClassData(ecgClass)
unseenClassOH = processClassData(unseenClass)

sess = tf.InteractiveSession()
tf.reset_default_graph()
tflearn.initializations.normal()

# Input layer:
net = tflearn.layers.core.input_data(shape=[None, 63, 19, 17, 1])

# First layer:
net = tflearn.layers.conv.conv_3d(net, 8, [5,5,5],  activation="leaky_relu", padding="valid")
net = tflearn.layers.conv.max_pool_3d(net, 3, padding='valid', strides=3)

# Second layer:
net = tflearn.layers.conv.conv_3d(net, 16, [5,5,5], activation="leaky_relu")
net = tflearn.layers.conv.max_pool_3d(net, 3, strides=3)

# Fully connected layer
net = tflearn.layers.core.fully_connected(net, 1024, regularizer="L2", weight_decay=0.001, activation="leaky_relu")
net = tflearn.layers.core.fully_connected(net, 1024, regularizer="L2", weight_decay=0.001, activation="leaky_relu")
# Dropout layer:
net = tflearn.layers.core.dropout(net, keep_prob=0.5)

# Output layer:
net = tflearn.layers.core.fully_connected(net, 2, activation="softmax")

net = tflearn.layers.estimator.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=3)
model.fit(ecgData[:,::32], ecgClassOH, batch_size=4, n_epoch=10, show_metric=True, validation_set=0.2)

sess = tf.InteractiveSession()
tf.reset_default_graph()
tflearn.initializations.normal()

# Input layer:
net = tflearn.layers.core.input_data(shape=[None, 500, 19, 17, 1])

# First layer:
net = tflearn.layers.conv.conv_3d(net, 8, [10,5,5],  activation="leaky_relu", padding="valid")
net = tflearn.layers.conv.max_pool_3d(net, 2, padding='valid', strides=2)

# Second layer:
net = tflearn.layers.conv.conv_3d(net, 16, [5,5,5], activation="leaky_relu")
net = tflearn.layers.conv.max_pool_3d(net, 2, strides=2)

# Fully connected layer
net = tflearn.layers.core.fully_connected(net, 1024, regularizer="L2", weight_decay=0.001, activation="leaky_relu")
net = tflearn.layers.core.fully_connected(net, 1024, regularizer="L2", weight_decay=0.001, activation="leaky_relu")
# Dropout layer:
net = tflearn.layers.core.dropout(net, keep_prob=0.5)

# Output layer:
net = tflearn.layers.core.fully_connected(net, 2, activation="softmax")

net = tflearn.layers.estimator.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=3)
model.fit(ecgData[:,::4], ecgClassOH, batch_size=32, n_epoch=10, show_metric=True, validation_set=0.2)

model.fit(ecgData[:,500::20], ecgClassOH, batch_size=32, n_epoch=150, show_metric=True, validation_set=0.2)

#model.save("./outData/models/3d_conv.tflearn")
#model.load("/media/john/SAMSUNG/CNN/3D-conv/3d_conv.tflearn")

def splitData(coilData, classData):
    """
    Split data into healthy and ill types.
    """
    illData = []
    healthData = []
    
    for index, item in enumerate(classData):
        if item == 1:
            illData.append(coilData[index])
        if item == 0:
            healthData.append(coilData[index])
            
    return illData, healthData

iUnseen, hUnseen = splitData(unseenData, unseenClass)
unseenHL = np.tile([1,0], (len(hUnseen), 1))
unseenIL = np.tile([0,1], (len(iUnseen), 1))
iUnseen = np.reshape(iUnseen, (-1,2000,19,17,1))
hUnseen = np.reshape(hUnseen, (-1,2000,19,17,1))

print(model.evaluate(unseenData[:,500::20], unseenClassOH),"\n",    model.evaluate(np.array(iUnseen)[:,500::20], unseenIL),"\n",    model.evaluate(np.array(hUnseen)[:,500::20], unseenHL))

# Get ROC curve
#if k == 1:
predicted = np.array(model.predict(np.array(unseenData)[:,500::20]))
fpr, tpr, th = roc_curve(unseenClass, predicted[:,1])
plt.plot(fpr,tpr)

[n.name for n in tf.get_default_graph().as_graph_def().node]

model.net.v

model.net.graph.get_tensor_by_name('Conv3D/W:0').re

var = [v for v in tf.trainable_variables() if v.name == "Conv3D/W:0"][0]

var.value

data = model.get_weights(var)

np.shape(data.T)

convlist=[]
for i in range(16):
    convlist.append(data.T[i])

np.shape(convlist[0][0].T[0])

for i in range(16):
    plt.imshow(convlist[i][0].T[1], cmap='hot', interpolation='nearest')
    plt.show()

for i in range(16):
    print "min ", (convlist[i][0].T[1]).min(), " max ", (convlist[i][0].T[0]).max()



