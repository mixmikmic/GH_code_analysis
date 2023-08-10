import tensorflow as tf
#import tensorflow.contrib.learn.python.learn as learn
import tflearn 
import scipy as sp
import numpy as np 
import matplotlib.pyplot as plt
import six
from sklearn.metrics import roc_curve, roc_auc_score
import datetime
get_ipython().magic('matplotlib inline')

comData = np.load("../inData/comData.npy")
comClass = np.load("../inData/comClass.npy")

def processClassData(classData):
    """
    Process classData.
    
    Returns a one-hot array of shape [len(classData), 2].
    """
    # Convert label data to one-hot array
          
    classDataOH = np.zeros((len(classData),2))
    classDataOH[np.arange(len(classData)), classData] = 1
    
    return classDataOH

comData = comData[..., np.newaxis]
comClassOH = processClassData(comClass)

# kfold x-validation...
k = 4
kfoldData = np.array_split(comData, k)
kfoldLabelsOH = np.array_split(comClassOH, k)
kfoldLabels = np.array_split(comClass, k)
print(kfoldData[0].shape)

spec = []
sens = []
roc = []

for i in np.arange(0,k,1):
    for j in np.arange(0,8,1):
        sess = tf.InteractiveSession()
        tf.reset_default_graph()
        tflearn.initializations.normal()

        # Input layer:
        net = tflearn.layers.core.input_data(shape=[None, 10, 19, 17, 1])

        # First layer:
        net = tflearn.layers.conv.conv_3d(net, 8, [5,5,5],  activation="leaky_relu")
        net = tflearn.layers.conv.max_pool_3d(net, 2, strides=2)

        # Second layer:
        net = tflearn.layers.conv.conv_3d(net, 16, [5,5,5], activation="leaky_relu")
        net = tflearn.layers.conv.max_pool_3d(net, 2, strides=2)

        # Fully connected layer
        net = tflearn.layers.core.fully_connected(net, 1024, regularizer="L2", weight_decay=0.001, activation="leaky_relu")
        #net = tflearn.layers.core.fully_connected(net, 1024, regularizer="L2", weight_decay=0.001, activation="leaky_relu")

        # Dropout layer:
        net = tflearn.layers.core.dropout(net, keep_prob=0.5)

        # Output layer:
        net = tflearn.layers.core.fully_connected(net, 2, activation="softmax")

        net = tflearn.layers.estimator.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')

        model = tflearn.DNN(net, tensorboard_verbose=0)

        dummyData = np.reshape(np.concatenate(kfoldData[:i] + kfoldData[i+1:], axis=0), [-1, 2000, 19, 17, 1])
        dummyData = dummyData[:,::40]
        dummyData = dummyData[:,0 + j*5: 10 + j*5]
        print(dummyData[0,:].shape)
        dummyLabels = np.reshape(np.concatenate(kfoldLabelsOH[:i] + kfoldLabelsOH[i+1:], axis=0), [-1, 2])
        model.fit(dummyData, dummyLabels, batch_size=8, n_epoch=30, show_metric=True)

        illTest = []
        healthTest = []
        for index, item in enumerate(kfoldLabels[i]):
            if item == 1:
                illTest.append(kfoldData[i][index])
            if item == 0:
                healthTest.append(kfoldData[i][index])

        healthLabel = np.tile([1,0], (len(healthTest), 1))
        illLabel = np.tile([0,1], (len(illTest), 1))
        
        sens.append(model.evaluate(np.array(healthTest)[:,::40][:,0 + j*5: 10 + j*5], healthLabel))
        spec.append(model.evaluate(np.array(illTest)[:,::40][:,0 + j*5: 10 + j*5], illLabel))

        # Get roc curve data
        predicted = np.array(model.predict(np.array(kfoldData[i])[:,::40][:,0 + j*5: 10 + j*5]))
        auc = roc_auc_score(kfoldLabels[i], predicted[:,1])
        roc.append(auc)

spec = np.reshape(np.array(spec), (k,8))
sens = np.reshape(np.array(sens), (k,8))
roc = np.reshape(np.array(roc), (k,8))

#model.save("./outData/models/3d_conv8-16-50.tflearn")
#model.load("./outData/models/3d_conv.tflearn")

for i in np.arange(0,k,1):
    print("Specificity:", spec[i], "\nSensitivity:", sens[i])

xax = [5,10,15,20,25,30,35,40]
for i in np.arange(0,k,1):
    plt.plot(xax, roc[i], "x")

rocAv = []
for i in np.arange(8):
    rocAv.append(np.mean(roc[:,i]))
plt.plot(xax, rocAv, "o")
plt.savefig("/Users/controller/Desktop/diagAUC/roc1.png")

np.save("/Users/controller/Desktop/diagAUC/spec2", spec)
np.save("/Users/controller/Desktop/diagAUC/sens2", sens)
np.save("/Users/controller/Desktop/diagAUC/roc2", roc)



