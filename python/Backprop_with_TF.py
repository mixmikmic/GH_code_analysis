import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(1)

from sklearn.datasets import make_blobs

data, labels = make_blobs(n_features=2,                          centers=2,                          cluster_std=(0.1, 0.1),                           center_box=([-2,-2], [-1,1]),                          shuffle=True)
labels[labels==0] = -1
labels = labels[:,np.newaxis]

plt.scatter(data[:,0], data[:,1], c=labels)
plt.grid()
plt.show()

def plotSeperatingPLane(predictionFunc, data):
    #create a meshgrid
    nb_of_xs = 100
    xs1 = np.linspace(-4, 4, num=nb_of_xs)
    xs2 = np.linspace(-4, 4, num=nb_of_xs)
    xx, yy = np.meshgrid(xs1, xs2)
    
    #classify each point in this meshgrid using a "predictionFunction"
    classification_plane = predictionFunc(np.c_[xx.ravel(), yy.ravel()])
    #reshape it to make it plottable
    classification_plane = classification_plane.reshape(xx.shape)

    #plot both the points and "contours"
    plt.contourf(xx, yy, classification_plane, cmap=plt.cm.Accent)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.Oranges_r)
    plt.grid()

import sklearn.linear_model
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(data, labels.ravel())

plotSeperatingPLane(clf.predict, data)
plt.title("Logistic Regression")

# create some random weights and biases
w = np.random.normal(0, 1, data.shape[1])[np.newaxis] # 1*2
b = np.random.rand()*np.ones((data.shape[0],1)) # 100*1

# the input to our node is
z = data.dot(w.T) + b

# our f function/activation which yields the y values
def predict(weights, biases, data):
    z = data.dot(weights.T) + biases
    return np.where(z >= 0, 1, -1)

# let's define the (labels - y).x part as the gradient function first
def perceptronGradient(labels, y, data):
    # A lot of tranpose operations here to match the sizes
    # (100*1).T -> 1*100
    # (1*100)*(100*2) -> 1*2 i.e. equal to w.shape()
    return ((labels - y).T.dot(data.reshape(-1,2)))

# one iteration over all the samples, called an epoch, looks like this
w = np.random.normal(0, 1, data.shape[1])[np.newaxis] # 1*2
b = np.random.rand()*np.ones((data.shape[0],1)) # 100*1
eta = 0.05
y = predict(w, b, data)
delta_w = perceptronGradient(labels, y, data)
w += eta*delta_w
b += eta*(labels - y).sum()

class Perceptron(object):
    
    def __init__(self, nDims, eta=0.05, epochs=10):
        self.eta = eta
        self.epochs = epochs
        self.weights = np.random.normal(0, 1, nDims)[np.newaxis] # 1*2
        self.biases = np.random.rand()
    
    # we are partitioning our above defined predict() function into
    # nodeInput() and nodeActivation() functions.
    def nodeInput(self, data):
        # this reshaping is done so that the sizes match
        # independent of the number of input samples
        data = data.reshape(-1, self.weights.shape[1])
        # the usual multiplication and bias addition
        z = data.dot(self.weights.T) + self.biases*np.ones((len(data),1)) 
        return z
    
    def nodeActivation(self, z):
        # activation for a perceptron is just a unit step function
        # i.e. positive if z>0 and negative if z<0
        return np.where(z >= 0, 1, -1)    
    
    # merge the nodeInput() and nodeActivation() functions to 
    # recreate the prediction function. This is useful for testing the neuron
    def predict(self, data):
        return self.nodeActivation(self.nodeInput(data))
    
    def train(self, data, labels):
        # here, we are going to update the weights after each epoch
        for i in range(self.epochs):
            # instead of the predict() function here we use the nodeActivation()'s
            # output. For this case they are equivalent but when we use other 
            # training rules, this distinction will be important
            y = self.nodeActivation(self.nodeInput(data))
            delta_w = (labels - y).T.dot(data.reshape(-1,2))
            self.weights += self.eta*delta_w
            self.biases += self.eta*(labels - y).sum()
        
        return self    
    
    def trainMiniBatch(self, data, labels, miniBatchSize = 1):
        # here, we are going to update the weights after each minibatch
        for i in range(self.epochs):
            # for each epoch create the minibatches
            dataMiniBatches, labelMiniBatches = self.getMiniBatch(data, labels, miniBatchSize) 
            
            # loop through all the minibatches
            for dataMiniBatch, labelMiniBatch in zip(dataMiniBatches, labelMiniBatches):
                # and for each minibatch, train the neuron as usual
                self.train(dataMiniBatch, labelMiniBatch)
                
        return self
    
    def getMiniBatch(self, data, labels, miniBatchSize):
        # create a random permutation of data abd labels
        idx = np.random.permutation(len(data))
        data,labels = data[idx], labels[idx]
        
        # create a list of mini data batches of size batchSize
        dataMiniBatches = [data[k:k+miniBatchSize]                            for k in range(0, len(data), miniBatchSize)]
        # create a list of mini label batches of size batchSize
        labelMiniBatches = [labels[k:k+miniBatchSize]                             for k in range(0, len(labels), miniBatchSize)]
        
        return dataMiniBatches, labelMiniBatches    

nn = Perceptron(nDims=2, epochs=30)

nn.trainMiniBatch(data,labels)
plotSeperatingPLane(nn.predict, data)

nn = Perceptron(nDims=2, epochs=30)
nn.train(data,labels)
plotSeperatingPLane(nn.predict, data)

class DeltaNeuron(object):
    
    def __init__(self, nDims, eta=0.005, epochs=10):
        self.eta = eta
        self.epochs = epochs
        self.weights = np.random.normal(0, 1, nDims)[np.newaxis] # 1*2
        self.biases = np.random.rand()
    
    # same as the perceptron
    def nodeInput(self, data):
        # this reshaping is done so that the sizes match
        # independent of the number of input samples
        data = data.reshape(-1, self.weights.shape[1])
        # the usual multiplication and bias addition
        z = data.dot(self.weights.T) + self.biases*np.ones((len(data),1)) 
        return z
    
    def nodeActivation(self, z):
        # activation for our delta neuron is just the input value itself
        return z        
    
    # We still need to get some label predictions using the activation results
    # and the obvious choice again is the unit step function
    def predict(self, data):
        y = self.nodeActivation(self.nodeInput(data))
        return np.where(y >= 0, 1, -1)
    
    def train(self, data, labels):
        # here, we are going to update the weights after each epoch
        for i in range(self.epochs):
            # instead of the predict() function here we use the nodeActivation()'s
            # output. For this case they are equivalent but when we use other 
            # training rules, this distinction will be important
            y = self.nodeActivation(self.nodeInput(data))
            delta_w = (labels - y).T.dot(data.reshape(-1,2))
            self.weights += self.eta*delta_w
            self.biases += self.eta*(labels - y).sum()
        
        return self    
    
    def trainMiniBatch(self, data, labels, miniBatchSize = 1):
        # here, we are going to update the weights after each minibatch
        for i in range(self.epochs):
            # for each epoch create the minibatches
            dataMiniBatches, labelMiniBatches = self.getMiniBatch(data, labels, miniBatchSize) 
            
            # loop through all the minibatches
            for dataMiniBatch, labelMiniBatch in zip(dataMiniBatches, labelMiniBatches):
                # and for each minibatch, train the neuron as usual
                self.train(dataMiniBatch, labelMiniBatch)
                
        return self
    
    def getMiniBatch(self, data, labels, miniBatchSize):
        # create a random permutation of data abd labels
        idx = np.random.permutation(len(data))
        data,labels = data[idx], labels[idx]
        
        # create a list of mini data batches of size batchSize
        dataMiniBatches = [data[k:k+miniBatchSize]                            for k in range(0, len(data), miniBatchSize)]
        # create a list of mini label batches of size batchSize
        labelMiniBatches = [labels[k:k+miniBatchSize]                             for k in range(0, len(labels), miniBatchSize)]
        
        return dataMiniBatches, labelMiniBatches

#sigmoid implementation
def sigma(x):
    #everything is a "tensor", even the constants
    return tf.div(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))
#derivative of the sigmoid implementation
def dSigma(x):
    return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))

#placeholders for input and output variables
y = tf.placeholder(tf.float32, [None,1])
x = tf.placeholder(tf.float32, [None,1])

#our input-output relation is just a sigmoid
y = sigma(x)

#create some input values as a numpy array
input_series = np.linspace(-10,10,200)[:,np.newaxis]

#evaluations are run in a tf session
sess= tf.Session()
#put the input array into the x placeholder and evaluate y
output = sess.run(y, feed_dict={x: input_series})

plt.plot(input_series, output)

dataDim = 2
nClasses = 2
hiddenLayerSize = 3

#input tensor
a_0 = tf.placeholder(tf.float32, [None, dataDim])
#output tensor
y = tf.placeholder(tf.float32, [None, nClasses])

#weights and biases
w_1 = tf.Variable(tf.truncated_normal([dataDim, hiddenLayerSize]))
b_1 = tf.Variable(tf.truncated_normal([1, hiddenLayerSize]))
w_2 = tf.Variable(tf.truncated_normal([hiddenLayerSize, nClasses]))
b_2 = tf.Variable(tf.truncated_normal([1, nClasses]))

#the forward pass
z_1 = tf.add(tf.matmul(a_0, w_1), b_1) #inputs*hiddenLayerSize
a_1 = sigma(z_1) #inputs*hiddenLayerSize
z_2 = tf.add(tf.matmul(a_1, w_2), b_2) #inputs*nClasses
a_2 = sigma(z_2) #inputs*nClasses

cost_gradient = tf.subtract(a_2, y) #inputs*nClasses

delta_2 = tf.multiply(cost_gradient, dSigma(z_2)) #inputs*nClasses
grad_b_2 = delta_2
grad_w_2 = tf.matmul(tf.transpose(a_1), delta_2) #hiddenLayerSize*nClasses

delta_1 = tf.multiply(tf.matmul(delta_2, tf.transpose(w_2))                       , dSigma(z_1)) #inputs*hiddenLayerSize
grad_b_1 = delta_1 #inputs*hiddenLayerSize
grad_w_1 = tf.matmul(tf.transpose(a_0), delta_1) #dataDim*hiddenLayerSize

eta = tf.constant(0.05)
step = [
    tf.assign(w_1,
            tf.subtract(w_1, tf.multiply(eta, grad_w_1)))
  , tf.assign(b_1,
            tf.subtract(b_1, tf.multiply(eta,
                               tf.reduce_mean(grad_b_1, axis=[0]))))
  , tf.assign(w_2,
            tf.subtract(w_2, tf.multiply(eta, grad_w_2)))
  , tf.assign(b_2,
            tf.subtract(b_2, tf.multiply(eta,
                               tf.reduce_mean(grad_b_2, axis=[0]))))
]

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)
enc_labels = enc.fit_transform(labels[:,np.newaxis])

def nnpredictor(data):
    return np.argmax(sess.run(a_2, feed_dict = {a_0: data}),axis=1)

acct_mat = tf.equal(tf.argmax(a_2, 1), tf.argmax(y, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(20):
    sess.run(step, feed_dict = {a_0: data,
                            y : enc_labels})
plotSeperatingPLane(nnpredictor, data)
plt.title("NN")





