#Likhit Teja Valavala  -  2015A3PS0221P
#Shikhar Shiromani     -  2015A3PS0194P
#Pratyush Priyank      -  2015A3PS0188P

get_ipython().run_cell_magic('cmd', '', '\npip install jdc')

# Library imports
import random
import numpy as np
import jdc
import sklearn
from datasets import *

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import scipy

def pca(x,k):
    covar_x = np.dot(x.T,x)/x.shape[0]
    [U,S,V] = scipy.linalg.svd(covar_x)
    Z = np.dot(x,U[:,0:k])
    return Z

training_data = []
names = ["_jackson_","_theo_","_jason_"]
for k in range(0,2):
    for i in range(0,10):
        for j in range(0,40):
            string = str(i)+names[k]+str(j)+".wav"
            (rate,sig) = wav.read(string)
            mfcc_feat = mfcc(sig,rate,winlen=0.025,winstep=0.01,numcep=13,nfilt=26,nfft=1103,lowfreq=0,highfreq=None,preemph=0.97,
                    ceplifter=22,appendEnergy=True)  
            z = pca(mfcc_feat.T,1)
            training_data.append(z)
print(len(training_data))

outputs = []
for k in range(0,2):
    for i in range(0,10):
        temp = [0]*10
        temp[i] = 1
        temp = np.array([temp]).T
        for j in range(0,40):
            outputs.append(temp)
print(len(outputs))

test_data = []
names = ["_jackson_","_theo_","_jason_"]
for k in range(0,2):
    for i in range(0,10):
        for j in range(40,50):
            string = str(i)+names[k]+str(j)+".wav"
            (rate,sig) = wav.read(string)
            mfcc_feat = mfcc(sig,rate,winlen=0.025,winstep=0.01,numcep=13,nfilt=26,nfft=1103,lowfreq=0,highfreq=None,preemph=0.97,
                    ceplifter=22,appendEnergy=True)  
            z = pca(mfcc_feat.T,1)
            test_data.append(z)
print(len(test_data))

test_outputs = []
for k in range(0,2):
    for i in range(0,10):
        temp = [0]*10
        temp[i] = 1
        temp = np.array([temp]).T
        for j in range(0,10):
            test_outputs.append(temp)
print(len(test_outputs))

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network. For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.initialize_biases()
        self.initialize_weights()

get_ipython().run_cell_magic('add_to', 'Network', 'def initialize_biases(self):\n    \n    self.biases = [np.random.uniform(-0.5,0.5,(self.sizes[b],1)) for b in range(1,self.num_layers)]\n    self.delta_b = [np.zeros((self.sizes[b],1)) for b in range(1,self.num_layers)]')

get_ipython().run_cell_magic('add_to', 'Network', 'def initialize_weights(self):\n    \n    self.weights = [np.random.uniform(-0.5,0.5,(self.sizes[b],self.sizes[b-1])) for b in range(1,self.num_layers)]\n    self.delta_w = [np.zeros((self.sizes[b],self.sizes[b-1])) for b in range(1,self.num_layers)]')

get_ipython().run_cell_magic('add_to', 'Network', 'def train(self, training_data, epochs, mini_batch_size, learning_rate,momentum):\n    """Train the neural network using gradient descent.  \n    ``training_data`` is a list of tuples ``(x, y)``\n    representing the training inputs and the desired\n    outputs.  The other parameters are self-explanatory."""\n\n    # training_data is a list and is passed by reference\n    # To prevernt affecting the original data we use \n    # this hack to create a copy of training_data\n    # https://stackoverflow.com/a/2612815\n    training_data = list(training_data)\n    \n    for i in range(epochs):\n        # Get mini-batches    \n        mini_batches = self.create_mini_batches(training_data, mini_batch_size)\n        \n        # Itterate over mini-batches to update pramaters   \n        cost = sum(map(lambda mini_batch: self.update_params(mini_batch, learning_rate,momentum), mini_batches))\n        \n        # Find accuracy of the model at the end of epoch         \n        acc = self.evaluate(training_data)\n        \n        if(i%100==0):\n            print("Epoch {} complete. Total Accuracy: {}".format(i,acc))')

get_ipython().run_cell_magic('add_to', 'Network', 'def create_mini_batches(self, training_data, mini_batch_size):\n    # Shuffling data helps a lot in mini-batch SGD\n    random.shuffle(training_data)\n    # YOUR CODE HERE\n    mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,len(training_data),mini_batch_size)]\n    return mini_batches')

get_ipython().run_cell_magic('add_to', 'Network', 'def update_params(self, mini_batch, learning_rate,momentum):\n    """Update the network\'s weights and biases by applying\n    gradient descent using backpropagation."""\n    #print(mini_batch)\n    # Initialize gradients     \n    delta_b = [np.zeros(b.shape) for b in self.biases]\n    delta_w = [np.zeros(w.shape) for w in self.weights]\n    \n    total_cost = 0\n    \n    for x,y in mini_batch:\n        # Obtain the mean squared error and the gradients\n        # with resepect to biases and weights \n        \n        cost, del_b, del_w = self.backprop(x, y)\n        \n        # Add the gradients for each sample in mini-batch        \n        delta_b = [nb + dnb for nb, dnb in zip(delta_b, del_b)]\n        delta_w = [nw + dnw for nw, dnw in zip(delta_w, del_w)]\n        \n        total_cost += cost\n\n    # Update self.biases and self.weights\n    # using delta_b, delta_w and learning_rate \n    #Momentum\n    self.delta_b = [(learning_rate*delta + momentum*db) for delta,db in zip(delta_b,self.delta_b)]\n    self.biases = [b - (1 / len(mini_batch)) * db\n                   for b, db in zip(self.biases, self.delta_b)]\n    self.delta_w = [(learning_rate*delta + momentum*dw) for delta,dw in zip(delta_w,self.delta_w)]\n    self.weights = [w - (1 / len(mini_batch)) * dw\n                    for w, dw in zip(self.weights, self.delta_w)]\n\n    return total_cost')

get_ipython().run_cell_magic('add_to', 'Network', 'def backprop(self, x, y):\n    """Return arry containiing cost, del_b, del_w representing the\n    cost function C(x) and gradient for cost function.  ``del_b`` and\n    ``del_w`` are layer-by-layer lists of numpy arrays, similar\n    to ``self.biases`` and ``self.weights``."""\n    # Forward pass\n    zs, activations = self.forward(x)\n    \n    # Backward pass     \n    cost, del_b, del_w = self.backward(activations, zs, y)\n\n    return cost, del_b, del_w')

get_ipython().run_cell_magic('add_to', 'Network', 'def sigmoid(self, z):\n    """The sigmoid function."""\n    # YOUR CODE HERE\n    return 1/(1+np.exp(-z))')

get_ipython().run_cell_magic('add_to', 'Network', 'def sigmoid_derivative(self, z):\n    """Derivative of the sigmoid function."""\n    # YOUR CODE HERE\n    return self.sigmoid(z)*(1-self.sigmoid(z))')

get_ipython().run_cell_magic('add_to', 'Network', 'def forward(self, x):\n    """Compute Z and activation for each layer."""\n    \n    # list to store all the activations, layer by layer\n    zs = []\n    \n    # current activation\n    activation = x\n    # list to store all the activations, layer by layer\n    activations = [x]\n    \n    # Loop through each layer to compute activations and Zs  \n    for b, w in zip(self.biases, self.weights):\n        # YOUR CODE HERE\n        # Calculate z\n        # watch out for the dimensions of multiplying matrices\n        #print(w)\n        #print(activations[-1])\n        z = np.dot(w,activations[-1])+b\n        zs.append(z)\n        # Calculate activation\n        activation = self.sigmoid(z)\n        activations.append(activation)\n        \n    return zs, activations')

get_ipython().run_cell_magic('add_to', 'Network', 'def lre(self, output_activations, y):\n    """Returns mean square error."""\n    return -(y*np.log(output_activations) + (1-y)*np.log(1-output_activations))')

get_ipython().run_cell_magic('add_to', 'Network', 'def lre_derivative(self, output_activations, y):\n    """Return the vector of partial derivatives \\partial C_x /\n    \\partial a for the output activations. """\n    return -(y/output_activations - (1-y)/(1-output_activations))')

get_ipython().run_cell_magic('add_to', 'Network', 'def backward(self, activations, zs, y):\n    """Compute and return cost funcation, gradients for \n    weights and biases for each layer."""\n    # Initialize gradient arrays\n    del_b = [np.zeros(b.shape) for b in self.biases]\n    del_w = [np.zeros(w.shape) for w in self.weights]\n    \n    # Compute for last layer\n    cost = self.lre(activations[-1], y)\n    \n    delta = self.lre_derivative(activations[-1],y)*self.sigmoid_derivative(zs[-1])\n    #print(delta.shape)\n    del_b[-1] = delta\n    del_w[-1] = np.dot(delta, activations[-2].transpose())\n    #print(del_w[-1].shape)\n    \n    # Loop through each layer in reverse direction to \n    # populate del_b and del_w   \n    for l in range(2, self.num_layers):\n        #print(delta.shape);print(self.sigmoid_derivative(activations[-l]).shape); print(np.dot(self.weights[-l+1].T,delta).shape)\n        delta = np.dot(self.weights[-l+1].T,delta)*self.sigmoid_derivative(zs[-l])\n        #print(delta.shape)\n        del_b[-l] = delta\n        del_w[-l] = np.dot(delta, activations[-l -1].transpose())\n        #print(del_w[-l].shape)\n    \n    return cost, del_b, del_w')

get_ipython().run_cell_magic('add_to', 'Network', 'def evaluate(self, test_data):\n    """Return the accuracy of Network. Note that the neural\n    network\'s output is assumed to be the index of whichever\n    neuron in the final layer has the highest activation."""\n    test_results = [(np.argmax(self.forward(x)[1][-1]), np.argmax(y))\n                    for (x, y) in test_data]\n    return sum(int(x == y) for (x, y) in test_results) * 100 / len(test_results)')

training_data = [sklearn.preprocessing.normalize(a) for a in training_data]
data = list(zip(training_data,outputs))
#print(data)
network = Network([13, 11,8, 10])
network.train(data,3001,100,1,0.2)
#network.evaluate(list(zip(test_data,test_outputs)))
predictions = list(map(lambda sample: np.argmax(network.forward(sample)[1][-1]), test_data))
#print(predictions)

network.evaluate(list(zip(test_data,test_outputs)))



