import numpy as np
import numpy.random as nr
import matplotlib.pyplot as pl
get_ipython().run_line_magic('matplotlib', 'inline')

class Perceptron:
    def __init__(self,X,Y,eta,X_test):
        self.w=np.random.uniform(1,2,X.shape[1]+1) #one weight for the bias
        self.x=X
        self.y=Y
        self.e=eta
        Perceptron.Train(self)
        Perceptron.Test(self,X_test)
    
    def Train(self):
        x_fin=np.array([np.ones(self.x.shape[0])])
        self.x=np.concatenate((self.x,np.transpose(x_fin)),axis=1)
        for i in range(100):
            y_pred=np.dot(self.x,np.transpose(self.w))
            y_pred[y_pred>=0]=1
            y_pred[y_pred<0]=-1
            err=self.y-y_pred
            for j in range(3):
                self.w=self.w + self.e*err[j]*self.x[j]

    
    def Test(self,X_test):
        x_bias=np.array([np.ones(X_test.shape[0])])
        X_test=np.concatenate((X_test,np.transpose(x_bias)),axis=1)
        y_test_pred=np.dot(X_test,np.transpose(self.w))
        y_test_pred[y_test_pred>=0]=1
        y_test_pred[y_test_pred<0]=-1
        print(y_test_pred)
        

X=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
Y=np.array([-1,-1,-1,1])
X_test=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
Perceptron(X,Y,0.01,X_test)


#Input array
X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1],[0,0,0,1]])

#Output
y=np.array([[0],[1],[0],[1]])

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

#Variable initialization
epoch=5000 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = X.shape[1] #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
np.random.seed(420)
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):

    #Forward Propogation
    hidden_layer_input1=np.dot(X,wh)
    hidden_layer_input=hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1=np.dot(hiddenlayer_activations,wout)
    output_layer_input= output_layer_input1+ bout
    output = sigmoid(output_layer_input)

    #Backpropagation

    Error = y - output
    
    # Slopes are found using derivatives of its output
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    
    # Found using multiplication of Error and ____
    delta_output = Error * slope_output_layer
    
    
    Error_at_hidden_layer = delta_output.dot(wout.T)
    
    # Similar to delta_output
    delta_hidden_layer = Error_at_hidden_layer * slope_hidden_layer
    
    # Weight and Bias Update
    wout += hiddenlayer_activations.T.dot(delta_output)*lr
    bout += np.sum(delta_output, axis=0,keepdims=True) *lr
    wh += X.T.dot(delta_hidden_layer)*lr
    bh += np.sum(delta_hidden_layer, axis=0,keepdims=True) *lr

print (output)
 



