import numpy as np
import scipy
import scipy.misc
import scipy.ndimage
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

import resource


np.set_printoptions(suppress=True, precision=5)



get_ipython().magic('matplotlib inline')

class Laptimer:    
    def __init__(self):
        self.start = datetime.now()
        self.lap = 0
        
    def click(self, message):
        td = datetime.now() - self.start
        td = (td.days*86400000 + td.seconds*1000 + td.microseconds / 1000) / 1000
        memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
        print("[%d] %s, %.2fs, memory: %dmb" % (self.lap, message, td, memory))
        self.start = datetime.now()
        self.lap = self.lap + 1
        return td
        
    def reset(self):
        self.__init__()
    
    def __call__(self, message = None):
        return self.click(message)
        
timer = Laptimer()
timer()

def normalize_fetures(X):
    return X * 0.98 / 255 + 0.01

def normalize_labels(y):
    y = OneHotEncoder(sparse=False).fit_transform(y)
    y[y == 0] = 0.01
    y[y == 1] = 0.99
    return y

url = "https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/mnist_train_100.csv"
train = pd.read_csv(url, header=None, dtype="float64")
train.sample(10)

X_train = normalize_fetures(train.iloc[:, 1:].values)
y_train = train.iloc[:, [0]].values.astype("int32")
y_train_ohe = normalize_labels(y_train)

fig, _ = plt.subplots(5, 6, figsize = (15, 10))
for i, ax in enumerate(fig.axes):
    ax.imshow(X_train[i].reshape(28, 28), cmap="Greys", interpolation="none")
    ax.set_title("T: %d" % y_train[i])

plt.tight_layout()

url = "https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/mnist_test_10.csv"
test = pd.read_csv(url, header=None, dtype="float64")
test.sample(10)

X_test = normalize_fetures(test.iloc[:, 1:].values)
y_test = test.iloc[:, 0].values.astype("int32")

class NeuralNetwork:

    def __init__(self, layers, learning_rate, random_state = None):
        self.layers_ = layers
        self.num_features = layers[0]
        self.num_classes = layers[-1]
        self.hidden = layers[1:-1]
        self.learning_rate = learning_rate
        
        if not random_state:
            np.random.seed(random_state)
        
        self.W_sets = []
        for i in range(len(self.layers_) - 1):
            n_prev = layers[i]
            n_next = layers[i + 1]
            m = np.random.normal(0.0, pow(n_next, -0.5), (n_next, n_prev))
            self.W_sets.append(m)
    
    def activation_function(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, training, targets):
        inputs0 = inputs = np.array(training, ndmin=2).T
        assert inputs.shape[0] == self.num_features,                 "no of features {0}, it must be {1}".format(inputs.shape[0], self.num_features)

        targets = np.array(targets, ndmin=2).T
        
        assert targets.shape[0] == self.num_classes,                 "no of classes {0}, it must be {1}".format(targets.shape[0], self.num_classes)

        
        outputs  = []
        for i in range(len(self.layers_) - 1):
            W = self.W_sets[i]
            inputs = self.activation_function(W.dot(inputs))
            outputs.append(inputs)
        
        errors = [None] * (len(self.layers_) - 1)
        errors[-1] = targets - outputs[-1]
        #print("Last layer", targets.shape, outputs[-1].shape, errors[-1].shape)
        #print("Last layer", targets, outputs[-1])
        
        #Back propagation
        for i in range(len(self.layers_) - 1)[::-1]:
            W = self.W_sets[i]
            E = errors[i]
            O = outputs[i] 
            I = outputs[i - 1] if i > 0 else inputs0
            #print("i: ", i, ", E: ", E.shape, ", O:", O.shape, ", I: ", I.shape, ",W: ", W.shape)
            W += self.learning_rate * (E * O * (1 - O)).dot(I.T)
            if i > 0:
                errors[i-1] = W.T.dot(E)
        
    
    def predict(self, inputs, cls = False):
        inputs = np.array(inputs, ndmin=2).T        
        assert inputs.shape[0] == self.num_features,                 "no of features {0}, it must be {1}".format(inputs.shape[0], self.num_features) 
        
        for i in range(len(self.layers_) - 1):
            W = self.W_sets[i]
            input_next = W.dot(inputs)
            inputs = activated = self.activation_function(input_next)
            
            
        return np.argmax(activated.T, axis=1) if cls else activated.T 
    
    def score(self, X_test, y_test):
        y_test = np.array(y_test).flatten()
        y_test_pred = nn.predict(X_test, cls=True)
        return np.sum(y_test_pred == y_test) / y_test.shape[0]



nn = NeuralNetwork([784,100,10], 0.3, random_state=0)
for i in np.arange(X_train.shape[0]):
    nn.fit(X_train[i], y_train_ohe[i])
    
nn.predict(X_train[2]), nn.predict(X_train[2], cls=True)
print("Testing accuracy: ", nn.score(X_test, y_test), ", training accuracy: ", nn.score(X_train, y_train))
#list(zip(y_test_pred, y_test))

train = pd.read_csv("../data/MNIST/mnist_train.csv", header=None, dtype="float64")
X_train = normalize_fetures(train.iloc[:, 1:].values)
y_train = train.iloc[:, [0]].values.astype("int32")
y_train_ohe = normalize_labels(y_train)
print(y_train.shape, y_train_ohe.shape)

test = pd.read_csv("../data/MNIST/mnist_test.csv", header=None, dtype="float64")
X_test = normalize_fetures(test.iloc[:, 1:].values)
y_test = test.iloc[:, 0].values.astype("int32")

timer.reset()
nn = NeuralNetwork([784,100,10], 0.3, random_state=0)
for i in range(X_train.shape[0]):
    nn.fit(X_train[i], y_train_ohe[i])
timer("training time")
accuracy = nn.score(X_test, y_test)
print("Testing accuracy: ", nn.score(X_test, y_test), ", Training accuracy: ", nn.score(X_train, y_train))

params = 10 ** - np.linspace(0.01, 2, 10)
scores_train = []
scores_test = []

timer.reset()
for p in params:
    nn = NeuralNetwork([784,100,10], p, random_state = 0)
    for i in range(X_train.shape[0]):
        nn.fit(X_train[i], y_train_ohe[i])
    scores_train.append(nn.score(X_train, y_train))
    scores_test.append(nn.score(X_test, y_test))
    timer()
    
plt.plot(params, scores_test, label = "Test score")
plt.plot(params, scores_train, label = "Training score")
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Effect of learning rate")

print("Accuracy scores")
pd.DataFrame({"learning_rate": params, "train": scores_train, "test": scores_test})

epochs = np.arange(20)
learning_rate = 0.077
scores_train, scores_test = [], []
nn = NeuralNetwork([784,100,10], learning_rate, random_state = 0)
indices = np.arange(X_train.shape[0])

timer.reset()
for _ in epochs:
    np.random.shuffle(indices)
    for i in indices:
        nn.fit(X_train[i], y_train_ohe[i])
    scores_train.append(nn.score(X_train, y_train))
    scores_test.append(nn.score(X_test, y_test))
    timer("test score: %f, training score: %f" % (scores_test[-1], scores_train[-1]))

plt.plot(epochs, scores_test, label = "Test score")
plt.plot(epochs, scores_train, label = "Training score")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc = "lower right")
plt.title("Effect of Epochs")

print("Accuracy scores")
pd.DataFrame({"epochs": epochs, "train": scores_train, "test": scores_test})

num_layers = 50 * (np.arange(10) + 1)
learning_rate = 0.077
scores_train, scores_test = [], []

timer.reset()
for p in num_layers:
    nn = NeuralNetwork([784, p,10], learning_rate, random_state = 0)
    indices = np.arange(X_train.shape[0])
    for i in indices:
        nn.fit(X_train[i], y_train_ohe[i])
    scores_train.append(nn.score(X_train, y_train))
    scores_test.append(nn.score(X_test, y_test))
    timer("size: %d, test score: %f, training score: %f" % (p, scores_test[-1], scores_train[-1]))

plt.plot(num_layers, scores_test, label = "Test score")
plt.plot(num_layers, scores_train, label = "Training score")
plt.xlabel("Hidden Layer Size")
plt.ylabel("Accuracy")
plt.legend(loc = "lower right")
plt.title("Effect of size (num of nodes) of the hidden layer")

print("Accuracy scores")
pd.DataFrame({"layer": num_layers, "train": scores_train, "test": scores_test})

num_layers = np.arange(5) + 1
learning_rate = 0.077
scores_train, scores_test = [], []

timer.reset()
for p in num_layers:
    layers = [100] * p
    layers.insert(0, 784)
    layers.append(10)
    
    nn = NeuralNetwork(layers, learning_rate, random_state = 0)
    indices = np.arange(X_train.shape[0])
    for i in indices:
        nn.fit(X_train[i], y_train_ohe[i])
    scores_train.append(nn.score(X_train, y_train))
    scores_test.append(nn.score(X_test, y_test))
    timer("size: %d, test score: %f, training score: %f" % (p, scores_test[-1], scores_train[-1]))

plt.plot(num_layers, scores_test, label = "Test score")
plt.plot(num_layers, scores_train, label = "Training score")
plt.xlabel("No of hidden layers")
plt.ylabel("Accuracy")
plt.legend(loc = "upper right")
plt.title("Effect of using multiple hidden layers, \nNodes per layer=100")

print("Accuracy scores")
pd.DataFrame({"layer": num_layers, "train": scores_train, "test": scores_test})

img = scipy.ndimage.interpolation.rotate(X_train[110].reshape(28, 28), -10, reshape=False)
print(img.shape)
plt.imshow(img, interpolation=None, cmap="Greys")

epochs = np.arange(10)
learning_rate = 0.077
scores_train, scores_test = [], []
nn = NeuralNetwork([784,250,10], learning_rate, random_state = 0)
indices = np.arange(X_train.shape[0])

timer.reset()
for _ in epochs:
    np.random.shuffle(indices)
    for i in indices:
        for rotation in [-10, 0, 10]:
            img = scipy.ndimage.interpolation.rotate(X_train[i].reshape(28, 28), rotation, cval=0.01, order=1, reshape=False)
            nn.fit(img.flatten(), y_train_ohe[i])
    scores_train.append(nn.score(X_train, y_train))
    scores_test.append(nn.score(X_test, y_test))
    timer("test score: %f, training score: %f" % (scores_test[-1], scores_train[-1]))

plt.plot(epochs, scores_test, label = "Test score")
plt.plot(epochs, scores_train, label = "Training score")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc = "lower right")
plt.title("Trained with rotation (+/- 10)\n Hidden Nodes: 250, LR: 0.077")

print("Accuracy scores")
pd.DataFrame({"epochs": epochs, "train": scores_train, "test": scores_test})

missed = y_test_pred != y_test
pd.Series(y_test[missed]).value_counts().plot(kind = "bar")
plt.title("No of mis classification by digit")
plt.ylabel("No of misclassification")
plt.xlabel("Digit")

fig, _ = plt.subplots(6, 4, figsize = (15, 10))
for i, ax in enumerate(fig.axes):
    ax.imshow(X_test[missed][i].reshape(28, 28), interpolation="nearest", cmap="Greys")
    ax.set_title("T: %d, P: %d" % (y_test[missed][i], y_test_pred[missed][i]))
plt.tight_layout()

img = scipy.ndimage.imread("/Users/abulbasar/Downloads/9-03.png", mode="L")
print("Original size:", img.shape)
img = normalize_fetures(scipy.misc.imresize(img, (28, 28)))
img = np.abs(img - 0.99)
plt.imshow(img, cmap="Greys", interpolation="none")
print("Predicted value: ", nn.predict(img.flatten(), cls=True))



