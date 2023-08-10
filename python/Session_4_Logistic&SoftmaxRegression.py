from IPython.display import Latex
from IPython.display import Math
import sys
import numpy
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))

def softmax(x):
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:  
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2

class LogisticRegression(object):
    def __init__(self, input, label, n_in, n_out):
        self.x = input
        self.y = label
        self.Weights = numpy.zeros((n_in, n_out))  # initialize W 0
        self.biases = numpy.zeros(n_out)          # initialize bias 0

        # self.params = [self.Weights, self.biases]

    def train(self, lr=0.1, input=None, L2_regularization=0.00):
        if input is not None:
            self.x = input

        # p_y_given_x = sigmoid(numpy.dot(self.x, self.Weights) + self.biases)
        p_y_given_x = softmax(numpy.dot(self.x, self.Weights) + self.biases)
        d_y = self.y - p_y_given_x
        
        self.Weights += lr * numpy.dot(self.x.T, d_y) 
        - lr * L2_regularization * self.Weights
        self.biases += lr * numpy.mean(d_y, axis=0)
        
        # cost = self.negative_log_likelihood()
        # return cost

    def negative_log_likelihood(self):
        # sigmoid_activation = sigmoid(numpy.dot(self.x, self.Weights) + self.biases)
        sigmoid_activation = softmax(numpy.dot(self.x, self.Weights) + self.biases)

        cross_entropy = - numpy.mean(
            numpy.sum(self.y * numpy.log(sigmoid_activation) +
            (1 - self.y) * numpy.log(1 - sigmoid_activation),
                      axis=1))

        return cross_entropy


    def predict(self, x):
        # return sigmoid(numpy.dot(x, self.Weights) + self.biases)
        return softmax(numpy.dot(x, self.Weights) + self.biases)

costlist=list()
def test_lr(learning_rate=0.01, n_epochs=2000):
    # training data
    x = numpy.array([[1,1,1,0,0,0],
                     [1,0,1,0,0,0],
                     [1,1,1,0,0,0],
                     [0,0,1,1,1,0],
                     [0,0,1,1,0,0],
                     [0,0,1,1,1,0]])
    y = numpy.array([[1, 0],
                     [1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 1],
                     [0, 1]])


    # construct LogisticRegression
    classifier = LogisticRegression(input=x, label=y, n_in=6, n_out=2)

    # train
    for epoch in range(n_epochs):
        classifier.train(lr=learning_rate)
        cost = classifier.negative_log_likelihood()
        costlist.append(cost)
        print ( 'Training epoch %d, cost is ' % epoch, cost)
        #learning_rate *= 0.95


    # test
    x = numpy.array([1, 1, 0, 0, 0, 0])
    print ( classifier.predict(x))

if __name__ == "__main__":
    test_lr()

plt.plot(costlist)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()



