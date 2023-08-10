import pickle
get_ipython().magic('run NeuralNet2.ipynb')
train_x = pickle.load(open("MNIST_train_x.pkl", 'rb'))
train_y = pickle.load(open("MNIST_train_y.pkl", 'rb'))
test_x = pickle.load(open("MNIST_test_x.pkl", 'rb'))
test_y = pickle.load(open("MNIST_test_y.pkl", 'rb'))
short_train_x = train_x[0:5000,:]
short_train_y = train_y[0:5000,:]

net2 = NeuralNet((784,100,10), QuadraticCost, SigmoidActivation, SigmoidActivation)
net2.initialize_variables()
learning_rate = 1
epochs = 61
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = NeuralNet((784,100,10), QuadraticCost, SigmoidActivation, SigmoidActivation)
net2.initialize_variables()
learning_rate = 0.1
epochs = 61
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = NeuralNet((784,100,10), QuadraticCost, SigmoidActivation, SigmoidActivation)
net2.initialize_variables()
learning_rate = 0.01
epochs = 61
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = NeuralNet((784,100,10), QuadraticCost, SigmoidActivation, SigmoidActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 61
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = NeuralNet((784,100,10), CrossEntropyCost, SigmoidActivation, SigmoidActivation)
net2.initialize_variables()
learning_rate = 1
epochs = 61
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = NeuralNet((784,100,10), CrossEntropyCost, SigmoidActivation, SigmoidActivation)
net2.initialize_variables()
learning_rate = 0.1
epochs = 61
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = NeuralNet((784,100,10), CrossEntropyCost, SigmoidActivation, SigmoidActivation)
net2.initialize_variables()
learning_rate = 0.01
epochs = 61
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = NeuralNet((784,100,10), CrossEntropyCost, SigmoidActivation, SigmoidActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 61
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = NeuralNet((784,100,10), LogLikelihoodCost, SigmoidActivation, SoftmaxActivation)
net2.initialize_variables()
learning_rate = 1
epochs = 61
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = NeuralNet((784,100,10), LogLikelihoodCost, SigmoidActivation, SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.1
epochs = 61
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = NeuralNet((784,100,10), LogLikelihoodCost, SigmoidActivation, SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.01
epochs = 61
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = NeuralNet((784,100,10), LogLikelihoodCost, SigmoidActivation, SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 61
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = NeuralNet((784,100,10), QuadraticCost, SigmoidActivation, SigmoidActivation)
net2.initialize_variables()
learning_rate = 0.05
epochs = 61
reporting_rate = 20
lmda = 0
batch_size = 100
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = NeuralNet((784,100,10), QuadraticCost, SigmoidActivation, SigmoidActivation)
net2.initialize_variables()
learning_rate = 0.03
epochs = 61
reporting_rate = 20
lmda = 0
batch_size = 50
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = NeuralNet((784,100,10), QuadraticCost, SigmoidActivation, SigmoidActivation)
net2.initialize_variables()
learning_rate = 0.01
epochs = 61
reporting_rate = 20
lmda = 0
batch_size = 25
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

