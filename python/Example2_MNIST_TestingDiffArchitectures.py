import pickle
import NeuralNetwork as nn

train_x = pickle.load(open("MNIST_train_x.pkl", 'rb'))
train_y = pickle.load(open("MNIST_train_y.pkl", 'rb'))
test_x = pickle.load(open("MNIST_test_x.pkl", 'rb'))
test_y = pickle.load(open("MNIST_test_y.pkl", 'rb'))
short_train_x = train_x[0:5000,:]
short_train_y = train_y[0:5000,:]

net2 = nn.NeuralNet((784,100,10), nn.LogLikelihoodCost, nn.ReluActivation, nn.SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 101
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = nn.NeuralNet((784,50,30,10), nn.LogLikelihoodCost, nn.ReluActivation, nn.SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 101
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)
print()

net2 = nn.NeuralNet((784,50,30,10), nn.LogLikelihoodCost, nn.ReluActivation, nn.SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 101
reporting_rate = 20
lmda = 0.5
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = nn.NeuralNet((784,50,50,10), nn.LogLikelihoodCost, nn.ReluActivation, nn.SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 101
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)
print()

net2 = nn.NeuralNet((784,50,50,10), nn.LogLikelihoodCost, nn.ReluActivation, nn.SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 101
reporting_rate = 20
lmda = 0.5
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = nn.NeuralNet((784,70,50,10), nn.LogLikelihoodCost, nn.ReluActivation, nn.SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 101
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)
print()

net2 = nn.NeuralNet((784,70,50,10), nn.LogLikelihoodCost, nn.ReluActivation, nn.SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 101
reporting_rate = 20
lmda = 1
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = nn.NeuralNet((784,100,60,10), nn.LogLikelihoodCost, nn.ReluActivation, nn.SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 101
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)
print()

net2 = nn.NeuralNet((784,100,60,10), nn.LogLikelihoodCost, nn.ReluActivation, nn.SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 101
reporting_rate = 20
lmda = 1
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = nn.NeuralNet((784,20,20,20,10), nn.LogLikelihoodCost, nn.ReluActivation, nn.SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 101
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)
print()

net2 = nn.NeuralNet((784,20,20,20,10), nn.LogLikelihoodCost, nn.ReluActivation, nn.SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 101
reporting_rate = 20
lmda = 0.5
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = nn.NeuralNet((784,30,30,20,10), nn.LogLikelihoodCost, nn.ReluActivation, nn.SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 101
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)
print()

net2 = nn.NeuralNet((784,30,30,20,10), nn.LogLikelihoodCost, nn.ReluActivation, nn.SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 101
reporting_rate = 20
lmda = 0.5
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = nn.NeuralNet((784,40,40,20,10), nn.LogLikelihoodCost, nn.ReluActivation, nn.SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 101
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)
print()

net2 = nn.NeuralNet((784,40,40,20,10), nn.LogLikelihoodCost, nn.ReluActivation, nn.SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 101
reporting_rate = 20
lmda = 0.5
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = nn.NeuralNet((784,80,60,40,10), nn.LogLikelihoodCost, nn.ReluActivation, nn.SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 101
reporting_rate = 20
lmda = 0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)
print()

net2 = nn.NeuralNet((784,80,60,40,10), nn.LogLikelihoodCost, nn.ReluActivation, nn.SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 101
reporting_rate = 20
lmda = 0.5
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

net2 = nn.NeuralNet((784,60,40,20,10), nn.LogLikelihoodCost, nn.ReluActivation, nn.SoftmaxActivation)
net2.initialize_variables()
learning_rate = 0.001
epochs = 101
reporting_rate = 20
lmda = 2.0
batch_size = 200
training_cost, valid_cost = net2.SGD(short_train_x, short_train_y, test_x, test_y, learning_rate,         epochs, reporting_rate, lmda, batch_size, verbose=False)

