import pickle
import NeuralNetwork as nn

train_x = pickle.load( open( "nn_donutballdata_train_x.pkl", "rb" ) )
train_y = pickle.load( open( "nn_donutballdata_train_y.pkl", "rb" ) )
test_x = pickle.load( open( "nn_donutballdata_test_x.pkl", "rb" ) )
test_y = pickle.load( open( "nn_donutballdata_test_y.pkl", "rb" ) )

net = nn.NeuralNet((2,2,1), nn.QuadraticCost, nn.SigmoidActivation, nn.SigmoidActivation)
net.initialize_variables()
learning_rate = 0.1
batch_size = 10
lmda = 0
epochs = 2001
reporting_rate = 200
print("Results for (2,2,1) neural network with quadratic cost and sigmoid activations")
training_cost, valid_cost = net.SGD(train_x, train_y, test_x, test_y, learning_rate, epochs, 
                                    reporting_rate, lmda, batch_size)
print()

net = nn.NeuralNet((2,2,1), nn.CrossEntropyCost, nn.SigmoidActivation, nn.SigmoidActivation)
net.initialize_variables()
learning_rate = 0.1
batch_size = 10
lmda = 0
epochs = 2001
reporting_rate = 200
print("Results for (2,2,1) neural network with cross entropy cost and sigmoid activations")
training_cost, valid_cost = net.SGD(train_x, train_y, test_x, test_y, learning_rate, epochs, 
                                    reporting_rate, lmda, batch_size)

net = nn.NeuralNet((2,3,1), nn.QuadraticCost, nn.SigmoidActivation, nn.SigmoidActivation)
net.initialize_variables()
learning_rate = 0.1
batch_size = 10
lmda = 0
epochs = 1001
reporting_rate = 200
print("Results for (2,3,1) neural network with quadratic cost and sigmoid activations")
training_cost, valid_cost = net.SGD(train_x, train_y, test_x, test_y, learning_rate, epochs, 
                                    reporting_rate, lmda, batch_size)
print()

net = nn.NeuralNet((2,3,1), nn.CrossEntropyCost, nn.SigmoidActivation, nn.SigmoidActivation)
net.initialize_variables()
learning_rate = 0.1
batch_size = 10
lmda = 0
epochs = 1001
reporting_rate = 200
print("Results for (2,3,1) neural network with cross entropy cost and sigmoid activations")
training_cost, valid_cost = net.SGD(train_x, train_y, test_x, test_y, learning_rate, epochs, 
                                    reporting_rate, lmda, batch_size)

