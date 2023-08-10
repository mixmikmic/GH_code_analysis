import pickle
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
get_ipython().magic('run NeuralNet2.ipynb')

train_x = pickle.load( open( "nn_donutballdata_train_x.pkl", "rb" ) )
train_y = pickle.load( open( "nn_donutballdata_train_y.pkl", "rb" ) )
test_x = pickle.load( open( "nn_donutballdata_test_x.pkl", "rb" ) )
test_y = pickle.load( open( "nn_donutballdata_test_y.pkl", "rb" ) )

net = NeuralNet((2,4,1), QuadraticCost, SigmoidActivation, SigmoidActivation)
net.initialize_variables()
learning_rate = 0.1
batch_size = 10
lmda = 0
epochs = 1001
reporting_rate = 200
training_cost, valid_cost =     net.SGD(train_x, train_y, test_x, test_y, learning_rate, epochs, reporting_rate, lmda, batch_size)

net = NeuralNet((2,4,1), QuadraticCost, SigmoidActivation, SigmoidActivation)
net.initialize_variables_normalized()
learning_rate = 0.1
batch_size = 10
lmda = 0
epochs = 1001
reporting_rate = 200
training_cost, valid_cost =     net.SGD(train_x, train_y, test_x, test_y, learning_rate, epochs, reporting_rate, lmda, batch_size)

net = NeuralNet((2,4,1), QuadraticCost, SigmoidActivation, SigmoidActivation)
net.initialize_variables_alt()
learning_rate = 0.1
batch_size = 10
lmda = 0
epochs = 1001
reporting_rate = 200
training_cost, valid_cost =     net.SGD(train_x, train_y, test_x, test_y, learning_rate, epochs, reporting_rate, lmda, batch_size)

net = NeuralNet((2,4,1), CrossEntropyCost, SigmoidActivation, SigmoidActivation)
net.initialize_variables()
learning_rate = 0.1
batch_size = 10
lmda = 0
epochs = 1001
reporting_rate = 200
training_cost, valid_cost =     net.SGD(train_x, train_y, test_x, test_y, learning_rate, epochs, reporting_rate, lmda, batch_size)

net = NeuralNet((2,4,1), CrossEntropyCost, SigmoidActivation, SigmoidActivation)
net.initialize_variables()
learning_rate = 0.1
batch_size = 50
lmda = 0
epochs = 1001
reporting_rate = 200
training_cost, valid_cost =     net.SGD(train_x, train_y, test_x, test_y, learning_rate, epochs, reporting_rate, lmda, batch_size)

net = NeuralNet((2,4,1), CrossEntropyCost, SigmoidActivation, SigmoidActivation)
net.initialize_variables()
learning_rate = 0.1
batch_size = 100
lmda = 0
epochs = 1001
reporting_rate = 200
training_cost, valid_cost =     net.SGD(train_x, train_y, test_x, test_y, learning_rate, epochs, reporting_rate, lmda, batch_size)

net = NeuralNet((2,4,1), CrossEntropyCost, SigmoidActivation, SigmoidActivation)
net.initialize_variables_normalized()
learning_rate = 0.1
batch_size = 10
lmda = 0
epochs = 1001
reporting_rate = 200
training_cost, valid_cost =     net.SGD(train_x, train_y, test_x, test_y, learning_rate, epochs, reporting_rate, lmda, batch_size)

net = NeuralNet((2,4,1), CrossEntropyCost, ReluActivation, SigmoidActivation)
net.initialize_variables()
learning_rate = 0.05
batch_size = 10
lmda = 0
epochs = 1001
reporting_rate = 200
training_cost, valid_cost =     net.SGD(train_x, train_y, test_x, test_y, learning_rate, epochs, reporting_rate, lmda, batch_size)

net = NeuralNet((2,4,1), CrossEntropyCost, ReluActivation, SigmoidActivation)
net.initialize_variables_normalized()
learning_rate = 0.05
batch_size = 10
lmda = 0
epochs = 1001
reporting_rate = 200
training_cost, valid_cost =     net.SGD(train_x, train_y, test_x, test_y, learning_rate, epochs, reporting_rate, lmda, batch_size)

net = NeuralNet((2,4,1), CrossEntropyCost, ReluActivation, SigmoidActivation)
net.initialize_variables_alt()
learning_rate = 0.05
batch_size = 10
lmda = 0
epochs = 1001
reporting_rate = 200
training_cost, valid_cost =     net.SGD(train_x, train_y, test_x, test_y, learning_rate, epochs, reporting_rate, lmda, batch_size)

def network_predictions(net, train_x, train_y, test_x, test_y, learning_rate, epochs, reporting_rate):
    train_results = []
    test_results = []
    training_cost_all = []
    valid_cost_all = []
    
    i = 6
    while(i>0):
        training_cost, valid_cost = net.SGD(train_x, train_y, test_x, test_y, learning_rate, epochs, reporting_rate)
        print()
        training_cost_all.extend(training_cost)
        valid_cost_all.extend(valid_cost)

        prediction_train = net.predict(train_x)
        train_result = np.hstack((train_x, prediction_train))
        train_results.append(train_result)

        prediction_test = net.predict(test_x)
        test_result = np.hstack((test_x, prediction_test))
        test_results.append(test_result)
        i -= 1

    return train_results, test_results, training_cost_all, valid_cost_all
    
def plot_subfig(ax, data, title):
    positives = data[data[:,2]==1]
    negatives = data[data[:,2]==0]
    ax.scatter(positives[:,0], positives[:,1], s=20, alpha=0.5, c='r', marker="o", label='positive')
    ax.scatter(negatives[:,0], negatives[:,1], s=20, alpha=0.5, c='b', marker="o", label='negative')
    ax.legend(loc='upper left')
    ax.set_title(title)

def plot_subplotpair(fig, train_result, test_result, training_title, test_title):
    ax1 = fig.add_subplot(221)
    plot_subfig(ax1, train_result, training_title)
    ax2 = fig.add_subplot(222)
    plot_subfig(ax2, test_result, test_title)
    
def plot_all(num_results, train_results, test_results, training_titles, test_titles, name):
    for i in range(6):
        fig = plt.figure(figsize=(12,12))
        newname = name + '-' + str(i) +'.png'
        plot_subplotpair(fig, train_results[i], test_results[i], training_titles[i], test_titles[i])
        plt.savefig(newname, dpi=200)
    plt.show()

net = NeuralNet((2,4,1), QuadraticCost, SigmoidActivation, SigmoidActivation)
net.initialize_variables()
learning_rate = 0.1
batch_size = 10
lmda = 0
epochs = 160
reporting_rate = 200

train_results, test_results, training_cost_all, valid_cost_all =                 network_predictions(net, train_x, train_y, test_x, test_y, learning_rate, epochs, reporting_rate)
training_titles = ["Training data: epoch 160","Training data: epoch 320","Training data: epoch 480",                  "Training data: epoch 640","Training data: epoch 800","Training data: epoch 960"]
test_titles = ["Test data: epoch 160","Test data: epoch 320","Test data: epoch 480",                  "Test data: epoch 640","Test data: epoch 800","Test data: epoch 960"]
name = 'DonutBall_slowlearning'
plot_all(6, train_results, test_results, training_titles, test_titles, name)

net = NeuralNet((2,4,1), CrossEntropyCost, SigmoidActivation, SigmoidActivation)
net.initialize_variables_normalized()
learning_rate = 0.1
batch_size = 10
lmda = 0
epochs = 25
reporting_rate = 200

train_results, test_results, training_cost_all, valid_cost_all =                 network_predictions(net, train_x, train_y, test_x, test_y, learning_rate, epochs, reporting_rate)
training_titles = ["Training data: epoch 25","Training data: epoch 50","Training data: epoch 75",                  "Training data: epoch 100","Training data: epoch 125","Training data: epoch 150"]
test_titles = ["Test data: epoch 25","Test data: epoch 50","Test data: epoch 75",                  "Test data: epoch 100","Test data: epoch 125","Test data: epoch 150"]
name = 'DonutBall_fastlearning'
plot_all(6, train_results, test_results, training_titles, test_titles, name)



