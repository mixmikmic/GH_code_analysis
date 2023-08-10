import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Load the data and shuffle
import  mnist_loader3 as mnl
train, val, test = mnl.load_data_wrapper()
np.random.shuffle(train)
np.random.shuffle(test)

import network_simple as network  # Implementation of our simple network

# Initialize and train the neural network
nn = network.Network([784, 30, 10])
iters = 20
train_cost, test_accuracy = nn.SGD(train, iters=iters, eta=0.05, test_data=test)
# Print the results
n_test = len(test)
for i in range(iters):
    print('Iter %*d:\t\t train cost = %f\t\t test accuracy = %d / %d' %
          (2, i+1, train_cost[i], test_accuracy[i], n_test))

# Plot train cost and test accuracy
f, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
f.tight_layout(pad=1.0, w_pad=4.0, h_pad=1.0)
ax1.plot(range(1, iters+1), train_cost, '-o')
ax1.set_xlabel('Iteration #')
ax1.set_ylabel('Train Cost (Mean Square Error)')
ax1.set_xlim((1,iters))
ax2.plot(range(1, iters+1), test_accuracy, '-o')
ax2.set_xlabel('Iteration #')
ax2.set_ylabel('Test Accuracy')
ax2.set_xlim((1,iters))
ax2.set_ylim((8000, 9800))
#f.show()

import network_simple2 as network2

# Initialize and train the neural network
nn = network2.Network([784, 30, 10])
iters = 20
train_cost, test_accuracy = nn.SGD(train, iters=iters, eta=0.05, test_data=test)
# Print the results
n_test = len(test)
for i in range(iters):
    print('Iter %*d:\t\t train cost = %f\t\t test accuracy = %d / %d' %
          (2, i+1, train_cost[i], test_accuracy[i], n_test))

# Plot train cost and test accuracy
f, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
f.tight_layout(pad=1.0, w_pad=4.0, h_pad=1.0)
ax1.plot(range(1, iters+1), train_cost, '-o')
ax1.set_xlabel('Iteration #')
ax1.set_ylabel('Train Cost (Cross-Entropy)')
ax1.set_xlim((1,iters))
ax2.plot(range(1, iters+1), test_accuracy, '-o')
ax2.set_xlabel('Iteration #')
ax2.set_ylabel('Test Accuracy')
ax2.set_xlim((1,iters))
ax2.set_ylim((8000, 9800))

# Initialize and train the neural network
nn = network2.Network([784, 30, 10])
iters = 20
train_cost, test_accuracy = nn.SGD(train, iters=iters, eta=0.05, lmbda=5.0, test_data=test)
# Print the results
n_test = len(test)
for i in range(iters):
    print('Iter %*d:\t\t train cost = %f\t\t test accuracy = %d / %d' %
          (2, i+1, train_cost[i], test_accuracy[i], n_test))

# Plot train cost and test accuracy
f, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
f.tight_layout(pad=1.0, w_pad=4.0, h_pad=1.0)
ax1.plot(range(1, iters+1), train_cost, '-o')
ax1.set_xlabel('Iteration #')
ax1.set_ylabel('Train Cost (Cross-Entropy Error + Regularization)')
ax1.set_xlim((1,iters))
ax2.plot(range(1, iters+1), test_accuracy, '-o')
ax2.set_xlabel('Iteration #')
ax2.set_ylabel('Test Accuracy')
ax2.set_xlim((1,iters))
ax2.set_ylim((8000, 9800))

import network_simple3 as network3

# Initialize and train the neural network
nn = network3.Network([784, 30, 10])
iters = 20
train_cost, test_accuracy = nn.SGD(train, iters=iters, eta=0.05, lmbda=5.0, batch_size=10, test_data=test)
# Print the results
n_test = len(test)
for i in range(iters):
    print('Iter %*d:\t\t train cost = %f\t\t test accuracy = %d / %d' %
          (2, i+1, train_cost[i], test_accuracy[i], n_test))

# Plot train cost and test accuracy
f, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
f.tight_layout(pad=1.0, w_pad=4.0, h_pad=1.0)
ax1.plot(range(1, iters+1), train_cost, '-o')
ax1.set_xlabel('Iteration #')
ax1.set_ylabel('Train Cost (Cross-Entropy Error + Regularization)')
ax1.set_xlim((1,iters))
ax2.plot(range(1, iters+1), test_accuracy, '-o')
ax2.set_xlabel('Iteration #')
ax2.set_ylabel('Test Accuracy')
ax2.set_xlim((1,iters))
ax2.set_ylim((8000, 9800))

get_ipython().run_cell_magic('timeit', 'nn = network3.Network([784, 30, 10]); iters = 5', 'train_cost, test_accuracy = nn.SGD(train, iters=iters, eta=0.05, lmbda=5.0)')

get_ipython().run_cell_magic('timeit', 'nn = network3.Network([784, 30, 10]); iters = 5', 'train_cost, test_accuracy = nn.SGD(train, iters=iters, eta=0.05, lmbda=5.0, batch_size=10)')

# Initialize and train the neural network
nn = network3.Network([784, 30, 10], tight_weights=True)
iters = 20
train_cost, test_accuracy = nn.SGD(train, iters=iters, eta=0.05, lmbda=5.0, batch_size=10, test_data=test)
# Print the results
n_test = len(test)
for i in range(iters):
    print('Iter %*d:\t\t train cost = %f\t\t test accuracy = %d / %d' %
          (2, i+1, train_cost[i], test_accuracy[i], n_test))

# Plot train cost and test accuracy
f, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
f.tight_layout(pad=1.0, w_pad=4.0, h_pad=1.0)
ax1.plot(range(1, iters+1), train_cost, '-o')
ax1.set_xlabel('Iteration #')
ax1.set_ylabel('Train Cost (Cross-Entropy Error + Regularization)')
ax1.set_xlim((1,iters))
ax2.plot(range(1, iters+1), test_accuracy, '-o')
ax2.set_xlabel('Iteration #')
ax2.set_ylabel('Test Accuracy')
ax2.set_xlim((1,iters))
ax2.set_ylim((8000, 9800))

# Initialize and train the neural network
nn = network3.Network([784, 100, 10], tight_weights=True)
iters = 70
train_cost, test_accuracy = nn.SGD(train, iters=iters, eta=0.05, lmbda=5.0, batch_size=10, test_data=test)
# Print the results
n_test = len(test)
for i in range(iters):
    print('Iter %*d:\t\t train cost = %f\t\t test accuracy = %d / %d' %
          (2, i+1, train_cost[i], test_accuracy[i], n_test))

# Plot train cost and test accuracy
f, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
f.tight_layout(pad=1.0, w_pad=4.0, h_pad=1.0)
ax1.plot(range(1, iters+1), train_cost, '-o')
ax1.set_xlabel('Iteration #')
ax1.set_ylabel('Train Cost (Cross-Entropy Error + Regularization)')
ax1.set_xlim((1,iters))
ax2.plot(range(1, iters+1), test_accuracy, '-o')
ax2.set_xlabel('Iteration #')
ax2.set_ylabel('Test Accuracy')
ax2.set_xlim((1,iters))
ax2.set_ylim((8000, 9800))

# Initialize and train the neural network
nn = network3.Network([784, 100, 100, 10], tight_weights=True)
iters = 200
train_cost, test_accuracy = nn.SGD(train, iters=iters, eta=0.05, lmbda=5.0, batch_size=10, test_data=test, 
                                   verbose=True)
# Print the results
n_test = len(test)
for i in range(iters):
    print('Iter %*d:\t\t train cost = %f\t\t test accuracy = %d / %d' %
          (2, i+1, train_cost[i], test_accuracy[i], n_test))

# Plot train cost and test accuracy
f, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
f.tight_layout(pad=1.0, w_pad=4.0, h_pad=1.0)
ax1.plot(range(1, iters+1), train_cost, '-o')
ax1.set_xlabel('Iteration #')
ax1.set_ylabel('Train Cost (Cross-Entropy Error + Regularization)')
ax1.set_xlim((1,iters))
ax2.plot(range(1, iters+1), test_accuracy, '-o')
ax2.set_xlabel('Iteration #')
ax2.set_ylabel('Test Accuracy')
ax2.set_xlim((1,iters))
ax2.set_ylim((8000, 9900))

import network_simple4 as network4

# Initialize and train the neural network
nn = network4.Network([784, 100, 100, 10], tight_weights=True, schedule_learning=True)
train_cost, test_accuracy = nn.SGD(train, eta_inv=10, lmbda=5.0, batch_size=10, no_improvement_stop=50, 
                                   test_data=val, verbose=True)
# Print the results
print('Max validation accuracy was: %f%%' % (max(test_accuracy)/len(val)*100))
print('Evaluation on test data  is: %f%%' % (nn.evaluate(test)/len(test)*100))

# Plot train cost and test accuracy
f, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
f.tight_layout(pad=1.0, w_pad=4.0, h_pad=1.0)
iters = len(train_cost)
ax1.plot(range(1, iters+1), train_cost, '-o')
ax1.set_xlabel('Iteration #')
ax1.set_ylabel('Train Cost (Cross-Entropy Error + Regularization)')
ax1.set_xlim((1,iters))
ax2.plot(range(1, iters+1), test_accuracy, '-o')
ax2.set_xlabel('Iteration #')
ax2.set_ylabel('Test Accuracy')
ax2.set_xlim((1,iters))
ax2.set_ylim((8000, 9900))



