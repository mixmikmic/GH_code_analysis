get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import load_iris

# a helper function for plotting
def scatter_plot(data, labels, legends, title, x_label, y_label):
    markers = ['o', '^', 'x']
    colors = ['r', 'g', 'b']
    
    for target in np.unique(labels):
        marker = markers[target]
        color = colors[target]
        legend = legends[target]
        
        x = data[labels == target, 0]
        y = data[labels == target, 1]
    
        plt.scatter(x, y, marker = marker, 
                    color = color, label = legend)
    
    plt.title(title)
    plt.legend(loc = 'upper left')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

# normalize the data to zero mean and unit standard deviation
def normalize(data):
    return (data - data.mean()) / data.std()

# load iris data
iris = load_iris()
# take only two out of four features available
feature_indices = [1, 3]
# only take samples from classes 0 and 1
indices = np.logical_or(iris.target == 0, iris.target == 1)
# get only 2 features from the samples of the classes 0 and 1
linear_iris_data = iris.data[indices][:, feature_indices]
# normalize the data
linear_iris_data[:, 0] = normalize(linear_iris_data[:, 0])
linear_iris_data[:, 1] = normalize(linear_iris_data[:, 1])
# get targets (labels)
linear_iris_target = iris.target[indices]
# plot the data
scatter_plot(linear_iris_data, linear_iris_target,
            legends = iris.target_names, 
            title = 'Iris dataset',
            x_label = iris.feature_names[feature_indices[0]], 
            y_label = iris.feature_names[feature_indices[1]])

import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    # a placeholder for inputs, shape = (N_samples, n_features)
    x = tf.placeholder(tf.float32, shape = [None, 2])
    
    # weight matrix with shape (n_inputs, n_outputs)
    W = tf.Variable(
        initial_value = np.identity(2, dtype = np.float32))
    # bias vector with shape (n_outputs)
    b = tf.Variable(tf.zeros([2]))
    
    # linear model
    y_hat = tf.matmul(x, W) + b
    
with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    predictions = session.run(y_hat, 
                              feed_dict = {x: linear_iris_data})
    print(predictions[:10])

graph = tf.Graph()
with graph.as_default():
    # a placeholder for inputs, shape = (N_samples, n_features)
    x = tf.placeholder(tf.float32, shape = [None, 2])
    # placeholder for expected outputs, shape = (N_samples, n_classes)
    
    # weight matrix with shape (n_inputs, n_outputs)
    W = tf.Variable(
        initial_value = np.identity(2, dtype = np.float32))
    # bias vector with shape (n_outputs)
    b = tf.Variable(tf.zeros([2]))
    
    # linear model
    logits = tf.matmul(x, W) + b
    y_hat = tf.nn.softmax(logits)
    
    
with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    probabilities = session.run(y_hat, 
                              feed_dict = {x: linear_iris_data})
    predictions = session.run(tf.argmax(probabilities, 1))
    predictions = predictions.reshape(len(predictions), 1)
    print(np.hstack((probabilities, predictions))[:10])

def plot_decision_region(data, labels, model, session, 
                         resolution = 0.02):
    # setup marker and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(labels))])

    # plot the decision surface
    (x1_min, x1_max) = (data[:, 0].min() - 1, data[:, 0].max() + 1)
    (x2_min, x2_max) = (data[:, 1].min() - 1, data[:, 1].max() + 1)
    (xx1, xx2) = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution))

    model_x = np.array([xx1.ravel(), xx2.ravel()]).T
    Z = session.run(model, feed_dict = {x: model_x})
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for (idx, cl) in enumerate(np.unique(labels)):
        plt.scatter(x = data[labels == cl, 0], 
                    y = data[labels == cl, 1],
                    alpha = 0.8, c = cmap(idx),
                    marker = markers[idx], label = cl)

with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    model = tf.argmax(y_hat, 1)
    plot_decision_region(linear_iris_data, linear_iris_target,
                        model, session)

def target_to_one_hot_vector(labels):
    unique_labels = np.unique(labels)
    vectors = np.zeros((len(labels), len(unique_labels)))
    for (i, label) in enumerate(labels):
        vectors[i, label] = 1.0
        
    return vectors

graph = tf.Graph()
with graph.as_default():
    # a placeholder for inputs, shape = (N_samples, n_features)
    x = tf.placeholder(tf.float32, shape = [None, 2])
    # placeholder for expected outputs, shape = (N_samples, n_classes)
    y = tf.placeholder(tf.float32, shape = [None, 2])
    
    # weight matrix with shape (n_inputs, n_outputs)
    W = tf.Variable(
        initial_value = np.identity(2, dtype = np.float32))
    # bias vector with shape (n_outputs)
    b = tf.Variable(tf.zeros([2]))
    
    # linear model
    logits = tf.matmul(x, W) + b
    y_hat = tf.nn.softmax(logits)
    
    # cross-entropy loss using one-hot vectors
    cross_entropy_loss = tf.reduce_mean(
        -tf.reduce_sum(y * tf.log(y_hat), 
                       reduction_indices = [1]))
    
    # a fixed learning rate
    learning_rate = 0.5
    # optimize the cross-entropy loss using Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = optimizer.minimize(cross_entropy_loss)
    
with tf.Session(graph = graph) as session:
    # Initialize variables
    tf.global_variables_initializer().run()
    
    one_hot_vectors = target_to_one_hot_vector(linear_iris_target)
    
    # set a number of iterations to run the optimization procedure
    max_iters = 100
    for iteration in range(max_iters):
        _, iter_loss = session.run([optimizer, cross_entropy_loss], 
                              feed_dict = {x: linear_iris_data,
                                          y: one_hot_vectors})
        # log model loss every 10 iterations to view model's convergence
        if (iteration % 10) == 0:
            print('Iteration {}, loss = {}'.format(
                iteration, iter_loss))
            
    model = tf.argmax(y_hat, 1)
    plot_decision_region(linear_iris_data, linear_iris_target,
                        model, session)

# take only two out of four features available
feature_indices = [1, 2]
# get only 2 features
nonlinear_iris_data = iris.data[:, feature_indices]
# normalize the data
nonlinear_iris_data[:, 0] = normalize(nonlinear_iris_data[:, 0])
nonlinear_iris_data[:, 1] = normalize(nonlinear_iris_data[:, 1])
# get targets (labels)
nonlinear_iris_target = iris.target
# plot the data
scatter_plot(nonlinear_iris_data, nonlinear_iris_target,
            legends = iris.target_names, 
            title = 'Iris dataset',
            x_label = iris.feature_names[feature_indices[0]], 
            y_label = iris.feature_names[feature_indices[1]])

graph = tf.Graph()
with graph.as_default():
    # a placeholder for inputs, shape = (N_samples, n_features)
    x = tf.placeholder(tf.float32, shape = [None, 2])
    # placeholder for expected outputs, shape = (N_samples, n_classes)
    y = tf.placeholder(tf.float32, shape = [None, 3])
    
    # weight matrix with shape (n_inputs, n_outputs)
    W = tf.Variable(tf.ones([2, 3]))
    # bias vector with shape (n_outputs)
    b = tf.Variable(tf.zeros([3]))
    
    # linear model
    logits = tf.matmul(x, W) + b
    y_hat = tf.nn.softmax(logits)
    
    # cross-entropy loss using one-hot vectors
    cross_entropy_loss = tf.reduce_mean(
        -tf.reduce_sum(y * tf.log(y_hat), 
                       reduction_indices = [1]))
    
    # a fixed learning rate
    learning_rate = 0.5
    # optimize the cross-entropy loss using Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = optimizer.minimize(cross_entropy_loss)
    
with tf.Session(graph = graph) as session:
    # Initialize variables
    tf.global_variables_initializer().run()
    
    one_hot_vectors = target_to_one_hot_vector(nonlinear_iris_target)
    
    # set a number of iterations to run the optimization procedure
    max_iters = 100
    for iteration in range(max_iters):
        _, iter_loss = session.run([optimizer, cross_entropy_loss], 
                              feed_dict = {x: nonlinear_iris_data,
                                          y: one_hot_vectors})
        if (iteration % 10) == 0:
            print('Iteration {}, loss = {}'.format(
                iteration, iter_loss))
    
    # plot decision boundary
    model = tf.argmax(y_hat, 1)
    plot_decision_region(nonlinear_iris_data, nonlinear_iris_target,
                        model, session)
    
    # compute accuracy
    correct_predictions = tf.equal(
        tf.argmax(y_hat, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    print('Model accuracy {}'.format(
        accuracy.eval(feed_dict = {x: nonlinear_iris_data,
                                    y: one_hot_vectors})))

graph = tf.Graph()
with graph.as_default():
    # a placeholder for inputs, shape = (N_samples, n_features)
    x = tf.placeholder(tf.float32, shape = [None, 2])
    # placeholder for expected outputs, shape = (N_samples, n_classes)
    y = tf.placeholder(tf.float32, shape = [None, 3])
    
    # weight matrix for layer 1
    W_1 = tf.Variable(tf.ones([2, 5]))
    # bias vector for layer 1
    b_1 = tf.Variable(tf.zeros([5]))
    # layer 1
    h_1 = tf.matmul(x, W_1) + b_1
    
    # weight matrix for layer 2
    W_2 = tf.Variable(tf.ones([5, 3]))
    # bias vector for layer 2
    b_2 = tf.Variable(tf.zeros([3]))
    
    # linear model
    logits = tf.matmul(h_1, W_2) + b_2
    y_hat = tf.nn.softmax(logits)
    
    # cross-entropy loss using one-hot vectors
    cross_entropy_loss = tf.reduce_mean(
        -tf.reduce_sum(y * tf.log(y_hat), 
                       reduction_indices = [1]))
    
    # a fixed learning rate
    learning_rate = 0.5
    # optimize the cross-entropy loss using Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = optimizer.minimize(cross_entropy_loss)
    
with tf.Session(graph = graph) as session:
    # Initialize variables
    tf.global_variables_initializer().run()
    
    one_hot_vectors = target_to_one_hot_vector(nonlinear_iris_target)
    
    # set a number of iterations to run the optimization procedure
    max_iters = 1000
    for iteration in range(max_iters):
        _, iter_loss = session.run([optimizer, cross_entropy_loss], 
                              feed_dict = {x: nonlinear_iris_data,
                                          y: one_hot_vectors})
        if (iteration % 100) == 0:
            print('Iteration {}, loss = {}'.format(
                iteration, iter_loss))
            
    model = tf.argmax(y_hat, 1)
    plot_decision_region(nonlinear_iris_data, nonlinear_iris_target,
                        model, session)
    
    correct_predictions = tf.equal(
        tf.argmax(y_hat, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    print('Model accuracy {}'.format(
        accuracy.eval(feed_dict = {x: nonlinear_iris_data,
                                    y: one_hot_vectors})))

graph = tf.Graph()
with graph.as_default():
    # a placeholder for inputs, shape = (N_samples, n_features)
    x = tf.placeholder(tf.float32, shape = [None, 2])
    # placeholder for expected outputs, shape = (N_samples, n_classes)
    y = tf.placeholder(tf.float32, shape = [None, 3])
    
    # weight matrix for layer 1
    W_1 = tf.Variable(tf.ones([2, 5]))
    # bias vector for layer 1
    b_1 = tf.Variable(tf.zeros([5]))
    # layer 1 with a ReLU non linearity
    h_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)
    
    # weight matrix for layer 2
    W_2 = tf.Variable(tf.ones([5, 3]))
    # bias vector for layer 2
    b_2 = tf.Variable(tf.zeros([3]))
    # layer 2
    h_2 = tf.matmul(h_1, W_2) + b_2
    
    y_hat = tf.nn.softmax(h_2)
    
    # cross-entropy loss using one-hot vectors
    cross_entropy_loss = tf.reduce_mean(
        -tf.reduce_sum(y * tf.log(y_hat), 
                       reduction_indices = [1]))
    
    # a fixed learning rate
    learning_rate = 0.5
    # optimize the cross-entropy loss using Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = optimizer.minimize(cross_entropy_loss)
    
with tf.Session(graph = graph) as session:
    # Initialize variables
    tf.global_variables_initializer().run()
    
    one_hot_vectors = target_to_one_hot_vector(nonlinear_iris_target)
    
    # set a number of iterations to run the optimization procedure
    max_iters = 1000
    for iteration in range(max_iters):
        _, iter_loss = session.run([optimizer, cross_entropy_loss], 
                              feed_dict = {x: nonlinear_iris_data,
                                          y: one_hot_vectors})
        if (iteration % 100) == 0:
            print('Iteration {}, loss = {}'.format(
                iteration, iter_loss))
            
    model = tf.argmax(y_hat, 1)
    plot_decision_region(nonlinear_iris_data, nonlinear_iris_target,
                        model, session)
    
    correct_predictions = tf.equal(
        tf.argmax(y_hat, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    print('Model accuracy {}'.format(
        accuracy.eval(feed_dict = {x: nonlinear_iris_data,
                                    y: one_hot_vectors})))



